from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.baseline import optimize_thresholds, quadratic_weighted_kappa
from src.transformer_data import (
    DynamicPaddingCollator,
    EssayTokenDataset,
    TokenCache,
    load_cache_metadata,
)
from src.transformer_model import (
    EssayScoringModel,
    ScoringHeadConfig,
    ordinal_positive_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one ModernBERT CV fold.")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/modernbert-base")
    )
    parser.add_argument(
        "--token-cache", type=Path, default=Path("artifacts/token_cache/modernbert")
    )
    parser.add_argument(
        "--stage2-predictions",
        type=Path,
        default=Path("artifacts/stage2_predictions.npz"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/stage3"))
    parser.add_argument("--run-name", default="modernbert_multitask_1280")
    parser.add_argument("--mode", choices=["regression", "multitask"], default="multitask")
    parser.add_argument("--max-length", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimizer_groups(
    model: EssayScoringModel,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        is_backbone = name.startswith("backbone.")
        no_decay = name.endswith(".bias") or "norm" in name.lower()
        grouped.setdefault((is_backbone, no_decay), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": encoder_lr if is_backbone else head_lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (is_backbone, no_decay), parameters in grouped.items()
    ]


@torch.no_grad()
def predict(
    model: EssayScoringModel,
    loader: DataLoader,
    device: torch.device,
    ordinal_pos_weight: torch.Tensor,
) -> dict[str, np.ndarray | float]:
    model.eval()
    indices: list[np.ndarray] = []
    raw_scores: list[np.ndarray] = []
    regression_scores: list[np.ndarray] = []
    ordinal_scores: list[np.ndarray] = []
    total_loss = 0.0
    rows = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids,
                attention_mask,
                labels=labels,
                ordinal_pos_weight=ordinal_pos_weight,
            )
        batch_size = len(input_ids)
        if labels is not None:
            total_loss += float(output["loss"]) * batch_size
            rows += batch_size
        indices.append(batch["indices"].numpy())
        raw_scores.append(output["raw_score"].float().cpu().numpy())
        regression_scores.append(
            output["regression_score"].float().cpu().numpy()
        )
        ordinal_scores.append(output["ordinal_score"].float().cpu().numpy())
    return {
        "indices": np.concatenate(indices),
        "raw": np.concatenate(raw_scores),
        "regression": np.concatenate(regression_scores),
        "ordinal": np.concatenate(ordinal_scores),
        "loss": total_loss / rows if rows else float("nan"),
    }


def save_model(model: EssayScoringModel, path: Path) -> None:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    save_file(state, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage-3 training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU must support BF16")
    set_seed(args.seed + args.fold)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    output_dir = args.output_root / args.run_name / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2 = np.load(args.stage2_predictions)
    labels = stage2["labels"].astype(np.int8)
    fold_ids = stage2["fold_ids"].astype(np.int8)
    train_cache = TokenCache(args.token_cache, "train")
    test_cache = TokenCache(args.token_cache, "test")
    metadata = load_cache_metadata(args.token_cache)
    if not np.array_equal(train_cache.essay_ids, stage2["train_ids"]):
        raise ValueError("Train token cache IDs do not match stage-2 artifacts")
    if not np.array_equal(test_cache.essay_ids, stage2["test_ids"]):
        raise ValueError("Test token cache IDs do not match stage-2 artifacts")

    train_indices = np.flatnonzero(fold_ids != args.fold)
    valid_indices = np.flatnonzero(fold_ids == args.fold)
    if args.max_train_rows is not None:
        rng = np.random.default_rng(args.seed + args.fold)
        train_indices = np.sort(
            rng.choice(
                train_indices,
                size=min(args.max_train_rows, len(train_indices)),
                replace=False,
            )
        )

    collator = DynamicPaddingCollator(
        int(metadata["pad_token_id"]), args.max_length
    )
    train_loader = DataLoader(
        EssayTokenDataset(train_cache, train_indices, labels),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        EssayTokenDataset(train_cache, valid_indices, labels),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        EssayTokenDataset(test_cache, np.arange(len(test_cache))),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    head_config = ScoringHeadConfig(mode=args.mode)
    model = EssayScoringModel(args.model_path, head_config).to(device)
    training_label_tensor = torch.as_tensor(labels[train_indices], device=device)
    ordinal_pos_weight = ordinal_positive_weights(training_label_tensor)
    optimizer = AdamW(
        optimizer_groups(
            model, args.encoder_lr, args.head_lr, args.weight_decay
        )
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    run_config = {
        **vars(args),
        "model_path": str(args.model_path),
        "token_cache": str(args.token_cache),
        "stage2_predictions": str(args.stage2_predictions),
        "output_root": str(args.output_root),
        "head_config": asdict(head_config),
        "ordinal_pos_weight": ordinal_pos_weight.cpu().tolist(),
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "optimizer_steps": total_steps,
        "model_sha256": metadata["model_sha256"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "torch_version": torch.__version__,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    history: list[dict[str, float | int | list[float]]] = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint = output_dir / "best_model.safetensors"
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_rows = 0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids,
                    attention_mask,
                    labels=batch_labels,
                    ordinal_pos_weight=ordinal_pos_weight,
                )
                loss = output["loss"] / args.gradient_accumulation
            loss.backward()
            running_loss += float(output["loss"].detach()) * len(input_ids)
            seen_rows += len(input_ids)
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation = predict(model, valid_loader, device, ordinal_pos_weight)
        valid_raw = np.asarray(validation["raw"])
        valid_labels = labels[np.asarray(validation["indices"], dtype=np.int64)]
        rounded = np.clip(np.rint(valid_raw), 1, 6).astype(np.int8)
        rounded_qwk = quadratic_weighted_kappa(valid_labels, rounded)
        thresholds, optimized_qwk = optimize_thresholds(valid_labels, valid_raw)
        epoch_metrics: dict[str, float | int | list[float]] = {
            "epoch": epoch,
            "train_loss": running_loss / seen_rows,
            "valid_loss": float(validation["loss"]),
            "rounded_qwk": rounded_qwk,
            "optimized_qwk": optimized_qwk,
            "reporting_thresholds": thresholds.tolist(),
            "max_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if float(validation["loss"]) < best_loss - 1e-5:
            best_loss = float(validation["loss"])
            epochs_without_improvement = 0
            save_model(model, checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > args.patience:
                break

    model.load_state_dict(load_file(checkpoint, device=str(device)))
    validation = predict(model, valid_loader, device, ordinal_pos_weight)
    test_prediction = predict(model, test_loader, device, ordinal_pos_weight)
    valid_order = np.argsort(np.asarray(validation["indices"]))
    test_order = np.argsort(np.asarray(test_prediction["indices"]))
    np.savez_compressed(
        output_dir / "predictions.npz",
        valid_indices=np.asarray(validation["indices"])[valid_order],
        valid_raw=np.asarray(validation["raw"])[valid_order],
        valid_regression=np.asarray(validation["regression"])[valid_order],
        valid_ordinal=np.asarray(validation["ordinal"])[valid_order],
        test_indices=np.asarray(test_prediction["indices"])[test_order],
        test_raw=np.asarray(test_prediction["raw"])[test_order],
        test_regression=np.asarray(test_prediction["regression"])[test_order],
        test_ordinal=np.asarray(test_prediction["ordinal"])[test_order],
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
