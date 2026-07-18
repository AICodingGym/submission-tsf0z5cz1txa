from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


WHITESPACE_RE = re.compile(r"\s+")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        if parent != item:
            self.parent[item] = self.find(parent)
        return int(self.parent[item])

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_essay(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.lower()).strip()


def detect_duplicate_groups(
    texts: list[str], near_duplicate_threshold: float = 0.92
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Group exact and highly similar essays so they never cross folds."""
    normalized = [normalize_essay(text) for text in texts]
    union_find = UnionFind(len(texts))
    hashes: dict[str, int] = {}
    exact_pairs = 0
    for index, text in enumerate(normalized):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in hashes:
            union_find.union(index, hashes[digest])
            exact_pairs += 1
        else:
            hashes[digest] = index

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(normalized)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix, return_distance=True)
    near_pairs = 0
    for index in range(len(texts)):
        neighbor = int(indices[index, 1])
        similarity = 1.0 - float(distances[index, 1])
        if similarity >= near_duplicate_threshold:
            if union_find.find(index) != union_find.find(neighbor):
                near_pairs += 1
            union_find.union(index, neighbor)

    roots = [union_find.find(index) for index in range(len(texts))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    groups = np.asarray([root_to_group[root] for root in roots], dtype=np.int32)
    group_sizes = Counter(map(int, groups))
    audit: dict[str, int | float] = {
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_links": near_pairs,
        "duplicate_groups": sum(size > 1 for size in group_sizes.values()),
        "essays_in_duplicate_groups": sum(
            size for size in group_sizes.values() if size > 1
        ),
        "largest_duplicate_group": max(group_sizes.values()),
    }
    return groups, audit


def infer_topic_clusters(
    texts: list[str], seed: int = 42, n_clusters: int = 8
) -> np.ndarray:
    """Create coarse topic labels for fold-distribution auditing only."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        strip_accents="unicode",
        min_df=3,
        max_df=0.995,
        max_features=40_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    components = min(48, matrix.shape[1] - 1)
    reduced = TruncatedSVD(n_components=components, random_state=seed).fit_transform(
        matrix
    )
    reduced = normalize(reduced)
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        batch_size=1_024,
    )
    return clusterer.fit_predict(reduced).astype(np.int8)


def build_fixed_folds(
    labels: np.ndarray,
    duplicate_groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    placeholder = np.zeros(len(labels), dtype=np.int8)
    for fold, (_, valid_index) in enumerate(
        splitter.split(placeholder, labels, duplicate_groups)
    ):
        fold_ids[valid_index] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("Every essay must be assigned to exactly one fold")
    for group in np.unique(duplicate_groups):
        if len(np.unique(fold_ids[duplicate_groups == group])) != 1:
            raise RuntimeError("A duplicate group was split across folds")
    return fold_ids


def write_fold_manifest(
    path: Path,
    essay_ids: list[str],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    duplicate_groups: np.ndarray,
    topic_clusters: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["essay_id", "score", "fold", "duplicate_group", "topic_cluster"]
        )
        writer.writerows(
            zip(
                essay_ids,
                labels,
                fold_ids,
                duplicate_groups,
                topic_clusters,
                strict=True,
            )
        )


def load_fold_manifest(path: Path, essay_ids: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    manifest = {row["essay_id"]: int(row["fold"]) for row in rows}
    if len(manifest) != len(rows):
        raise ValueError("Duplicate essay IDs found in fold manifest")
    if set(manifest) != set(essay_ids):
        missing = len(set(essay_ids) - set(manifest))
        extra = len(set(manifest) - set(essay_ids))
        raise ValueError(f"Fold manifest ID mismatch: missing={missing}, extra={extra}")
    fold_ids = np.asarray([manifest[essay_id] for essay_id in essay_ids], dtype=np.int8)
    if np.any((fold_ids < 0) | (fold_ids > 4)):
        raise ValueError("Fold IDs must be between 0 and 4")
    return fold_ids


def build_fold_audit(
    labels: np.ndarray,
    fold_ids: np.ndarray,
    duplicate_groups: np.ndarray,
    topic_clusters: np.ndarray,
    duplicate_audit: dict[str, int | float],
) -> dict[str, object]:
    folds: dict[str, object] = {}
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        folds[str(int(fold))] = {
            "rows": int(mask.sum()),
            "scores": dict(sorted(Counter(map(int, labels[mask])).items())),
            "topics": dict(
                sorted(Counter(map(int, topic_clusters[mask])).items())
            ),
        }
    group_to_folds: dict[int, set[int]] = defaultdict(set)
    for group, fold in zip(duplicate_groups, fold_ids, strict=True):
        group_to_folds[int(group)].add(int(fold))
    return {
        "duplicate_audit": duplicate_audit,
        "topic_distribution": dict(
            sorted(Counter(map(int, topic_clusters)).items())
        ),
        "folds": folds,
        "duplicate_groups_split_across_folds": sum(
            len(values) > 1 for values in group_to_folds.values()
        ),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
