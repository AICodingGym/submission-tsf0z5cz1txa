from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
REPEATED_WORD_RE = re.compile(r"\b([a-z]+)\s+\1\b")
REPEATED_CHAR_RE = re.compile(r"([a-z])\1{2,}")
LOWERCASE_I_RE = re.compile(r"\bi\b")
MISSPACED_PUNCTUATION_RE = re.compile(r"\s+[,.!?;:]")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

TRANSITION_PHRASES = (
    "for example",
    "for instance",
    "in conclusion",
    "on the other hand",
    "as a result",
    "in addition",
    "first of all",
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "additionally",
    "consequently",
    "although",
    "because",
    "nevertheless",
    "finally",
    "secondly",
)

ARGUMENT_WORDS = frozenset(
    {
        "agree",
        "disagree",
        "believe",
        "claim",
        "evidence",
        "opinion",
        "reason",
        "reasons",
        "should",
        "must",
        "argue",
        "argument",
        "support",
        "conclude",
    }
)
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "we", "us", "our", "ours"})
SECOND_PERSON = frozenset({"you", "your", "yours"})
THIRD_PERSON = frozenset(
    {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
)


FEATURE_NAMES = (
    "char_count",
    "word_count",
    "unique_word_count",
    "sentence_count",
    "paragraph_count",
    "newline_count",
    "log_char_count",
    "log_word_count",
    "avg_word_length",
    "std_word_length",
    "max_word_length",
    "type_token_ratio",
    "hapax_ratio",
    "short_word_ratio",
    "long_word_ratio",
    "very_long_word_ratio",
    "avg_sentence_words",
    "std_sentence_words",
    "max_sentence_words",
    "short_sentence_ratio",
    "long_sentence_ratio",
    "avg_paragraph_words",
    "std_paragraph_words",
    "max_paragraph_words",
    "alpha_char_ratio",
    "uppercase_alpha_ratio",
    "digit_char_ratio",
    "whitespace_char_ratio",
    "newlines_per_100_words",
    "periods_per_100_words",
    "commas_per_100_words",
    "semicolons_per_100_words",
    "colons_per_100_words",
    "questions_per_100_words",
    "exclamations_per_100_words",
    "quotes_per_100_words",
    "apostrophes_per_100_words",
    "hyphens_per_100_words",
    "parentheses_per_100_words",
    "punctuation_char_ratio",
    "terminal_punctuation_per_sentence",
    "transition_phrases_per_100_words",
    "argument_words_per_100_words",
    "stopword_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "contractions_per_100_words",
    "repeated_words_per_100_words",
    "repeated_chars_per_100_words",
    "lowercase_i_per_100_words",
    "misspaced_punctuation_per_100_words",
    "sentence_start_upper_ratio",
    "syllables_per_word",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
)


def _safe_mean(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[int]) -> float:
    return float(np.std(values)) if values else 0.0


def _estimate_syllables(word: str) -> int:
    """Cheap deterministic approximation sufficient for aggregate readability."""
    word = word.lower()
    groups = len(VOWEL_GROUP_RE.findall(word))
    if len(word) > 2 and word.endswith("e") and not word.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


def extract_essay_features(text: str) -> np.ndarray:
    """Extract length, structure, style, discourse, and readability signals."""
    words_original = WORD_RE.findall(text)
    words = [word.lower() for word in words_original]
    word_count = len(words)
    char_count = len(text)
    word_denominator = max(word_count, 1)
    char_denominator = max(char_count, 1)

    word_lengths = [len(word) for word in words]
    word_frequencies = Counter(words)
    unique_word_count = len(word_frequencies)

    sentence_strings = [
        item.strip() for item in SENTENCE_RE.findall(text) if item.strip()
    ]
    sentence_word_counts = [
        len(WORD_RE.findall(sentence)) for sentence in sentence_strings
    ]
    sentence_word_counts = [count for count in sentence_word_counts if count > 0]
    sentence_count = max(len(sentence_word_counts), 1)
    sentence_starts_upper = sum(
        sentence.lstrip()[:1].isupper()
        for sentence in sentence_strings
        if sentence.lstrip()
    )

    paragraph_strings = [item.strip() for item in PARAGRAPH_RE.split(text) if item.strip()]
    paragraph_word_counts = [len(WORD_RE.findall(item)) for item in paragraph_strings]
    paragraph_count = max(len(paragraph_word_counts), 1)

    alpha_count = sum(character.isalpha() for character in text)
    upper_count = sum(character.isupper() for character in text)
    digit_count = sum(character.isdigit() for character in text)
    whitespace_count = sum(character.isspace() for character in text)
    punctuation_count = sum(
        not character.isalnum() and not character.isspace() for character in text
    )

    lower_text = text.lower()
    transition_count = sum(lower_text.count(phrase) for phrase in TRANSITION_PHRASES)
    argument_count = sum(word in ARGUMENT_WORDS for word in words)
    stopword_count = sum(word in ENGLISH_STOP_WORDS for word in words)
    first_person_count = sum(word in FIRST_PERSON for word in words)
    second_person_count = sum(word in SECOND_PERSON for word in words)
    third_person_count = sum(word in THIRD_PERSON for word in words)

    syllable_count = sum(_estimate_syllables(word) for word in words)
    syllables_per_word = syllable_count / word_denominator
    words_per_sentence = word_count / sentence_count
    flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

    per_100_words = 100.0 / word_denominator
    features = (
        char_count,
        word_count,
        unique_word_count,
        sentence_count,
        paragraph_count,
        text.count("\n"),
        np.log1p(char_count),
        np.log1p(word_count),
        _safe_mean(word_lengths),
        _safe_std(word_lengths),
        max(word_lengths, default=0),
        unique_word_count / word_denominator,
        sum(count == 1 for count in word_frequencies.values()) / word_denominator,
        sum(length <= 3 for length in word_lengths) / word_denominator,
        sum(length >= 7 for length in word_lengths) / word_denominator,
        sum(length >= 10 for length in word_lengths) / word_denominator,
        _safe_mean(sentence_word_counts),
        _safe_std(sentence_word_counts),
        max(sentence_word_counts, default=0),
        sum(count <= 8 for count in sentence_word_counts) / sentence_count,
        sum(count >= 30 for count in sentence_word_counts) / sentence_count,
        _safe_mean(paragraph_word_counts),
        _safe_std(paragraph_word_counts),
        max(paragraph_word_counts, default=0),
        alpha_count / char_denominator,
        upper_count / max(alpha_count, 1),
        digit_count / char_denominator,
        whitespace_count / char_denominator,
        text.count("\n") * per_100_words,
        text.count(".") * per_100_words,
        text.count(",") * per_100_words,
        text.count(";") * per_100_words,
        text.count(":") * per_100_words,
        text.count("?") * per_100_words,
        text.count("!") * per_100_words,
        (text.count('"') + text.count("“") + text.count("”")) * per_100_words,
        (text.count("'") + text.count("’")) * per_100_words,
        (text.count("-") + text.count("—")) * per_100_words,
        (text.count("(") + text.count(")")) * per_100_words,
        punctuation_count / char_denominator,
        sum(text.count(mark) for mark in ".!?") / sentence_count,
        transition_count * per_100_words,
        argument_count * per_100_words,
        stopword_count / word_denominator,
        first_person_count / word_denominator,
        second_person_count / word_denominator,
        third_person_count / word_denominator,
        sum("'" in word or "’" in word for word in words_original) * per_100_words,
        len(REPEATED_WORD_RE.findall(lower_text)) * per_100_words,
        len(REPEATED_CHAR_RE.findall(lower_text)) * per_100_words,
        len(LOWERCASE_I_RE.findall(text)) * per_100_words,
        len(MISSPACED_PUNCTUATION_RE.findall(text)) * per_100_words,
        sentence_starts_upper / sentence_count,
        syllables_per_word,
        flesch_reading_ease,
        flesch_kincaid_grade,
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Engineered essay features must be finite and match FEATURE_NAMES")
    return result


def extract_feature_matrix(texts: list[str]) -> np.ndarray:
    matrix = np.vstack([extract_essay_features(text) for text in texts])
    return matrix.astype(np.float32, copy=False)


def feature_target_correlations(
    features: np.ndarray, labels: np.ndarray
) -> list[dict[str, float | str]]:
    """Return simple OOF-safe descriptive correlations, not model importance."""
    correlations: list[dict[str, float | str]] = []
    label_values = labels.astype(np.float64)
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index].astype(np.float64)
        correlation = (
            0.0
            if np.std(values) == 0
            else float(np.corrcoef(values, label_values)[0, 1])
        )
        correlations.append({"feature": name, "correlation": correlation})
    return sorted(correlations, key=lambda item: abs(float(item["correlation"])), reverse=True)
