"""Unit tests for the ordered-features RL truncation helpers (nla/schema.py).

These cover the nested-dropout signal wired into the RL trainers: the AV's
explanation is truncated to a random prefix of its features before the critic
scores / co-trains on it. Pure-Python, no GPU — runs anywhere nla.schema imports.
"""

import random

from nla.schema import (
    split_features,
    truncate_explanation,
    truncate_explanations_for_reward,
)

# Real explanations come with blank lines between features ("only one newline"
# was the instruction, but the model emits double newlines in practice).
EXPL_BLANK = "feat one\n\nfeat two\n\nfeat three\n\nfeat four"
EXPL_TIGHT = "a\nb\nc\nd"


def test_split_features_ignores_blank_lines():
    assert split_features(EXPL_BLANK) == [
        "feat one", "feat two", "feat three", "feat four",
    ]
    assert split_features(EXPL_TIGHT) == ["a", "b", "c", "d"]


def test_truncate_keeps_first_k_and_preserves_separators():
    assert truncate_explanation(EXPL_BLANK, 1) == "feat one"
    assert truncate_explanation(EXPL_BLANK, 2) == "feat one\n\nfeat two"
    # Tight (single-newline) format preserved too.
    assert truncate_explanation(EXPL_TIGHT, 2) == "a\nb"


def test_truncate_k_beyond_count_returns_all():
    assert truncate_explanation(EXPL_BLANK, 99) == EXPL_BLANK
    assert truncate_explanation(EXPL_TIGHT, 4) == EXPL_TIGHT


def test_truncate_clamps_k_below_one():
    assert truncate_explanation(EXPL_BLANK, 0) == "feat one"
    assert truncate_explanation(EXPL_BLANK, -5) == "feat one"


def test_per_group_uses_one_k_per_group():
    expls = [EXPL_TIGHT] * 6
    groups = [0, 0, 0, 1, 1, 1]
    out = truncate_explanations_for_reward(
        expls, groups, max_lines=4, mode="per-group", rng=random.Random(0)
    )
    # Within a group every sample is truncated identically (same prefix length).
    assert out[0] == out[1] == out[2]
    assert out[3] == out[4] == out[5]


def test_per_sample_varies_and_stays_in_bounds():
    expls = ["\n".join("abcdefghij")] * 300  # 10 features each
    groups = list(range(300))
    out = truncate_explanations_for_reward(
        expls, groups, max_lines=10, mode="per-sample", rng=random.Random(1)
    )
    counts = {len(split_features(o)) for o in out}
    assert min(counts) >= 1
    assert max(counts) <= 10
    assert len(counts) > 1  # genuinely random, not a constant


def test_none_entries_pass_through():
    out = truncate_explanations_for_reward(
        [None, EXPL_TIGHT], [0, 0], max_lines=2, mode="per-group",
        rng=random.Random(0),
    )
    assert out[0] is None
    assert out[1] is not None


def test_deterministic_given_seed():
    expls = [EXPL_TIGHT] * 8
    groups = [0, 0, 1, 1, 2, 2, 3, 3]
    a = truncate_explanations_for_reward(
        expls, groups, max_lines=4, mode="per-group", rng=random.Random(42)
    )
    b = truncate_explanations_for_reward(
        expls, groups, max_lines=4, mode="per-group", rng=random.Random(42)
    )
    assert a == b


def test_unknown_mode_raises():
    try:
        truncate_explanations_for_reward(
            ["a\nb"], [0], max_lines=2, mode="bogus", rng=random.Random(0)
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
