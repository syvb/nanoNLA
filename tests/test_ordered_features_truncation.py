"""Unit tests for the ordered-features RL truncation helpers (nla/schema.py).

These cover the nested-dropout signal wired into the RL trainers: the AV's
explanation is truncated to a random prefix of its features before the critic
scores / co-trains on it. Pure-Python, no GPU — runs anywhere nla.schema imports.
"""

import random

from nla.schema import (
    EXPLANATION_CLOSE,
    EXPLANATION_OPEN,
    count_complete_features,
    extract_explanation,
    normalize_explanation,
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


def test_normalize_collapses_blank_lines_to_single_newline():
    assert normalize_explanation(EXPL_BLANK) == (
        "feat one\nfeat two\nfeat three\nfeat four"
    )
    # Already-tight text is unchanged.
    assert normalize_explanation(EXPL_TIGHT) == EXPL_TIGHT


def test_truncate_keeps_first_k_and_normalizes_to_single_newline():
    assert truncate_explanation(EXPL_BLANK, 1) == "feat one"
    # Blank-line separators collapse to a single newline.
    assert truncate_explanation(EXPL_BLANK, 2) == "feat one\nfeat two"
    assert truncate_explanation(EXPL_TIGHT, 2) == "a\nb"


def test_truncate_k_beyond_count_returns_all_normalized():
    assert truncate_explanation(EXPL_BLANK, 99) == (
        "feat one\nfeat two\nfeat three\nfeat four"
    )
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


def _gen(body: str) -> str:
    """A partial AV generation: opener + body (as the model would stream it)."""
    return f"{EXPLANATION_OPEN}\n{body}"


def test_count_features_zero_until_first_newline():
    # Opener not yet emitted.
    assert count_complete_features("<expl") == 0
    # Opener + newline, no feature yet (empty in-progress line).
    assert count_complete_features(EXPLANATION_OPEN + "\n") == 0
    # feat1 still being generated (it's the in-progress last line).
    assert count_complete_features(_gen("feat one")) == 0


def test_count_features_increments_after_each_completing_newline():
    # The newline AFTER feat1 completes it.
    assert count_complete_features(_gen("feat one\n")) == 1
    # feat2 in-progress -> still 1.
    assert count_complete_features(_gen("feat one\nfeat two")) == 1
    assert count_complete_features(_gen("feat one\nfeat two\n")) == 2
    assert count_complete_features(_gen("feat one\nfeat two\nfeat three\n")) == 3


def test_count_features_ignores_blank_lines_from_drift():
    # Double newline (drift) must not inflate the count.
    assert count_complete_features(_gen("feat one\n\nfeat two\n")) == 2


def test_count_features_with_close_tag_counts_last_feature():
    text = _gen("feat one\nfeat two\n") + EXPLANATION_CLOSE
    # Closing tag means feat two is complete even without a trailing newline.
    assert count_complete_features(text) == 2


def test_count_features_stop_threshold_semantics():
    # Simulate a stop at k=2: count reaches >=2 exactly when the newline after
    # feat2 is emitted, NOT while feat2 is still streaming.
    streaming = _gen("feat one\nfeat two")     # feat2 mid-stream
    completed = _gen("feat one\nfeat two\n")   # newline after feat2
    assert count_complete_features(streaming) < 2
    assert count_complete_features(completed) >= 2


def test_force_stopped_generation_needs_close_tag_to_extract():
    # generate-K stops at the newline completing feature K, BEFORE the model
    # emits </explanation>. extract_explanation requires the close tag, so the
    # trainer appends it to the decoded text; without that, every force-stopped
    # rollout parses to None and the reward collapses to the failure sentinel.
    forced = _gen("feat one\nfeat two\n")  # no </explanation>
    assert extract_explanation(forced) is None
    assert extract_explanation(forced + EXPLANATION_CLOSE) == "feat one\nfeat two"


def test_unknown_mode_raises():
    try:
        truncate_explanations_for_reward(
            ["a\nb"], [0], max_lines=2, mode="bogus", rng=random.Random(0)
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
