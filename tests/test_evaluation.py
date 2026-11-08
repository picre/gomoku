"""Testovi funkcije evaluacije i inkrementalne evaluacije."""

import random
import re

import numpy as np
import pytest

from engine.evaluation import (
    BLACK,
    EMPTY,
    IMPORTANT_PATTERNS,
    PATTERN_CATEGORIES,
    PATTERN_CATEGORY_INDEX,
    WHITE,
    count_span,
    evaluate,
    get_spans,
)
from engine.incremental import IncrementalEvaluator
from engine.weights import DEFAULT_WEIGHTS


def regex_count_span(span, own, opponent):
    """Referentna implementacija pomoću regularnih izraza (sa preklapanjem)."""
    digits = "2" + "".join(
        "1" if v == own else "2" if v == opponent else "0" for v in span
    ) + "2"
    counts = np.zeros(len(PATTERN_CATEGORIES), dtype=np.int64)
    for i, category in enumerate(PATTERN_CATEGORIES):
        for pattern in IMPORTANT_PATTERNS[category]:
            counts[i] += len(re.findall(f"(?={pattern})", digits))
    return counts


def random_board(rng, n=15, fill=0.3):
    """Nasumična pozicija sa približno 'fill' udelom zauzetih polja."""
    arr = np.full((n, n), EMPTY, dtype=np.int8)
    for r in range(n):
        for c in range(n):
            if rng.random() < fill:
                arr[r, c] = rng.choice((BLACK, WHITE))
    return arr


def span_of(text):
    """Pravi pravac iz teksta: 'B' crni, 'W' beli, '.' prazno."""
    codes = {"B": BLACK, "W": WHITE, ".": EMPTY}
    return np.array([codes[ch] for ch in text], dtype=np.int8)


@pytest.mark.parametrize("text, category", [
    (".BBBBB.", "five"),
    (".BBBB..", "open_four"),
    ("WBBBB..", "four"),
    ("..BBB...", "open_three"),
    ("WBBB....", "three"),
])
def test_known_patterns_are_recognised(text, category):
    counts = count_span(span_of(text), BLACK, WHITE)
    assert counts[PATTERN_CATEGORY_INDEX[category]] >= 1


def test_wall_blocks_open_four():
    # četvorka uz ivicu table nije otvorena
    counts = count_span(span_of("BBBB.........."), BLACK, WHITE)
    assert counts[PATTERN_CATEGORY_INDEX["open_four"]] == 0
    assert counts[PATTERN_CATEGORY_INDEX["four"]] == 1


def test_lookup_tables_match_regex_reference():
    rng = random.Random(1)
    for _ in range(30):
        arr = random_board(rng)
        for rows, cols in get_spans(arr.shape[0]):
            span = arr[rows, cols]
            for own, opponent in ((BLACK, WHITE), (WHITE, BLACK)):
                expected = regex_count_span(span, own, opponent)
                assert np.array_equal(count_span(span, own, opponent), expected)


def test_spans_cover_rows_columns_and_long_diagonals():
    n = 15
    # 15 vrsta + 15 kolona + 2 * (2n - 1 - 2 * 4) dijagonala dužine >= 5
    assert len(get_spans(n)) == 2 * n + 2 * (2 * n - 1 - 8)


def test_evaluation_is_zero_on_empty_board_and_antisymmetric_in_sign():
    empty = np.full((15, 15), EMPTY, dtype=np.int8)
    assert evaluate(empty, BLACK, DEFAULT_WEIGHTS) == 0
    arr = empty.copy()
    arr[7, 5:8] = BLACK
    assert evaluate(arr, BLACK, DEFAULT_WEIGHTS) > 0
    assert evaluate(arr, WHITE, DEFAULT_WEIGHTS) < 0


def test_incremental_matches_full_evaluation():
    rng = random.Random(7)
    arr = random_board(rng, fill=0.15)
    evaluator = IncrementalEvaluator(arr, DEFAULT_WEIGHTS)
    start_hash = evaluator.hash
    placed = []
    for _ in range(40):
        empty_cells = list(zip(*np.nonzero(arr == EMPTY)))
        r, c = map(int, rng.choice(empty_cells))
        stone = rng.choice((BLACK, WHITE))
        evaluator.place(r, c, stone)
        placed.append((r, c))
        for own in (BLACK, WHITE):
            full = evaluate(arr.copy(), own, DEFAULT_WEIGHTS)
            assert evaluator.value(own) == pytest.approx(full)
    for r, c in reversed(placed):
        evaluator.remove(r, c)
    assert evaluator.hash == start_hash
    assert evaluator.value(BLACK) == pytest.approx(evaluate(arr, BLACK, DEFAULT_WEIGHTS))
