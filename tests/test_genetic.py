"""Testovi težina i genetskog algoritma."""

import random

import numpy as np

from engine.weights import (
    DEFAULT_WEIGHTS,
    SCALE_TARGET,
    WEIGHT_FIELDS,
    load_weights,
    normalize_weights,
    save_weights,
    vector_to_weights,
    weights_to_vector,
)
from genetic.arena import outcome_for, play_game
from genetic.genetic_algorithm import crossover, mutate, round_robin_scores
from lib.constants import Stone


def test_vector_round_trip_preserves_order():
    vector = weights_to_vector(DEFAULT_WEIGHTS)
    assert list(vector_to_weights(vector)) == list(WEIGHT_FIELDS)
    assert vector_to_weights(vector) == {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}


def test_normalization_scales_absolute_genes_but_not_defense():
    halved = {k: v / 2 for k, v in DEFAULT_WEIGHTS.items()}
    normalized = normalize_weights(halved)
    assert max(v for k, v in normalized.items() if k != "defense") == SCALE_TARGET
    assert normalized["defense"] == halved["defense"]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "w.json"
    save_weights(DEFAULT_WEIGHTS, path)
    assert load_weights(path) == normalize_weights(DEFAULT_WEIGHTS)
    assert load_weights(tmp_path / "missing.json") == DEFAULT_WEIGHTS


def test_crossover_takes_each_gene_from_a_parent():
    rng = random.Random(0)
    a, b = np.zeros(11), np.ones(11)
    child = crossover(a, b, rng)
    assert set(child.tolist()) <= {0.0, 1.0}


def test_mutation_keeps_weights_non_negative():
    rng = random.Random(0)
    vector = weights_to_vector(DEFAULT_WEIGHTS)
    for _ in range(50):
        vector = mutate(vector, rng, rate=1.0, scale=1.0)
        assert np.all(vector >= 0)


def test_outcome_for():
    assert outcome_for(Stone.BLACK, Stone.BLACK) == 1
    assert outcome_for(Stone.BLACK, Stone.WHITE) == -1
    assert outcome_for(None, Stone.WHITE) == 0


def test_play_game_returns_result():
    result = play_game(DEFAULT_WEIGHTS, DEFAULT_WEIGHTS, depth=1, size=9, seed=1)
    assert result in (Stone.BLACK, Stone.WHITE, None)


def test_round_robin_is_zero_sum():
    rng = random.Random(0)
    population = [weights_to_vector(DEFAULT_WEIGHTS) * f for f in (1.0, 0.5, 1.5)]
    scores = round_robin_scores(population, rng, depth=1, size=9, workers=1)
    assert scores.sum() == 0
    assert np.all(np.abs(scores) <= 2 * (len(population) - 1))
