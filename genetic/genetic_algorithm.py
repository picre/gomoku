"""Genetski algoritam: round-robin turnir težina + prolaz najboljih.

Svaka jedinka je vektor heurističkih težina ('WEIGHT_FIELDS'). U generaciji
svi igraju protiv svih (obe boje). Skor: pobeda +1, poraz -1, nerešeno 0.
Najboljih 'elite' jedinki prelazi u sledeću generaciju i proizvodi decu
(ukrštanje + mutacija). Mečevi se paralelizuju.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm, trange

from engine.weights import (
    DEFAULT_WEIGHTS,
    load_weights,
    save_weights,
    vector_to_weights,
    weights_to_vector,
)
from genetic.arena import play_game
from lib.constants import Stone


def random_individual(rng, base=None, spread=0.5):
    """Vektor težina nasumično poremećen oko 'base' za najviše +/- 'spread'."""
    base = base if base is not None else weights_to_vector(DEFAULT_WEIGHTS)
    factors = np.array([rng.uniform(1 - spread, 1 + spread) for _ in base])
    return np.clip(base * factors, 0.0, None)


def crossover(parent_a, parent_b, rng):
    """Uniformno ukrštanje: svaki gen nasumično dolazi od jednog od roditelja."""
    mask = np.array([rng.random() < 0.5 for _ in parent_a])
    return np.where(mask, parent_a, parent_b).astype(float)


def mutate(vector, rng, rate=0.2, scale=0.2):
    """Multiplikativna Gaussova mutacija, ograničena na nenegativne težine."""
    out = vector.copy()
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] *= 1.0 + rng.gauss(0.0, scale)
    return np.clip(out, 0.0, None)


def _rr_game_task(payload):
    """Jedna partija round-robin-a.

    'payload': (i, j, wi, wj, depth, radius, size, seed, i_is_black)
    Vraća (i, j, delta_i, delta_j) sa vrednostima u {-1, 0, +1}.
    """
    i, j, wi, wj, depth, radius, size, seed, i_is_black = payload
    if i_is_black:
        result = play_game(wi, wj, depth, radius, size, seed=seed)
        if result is None:
            return i, j, 0, 0
        if result == Stone.BLACK:
            return i, j, 1, -1
        return i, j, -1, 1

    result = play_game(wj, wi, depth, radius, size, seed=seed)
    if result is None:
        return i, j, 0, 0
    if result == Stone.WHITE:
        return i, j, 1, -1
    return i, j, -1, 1


def round_robin_scores(
    pop, rng, depth=2, radius=1, size=15, workers=None, verbose=False,
):
    """Skor svake jedinke nakon turnira svaki-protiv-svakog (obe boje)."""
    n = len(pop)
    scores = np.zeros(n, dtype=float)
    weights = [vector_to_weights(ind) for ind in pop]

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            seed = rng.randint(0, 1_000_000)
            tasks.append((
                i, j, weights[i], weights[j], depth, radius, size, seed, True,
            ))
            tasks.append((
                i, j, weights[i], weights[j], depth, radius, size, seed + 1, False,
            ))

    if not tasks:
        return scores

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, int(workers))

    def _apply(result) -> None:
        i, j, di, dj = result
        scores[i] += di
        scores[j] += dj

    if workers == 1 or len(tasks) == 1:
        iterator = tasks
        if verbose:
            iterator = tqdm(tasks, desc="games", leave=False, unit="game")
        for task in iterator:
            _apply(_rr_game_task(task))
    else:
        workers = min(workers, len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_rr_game_task, task) for task in tasks]
            pending = as_completed(futures)
            if verbose:
                pending = tqdm(
                    pending, total=len(futures), desc="games",
                    leave=False, unit="game",
                )
            for future in pending:
                _apply(future.result())

    return scores


def _next_population(elites, pop_size, rng, mutation_rate):
    """Elita + deca nastala ukrštanjem/mutacijom elitnih roditelja."""
    new_pop = [ind.copy() for ind in elites]
    while len(new_pop) < pop_size:
        parent_a = elites[rng.randrange(len(elites))]
        parent_b = elites[rng.randrange(len(elites))]
        child = mutate(
            crossover(parent_a, parent_b, rng), rng, rate=mutation_rate,
        )
        new_pop.append(child)
    return new_pop


def run_genetic_algorithm(
    pop_size=10, generations=100, opponents=3, depth=2, radius=1,
    size=15, elite=2, mutation_rate=0.2, seed=None, out_path=None,
    workers=None, verbose=True,
):
    """Pokreće round-robin evoluciju i snima najbolje težine.

    'opponents' se ignoriše (zadržan radi kompatibilnosti CLI-ja). Fitnes je
    netto skor (+1/−1/0) iz kompletnog turnira. Najboljih 'elite' jedinki
    proizvodi sledeću generaciju.
    """
    del opponents  # round-robin ne koristi uzorkovanje protivnika

    rng = random.Random(seed)
    # polazna tačka: postojeći weights.json ili podrazumevane težine
    base = weights_to_vector(load_weights())
    pop = [base.copy()]
    while len(pop) < pop_size:
        pop.append(random_individual(rng, base))

    best_vec, best_fit = base.copy(), float("-inf")
    gens = (
        trange(generations, desc="generations") if verbose
        else range(generations)
    )
    for gen_index in gens:
        scores = round_robin_scores(
            pop, rng, depth=depth, radius=radius, size=size,
            workers=workers, verbose=verbose,
        )
        order = np.argsort(scores)[::-1]
        pop = [pop[i] for i in order]
        scores = scores[order]

        if verbose:
            top = ", ".join(f"{scores[i]:+.0f}" for i in range(min(elite, pop_size)))
            print(f"\ngen {gen_index + 1}: top-{elite} scores [{top}]")

        if scores[0] > best_fit:
            best_fit = float(scores[0])
            best_vec = pop[0].copy()
            save_weights(vector_to_weights(best_vec), out_path)

        n_elite = min(elite, pop_size)
        elites = [pop[i].copy() for i in range(n_elite)]
        pop = _next_population(elites, pop_size, rng, mutation_rate)

    best_weights = vector_to_weights(best_vec)
    save_weights(best_weights, out_path)
    return best_weights, best_fit
