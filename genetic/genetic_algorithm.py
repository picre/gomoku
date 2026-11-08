"""Genetski algoritam: turnir svako-protiv-svakog (round-robin) + elitizam.

Svaka jedinka je vektor heurističkih težina ('WEIGHT_FIELDS'). U generaciji
svi igraju protiv svih, obema bojama. Skor: pobeda +1, poraz -1, nerešeno 0.
Najboljih 'elite' jedinki prelazi u sledeću generaciju i proizvodi decu
(ukrštanje + mutacija). Partije se izvršavaju paralelno.

Skor iz turnira je **relativan** — meri jedinku samo u odnosu na ostale članove
iste generacije — pa se skorovi iz različitih generacija ne porede. Kao
rezultat se uzima šampion poslednje generacije: zahvaljujući elitizmu, on je
pobedio (ili izjednačio sa) šampionom prethodne generacije u zajedničkom turniru.
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
from genetic.arena import outcome_for, play_game
from lib.constants import Stone


def random_individual(rng, base=None, spread=0.5):
    """Vektor težina nasumično poremećen oko 'base' za najviše ±'spread' (relativno)."""
    base = base if base is not None else weights_to_vector(DEFAULT_WEIGHTS)
    factors = np.array([rng.uniform(1 - spread, 1 + spread) for _ in base])
    return np.clip(base * factors, 0.0, None)


def crossover(parent_a, parent_b, rng):
    """Uniformno ukrštanje: svaki gen nasumično dolazi od jednog od roditelja."""
    mask = np.array([rng.random() < 0.5 for _ in parent_a])
    return np.where(mask, parent_a, parent_b).astype(float)


def mutate(vector, rng, rate=0.2, scale=0.2):
    """Multiplikativna Gausova mutacija, ograničena na nenegativne težine."""
    out = vector.copy()
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] *= 1.0 + rng.gauss(0.0, scale)
    return np.clip(out, 0.0, None)


def _round_robin_game(payload):
    """Jedna partija turnira.

    'payload': (i, j, weights_i, weights_j, depth, radius, size, seed, i_is_black)
    Vraća (i, j, delta_i, delta_j) sa vrednostima u {-1, 0, +1}.
    """
    i, j, weights_i, weights_j, depth, radius, size, seed, i_is_black = payload
    if i_is_black:
        result = play_game(weights_i, weights_j, depth, radius, size, seed=seed)
        delta_i = outcome_for(result, Stone.BLACK)
    else:
        result = play_game(weights_j, weights_i, depth, radius, size, seed=seed)
        delta_i = outcome_for(result, Stone.WHITE)
    return i, j, delta_i, -delta_i


def round_robin_scores(
    population, rng, depth=2, radius=1, size=15, workers=None, verbose=False,
):
    """Skor svake jedinke nakon turnira svako-protiv-svakog (obe boje)."""
    n = len(population)
    scores = np.zeros(n, dtype=float)
    weights = [vector_to_weights(individual) for individual in population]

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
    workers = max(1, min(int(workers), len(tasks)))

    def apply_result(result) -> None:
        i, j, delta_i, delta_j = result
        scores[i] += delta_i
        scores[j] += delta_j

    if workers == 1:
        iterator = tqdm(tasks, desc="games", leave=False, unit="game") if verbose else tasks
        for task in iterator:
            apply_result(_round_robin_game(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_round_robin_game, task) for task in tasks]
            pending = as_completed(futures)
            if verbose:
                pending = tqdm(
                    pending, total=len(futures), desc="games",
                    leave=False, unit="game",
                )
            for future in pending:
                apply_result(future.result())

    return scores


def _next_population(elites, population_size, rng, mutation_rate):
    """Elita + deca nastala ukrštanjem i mutacijom elitnih roditelja."""
    new_population = [individual.copy() for individual in elites]
    while len(new_population) < population_size:
        parent_a = elites[rng.randrange(len(elites))]
        parent_b = elites[rng.randrange(len(elites))]
        child = mutate(
            crossover(parent_a, parent_b, rng), rng, rate=mutation_rate,
        )
        new_population.append(child)
    return new_population


def run_genetic_algorithm(
    population_size=10, generations=100, depth=2, radius=1,
    size=15, elite=2, mutation_rate=0.2, seed=None, out_path=None,
    workers=None, verbose=True,
):
    """Pokreće evoluciju i snima težine šampiona posle svake generacije.

    Vraća par ('weights', 'history'): težine šampiona poslednje generacije i
    listu skorova šampiona po generacijama (svaki skor je relativan u odnosu
    na svoju generaciju, maksimum je 2 * (population_size - 1)).
    """
    rng = random.Random(seed)
    # polazna tačka: postojeći weights.json ili podrazumevane težine
    base = weights_to_vector(load_weights())
    population = [base.copy()]
    while len(population) < population_size:
        population.append(random_individual(rng, base))

    n_elite = max(1, min(elite, population_size))
    history = []
    generation_range = (
        trange(generations, desc="generations") if verbose
        else range(generations)
    )
    for generation in generation_range:
        scores = round_robin_scores(
            population, rng, depth=depth, radius=radius, size=size,
            workers=workers, verbose=verbose,
        )
        # sortira populaciju opadajuće po skoru
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = scores[order]
        history.append(float(scores[0]))

        if verbose:
            top = ", ".join(f"{scores[i]:+.0f}" for i in range(n_elite))
            print(f"\ngen {generation + 1}: top-{n_elite} scores [{top}]")

        # šampion tekuće generacije se snima odmah, da prekid treninga ne izgubi napredak
        save_weights(vector_to_weights(population[0]), out_path)

        if generation < generations - 1:
            elites = [population[i].copy() for i in range(n_elite)]
            population = _next_population(elites, population_size, rng, mutation_rate)

    return vector_to_weights(population[0]), history
