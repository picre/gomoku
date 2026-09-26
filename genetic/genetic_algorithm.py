"""Genetski algoritam koji optimizuje težine funkcije evaluacije.

Svaka jedinka je vektor heurističkih težina (videti
'engine.weights.WEIGHT_FIELDS'). Jedinke igraju partije jedna protiv
druge; fitnes je broj partija koje osvoje. Kroz generacije, turnirska selekcija,
uniformno ukrštanje i Gaussova mutacija traže kombinaciju težina koja igra
najjači Gomoku.
"""

import random

import numpy as np
from tqdm import trange

from engine.weights import (
    DEFAULT_WEIGHTS,
    save_weights,
    vector_to_weights,
    weights_to_vector,
)
from genetic.arena import match_score


def random_individual(rng, base=None, spread=0.5):
    """Vektor težina nasumično poremećen oko 'base' za najviše +/- 'spread'."""
    base = base if base is not None else weights_to_vector(DEFAULT_WEIGHTS)
    factors = np.array([rng.uniform(1 - spread, 1 + spread) for _ in base])
    return np.clip(base * factors, 0.0, None)


def evaluate_population(pop, rng, opponents=3, depth=1, radius=1, size=15):
    """Fitnes svake jedinke = poeni osvojeni protiv nasumičnih protivnika."""
    n = len(pop)
    fitness = np.zeros(n)
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        for j in rng.sample(pool, min(opponents, len(pool))):
            fitness[i] += match_score(
                vector_to_weights(pop[i]),
                vector_to_weights(pop[j]),
                depth=depth, radius=radius, size=size,
                seed=rng.randint(0, 1_000_000),
            )
    return fitness


def tournament_select(pop, fitness, rng, k=3):
    """Bira najsposobniju od 'k' nasumičnih takmičarki."""
    idxs = rng.sample(range(len(pop)), min(k, len(pop)))
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best]


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


def run_ga(pop_size=12, generations=10, opponents=3, depth=1, radius=1,
           size=15, elite=2, mutation_rate=0.2, seed=None, out_path=None,
           verbose=True):
    """Pokreće genetsku pretragu i snima najbolje pronađene težine.

    Vraća '(weights, fitness)' i upisuje najbolje
    težine u 'out_path' (podrazumevano projektni 'weights.json').
    """
    rng = random.Random(seed)
    base = weights_to_vector(DEFAULT_WEIGHTS)
    pop = [random_individual(rng, base) for _ in range(pop_size)]

    best_vec, best_fit = base.copy(), -1.0
    gens = trange(generations, desc="GA generacije") if verbose else range(generations)
    for _ in gens:
        fitness = evaluate_population(pop, rng, opponents, depth, radius, size)
        order = np.argsort(fitness)[::-1]
        pop = [pop[i] for i in order]
        fitness = fitness[order]

        if fitness[0] > best_fit:
            best_fit = float(fitness[0])
            best_vec = pop[0].copy()
            # snima najbolju-do-sada pri svakom poboljšanju
            save_weights(vector_to_weights(best_vec), out_path)

        # elitizam: prenosi najbolje jedinke nepromenjene
        new_pop = [pop[i].copy() for i in range(min(elite, pop_size))]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, rng)
            p2 = tournament_select(pop, fitness, rng)
            child = mutate(crossover(p1, p2, rng), rng, rate=mutation_rate)
            new_pop.append(child)
        pop = new_pop

    best_weights = vector_to_weights(best_vec)
    save_weights(best_weights, out_path)
    return best_weights, best_fit
