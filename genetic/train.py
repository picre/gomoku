"""Ulazna tačka iz komandne linije za evoluciju težina funkcije evaluacije.

Turnir svako-protiv-svakog (obe boje); najboljih nekoliko jedinki proizvodi
sledeću generaciju.

Pokreće se iz korena projekta::

    python -m genetic.train --workers 12 --seed 42
"""

import argparse
import os
import sys

from engine.weights import WEIGHTS_PATH
from genetic.genetic_algorithm import run_genetic_algorithm


def parse_args(argv=None):
    """Definiše i parsira argumente komandne linije za trening."""
    parser = argparse.ArgumentParser(
        description="Optimize Gomoku weights via a round-robin genetic tournament.",
    )
    parser.add_argument(
        "--population", type=int, default=10,
        help="individuals per generation (round-robin)",
    )
    parser.add_argument(
        "--generations", type=int, default=100, help="number of generations",
    )
    parser.add_argument(
        "--depth", type=int, default=2,
        help="minimax depth during self-play",
    )
    parser.add_argument(
        "--radius", type=int, default=1,
        help="candidate-move radius around stones",
    )
    parser.add_argument(
        "--size", type=int, default=15,
        help="board size for self-play games",
    )
    parser.add_argument(
        "--elite", type=int, default=2,
        help="top individuals that breed the next generation",
    )
    parser.add_argument(
        "--mutation-rate", type=float, default=0.2,
        help="per-gene mutation probability",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="worker processes for parallel games (default: CPU count)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="output JSON path (default: weights.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Pokreće genetsku pretragu sa zadatim argumentima i ispisuje rezultat."""
    # UTF-8 na Windows konzoli (podrazumevano je cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    out_path = args.out if args.out is not None else str(WEIGHTS_PATH)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    # broj partija po generaciji: C(n, 2) parova * 2 boje
    games_per_generation = args.population * (args.population - 1)

    print(
        f"Weight evolution (round-robin): pop={args.population}, "
        f"elite={args.elite}, gen={args.generations}, depth={args.depth}, "
        f"board={args.size}x{args.size}, workers={workers}, "
        f"games/gen={games_per_generation}"
    )
    best_weights, history = run_genetic_algorithm(
        population_size=args.population,
        generations=args.generations,
        depth=args.depth,
        radius=args.radius,
        size=args.size,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        out_path=out_path,
        workers=workers,
    )

    max_score = 2 * (args.population - 1)
    print(f"\nChampion score per generation (max {max_score}):")
    print("  " + " ".join(f"{score:+.0f}" for score in history))
    print(f"Final champion weights saved to: {out_path}")
    for name, value in best_weights.items():
        print(f"  {name:>13}: {value:.3f}")


if __name__ == "__main__":
    main()
