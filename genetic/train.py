"""Ulazna tačka iz komandne linije za evoluciju težina funkcije evaluacije.

Round-robin turnir (svaki protiv svakog, obe boje); najboljih nekoliko jedinki
proizvodi sledeću generaciju.

Pokreće se iz korena projekta::

    python -m genetic.train --workers 12 --seed 42
"""

import argparse
import os
import sys

# obezbeđuje da koren projekta bude uvoziv kada se pokreće kao obična skripta
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# UTF-8 na Windows konzoli (cp1252 podrazumevano)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from engine.weights import WEIGHTS_PATH
from genetic.genetic_algorithm import run_genetic_algorithm


def parse_args(argv=None):
    """Definiše i parsira argumente komandne linije za trening."""
    parser = argparse.ArgumentParser(
        description=(
            "Optimize Gomoku weights via round-robin genetic tournament."
        ),
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
    """Pokreće genetsku pretragu sa zadatim argumentima i ispisuje najbolji rezultat."""
    args = parse_args(argv)
    out_path = args.out if args.out is not None else str(WEIGHTS_PATH)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    # partija po generaciji: C(n,2)*2
    games_per_gen = args.population * (args.population - 1)

    print(
        f"Weight evolution (round-robin): pop={args.population}, "
        f"elite={args.elite}, gen={args.generations}, depth={args.depth}, "
        f"board={args.size}x{args.size}, workers={workers}, "
        f"games/gen={games_per_gen}"
    )
    best_weights, best_fit = run_genetic_algorithm(
        pop_size=args.population,
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

    print(f"\nBest score: {best_fit:.0f}")
    print(f"Best weights saved to: {out_path}")
    for name, value in best_weights.items():
        print(f"  {name:>13}: {value:.3f}")


if __name__ == "__main__":
    main()
