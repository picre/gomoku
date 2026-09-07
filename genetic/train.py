"""Ulazna tačka iz komandne linije za evoluciju težina funkcije evaluacije.

Pokreće se iz korena projekta, kao modul ili kao skripta::

    python -m genetic.train --generations 15 --population 16
    python genetic/train.py --generations 15 --population 16

Najbolje pronađene težine upisuju se u ``weights.json`` u korenu projekta, koje
igra (i AI) automatski preuzimaju pri sledećem pokretanju.
"""

import argparse
import os
import sys

# obezbedi da koren projekta bude uvoziv kada se pokreće kao obična skripta
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ispisuj UTF-8 (srpska slova) i na Windows konzoli koja podrazumeva cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from engine.evaluation import WEIGHTS_PATH  # noqa: E402
from genetic.ga import run_ga  # noqa: E402


def parse_args(argv=None):
    """Definiše i parsira argumente komandne linije za trening."""
    parser = argparse.ArgumentParser(description="Optimizacija težina Gomoku evaluacije genetskim algoritmom.")
    parser.add_argument("--population", type=int, default=12, help="broj jedinki po generaciji")
    parser.add_argument("--generations", type=int, default=10, help="broj generacija")
    parser.add_argument("--opponents", type=int, default=3, help="mečeva po jedinki u svakoj generaciji")
    parser.add_argument("--depth", type=int, default=1, help="dubina minimax pretrage tokom self-play-a")
    parser.add_argument("--radius", type=int, default=1, help="radijus kandidat-poteza oko kamenčića")
    parser.add_argument("--size", type=int, default=15, help="veličina table za self-play partije")
    parser.add_argument("--elite", type=int, default=2, help="jedinke koje se prenose nepromenjene")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="verovatnoća mutacije po genu")
    parser.add_argument("--seed", type=int, default=None, help="seme RNG-a radi ponovljivosti")
    parser.add_argument("--out", type=str, default=None, help="izlazna JSON putanja (podrazumevano: weights.json)")
    return parser.parse_args(argv)


def main(argv=None):
    """Pokreće genetsku pretragu sa zadatim argumentima i ispisuje najbolji rezultat."""
    args = parse_args(argv)
    out_path = args.out if args.out is not None else str(WEIGHTS_PATH)

    print(f"Evolucija težina: pop={args.population}, gen={args.generations}, "
          f"dubina={args.depth}, tabla={args.size}x{args.size}")
    best_weights, best_fit = run_ga(
        pop_size=args.population,
        generations=args.generations,
        opponents=args.opponents,
        depth=args.depth,
        radius=args.radius,
        size=args.size,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        out_path=out_path,
    )

    print(f"\nNajbolji fitnes: {best_fit:.2f}")
    print(f"Najbolje težine sačuvane u: {out_path}")
    for name, value in best_weights.items():
        print(f"  {name:>13}: {value:.3f}")


if __name__ == "__main__":
    main()
