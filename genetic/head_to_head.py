"""Head-to-head: 'weights_initial' protiv 'weights_latest'.

Meri koliko partija pobedi početna (stara) matematika, a koliko evoluirane
težine. Svaka runda igra dve partije (zamena boja) da se poništi prednost crnog.
Progress prati pojedinačne partije i tekući skor.

Pokretanje iz korena projekta::

    python -m genetic.head_to_head --rounds 20 --workers 12
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# koren projekta u sys.path kada se pokreće kao skripta
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from engine.weights import (
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    load_weights,
    save_weights,
)
from genetic.arena import play_game
from lib.constants import Stone

INITIAL_PATH = PROJECT_ROOT / "weights_initial.json"
LATEST_PATH = PROJECT_ROOT / "weights_latest.json"


def _ensure_initial_weights(path: Path) -> None:
    """Ako početne težine ne postoje, snima 'DEFAULT_WEIGHTS' na 'path'."""
    if path.exists():
        return
    save_weights(DEFAULT_WEIGHTS, path)
    print(f"Wrote default initial weights to: {path}")


def _resolve_latest_path(path: Path) -> Path:
    """Vraća putanju najnovijih težina; pada na 'weights.json' ako treba."""
    if path.exists():
        return path
    fallback = PROJECT_ROOT / "weights.json"
    if fallback.exists():
        print(f"{path.name} missing; using {fallback.name}")
        return fallback
    raise FileNotFoundError(
        f"Neither {path} nor {fallback} found; run training first."
    )


def _game_task(payload):
    """Jedna partija; vraća 'latest' / 'initial' / 'draw'.

    'payload': (black_w, white_w, depth, radius, size, seed, latest_is_black)
    """
    black_w, white_w, depth, radius, size, seed, latest_is_black = payload
    result = play_game(black_w, white_w, depth, radius, size, seed=seed)
    if result is None:
        return "draw"
    if latest_is_black:
        return "latest" if result == Stone.BLACK else "initial"
    return "latest" if result == Stone.WHITE else "initial"


def parse_args(argv=None):
    """Definiše argumente komandne linije za head-to-head test."""
    parser = argparse.ArgumentParser(
        description="Head-to-head: weights_initial vs weights_latest.",
    )
    parser.add_argument(
        "--initial", type=str, default=str(INITIAL_PATH),
        help="path to initial (baseline) weights JSON",
    )
    parser.add_argument(
        "--latest", type=str, default=str(LATEST_PATH),
        help="path to latest (evolved) weights JSON",
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="color-swapped match pairs (2 games each)",
    )
    parser.add_argument(
        "--depth", type=int, default=2,
        help="minimax depth",
    )
    parser.add_argument(
        "--radius", type=int, default=1,
        help="candidate-move radius",
    )
    parser.add_argument(
        "--size", type=int, default=15,
        help="board size",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="base RNG seed",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Pokreće seriju partija i ispisuje zbir pobeda / nerešenih."""
    args = parse_args(argv)
    initial_path = Path(args.initial)
    latest_path = Path(args.latest)

    _ensure_initial_weights(initial_path)
    latest_path = _resolve_latest_path(latest_path)

    initial = load_weights(initial_path)
    latest = load_weights(latest_path)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, int(workers))

    n_games = args.rounds * 2
    print(
        f"Head-to-head: {initial_path.name} vs {latest_path.name} | "
        f"rounds={args.rounds} ({n_games} games), depth={args.depth}, "
        f"board={args.size}x{args.size}, workers={workers}"
    )

    # dve partije po rundi: latest crni, zatim latest beli
    tasks = []
    for i in range(args.rounds):
        seed = None if args.seed is None else args.seed + i * 10
        tasks.append((
            latest, initial, args.depth, args.radius, args.size, seed, True,
        ))
        tasks.append((
            initial, latest, args.depth, args.radius, args.size,
            None if seed is None else seed + 1, False,
        ))

    latest_wins = initial_wins = draws = 0
    bar = tqdm(total=len(tasks), desc="games", unit="game")

    def _consume(outcome: str) -> None:
        nonlocal latest_wins, initial_wins, draws
        if outcome == "latest":
            latest_wins += 1
        elif outcome == "initial":
            initial_wins += 1
        else:
            draws += 1
        bar.set_postfix(
            L=latest_wins, I=initial_wins, D=draws,
            rate=f"{100.0 * latest_wins / max(1, latest_wins + initial_wins + draws):.0f}%",
        )
        bar.update(1)

    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            _consume(_game_task(task))
    else:
        workers = min(workers, len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_game_task, task) for task in tasks]
            for future in as_completed(futures):
                _consume(future.result())

    bar.close()

    games = latest_wins + initial_wins + draws
    print(f"\nGames:          {games}")
    print(f"Latest wins:    {latest_wins}")
    print(f"Initial wins:   {initial_wins}")
    print(f"Draws:          {draws}")
    if games:
        decided = latest_wins + initial_wins
        print(
            f"Latest win rate: {100.0 * latest_wins / games:.1f}% "
            f"(excl. draws: "
            f"{100.0 * latest_wins / max(1, decided):.1f}%)"
        )


if __name__ == "__main__":
    main()
