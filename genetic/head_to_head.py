"""Eksperiment: početne težine protiv evoluiranih (meč dva agenta).

Meri koliko partija dobijaju početne (ručno izabrane) težine, a koliko težine
dobijene genetskim algoritmom. Svaka runda se sastoji od dve partije sa
zamenjenim bojama, čime se poništava prednost prvog poteza (crni).

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

from engine.weights import DEFAULT_WEIGHTS, PROJECT_ROOT, WEIGHTS_PATH, load_weights, save_weights
from genetic.arena import outcome_for, play_game
from lib.constants import Stone

INITIAL_PATH = PROJECT_ROOT / "weights_initial.json"


def _ensure_initial_weights(path: Path) -> None:
    """Ako početne težine ne postoje, snima 'DEFAULT_WEIGHTS' na 'path'."""
    if path.exists():
        return
    save_weights(DEFAULT_WEIGHTS, path)
    print(f"Wrote default initial weights to: {path}")


def _game_task(payload):
    """Jedna partija; vraća ishod iz ugla evoluiranih težina (+1 / 0 / -1).

    'payload': (black_weights, white_weights, depth, radius, size, seed, evolved_is_black)
    """
    black_weights, white_weights, depth, radius, size, seed, evolved_is_black = payload
    result = play_game(black_weights, white_weights, depth, radius, size, seed=seed)
    evolved_stone = Stone.BLACK if evolved_is_black else Stone.WHITE
    return outcome_for(result, evolved_stone)


def parse_args(argv=None):
    """Definiše argumente komandne linije za eksperiment."""
    parser = argparse.ArgumentParser(
        description="Head-to-head match: initial weights vs evolved weights.",
    )
    parser.add_argument(
        "--initial", type=str, default=str(INITIAL_PATH),
        help="path to initial (baseline) weights JSON",
    )
    parser.add_argument(
        "--evolved", type=str, default=str(WEIGHTS_PATH),
        help="path to evolved weights JSON (default: weights.json)",
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="color-swapped match pairs (2 games each)",
    )
    parser.add_argument("--depth", type=int, default=2, help="minimax depth")
    parser.add_argument("--radius", type=int, default=1, help="candidate-move radius")
    parser.add_argument("--size", type=int, default=15, help="board size")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="worker processes (default: CPU count)",
    )
    parser.add_argument("--seed", type=int, default=None, help="base RNG seed")
    return parser.parse_args(argv)


def main(argv=None):
    """Pokreće seriju partija i ispisuje broj pobeda i nerešenih partija."""
    # UTF-8 na Windows konzoli (podrazumevano je cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    initial_path = Path(args.initial)
    evolved_path = Path(args.evolved)
    _ensure_initial_weights(initial_path)
    if not evolved_path.exists():
        raise FileNotFoundError(f"{evolved_path} not found; run training first.")

    initial = load_weights(initial_path)
    evolved = load_weights(evolved_path)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, int(workers))

    n_games = args.rounds * 2
    print(
        f"Head-to-head: {initial_path.name} vs {evolved_path.name} | "
        f"rounds={args.rounds} ({n_games} games), depth={args.depth}, "
        f"board={args.size}x{args.size}, workers={workers}"
    )

    # dve partije po rundi: evoluirane težine kao crni, zatim kao beli
    tasks = []
    for i in range(args.rounds):
        seed = None if args.seed is None else args.seed + i * 10
        tasks.append((evolved, initial, args.depth, args.radius, args.size, seed, True))
        tasks.append((
            initial, evolved, args.depth, args.radius, args.size,
            None if seed is None else seed + 1, False,
        ))

    tally = {1: 0, -1: 0, 0: 0}  # pobede evoluiranih / pobede početnih / nerešeno
    progress = tqdm(total=len(tasks), desc="games", unit="game")

    def consume(outcome: int) -> None:
        tally[outcome] += 1
        progress.set_postfix(evolved=tally[1], initial=tally[-1], draws=tally[0])
        progress.update(1)

    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            consume(_game_task(task))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_game_task, task) for task in tasks]
            for future in as_completed(futures):
                consume(future.result())
    progress.close()

    games = sum(tally.values())
    decided = tally[1] + tally[-1]
    print(f"\nGames:            {games}")
    print(f"Evolved wins:     {tally[1]}")
    print(f"Initial wins:     {tally[-1]}")
    print(f"Draws:            {tally[0]}")
    if games:
        print(
            f"Evolved win rate: {100.0 * tally[1] / games:.1f}% "
            f"(excl. draws: {100.0 * tally[1] / max(1, decided):.1f}%)"
        )


if __name__ == "__main__":
    main()
