import random

import numpy as np

from engine.board import GomokuBoard
from lib.constants import Stone


class GomokuAI:
    """A move-picking agent. It does not own board state; it reads a board
    and returns a move. The game loop is responsible for applying it."""

    def __init__(self, seed: int | None = None):
        # own RNG so games are reproducible without touching global random state
        self._rng = random.Random(seed)

    def choose_move(self, board: GomokuBoard) -> tuple[int, int] | None:
        # for now: just pick a random empty cell (a legal move)
        empty = self.legal_moves(board)
        if not empty:
            return None
        return self._rng.choice(empty)

    @staticmethod
    def legal_moves(board: GomokuBoard) -> list[tuple[int, int]]:
        rows, cols = np.where(board.board == Stone.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))
