import random

import numpy as np

from engine.board import GomokuBoard
from engine.evaluation import load_weights
from engine.search import board_to_int, search_best_move
from lib.constants import Stone


class GomokuAI:
    """Agent koji bira potez. Ne poseduje stanje table; čita tablu i vraća
    potez. Za primenu poteza zadužena je petlja igre.

    Potez se bira Minimax algoritmom + alfa-beta odsecanjem nad heuristikom
    zasnovanom na obrascima (videti :mod:`engine.search` i
    :mod:`engine.evaluation`). Heurističke ``weights`` mogu se proslediti
    direktno (koristi ih genetski algoritam); u suprotnom se učitavaju
    evoluirane ``weights.json`` ako postoje, uz vraćanje na razumne podrazumevane."""

    def __init__(self, seed: int | None = None, depth: int = 2, radius: int = 1,
                 weights: dict | None = None):
        # sopstveni RNG da partije budu ponovljive bez diranja globalnog random stanja
        self._rng = random.Random(seed)
        self.depth = depth
        self.radius = radius
        self.weights = weights if weights is not None else load_weights()

    def choose_move(self, board: GomokuBoard) -> tuple[int, int] | None:
        """Vraća najbolji potez za igrača koji je na potezu (ili None ako nema poteza).

        Tablu prevodi u int niz i prepušta izbor minimax pretrazi; ako pretraga
        ništa ne vrati, bira nasumičan legalan potez kao rezervu."""
        arr = board_to_int(board)
        ai_stone = int(board.current_stone.value)
        move = search_best_move(
            arr, ai_stone, self.weights,
            depth=self.depth, radius=self.radius, rng=self._rng,
        )
        if move is None:
            # vrati se na nasumičan legalan potez ako pretraga ništa nije našla
            legal = self.legal_moves(board)
            return self._rng.choice(legal) if legal else None
        return (int(move[0]), int(move[1]))

    @staticmethod
    def legal_moves(board: GomokuBoard) -> list[tuple[int, int]]:
        """Vraća listu svih praznih polja (row, col) na koja je potez dozvoljen."""
        rows, cols = np.where(board.board == Stone.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))
