"""AI agent: spaja knjigu otvaranja, taktičku analizu i minimax pretragu."""

import random

import numpy as np

from engine.board import GomokuBoard
from engine.incremental import IncrementalEvaluator
from engine.opening_book import BOARD_SIZE as BOOK_BOARD_SIZE
from engine.opening_book import lookup as opening_lookup
from engine.search import board_to_int, find_tactical_move, search_best_move
from engine.weights import load_weights
from lib.constants import Stone


class GomokuAI:
    """Agent koji bira potez. Ne poseduje stanje table; čita tablu i vraća
    potez. Za primenu poteza zadužena je petlja igre.

    Redosled: knjiga otvaranja → taktički forsiran potez → minimax pretraga.
    Poslednji izvor poteza je u 'last_phase' ('opening' / 'tactical' / 'search').
    """

    def __init__(
        self,
        seed: int | None = None,
        depth: int = 3,
        radius: int = 1,
        weights: dict | None = None,
        use_opening_book: bool = True,
        force_depth: int = 6,
    ):
        # sopstveni RNG da partije budu ponovljive bez diranja globalnog random stanja
        self._rng = random.Random(seed)
        self.depth = depth
        self.radius = radius
        self.weights = weights if weights is not None else load_weights()
        self.use_opening_book = use_opening_book
        # dubina VCF forsiranja; 0 = samo pobeda/blokada/otvorena četvorka
        self.force_depth = force_depth
        self.last_phase = "search"

    def choose_move(
        self,
        board: GomokuBoard,
        history: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Vraća najbolji potez za igrača na potezu (ili None ako nema).

        'history' je lista odigranih (row, col) radi knjige otvaranja.
        """
        history = history if history is not None else []

        # knjiga je zapisana za tablu 15x15, pa se na drugim veličinama ne koristi
        if self.use_opening_book and board.board_size == BOOK_BOARD_SIZE:
            book_move = opening_lookup(history)
            if book_move is not None and board.is_valid_move(*book_move):
                self.last_phase = "opening"
                return book_move

        arr = board_to_int(board)
        ai_stone = int(board.current_stone)
        evaluator = IncrementalEvaluator(arr, self.weights)
        tactical = find_tactical_move(
            evaluator, ai_stone, self.radius, force_depth=self.force_depth,
        )
        if tactical is not None:
            self.last_phase = "tactical"
            return (int(tactical[0]), int(tactical[1]))

        self.last_phase = "search"
        move = search_best_move(
            arr, ai_stone, self.weights,
            depth=self.depth, radius=self.radius, rng=self._rng,
        )
        if move is None:
            if self.legal_moves(board):
                raise RuntimeError("search returned no move though empty cells remain")
            return None
        return (int(move[0]), int(move[1]))

    @staticmethod
    def legal_moves(board: GomokuBoard) -> list[tuple[int, int]]:
        """Vraća listu svih praznih polja (row, col) na koja je potez dozvoljen."""
        rows, cols = np.where(board.board == Stone.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))
