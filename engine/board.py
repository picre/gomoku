"""Model igre: stanje table, potezi i provera pobede."""

import numpy as np

from lib.constants import Stone

WIN_LENGTH = 5
# smerovi (dr, dc): horizontala, vertikala i obe dijagonale
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class GomokuBoard:
    """Stanje jedne partije: tabla, igrač na potezu i poslednji potez.

    Tabla se čuva kao 'numpy' niz tipa 'int8' sa vrednostima 'Stone'
    (BLACK=0, WHITE=1, EMPTY=2), pa je pretraga može kopirati bez konverzije.
    """

    def __init__(self, board_size: int = 15):
        """Inicijalizuje praznu tablu zadate veličine; crni je prvi na potezu."""
        self.board_size = board_size
        self.board = np.full((board_size, board_size), Stone.EMPTY, dtype=np.int8)
        self.current_stone = Stone.BLACK  # crni igra prvi
        self.last_move: tuple[int, int] | None = None

    def stone_at(self, row: int, col: int) -> Stone:
        """Vraća sadržaj polja (row, col) kao 'Stone'."""
        return Stone(int(self.board[row, col]))

    def make_move(self, row: int, col: int) -> None:
        """Postavlja kamenčić trenutnog igrača na (row, col), pamti potez i
        predaje potez protivniku.

        Pretpostavlja se da je potez legalan; legalnost proverava
        'is_valid_move' (GUI) ili je garantuje izbor poteza agenta.
        """
        self.board[row, col] = self.current_stone
        self.last_move = (row, col)
        self.current_stone = self.current_stone.opponent

    def is_valid_move(self, row: int, col: int) -> bool:
        """Tačno ako je (row, col) unutar table i to polje je prazno."""
        return (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row, col] == Stone.EMPTY
        )

    def is_a_tie(self) -> bool:
        """Tačno ako je tabla popunjena (nema praznih polja), tj. nerešeno je."""
        return not np.any(self.board == Stone.EMPTY)

    def _count_in_direction(self, row, col, stone, row_step, col_step):
        """Broji uzastopne kamenčiće 'stone' počev od jednog koraka dalje u
        smeru (row_step, col_step)."""
        matches = 0
        current_row, current_col = row + row_step, col + col_step
        # korača dok je polje na tabli i sadrži isti kamenčić
        while (
            0 <= current_row < self.board_size
            and 0 <= current_col < self.board_size
            and self.board[current_row, current_col] == stone
        ):
            matches += 1
            current_row += row_step
            current_col += col_step
        return matches

    def player_has_won(self) -> bool:
        """Vraća True ako poslednji potez formira niz dužine ≥ 'WIN_LENGTH'.

        Pravilo je slobodni gomoku (≥ 5, uključujući niz duži od pet);
        varijanta sa tačno pet u nizu zahteva uslov '== WIN_LENGTH'.
        """
        if self.last_move is None:
            raise RuntimeError("win check before any move")
        row, col = self.last_move
        stone = self.board[row, col]
        if stone == Stone.EMPTY:
            raise RuntimeError(f"last_move ({row}, {col}) points to an empty cell")
        for row_step, col_step in DIRECTIONS:
            # dužina niza = postavljeni kamenčić + produžeci u oba smera
            streak = (
                1
                + self._count_in_direction(row, col, stone, row_step, col_step)
                + self._count_in_direction(row, col, stone, -row_step, -col_step)
            )
            if streak >= WIN_LENGTH:
                return True
        return False
