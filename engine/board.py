import numpy as np

from lib.constants import Stone


class GomokuBoard:
    def __init__(self, board_size: int = 15):
        """Pravi praznu tablu zadate veličine; crni je prvi na potezu."""
        self.board_size = board_size
        self.board = np.full((board_size, board_size), Stone.EMPTY)
        self.current_stone = Stone.BLACK  # crni igra prvi
        self.last_move = None

    def move_attempt(self, row, col):
        """Pokušava potez; vraća False ako je nelegalan, inače ga odigrava i vraća True."""
        if not self.is_valid_move(row, col):
            return False
        self.make_move(row, col)
        return True

    def make_move(self, row, col):
        """Postavlja kamenčić trenutnog igrača na (row, col), pamti potez i
        predaje potez protivniku.

        Pozivalac mora da proveri legalnost (npr. ``is_valid_move`` / ``move_attempt``).
        """
        self.board[row][col] = self.current_stone
        self.last_move = (row, col)
        self.current_stone = (
            Stone.WHITE if self.current_stone == Stone.BLACK else Stone.BLACK
        )

    def is_valid_move(self, row, col):
        """Tačno ako je (row, col) unutar table i to polje je prazno."""
        return (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row][col] == Stone.EMPTY
        )

    def is_a_tie(self):
        """Tačno ako je tabla popunjena (nema praznih polja), tj. nerešeno je."""
        return np.count_nonzero(self.board == Stone.EMPTY) == 0

    def _count_in_direction(self, row, col, stone, row_step, col_step):
        # broji uzastopne odgovarajuće kamenčiće počev od jednog koraka dalje
        # u pravcu (row_step, col_step)
        matches = 0
        current_row, current_col = row + row_step, col + col_step
        # korača dok je polje na tabli i sadrži isti kamenčić
        while (
            0 <= current_row < self.board_size
            and 0 <= current_col < self.board_size
            and self.board[current_row][current_col] == stone
        ):
            matches += 1
            current_row += row_step  # pomera se još jedan korak u istom pravcu
            current_col += col_step
        return matches

    def player_has_won(self):
        if self.last_move is None:
            raise RuntimeError("provera pobede pre bilo kog poteza")
        row, col = self.last_move
        stone = self.board[row][col]
        if stone == Stone.EMPTY:
            raise RuntimeError(
                f"last_move ({row}, {col}) pokazuje na prazno polje"
            )
        # proverava horizontalu, vertikalu i obe dijagonale za 5 u nizu
        for row_step, col_step in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # 4 smera
            # streak = postavljeni kamenčić (1) + poklapanja sa jedne strane
            # + poklapanja sa suprotne strane
            streak = (
                1
                + self._count_in_direction(row, col, stone, row_step, col_step)
                + self._count_in_direction(row, col, stone, -row_step, -col_step)
            )
            if streak >= 5:  # == 5 u nekim verzijama igre
                return True
        return False
