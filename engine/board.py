import numpy as np
from lib.constants import Stone


class GomokuBoard:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board = np.full((board_size, board_size), Stone.EMPTY)
        self.current_stone = Stone.BLACK # black goes first
        self.last_move = None

    def move_attempt(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        self.make_move(row, col)
        return True

    def make_move(self, row, col):
        self.board[row][col] = self.current_stone
        self.last_move = (row, col)
        self.current_stone = Stone.WHITE if self.current_stone == Stone.BLACK else Stone.BLACK

    def is_valid_move(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == Stone.EMPTY

    def is_a_tie(self):
        return np.count_nonzero(self.board == Stone.EMPTY) == 0

    def _count_in_direction(self, row, col, stone, row_step, col_step):
        # count consecutive matching stones starting one step away in direction (row_step, col_step)
        matches = 0
        current_row, current_col = row + row_step, col + col_step
        # keep stepping while we stay on the board and the cell holds the same stone
        while 0 <= current_row < self.board_size and 0 <= current_col < self.board_size and self.board[current_row][current_col] == stone:
            matches += 1
            current_row += row_step  # move one step further in the same direction
            current_col += col_step
        return matches

    def player_has_won(self):
        if self.last_move is None:
            return False
        row, col = self.last_move
        stone = self.board[self.last_move]
        if stone == Stone.EMPTY:
            return False
        # check horizontal, vertical, and both diagonals for 5 in a row
        for row_step, col_step in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # streak = the placed stone (1) + matches on one side + matches on the opposite side
            streak = (
                1
                + self._count_in_direction(row, col, stone, row_step, col_step)
                + self._count_in_direction(row, col, stone, -row_step, -col_step)
            )
            if streak >= 5: # == 5 in some version of the game
                return True
        return False
