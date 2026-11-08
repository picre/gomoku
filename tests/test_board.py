"""Testovi pravila igre ('engine.board')."""

import pytest

from engine.board import DIRECTIONS, GomokuBoard
from lib.constants import Stone


def play_line(board, start, step, length, filler_row):
    """Naizmenično postavlja 'length' crnih kamenčića u smeru 'step' i bele
    kamenčiće u udaljenom redu 'filler_row', tako da crni igra poslednji."""
    row, col = start
    for i in range(length):
        board.make_move(row + i * step[0], col + i * step[1])
        if i < length - 1:
            board.make_move(filler_row, i)


@pytest.mark.parametrize("step", DIRECTIONS)
def test_five_in_a_row_wins_in_every_direction(step):
    board = GomokuBoard(15)
    start = (5, 7) if step != (1, -1) else (5, 9)
    play_line(board, start, step, 5, filler_row=14)
    assert board.player_has_won()
    assert board.stone_at(*board.last_move) == Stone.BLACK


def test_four_in_a_row_does_not_win():
    board = GomokuBoard(15)
    play_line(board, (5, 5), (0, 1), 4, filler_row=14)
    assert not board.player_has_won()


def test_overline_counts_as_win():
    board = GomokuBoard(15)
    # crni: (7, 0..2) i (7, 4..5), zatim popunjava (7, 3) -> šest u nizu
    black = [(7, 0), (7, 1), (7, 2), (7, 4), (7, 5), (7, 3)]
    white = [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]
    for i, move in enumerate(black):
        board.make_move(*move)
        if i < len(white):
            board.make_move(*white[i])
    assert board.player_has_won()


def test_turns_alternate_and_cells_are_filled():
    board = GomokuBoard(15)
    assert board.current_stone == Stone.BLACK
    board.make_move(7, 7)
    assert board.current_stone == Stone.WHITE
    assert board.stone_at(7, 7) == Stone.BLACK
    assert not board.is_valid_move(7, 7)
    assert not board.is_valid_move(-1, 0)
    assert not board.is_valid_move(0, 15)
    assert board.is_valid_move(0, 0)


def test_full_board_is_a_tie():
    board = GomokuBoard(5)
    assert not board.is_a_tie()
    board.board[:, :] = Stone.BLACK
    assert board.is_a_tie()


def test_win_check_before_any_move_raises():
    with pytest.raises(RuntimeError):
        GomokuBoard(15).player_has_won()


def test_stone_opponent():
    assert Stone.BLACK.opponent == Stone.WHITE
    assert Stone.WHITE.opponent == Stone.BLACK
    with pytest.raises(ValueError):
        _ = Stone.EMPTY.opponent
