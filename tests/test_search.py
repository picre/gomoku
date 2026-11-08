"""Testovi pretrage, taktike i AI agenta."""

import random

from engine.ai import GomokuAI
from engine.board import GomokuBoard
from engine.evaluation import BLACK, WHITE
from engine.incremental import IncrementalEvaluator
from engine.opening_book import algebraic_to_rc
from engine.search import board_to_int, find_tactical_move, search_best_move
from engine.weights import DEFAULT_WEIGHTS


def board_with(black, white, size=15):
    """Pravi tablu sa zadatim crnim i belim kamenčićima; crni je na potezu
    ako su brojevi jednaki, inače beli."""
    board = GomokuBoard(size)
    for move in black:
        board.board[move] = BLACK
    for move in white:
        board.board[move] = WHITE
    if len(black) > len(white):
        board.current_stone = board.current_stone.opponent
    return board


def test_board_to_int_returns_independent_copy():
    board = GomokuBoard(15)
    arr = board_to_int(board)
    arr[0, 0] = BLACK
    assert board.is_valid_move(0, 0)


def test_tactical_takes_immediate_win():
    board = board_with([(7, 3), (7, 4), (7, 5), (7, 6)], [(0, 0), (0, 2), (0, 4), (0, 6)])
    evaluator = IncrementalEvaluator(board_to_int(board), DEFAULT_WEIGHTS)
    assert find_tactical_move(evaluator, BLACK, radius=1) in {(7, 2), (7, 7)}


def test_tactical_blocks_opponent_four():
    # beli je na potezu, crni preti petorkom na (7, 7)
    board = board_with([(7, 3), (7, 4), (7, 5), (7, 6)], [(7, 2), (0, 0), (0, 14)])
    evaluator = IncrementalEvaluator(board_to_int(board), DEFAULT_WEIGHTS)
    assert find_tactical_move(evaluator, WHITE, radius=1) == (7, 7)


def test_search_returns_legal_move():
    board = board_with([(7, 7), (8, 8)], [(7, 8), (6, 6)])
    move = search_best_move(
        board_to_int(board), BLACK, DEFAULT_WEIGHTS, depth=2, rng=random.Random(0),
    )
    assert move is not None
    assert board.is_valid_move(*move)


def test_ai_uses_opening_book_on_standard_board():
    ai = GomokuAI(seed=0, weights=DEFAULT_WEIGHTS)
    move = ai.choose_move(GomokuBoard(15), [])
    assert ai.last_phase == "opening"
    assert move == algebraic_to_rc("F6")


def test_ai_skips_opening_book_on_other_board_sizes():
    ai = GomokuAI(seed=0, weights=DEFAULT_WEIGHTS)
    move = ai.choose_move(GomokuBoard(9), [])
    assert ai.last_phase == "search"
    assert move == (4, 4)  # centar prazne table


def test_ai_is_reproducible_with_seed():
    def play(seed):
        board, history = GomokuBoard(9), []
        ai = GomokuAI(seed=seed, depth=1, weights=DEFAULT_WEIGHTS, use_opening_book=False)
        for _ in range(8):
            move = ai.choose_move(board, history)
            board.make_move(*move)
            history.append(move)
        return history

    assert play(3) == play(3)
