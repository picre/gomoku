"""Dva kandidat-skupa težina odigravaju punu partiju jedan protiv drugog.

Koristi ga genetski algoritam za merenje fitnesa: jedinka vredi onoliko koliko
partija može da dobije, pa se agenti sučeljavaju direktno, a pobednik osvaja poene.
"""

from engine.ai import GomokuAI
from engine.board import GomokuBoard
from lib.constants import Stone


def play_game(black_weights, white_weights, depth=1, radius=1, size=15,
              max_moves=None, seed=None):
    """Odigra jednu partiju; vraća pobednički ``Stone`` ili ``None`` za nerešeno.

    Podrazumevano se igra na punoj tabli 15x15; plitka pretraga (``depth=1``)
    drži self-play dovoljno brzim za veliki broj partija koje GA zahteva.
    """
    board = GomokuBoard(size)
    black = GomokuAI(seed=seed, depth=depth, radius=radius, weights=black_weights)
    white = GomokuAI(
        seed=None if seed is None else seed + 1,
        depth=depth, radius=radius, weights=white_weights,
    )

    limit = max_moves if max_moves is not None else size * size
    for _ in range(limit):
        agent = black if board.current_stone == Stone.BLACK else white
        move = agent.choose_move(board)
        if move is None:
            break
        board.make_move(*move)
        if board.player_has_won():
            return board.board[board.last_move]
    return None


def _points(result, playing_as):
    """Poeni iz jedne partije za boju ``playing_as``: pobeda 1.0, nerešeno 0.5, poraz 0.0."""
    if result is None:
        return 0.5
    return 1.0 if result == playing_as else 0.0


def match_score(a_weights, b_weights, depth=1, radius=1, size=15, seed=None):
    """Rezultat ``a`` u meču od dve partije protiv ``b`` (svaka boja po jednom).

    Vraća vrednost u [0, 2]: 1 poen po pobedi, 0.5 po nerešenom. Igranje obe
    boje uklanja prednost prvog poteza (crni) iz poređenja.
    """
    as_black = play_game(a_weights, b_weights, depth, radius, size, seed=seed)
    as_white = play_game(b_weights, a_weights, depth, radius, size, seed=seed)
    return _points(as_black, Stone.BLACK) + _points(as_white, Stone.WHITE)
