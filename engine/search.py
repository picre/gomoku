"""Izbor poteza pomoću Minimax algoritma sa alfa-beta odsecanjem.

Pretraga radi nad običnim ``int`` numpy nizom (za kodiranje videti
:mod:`engine.evaluation`), a ne nad tablom iz GUI-ja, tako da pozicije mogu da
se isprobavaju i poništavaju u mestu bez alociranja objekta table po čvoru.
Da bi faktor grananja ostao razuman na tabli 15x15, kao kandidati se razmatraju
samo prazna polja u okviru ``radius`` od nekog postojećeg kamenčića.
"""

import numpy as np

from engine.evaluation import BLACK, EMPTY, WHITE, WIN_LENGTH, evaluate

INF = float("inf")
WIN_SCORE = 10_000_000.0
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def board_to_int(board):
    """Konvertuje :class:`~engine.board.GomokuBoard` u int niz."""
    n = board.board_size
    arr = np.empty((n, n), dtype=np.int8)
    for r in range(n):
        for c in range(n):
            arr[r, c] = int(board.board[r][c].value)
    return arr


def _wins(arr, row, col):
    """Tačno ako upravo postavljeni kamenčić na (row, col) formira pet u nizu."""
    stone = arr[row, col]
    if stone == EMPTY:
        return False
    n = arr.shape[0]
    for dr, dc in DIRECTIONS:
        count = 1
        rr, cc = row + dr, col + dc
        while 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
            count += 1
            rr += dr
            cc += dc
        rr, cc = row - dr, col - dc
        while 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
            count += 1
            rr -= dr
            cc -= dc
        if count >= WIN_LENGTH:
            return True
    return False


def candidate_moves(arr, radius=1):
    """Prazna polja u okviru ``radius`` od nekog kamenčića (centar ako je tabla prazna)."""
    n = arr.shape[0]
    occupied = np.argwhere(arr != EMPTY)
    if occupied.size == 0:
        return [(n // 2, n // 2)]
    cand = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = int(row) + dr, int(col) + dc
                if 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == EMPTY:
                    cand.add((rr, cc))
    return list(cand)


def _minimax(arr, depth, alpha, beta, maximizing, ai_stone, opp, weights, radius):
    """Standardni alfa-beta minimax; vraća ocenu iz ugla ``ai_stone``."""
    moves = candidate_moves(arr, radius)
    if not moves:
        return evaluate(arr, ai_stone, weights)

    if maximizing:
        stone, best = ai_stone, -INF
        for row, col in moves:
            arr[row, col] = stone
            if _wins(arr, row, col):
                arr[row, col] = EMPTY
                return WIN_SCORE + depth
            score = (
                evaluate(arr, ai_stone, weights)
                if depth <= 1
                else _minimax(arr, depth - 1, alpha, beta, False, ai_stone, opp, weights, radius)
            )
            arr[row, col] = EMPTY
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    stone, best = opp, INF
    for row, col in moves:
        arr[row, col] = stone
        if _wins(arr, row, col):
            arr[row, col] = EMPTY
            return -(WIN_SCORE + depth)
        score = (
            evaluate(arr, ai_stone, weights)
            if depth <= 1
            else _minimax(arr, depth - 1, alpha, beta, True, ai_stone, opp, weights, radius)
        )
        arr[row, col] = EMPTY
        if score < best:
            best = score
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return best


def search_best_move(arr, ai_stone, weights, depth=2, radius=1, rng=None):
    """Bira najbolji potez za ``ai_stone`` na ``arr`` pomoću minimax-a.

    Vraća (row, col) torku, ili ``None`` ako nema legalnog poteza. Izjednačeni
    potezi se biraju pomoću ``rng`` kada je prosleđen, tako da partije nisu
    potpuno deterministične.
    """
    opp = WHITE if ai_stone == BLACK else BLACK
    moves = candidate_moves(arr, radius)
    if not moves:
        return None

    # sortiraj poteze po plitkoj statičkoj oceni (i uhvati momentalnu pobedu)
    # da bi alfa-beta agresivno odsecalo.
    scored = []
    for row, col in moves:
        arr[row, col] = ai_stone
        if _wins(arr, row, col):
            arr[row, col] = EMPTY
            return (row, col)
        scored.append((evaluate(arr, ai_stone, weights), (row, col)))
        arr[row, col] = EMPTY
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [move for _, move in scored]

    best, best_moves = -INF, []
    alpha, beta = -INF, INF
    for row, col in ordered:
        arr[row, col] = ai_stone
        score = (
            evaluate(arr, ai_stone, weights)
            if depth <= 1
            else _minimax(arr, depth - 1, alpha, beta, False, ai_stone, opp, weights, radius)
        )
        arr[row, col] = EMPTY
        if score > best:
            best, best_moves = score, [(row, col)]
        elif score == best:
            best_moves.append((row, col))
        if best > alpha:
            alpha = best

    if rng is not None:
        return rng.choice(best_moves)
    return best_moves[0]
