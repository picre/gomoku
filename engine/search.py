"""Izbor poteza pomoću Minimax algoritma sa alfa-beta odsecanjem.

Pretraga radi nad običnim 'int' numpy nizom (za kodiranje videti
'engine.evaluation'), a ne nad tablom iz GUI-ja, tako da pozicije mogu da
se isprobavaju i poništavaju u mestu bez alociranja objekta table po čvoru.
Da bi faktor grananja ostao razuman na tabli 15x15, kao kandidati se razmatraju
samo prazna polja u okviru 'radius' od nekog postojećeg kamenčića.

Optimizacije koje ubrzavaju pretragu:

* **Inkrementalna evaluacija** ('engine.incremental.IncrementalEvaluator') —
  postavljanje ili uklanjanje kamenčića menja samo 4 pravca kroz to polje.
* **Transpoziciona tabela** (Zobrist heš) — kešira ocene pozicija.
* **Uređivanje poteza** — potezi se sortiraju po plitkoj oceni radi jačeg
  alfa-beta odsecanja.
"""

import numpy as np

from engine.evaluation import BLACK, WHITE, EMPTY, PATTERN_CATEGORY_INDEX, WIN_LENGTH
from engine.incremental import IncrementalEvaluator

INF = float("inf")
WIN_SCORE = 10_000_000.0
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# oznake za transpozicionu tabelu (tačna vrednost / donja granica / gornja granica)
_EXACT, _LOWER, _UPPER = 0, 1, 2


def board_to_int(board):
    """Konvertuje 'engine.board.GomokuBoard' u int niz."""
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
    """Prazna polja u okviru 'radius' od nekog kamenčića (centar ako je tabla prazna).

    Samostalna verzija (npr. za testove); pretraga koristi bržu varijantu iz
    'engine.incremental.IncrementalEvaluator' koja održava skup zauzetih polja.
    """
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


class _Searcher:
    """Minimax sa alfa-beta odsecanjem, transpozicionom tabelom i uređivanjem poteza."""

    def __init__(self, ev, ai_stone, radius):
        self.ev = ev
        self.ai = ai_stone
        self.radius = radius
        self.tt = {}

    def _order(self, moves, player, maximizing):
        """Sortira poteze po plitkoj (inkrementalnoj) oceni radi boljeg odsecanja."""
        scored = []
        for (r, c) in moves:
            self.ev.place(r, c, player)
            scored.append((self.ev.value(self.ai), (r, c)))
            self.ev.remove(r, c)
        scored.sort(key=lambda item: item[0], reverse=maximizing)
        return [move for _, move in scored]

    def _leaf_value(self, maximizing, player):
        """Ocena u listu: forsirani ishod preko brojača pretnji, inače heuristika.

        Na neparnoj dubini obična evaluacija ne vidi otvorenu četvorku: AI odbrani
        jedan kraj, pretraga stane, a drugi kraj i dalje pobeđuje. Zato u listu
        proverava se da li igrač na potezu već ima četvorku / otvorenu četvorku.
        """
        four_i = PATTERN_CATEGORY_INDEX["four"]
        open_four_i = PATTERN_CATEGORY_INDEX["open_four"]
        mine = self.ev.total[player]
        if int(mine[open_four_i]) > 0 or int(mine[four_i]) > 0:
            return WIN_SCORE if maximizing else -WIN_SCORE
        other = WHITE if player == BLACK else BLACK
        if int(self.ev.total[other][open_four_i]) > 0:
            return -WIN_SCORE if maximizing else WIN_SCORE
        return self.ev.value(self.ai)

    def search(self, depth, alpha, beta, maximizing, player):
        """Rekurzivni alfa-beta minimax; vraća ocenu iz ugla 'self.ai'."""
        key = (self.ev.hash, maximizing)
        entry = self.tt.get(key)
        if entry is not None and entry[0] >= depth:
            flag, val = entry[1], entry[2]
            if flag == _EXACT:
                return val
            if flag == _LOWER and val > alpha:
                alpha = val
            elif flag == _UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val

        moves = self.ev.candidates(self.radius)
        if depth == 0 or not moves:
            return self._leaf_value(maximizing, player)

        alpha0, beta0 = alpha, beta
        other = WHITE if player == BLACK else BLACK
        if depth > 1:
            moves = self._order(moves, player, maximizing)

        if maximizing:
            best = -INF
            for (r, c) in moves:
                self.ev.place(r, c, player)
                if _wins(self.ev.arr, r, c):
                    val = WIN_SCORE + depth
                else:
                    val = self.search(depth - 1, alpha, beta, False, other)
                self.ev.remove(r, c)
                if val > best:
                    best = val
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        else:
            best = INF
            for (r, c) in moves:
                self.ev.place(r, c, player)
                if _wins(self.ev.arr, r, c):
                    val = -(WIN_SCORE + depth)
                else:
                    val = self.search(depth - 1, alpha, beta, True, other)
                self.ev.remove(r, c)
                if val < best:
                    best = val
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break

        if best <= alpha0:
            flag = _UPPER
        elif best >= beta0:
            flag = _LOWER
        else:
            flag = _EXACT
        self.tt[key] = (depth, flag, best)
        return best


def search_best_move(arr, ai_stone, weights, depth=3, radius=1, rng=None):
    """Bira najbolji potez za 'ai_stone' na 'arr' pomoću minimax-a.

    Vraća (row, col) torku, ili 'None' ako nema legalnog poteza. Izjednačeni
    potezi se biraju pomoću 'rng' kada je prosleđen, tako da partije nisu
    potpuno deterministične.
    """
    opponent = WHITE if ai_stone == BLACK else BLACK
    ev = IncrementalEvaluator(arr, weights)
    moves = ev.candidates(radius)
    if not moves:
        if np.any(arr == EMPTY):
            raise RuntimeError("nema kandidat-poteza iako tabla nije puna")
        return None

    # momentalna pobeda ako postoji
    for (r, c) in moves:
        ev.place(r, c, ai_stone)
        won = _wins(ev.arr, r, c)
        ev.remove(r, c)
        if won:
            return (r, c)

    # uređivanje poteza na korenu po plitkoj oceni
    scored = []
    for (r, c) in moves:
        ev.place(r, c, ai_stone)
        scored.append((ev.value(ai_stone), (r, c)))
        ev.remove(r, c)
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [move for _, move in scored]

    searcher = _Searcher(ev, ai_stone, radius)
    best, best_moves = -INF, []
    # Na korenu svaki kandidat ide sa punim prozorom (-inf, +inf).
    # Ako se alpha prenosi sa prethodnog poteza, fail-soft alfa-beta često
    # vraća istu granicu i za loše poteze, pa RNG izjednačava dobar i loš potez.
    for (r, c) in ordered:
        ev.place(r, c, ai_stone)
        if _wins(ev.arr, r, c):
            val = WIN_SCORE + depth
        else:
            val = searcher.search(depth - 1, -INF, INF, False, opponent)
        ev.remove(r, c)
        if val > best:
            best, best_moves = val, [(r, c)]
        elif val == best:
            best_moves.append((r, c))

    if rng is not None:
        return rng.choice(best_moves)
    return best_moves[0]
