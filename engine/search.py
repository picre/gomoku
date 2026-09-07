"""Izbor poteza pomoću Minimax algoritma sa alfa-beta odsecanjem.

Pretraga radi nad običnim ``int`` numpy nizom (za kodiranje videti
:mod:`engine.evaluation`), a ne nad tablom iz GUI-ja, tako da pozicije mogu da
se isprobavaju i poništavaju u mestu bez alociranja objekta table po čvoru.
Da bi faktor grananja ostao razuman na tabli 15x15, kao kandidati se razmatraju
samo prazna polja u okviru ``radius`` od nekog postojećeg kamenčića.

Optimizacije koje ubrzavaju pretragu:

* **Inkrementalna evaluacija** (:class:`IncrementalEvaluator`) — postavljanje ili
  uklanjanje kamenčića menja samo 4 linije kroz to polje, pa se tekuća ocena
  ažurira u O(4) linija umesto ponovnog skeniranja cele table u svakom čvoru.
* **Transpoziciona tabela** (Zobrist heš) — kešira ocene pozicija, tako da se
  ista pozicija dostignuta različitim redosledom poteza ne pretražuje ponovo.
* **Uređivanje poteza** — potezi se sortiraju po plitkoj oceni radi jačeg
  alfa-beta odsecanja.
"""

import random

import numpy as np

from engine.evaluation import (
    BLACK,
    EMPTY,
    WHITE,
    WIN_LENGTH,
    _adjacent_pairs,
    count_line,
    get_cell_lines,
    get_lines,
    score_from_counts,
    NUM_CATS,
)

INF = float("inf")
WIN_SCORE = 10_000_000.0
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# oznake za transpozicionu tabelu (tačna vrednost / donja granica / gornja granica)
_EXACT, _LOWER, _UPPER = 0, 1, 2

# keširane Zobrist tabele po veličini table
_ZOBRIST_CACHE = {}


def _get_zobrist(n):
    """Vraća (keširanu) Zobrist tabelu slučajnih ključeva oblika [n][n][2]."""
    if n not in _ZOBRIST_CACHE:
        rng = random.Random(0xC0FFEE ^ n)
        _ZOBRIST_CACHE[n] = [
            [[rng.getrandbits(63) for _ in range(2)] for _ in range(n)]
            for _ in range(n)
        ]
    return _ZOBRIST_CACHE[n]


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
    """Prazna polja u okviru ``radius`` od nekog kamenčića (centar ako je tabla prazna).

    Samostalna verzija (npr. za testove); pretraga koristi bržu varijantu iz
    :class:`IncrementalEvaluator` koja održava skup zauzetih polja.
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


class IncrementalEvaluator:
    """Održava tekuću ocenu pozicije i ažurira je pri svakom potezu u O(4) linija.

    Deli isti niz (``arr``) sa pretragom i menja ga kroz :meth:`place`/:meth:`remove`.
    Uz brojače obrazaca po perspektivi, održava i kontrolu centra, povezanost,
    skup zauzetih polja i Zobrist heš pozicije.
    """

    def __init__(self, arr, weights):
        """Inicijalizuje evaluator iz zadate pozicije (jednom skenira sve linije)."""
        self.arr = arr
        self.weights = weights
        n = arr.shape[0]
        self.n = n
        self.lines = get_lines(n)
        self.cell_lines = get_cell_lines(n)
        self.center_ref = (n - 1) / 2.0
        self.zob = _get_zobrist(n)

        self.line_counts = {
            BLACK: [None] * len(self.lines),
            WHITE: [None] * len(self.lines),
        }
        self.total = {BLACK: np.zeros(NUM_CATS, dtype=np.int64),
                      WHITE: np.zeros(NUM_CATS, dtype=np.int64)}
        self.center = {BLACK: 0.0, WHITE: 0.0}
        self.conn = {BLACK: 0, WHITE: 0}
        self.occupied = set()
        self.hash = 0

        # brojači obrazaca po linijama i njihov zbir
        for li, (rows, cols) in enumerate(self.lines):
            line = arr[rows, cols]
            for color in (BLACK, WHITE):
                opp = WHITE if color == BLACK else BLACK
                vec = count_line(line, color, opp)
                self.line_counts[color][li] = vec
                self.total[color] += vec

        # kontrola centra, zauzetost i Zobrist heš iz postojećih kamenčića
        ys, xs = np.nonzero(arr != EMPTY)
        for r, c in zip(ys.tolist(), xs.tolist()):
            stone = int(arr[r, c])
            self.occupied.add((r, c))
            self.center[stone] += self._cell_center(r, c)
            self.hash ^= self.zob[r][c][stone]
        # povezanost (broj susednih parova) po boji
        for color in (BLACK, WHITE):
            self.conn[color] = _adjacent_pairs(arr == color)

    def _cell_center(self, r, c):
        """Doprinos jednog polja kontroli centra (bliže centru = veći)."""
        return self.n - (abs(r - self.center_ref) + abs(c - self.center_ref))

    def _neighbors(self, r, c, stone):
        """Broj susednih polja (8 pravaca) koja sadrže kamenčić ``stone``."""
        n, arr, cnt = self.n, self.arr, 0
        for dr, dc in NEIGHBORS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
                cnt += 1
        return cnt

    def _recompute_cell(self, r, c):
        """Preračunava brojače obrazaca za sve linije koje prolaze kroz (r, c)."""
        for li in self.cell_lines.get((r, c), ()):
            rows, cols = self.lines[li]
            line = self.arr[rows, cols]
            for color in (BLACK, WHITE):
                opp = WHITE if color == BLACK else BLACK
                vec = count_line(line, color, opp)
                self.total[color] += vec - self.line_counts[color][li]
                self.line_counts[color][li] = vec

    def place(self, r, c, stone):
        """Postavlja ``stone`` na (r, c) i inkrementalno ažurira sve pokazatelje."""
        self.arr[r, c] = stone
        self.occupied.add((r, c))
        self.center[stone] += self._cell_center(r, c)
        self.conn[stone] += self._neighbors(r, c, stone)
        self.hash ^= self.zob[r][c][stone]
        self._recompute_cell(r, c)

    def remove(self, r, c):
        """Uklanja kamenčić sa (r, c) i poništava sve inkrementalne promene."""
        stone = int(self.arr[r, c])
        self.conn[stone] -= self._neighbors(r, c, stone)
        self.center[stone] -= self._cell_center(r, c)
        self.hash ^= self.zob[r][c][stone]
        self.arr[r, c] = EMPTY
        self.occupied.discard((r, c))
        self._recompute_cell(r, c)

    def value(self, own):
        """Trenutna ocena pozicije iz ugla igrača ``own`` (bez ponovnog skeniranja)."""
        opp = WHITE if own == BLACK else BLACK
        center_diff = self.center[own] - self.center[opp]
        conn_diff = self.conn[own] - self.conn[opp]
        return score_from_counts(
            self.total[own], self.total[opp], center_diff, conn_diff, self.weights
        )

    def candidates(self, radius):
        """Prazna polja u okviru ``radius`` od zauzetih (centar ako je tabla prazna)."""
        if not self.occupied:
            return [(self.n // 2, self.n // 2)]
        n, arr, cand = self.n, self.arr, set()
        for (r, c) in self.occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
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

    def search(self, depth, alpha, beta, maximizing, player):
        """Rekurzivni alfa-beta minimax; vraća ocenu iz ugla ``self.ai``."""
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
            return self.ev.value(self.ai)

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


def search_best_move(arr, ai_stone, weights, depth=2, radius=1, rng=None):
    """Bira najbolji potez za ``ai_stone`` na ``arr`` pomoću minimax-a.

    Vraća (row, col) torku, ili ``None`` ako nema legalnog poteza. Izjednačeni
    potezi se biraju pomoću ``rng`` kada je prosleđen, tako da partije nisu
    potpuno deterministične.
    """
    opp = WHITE if ai_stone == BLACK else BLACK
    ev = IncrementalEvaluator(arr, weights)
    moves = ev.candidates(radius)
    if not moves:
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
    alpha, beta = -INF, INF
    for (r, c) in ordered:
        ev.place(r, c, ai_stone)
        if _wins(ev.arr, r, c):
            val = WIN_SCORE + depth
        else:
            val = searcher.search(depth - 1, alpha, beta, False, opp)
        ev.remove(r, c)
        if val > best:
            best, best_moves = val, [(r, c)]
        elif val == best:
            best_moves.append((r, c))
        if best > alpha:
            alpha = best

    if rng is not None:
        return rng.choice(best_moves)
    return best_moves[0]
