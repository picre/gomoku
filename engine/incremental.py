"""Inkrementalna evaluacija pozicije tokom pretrage.

'IncrementalEvaluator' održava brojače obrazaca, kontrolu centra,
povezanost, skup zauzetih polja i Zobrist heš, i ažurira ih u O(4) pravca pri
svakom 'place'/'remove'. Koristi ga Minimax u 'engine.search'.
"""

import random

import numpy as np

from engine.evaluation import (
    BLACK,
    EMPTY,
    NUM_PATTERN_CATEGORIES,
    WHITE,
    _adjacent_pairs,
    count_span,
    get_cell_spans,
    get_spans,
    score_from_counts,
)

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Zobrist tabele po veličini table
_ZOBRIST = {}


def get_zobrist(n):
    """Vraća Zobrist tabelu slučajnih ključeva oblika [n][n][2]."""
    if n not in _ZOBRIST:
        rng = random.Random(0xC0FFEE ^ n)
        _ZOBRIST[n] = [
            [[rng.getrandbits(63) for _ in range(2)] for _ in range(n)]
            for _ in range(n)
        ]
    return _ZOBRIST[n]


class IncrementalEvaluator:
    """Održava tekuću ocenu pozicije i ažurira je pri svakom potezu u O(4) pravca.

    Deli isti niz ('arr') sa pretragom i menja ga kroz 'place'/'remove'.
    Uz brojače obrazaca po perspektivi, održava i kontrolu centra, povezanost,
    skup zauzetih polja i Zobrist heš pozicije.
    """

    def __init__(self, arr, weights):
        """Inicijalizuje evaluator iz zadate pozicije (jednom skenira sve pravce)."""
        self.arr = arr
        self.weights = weights
        n = arr.shape[0]
        self.n = n
        self.spans = get_spans(n)
        self.cell_spans = get_cell_spans(n)
        self.center_ref = (n - 1) / 2.0
        self.zob = get_zobrist(n)

        self.span_counts = {
            BLACK: [None] * len(self.spans),
            WHITE: [None] * len(self.spans),
        }
        self.total = {BLACK: np.zeros(NUM_PATTERN_CATEGORIES, dtype=np.int64),
                      WHITE: np.zeros(NUM_PATTERN_CATEGORIES, dtype=np.int64)}
        self.center = {BLACK: 0.0, WHITE: 0.0}
        self.conn = {BLACK: 0, WHITE: 0}
        self.occupied = set()
        self.hash = 0

        # brojači obrazaca po pravcima i njihov zbir
        for si, (rows, cols) in enumerate(self.spans):
            span = arr[rows, cols]
            for color in (BLACK, WHITE):
                opponent = WHITE if color == BLACK else BLACK
                counts = count_span(span, color, opponent)
                self.span_counts[color][si] = counts
                self.total[color] += counts

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
        """Broj susednih polja (8 suseda) koja sadrže kamenčić 'stone'."""
        n, arr, cnt = self.n, self.arr, 0
        for dr, dc in NEIGHBORS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
                cnt += 1
        return cnt

    def _recompute_cell(self, r, c):
        """Preračunava brojače obrazaca za sve pravce koje prolaze kroz (r, c)."""
        for si in self.cell_spans.get((r, c), ()):
            rows, cols = self.spans[si]
            span = self.arr[rows, cols]
            for color in (BLACK, WHITE):
                opponent = WHITE if color == BLACK else BLACK
                counts = count_span(span, color, opponent)
                self.total[color] += counts - self.span_counts[color][si]
                self.span_counts[color][si] = counts

    def place(self, r, c, stone):
        """Postavlja 'stone' na (r, c) i inkrementalno ažurira sve pokazatelje."""
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
        """Trenutna ocena pozicije iz ugla igrača 'own' (bez ponovnog skeniranja)."""
        opponent = WHITE if own == BLACK else BLACK
        center_diff = self.center[own] - self.center[opponent]
        conn_diff = self.conn[own] - self.conn[opponent]
        return score_from_counts(
            self.total[own], self.total[opponent], center_diff, conn_diff, self.weights
        )

    def candidates(self, radius):
        """Prazna polja u okviru 'radius' od zauzetih (centar ako je tabla prazna)."""
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
