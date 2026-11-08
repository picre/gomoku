"""Knjiga otvaranja: mapa sekvenci poteza u algebarskoj notaciji → sledeći potez.

Algebarska notacija: kolone A–P bez I; red 1 je dno table. Ključ čine potezi
spojeni zarezom; prazan ključ označava početak partije.

Ako tačan ključ ne postoji, a prvi potez je u zoni ±'CENTER_OFFSET' oko
centra, pokušava se translacija: istorija se mapira na koren neke knjiške
linije u istoj zoni (najbliži koren), pa se odgovor vraća pomeranjem nazad.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

COLS = "ABCDEFGHJKLMNOP"
BOARD_SIZE = 15
CENTER = (BOARD_SIZE // 2, BOARD_SIZE // 2)  # (7, 7) == H8
CENTER_OFFSET = 2  # dozvoljeni ofset sidra od centra za translaciju
BOOK_PATH = Path(__file__).resolve().parents[1] / "data" / "opening_book.json"


@cache
def _load() -> dict[str, str]:
    """Učitava knjigu iz 'BOOK_PATH' (jednom, zatim iz keša).

    Ako datoteka ne postoji, vraća praznu knjigu, pa se svi potezi biraju pretragom.
    """
    if not BOOK_PATH.exists():
        return {}
    with open(BOOK_PATH, encoding="utf-8") as handle:
        return {str(k): str(v) for k, v in json.load(handle).items()}


def _in_center_zone(row: int, col: int) -> bool:
    """Tačno ako je polje unutar ±CENTER_OFFSET od geometrijskog centra."""
    cr, cc = CENTER
    return abs(row - cr) <= CENTER_OFFSET and abs(col - cc) <= CENTER_OFFSET


def _on_board(row: int, col: int) -> bool:
    """Tačno ako su koordinate unutar table."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


@cache
def _book_zone_roots() -> tuple[tuple[int, int], ...]:
    """Jedinstveni prvi potezi knjiških ključeva koji leže u centralnoj zoni."""
    book = _load()
    roots: set[tuple[int, int]] = set()
    for key in book:
        if not key:
            continue
        first = key.split(",", 1)[0]
        row, col = algebraic_to_rc(first)
        if _in_center_zone(row, col):
            roots.add((row, col))
    # bliži centru prvi pri istoj udaljenosti od sidra (videti 'lookup')
    return tuple(sorted(
        roots,
        key=lambda rc: abs(rc[0] - CENTER[0]) + abs(rc[1] - CENTER[1]),
    ))


def rc_to_algebraic(row: int, col: int) -> str:
    """Pretvara (row, col) sa indeksima od nule u algebarsku notaciju (npr. 7,7 → H8)."""
    return f"{COLS[col]}{BOARD_SIZE - row}"


def algebraic_to_rc(move: str) -> tuple[int, int]:
    """Pretvara algebarsku notaciju (npr. H8) u (row, col) sa indeksima od nule."""
    col = COLS.index(move[0].upper())
    row = BOARD_SIZE - int(move[1:])
    return row, col


def history_key(history: list[tuple[int, int]]) -> str:
    """Ključ knjige od liste (row, col) poteza."""
    return ",".join(rc_to_algebraic(r, c) for r, c in history)


def _exact_lookup(history: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Tačan unos u mapi, bez translacije."""
    book = _load()
    move = book.get(history_key(history))
    if move is None:
        return None
    return algebraic_to_rc(move)


def _translated_lookup(history: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Pretraživanje knjige sa translacijom na najbliži knjiški koren u centralnoj zoni.

    Istorija se pomeri tako da prvi potez padne na koren R; ako postoji knjiški
    nastavak, odgovor se pomeri nazad za isti ofset. Koren se bira po
    Manhattan udaljenosti od stvarnog prvog poteza, zatim po blizini centru.
    """
    h0 = history[0]
    if not _in_center_zone(*h0):
        return None

    book = _load()
    roots = _book_zone_roots()
    # najbliži stvarnom sidru; pri izjednačenju bliži geometrijskom centru
    ordered = sorted(
        roots,
        key=lambda rc: (
            abs(rc[0] - h0[0]) + abs(rc[1] - h0[1]),
            abs(rc[0] - CENTER[0]) + abs(rc[1] - CENTER[1]),
        ),
    )
    for root in ordered:
        dr, dc = h0[0] - root[0], h0[1] - root[1]
        if dr == 0 and dc == 0:
            # isti koren — tačno pretraživanje je već promašilo ovu granu
            continue
        mapped: list[tuple[int, int]] = []
        ok = True
        for r, c in history:
            mr, mc = r - dr, c - dc
            if not _on_board(mr, mc):
                ok = False
                break
            mapped.append((mr, mc))
        if not ok:
            continue
        move = book.get(history_key(mapped))
        if move is None:
            continue
        nr, nc = algebraic_to_rc(move)
        out_r, out_c = nr + dr, nc + dc
        if _on_board(out_r, out_c):
            return out_r, out_c
    return None


def lookup(history: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Vraća sledeći potez iz knjige, ili None ako nema nastavka.

    Redosled: tačan ključ, zatim translacija ako je prvi potez u zoni ±2
    oko centra.
    """
    exact = _exact_lookup(history)
    if exact is not None:
        return exact
    if not history:
        return None
    return _translated_lookup(history)
