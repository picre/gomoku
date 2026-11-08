"""Testovi knjige otvaranja."""

from engine.opening_book import _load, algebraic_to_rc, history_key, lookup, rc_to_algebraic


def test_algebraic_round_trip():
    for row in range(15):
        for col in range(15):
            assert algebraic_to_rc(rc_to_algebraic(row, col)) == (row, col)
    assert rc_to_algebraic(7, 7) == "H8"
    assert "I" not in "".join(rc_to_algebraic(0, c) for c in range(15))


def test_exact_lookup_follows_book():
    book = _load()
    assert lookup([]) == algebraic_to_rc(book[""])
    first = algebraic_to_rc(book[""])
    assert lookup([first]) == algebraic_to_rc(book[history_key([first])])


def test_translated_lookup_keeps_relative_answer():
    book = _load()
    # pomeraj (odgovor - prvi potez) za svaki knjiški koren sa jednim potezom
    offsets = set()
    for key, answer in book.items():
        if key and "," not in key:
            (r0, c0), (r1, c1) = algebraic_to_rc(key), algebraic_to_rc(answer)
            offsets.add((r1 - r0, c1 - c0))
    roots = {algebraic_to_rc(key) for key in book if key and "," not in key}
    # prvo polje centralne zone koje nije koren knjige
    first = next(
        (r, c) for r in range(5, 10) for c in range(5, 10) if (r, c) not in roots
    )
    answer = lookup([first])
    assert answer is not None
    assert (answer[0] - first[0], answer[1] - first[1]) in offsets


def test_far_from_center_has_no_book_move():
    assert lookup([(0, 0)]) is None
