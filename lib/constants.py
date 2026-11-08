"""Osnovne konstante modela igre."""

import enum


class Stone(enum.IntEnum):
    """Sadržaj polja na tabli: crni kamenčić, beli kamenčić ili prazno polje.

    Vrednosti 'BLACK' i 'WHITE' istovremeno označavaju i igrača. Pošto je
    'IntEnum', vrednosti se direktno porede sa celim brojevima u 'numpy'
    nizovima table, bez konverzije.
    """

    BLACK = 0
    WHITE = 1
    EMPTY = 2

    @property
    def opponent(self) -> "Stone":
        """Vraća protivničku boju ('BLACK' ↔ 'WHITE')."""
        if self is Stone.EMPTY:
            raise ValueError("EMPTY has no opponent")
        return Stone.WHITE if self is Stone.BLACK else Stone.BLACK
