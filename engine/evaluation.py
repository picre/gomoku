"""Statička ocena Gomoku pozicije.

Ocena se gradi prepoznavanjem karakterističnih obrazaca na tabli
(otvorene/zatvorene dvojke, trojke i četvorke, forkovi, plus pozicioni faktori
kao što su kontrola centra i povezanost kamenčića). Svaki obrazac nosi
određeni broj poena; težina svakog obrasca čuva se u ``weights`` rečniku kako
bi genetski algoritam mogao da ih podešava.

Tabla se prosleđuje kao mali ``int`` numpy niz sa istim vrednostima kao
:class:`lib.constants.Stone` (BLACK=0, WHITE=1, EMPTY=2), da bi pretraga mogla
jeftino da kopira i isprobava pozicije, nezavisno od table iz GUI-ja.
"""

import json
import re
from pathlib import Path

import numpy as np

from lib.constants import Stone

# --- kodiranje kamenčića (poklapa se sa vrednostima Stone enum-a) ---------
BLACK = int(Stone.BLACK.value)   # 0
WHITE = int(Stone.WHITE.value)   # 1
EMPTY = int(Stone.EMPTY.value)   # 2

WIN_LENGTH = 5

# razuman ručno izabran polazni skup; GA pretražuje u okolini ovih vrednosti
DEFAULT_WEIGHTS = {
    "five": 100000.0,
    "open_four": 10000.0,
    "four": 1000.0,
    "open_three": 1000.0,
    "three": 100.0,
    "open_two": 100.0,
    "two": 10.0,
    "center": 3.0,
    "connectivity": 5.0,
    "fork": 2000.0,
    "defense": 1.1,
}

# kanonski redosled gena u GA vektoru težina. Izveden iz DEFAULT_WEIGHTS
# (rečnici čuvaju redosled umetanja) tako da njih dva nikada ne mogu da se raziđu.
WEIGHT_FIELDS = tuple[str, ...](DEFAULT_WEIGHTS)

# Katalog obrazaca, iz perspektive igrača koji je na potezu:
#   '1' sopstveni kamenčić, '2' protivnički kamenčić ILI zid, '0' prazno polje.
# Linije se sa obe strane dopunjavaju '2' da bi ivice table brojale kao blokada.
PATTERNS = {
    "five": ["11111"],
    "open_four": ["011110"],
    "four": ["011112", "211110", "10111", "11011", "11101"],
    "open_three": ["011100", "001110", "011010", "010110"],
    "three": ["211100", "001112", "211010", "010112", "210110", "011012"],
    "open_two": ["001100", "011000", "000110", "010100", "001010"],
    "two": ["211000", "000112", "210100", "001012", "010010"],
}
_PATTERN_CATS = ["five", "open_four", "four", "open_three", "three", "open_two", "two"]

# unapred kompajlirani matcheri sa preklapanjem (lookahead dozvoljava da se obrasci preklapaju)
_COMPILED = {
    cat: [re.compile("(?=(%s))" % p) for p in pats] for cat, pats in PATTERNS.items()
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights.json"


# --- pomoćne funkcije za težine -------------------------------------------
def weights_to_vector(weights):
    """Pretvara rečnik težina u uređeni numpy vektor (za GA)."""
    return np.array([float(weights[f]) for f in WEIGHT_FIELDS], dtype=float)


def vector_to_weights(vector):
    """Inverzna funkcija od :func:`weights_to_vector`."""
    return {f: float(vector[i]) for i, f in enumerate(WEIGHT_FIELDS)}


def load_weights(path=None):
    """Učitava evoluirane težine iz JSON-a, uz vraćanje na podrazumevane."""
    p = Path(path) if path is not None else WEIGHTS_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {k: float(data.get(k, DEFAULT_WEIGHTS[k])) for k in WEIGHT_FIELDS}
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights, path=None):
    """Trajno upisuje rečnik težina u JSON."""
    p = Path(path) if path is not None else WEIGHTS_PATH
    with open(p, "w", encoding="utf-8") as handle:
        json.dump({k: float(weights[k]) for k in WEIGHT_FIELDS}, handle, indent=2)
    return p


# --- izdvajanje linija ----------------------------------------------------
def _iter_lines(arr):
    """Vraća (yield) svaku vrstu, kolonu i dijagonalu dužine >= WIN_LENGTH."""
    n = arr.shape[0]
    for i in range(n):
        yield arr[i, :]
        yield arr[:, i]
    flipped = np.fliplr(arr)
    for offset in range(-(n - 1), n):
        d = np.diagonal(arr, offset=offset)
        if d.size >= WIN_LENGTH:
            yield d
        a = np.diagonal(flipped, offset=offset)
        if a.size >= WIN_LENGTH:
            yield a


def _line_to_string(line, own, opp):
    """Kodira liniju iz perspektive ``own``, dopunjenu blokirajućim zidovima."""
    lookup = []
    for value in line:
        if value == own:
            lookup.append("1")
        elif value == opp:
            lookup.append("2")
        else:
            lookup.append("0")
    return "2" + "".join(lookup) + "2"


def _count_patterns(arr, own, opp):
    """Broji svaki katalogizovani obrazac za ``own`` kroz sve linije."""
    counts = {cat: 0 for cat in PATTERNS}
    for line in _iter_lines(arr):
        text = _line_to_string(line, own, opp)
        for cat, matchers in _COMPILED.items():
            total = 0
            for matcher in matchers:
                total += len(matcher.findall(text))
            counts[cat] += total
    return counts


# --- pozicioni faktori ----------------------------------------------------
def _adjacent_pairs(mask):
    """Broj ortogonalno/dijagonalno susednih parova kamenčića u ``mask``."""
    m = mask.astype(np.int32)
    pairs = 0
    pairs += int(np.sum(m[:, :-1] & m[:, 1:]))
    pairs += int(np.sum(m[:-1, :] & m[1:, :]))
    pairs += int(np.sum(m[:-1, :-1] & m[1:, 1:]))
    pairs += int(np.sum(m[:-1, 1:] & m[1:, :-1]))
    return pairs


def _positional(arr, own, opp):
    """Vraća (razlika u kontroli centra, razlika u povezanosti)."""
    n = arr.shape[0]
    center = (n - 1) / 2.0

    def center_score(mask):
        """Zbir blizine centru za sve kamenčiće u ``mask`` (bliže centru = više)."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return 0.0
        return float(np.sum(n - (np.abs(ys - center) + np.abs(xs - center))))

    own_mask = arr == own
    opp_mask = arr == opp
    center_diff = center_score(own_mask) - center_score(opp_mask)
    conn_diff = _adjacent_pairs(own_mask) - _adjacent_pairs(opp_mask)
    return center_diff, conn_diff


# --- glavna ulazna tačka --------------------------------------------------
def evaluate(arr, own, weights):
    """Ocenjuje ``arr`` iz ugla igrača ``own``.

    Pozitivne vrednosti idu u korist ``own``; negativne u korist protivnika.
    Protivnički obrasci se oduzimaju (skalirani ``defense`` težinom) tako da
    pretraga prirodno uči i da blokira pretnje, a ne samo da gradi svoje.
    """
    opp = WHITE if own == BLACK else BLACK
    own_counts = _count_patterns(arr, own, opp)
    opp_counts = _count_patterns(arr, opp, own)

    own_pat = sum(weights[c] * own_counts[c] for c in _PATTERN_CATS)
    opp_pat = sum(weights[c] * opp_counts[c] for c in _PATTERN_CATS)

    # fork = više istovremenih pretnji odjednom (dupla trojka / četvorka itd.)
    own_threats = own_counts["open_three"] + own_counts["four"] + 2 * own_counts["open_four"]
    opp_threats = opp_counts["open_three"] + opp_counts["four"] + 2 * opp_counts["open_four"]
    own_pat += weights["fork"] * max(0, own_threats - 1)
    opp_pat += weights["fork"] * max(0, opp_threats - 1)

    center_diff, conn_diff = _positional(arr, own, opp)

    return (
        own_pat
        - weights["defense"] * opp_pat
        + weights["center"] * center_diff
        + weights["connectivity"] * conn_diff
    )
