"""Statička ocena Gomoku pozicije.

Ocena se gradi prepoznavanjem karakterističnih obrazaca na tabli
(otvorene/zatvorene dvojke, trojke i četvorke, forkovi, plus pozicioni faktori
kao što su kontrola centra i povezanost kamenčića). Svaki obrazac nosi
određeni broj poena; težina svakog obrasca čuva se u ``weights`` rečniku kako
bi genetski algoritam mogao da ih podešava.

Tabla se prosleđuje kao mali ``int`` numpy niz sa istim vrednostima kao
:class:`lib.constants.Stone` (BLACK=0, WHITE=1, EMPTY=2), da bi pretraga mogla
jeftino da kopira i isprobava pozicije, nezavisno od table iz GUI-ja.

Prebrojavanje obrazaca je optimizovano: umesto regularnih izraza koristi se
**unapred izračunata tabela** za sve moguće prozore dužine 5 i 6 (kodirane u
bazi 3). Skeniranje jedne linije se svodi na klizanje prozora i sabiranje
vektora iz tabele — bez građenja niski i bez regexa. Koordinate svih linija i
mapiranje polje→linije keširaju se po veličini table (koristi ih i inkrementalni
evaluator u :mod:`engine.search`).
"""

import json
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

# uređene kategorije obrazaca (indeks u vektoru brojača); ``fork`` koristi
# indekse: open_four=1, four=2, open_three=3.
CATS = ("five", "open_four", "four", "open_three", "three", "open_two", "two")
CAT_INDEX = {name: i for i, name in enumerate(CATS)}
NUM_CATS = len(CATS)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights.json"


# --- unapred izračunate tabele obrazaca -----------------------------------
def _build_window_table(length):
    """Tabela vektora brojača za sve prozore date dužine (kodirane u bazi 3).

    Za kod prozora ``w[0..L-1]`` (gde je ``code = sum(w[k] * 3**k)``) vraća se
    koliko obrazaca **te dužine** iz kataloga počinje na poziciji 0 prozora.
    Sabiranjem po svim početnim pozicijama u liniji dobija se isti rezultat kao
    prebrojavanje preklapajućih pojava svakog obrasca (kao ranije sa regexom).
    """
    table = np.zeros((3 ** length, NUM_CATS), dtype=np.int64)
    for code in range(3 ** length):
        digits, x = [], code
        for _ in range(length):
            digits.append(x % 3)
            x //= 3
        window = "".join("012"[d] for d in digits)  # pozicija 0..L-1
        for ci, cat in enumerate(CATS):
            for pat in PATTERNS[cat]:
                if len(pat) == length and window.startswith(pat):
                    table[code, ci] += 1
    return table


_TABLE5 = _build_window_table(5)
_TABLE6 = _build_window_table(6)


# --- keš koordinata linija i mape polje→linije ----------------------------
_LINES_CACHE = {}
_CELL_LINES_CACHE = {}


def _build_lines(n):
    """Sve vrste, kolone i dijagonale (dužine >= WIN_LENGTH) kao (rows, cols) nizovi."""
    raw = []
    for r in range(n):                       # vrste
        raw.append([(r, c) for c in range(n)])
    for c in range(n):                       # kolone
        raw.append([(r, c) for r in range(n)])
    for start in range(n):                   # dijagonale (1, 1) — start u gornjoj vrsti
        coords, r, c = [], 0, start
        while r < n and c < n:
            coords.append((r, c)); r += 1; c += 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)
    for start in range(1, n):                # dijagonale (1, 1) — start u levoj koloni
        coords, r, c = [], start, 0
        while r < n and c < n:
            coords.append((r, c)); r += 1; c += 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)
    for start in range(n):                   # anti-dijagonale (1, -1) — start u gornjoj vrsti
        coords, r, c = [], 0, start
        while r < n and c >= 0:
            coords.append((r, c)); r += 1; c -= 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)
    for start in range(1, n):                # anti-dijagonale (1, -1) — start u desnoj koloni
        coords, r, c = [], start, n - 1
        while r < n and c >= 0:
            coords.append((r, c)); r += 1; c -= 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)
    return [
        (np.array([p[0] for p in cs]), np.array([p[1] for p in cs])) for cs in raw
    ]


def get_lines(n):
    """Vraća (keširane) koordinate svih linija za tablu veličine ``n``."""
    if n not in _LINES_CACHE:
        lines = _build_lines(n)
        _LINES_CACHE[n] = lines
        cell_lines = {}
        for li, (rows, cols) in enumerate(lines):
            for r, c in zip(rows.tolist(), cols.tolist()):
                cell_lines.setdefault((r, c), []).append(li)
        _CELL_LINES_CACHE[n] = cell_lines
    return _LINES_CACHE[n]


def get_cell_lines(n):
    """Vraća (keširanu) mapu polje (r, c) → indeksi linija koje ga sadrže."""
    get_lines(n)
    return _CELL_LINES_CACHE[n]


# --- weight helpers -------------------------------------------------------
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


# --- prebrojavanje obrazaca -----------------------------------------------
def count_line(line, own, opp):
    """Vektor brojača obrazaca za jednu liniju, iz perspektive ``own``.

    ``line`` je 1D niz vrednosti kamenčića; mapira se u cifre (own→1, opp→2,
    prazno→0), dopuni zidovima sa obe strane, pa se klizećim prozorima dužine
    5 i 6 sabiraju vektori iz unapred izračunatih tabela.
    """
    digit = np.where(line == own, 1, np.where(line == opp, 2, 0)).astype(np.int64)
    d = np.empty(digit.size + 2, dtype=np.int64)
    d[0] = 2
    d[-1] = 2
    d[1:-1] = digit
    m = d.size
    counts = np.zeros(NUM_CATS, dtype=np.int64)
    if m >= 6:
        c6 = (d[0:m - 5] + 3 * d[1:m - 4] + 9 * d[2:m - 3]
              + 27 * d[3:m - 2] + 81 * d[4:m - 1] + 243 * d[5:m])
        counts += _TABLE6[c6].sum(axis=0)
    if m >= 5:
        c5 = (d[0:m - 4] + 3 * d[1:m - 3] + 9 * d[2:m - 2]
              + 27 * d[3:m - 1] + 81 * d[4:m])
        counts += _TABLE5[c5].sum(axis=0)
    return counts


def board_counts(arr, own, opp, n=None):
    """Zbir brojača obrazaca za ``own`` preko svih linija cele table."""
    if n is None:
        n = arr.shape[0]
    total = np.zeros(NUM_CATS, dtype=np.int64)
    for rows, cols in get_lines(n):
        total += count_line(arr[rows, cols], own, opp)
    return total


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


# --- sastavljanje ocene iz brojača ----------------------------------------
def _pattern_weight_vector(weights):
    """Težine kategorija obrazaca kao numpy vektor (istim redosledom kao CATS)."""
    return np.array([weights[c] for c in CATS], dtype=float)


def _fork_bonus(counts, weights):
    """Bonus za višestruke istovremene pretnje (fork), izveden iz brojača."""
    threats = int(counts[CAT_INDEX["open_three"]]
                  + counts[CAT_INDEX["four"]]
                  + 2 * counts[CAT_INDEX["open_four"]])
    return weights["fork"] * max(0, threats - 1)


def score_from_counts(own_counts, opp_counts, center_diff, conn_diff, weights):
    """Konačna ocena iz brojača obrazaca i pozicionih razlika.

    Zajednička je za punu (:func:`evaluate`) i inkrementalnu evaluaciju, pa su
    im rezultati uvek identični.
    """
    patw = _pattern_weight_vector(weights)
    own_pat = float(patw.dot(own_counts)) + _fork_bonus(own_counts, weights)
    opp_pat = float(patw.dot(opp_counts)) + _fork_bonus(opp_counts, weights)
    return (
        own_pat
        - weights["defense"] * opp_pat
        + weights["center"] * center_diff
        + weights["connectivity"] * conn_diff
    )


# --- glavna ulazna tačka --------------------------------------------------
def evaluate(arr, own, weights):
    """Ocenjuje ``arr`` iz ugla igrača ``own``.

    Pozitivne vrednosti idu u korist ``own``; negativne u korist protivnika.
    Protivnički obrasci se oduzimaju (skalirani ``defense`` težinom) tako da
    pretraga prirodno uči i da blokira pretnje, a ne samo da gradi svoje.
    """
    n = arr.shape[0]
    opp = WHITE if own == BLACK else BLACK
    own_counts = board_counts(arr, own, opp, n)
    opp_counts = board_counts(arr, opp, own, n)
    center_diff, conn_diff = _positional(arr, own, opp)
    return score_from_counts(own_counts, opp_counts, center_diff, conn_diff, weights)
