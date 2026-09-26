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
**unapred izračunata tabela** za sve moguće linije dužine 5 i prozore dužine 6
(kodirane u bazi 3). Skeniranje jednog **span**-a / pravca (vrsta/kolona/dijagonala)
svodi se na klizanje tih segmenata i sabiranje vektora iz tabele. Koordinate
svih span-ova i mapa polje→span čuvaju se u ``_SPANS`` / ``_CELL_SPANS``.
Težine su u :mod:`engine.weights`.
"""

import numpy as np

from lib.constants import Stone

# --- kodiranje kamenčića (poklapa se sa vrednostima Stone enum-a) ---------
BLACK = int(Stone.BLACK.value)   # 0
WHITE = int(Stone.WHITE.value)   # 1
EMPTY = int(Stone.EMPTY.value)   # 2

WIN_LENGTH = 5

# Katalog obrazaca, iz perspektive igrača koji je na potezu:
#   '1' sopstveni kamenčić, '2' protivnički kamenčić ILI zid, '0' prazno polje.
# Span-ovi (pravci) se sa obe strane dopunjavaju '2' da bi ivice table brojale kao blokada.
# Terminologija: span = cela vrsta/kolona/dijagonala (pravac);
#                linija = segment dužine 5 unutar span-a (prozor 6 je za duže obrasce);
#                smer = (dr, dc), npr. u board/search.
IMPORTANT_PATTERNS = {
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
# ("five", "open_four", "four", "open_three", "three", "open_two", "two")
CATS = tuple(IMPORTANT_PATTERNS.keys())
CAT_INDEX = {name: i for i, name in enumerate(CATS)}
NUM_CATS = len(CATS)


# --- unapred izračunate tabele obrazaca -----------------------------------
def _build_window_table(length):
    """Lookup tabela: kod segmenta -> broj pogodaka po kategoriji.

    Za dužinu 5 (linija) ili 6 (duži prozor) prolazi kroz svih 3**length
    mogućih kombinacija 0/1/2 i za svaku proverava koje IMPORTANT_PATTERNS
    te dužine se poklapaju sa početkom. Rezultat je niz brojača po CATS.

    Kasnije count_span samo izračuna kod i sabere table[code],
    bez poređenja stringova u toku igre.
    """
    table = np.zeros((3 ** length, NUM_CATS), dtype=np.int64)
    for code in range(3 ** length):
        digits, x = [], code
        for _ in range(length):
            digits.append(x % 3)
            x //= 3
        window = "".join("012"[d] for d in digits)  # pozicija 0..L-1
        for ci, cat in enumerate(CATS):
            for pat in IMPORTANT_PATTERNS[cat]:
                if len(pat) == length and window.startswith(pat):
                    table[code, ci] += 1
    return table


_TABLE5 = _build_window_table(5)  # linije dužine 5
_TABLE6 = _build_window_table(6)  # prozori dužine 6 (za duže obrasce)


# --- span-ovi (pravci) po veličini table (grade se jednom, pa se samo čitaju) ---
_SPANS = {}       # n -> lista (rows, cols) za sve span-ove
_CELL_SPANS = {}  # n -> mapa (r, c) -> indeksi span-ova kroz to polje


def _build_spans(n):
    """Sve vrste, kolone i dijagonale (span-ovi / pravci na kojima može biti pobeda),
    dužine >= WIN_LENGTH, kao (rows, cols) nizovi."""
    raw = []

    # svi horizontalni span-ovi (jedan po vrsti)
    for r in range(n):
        raw.append([(r, c) for c in range(n)])

    # svi vertikalni span-ovi (jedan po koloni)
    for c in range(n):
        raw.append([(r, c) for r in range(n)])

    # dijagonale nadole-desno (\/ smer), koje počinju u gornjem redu
    for start in range(n):
        coords, r, c = [], 0, start
        while r < n and c < n:
            coords.append((r, c))
            r += 1
            c += 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)

    # iste dijagonale nadole-desno, koje počinju u levoj koloni (bez ugla 0,0 — već pokriven)
    for start in range(1, n):
        coords, r, c = [], start, 0
        while r < n and c < n:
            coords.append((r, c))
            r += 1
            c += 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)

    # anti-dijagonale nadole-levo (/\ smer), koje počinju u gornjem redu
    for start in range(n):
        coords, r, c = [], 0, start
        while r < n and c >= 0:
            coords.append((r, c))
            r += 1
            c -= 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)

    # iste anti-dijagonale, koje počinju u desnoj koloni (bez gornjeg reda — već pokriven)
    for start in range(1, n):
        coords, r, c = [], start, n - 1
        while r < n and c >= 0:
            coords.append((r, c))
            r += 1
            c -= 1
        if len(coords) >= WIN_LENGTH:
            raw.append(coords)

    # pretvara listu (row, col) parova u dva numpy niza radi brzog indeksiranja table (np.where oblik)
    spans = []
    for coords in raw:
        rows, cols = zip(*coords)
        spans.append((np.array(rows), np.array(cols)))
    return spans


def get_spans(n):
    """Vraća koordinate svih span-ova (pravaca) za tablu veličine ``n``."""
    if n not in _SPANS:
        spans = _build_spans(n)
        _SPANS[n] = spans
        cell_spans = {}
        for si, (rows, cols) in enumerate(spans):
            for r, c in zip(rows.tolist(), cols.tolist()):
                cell_spans.setdefault((r, c), []).append(si)
        _CELL_SPANS[n] = cell_spans
    return _SPANS[n]


def get_cell_spans(n):
    """Vraća mapu polje (r, c) → indeksi span-ova koje ga sadrže."""
    get_spans(n)
    return _CELL_SPANS[n]


# --- prebrojavanje obrazaca -----------------------------------------------
# stepeni trojke za kodiranje linije/prozora u bazi 3 (isti kod kao u _build_window_table)
_POW3 = (1, 3, 9, 27, 81, 243)


def _span_to_digits(span, own, opp):
    """Mapira span (pravac) u 0/1/2 i doda zid (2) sa obe strane."""
    # 1 = own, 2 = opp, 0 = prazno
    middle = np.where(span == own, 1, np.where(span == opp, 2, 0)).astype(np.int64)
    digits = np.empty(middle.size + 2, dtype=np.int64)
    digits[0] = 2
    digits[-1] = 2
    digits[1:-1] = middle
    return digits


def _segment_codes(digits, length):
    """Kodovi svih kliznih segmenata date dužine (baza 3).

    length=5 → linije; length=6 → duži prozori za obrasce tipa open_four.
    """
    # za svaki početak i: code = d[i] + 3*d[i+1] + 9*d[i+2] + ...
    code = np.zeros(digits.size - length + 1, dtype=np.int64)
    for k in range(length):
        code += _POW3[k] * digits[k: digits.size - length + 1 + k]
    return code


def count_span(span, own, opp):
    """Broj IMPORTANT_PATTERNS pogodaka na jednom span-u (pravcu), iz ugla ``own``.

    Koraci:
    1. pretvara polja span-a u 0/1/2 i dodaje zidove na krajeve
    2. klizi linije dužine 5 i prozore dužine 6
    3. za svaki segment uzima unapred izračunat vektor iz _TABLE5/_TABLE6
    4. sabira sve pogodke po kategorijama (CATS)
    """
    digits = _span_to_digits(span, own, opp)
    counts = np.zeros(NUM_CATS, dtype=np.int64)

    if digits.size >= 6:
        counts += _TABLE6[_segment_codes(digits, 6)].sum(axis=0)
    if digits.size >= 5:
        counts += _TABLE5[_segment_codes(digits, 5)].sum(axis=0)
    return counts


def board_counts(arr, own, opp, n=None):
    """Zbir brojača obrazaca za ``own`` preko svih span-ova cele table."""
    if n is None:
        n = arr.shape[0]
    total = np.zeros(NUM_CATS, dtype=np.int64)
    for rows, cols in get_spans(n):
        total += count_span(arr[rows, cols], own, opp)
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
