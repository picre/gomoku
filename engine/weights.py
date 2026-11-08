"""Težine heurističke funkcije evaluacije.

Čuva podrazumevane vrednosti, redosled gena za genetski algoritam i I/O
prema 'weights.json'. Samo ocena pozicije ostaje u 'engine.evaluation'.

Apsolutna skala gena nije bitna za rangiranje poteza — bitni su odnosi.
Pri snimanju se najveća apsolutna težina fiksira na 'SCALE_TARGET';
'defense' ostaje neskaliran jer je već relativni faktor.
"""

import json
from pathlib import Path

import numpy as np

# razuman ručno izabran polazni skup; genetski algoritam pretražuje u okolini
DEFAULT_WEIGHTS = {
    "five": 100_000,
    "open_four": 10_000,
    "four": 1_000,
    "open_three": 1_000,
    "three": 100,
    "open_two": 100,
    "two": 10,
    "center": 3,
    "connectivity": 5,
    "fork": 2000,
    "defense": 1.1,
}

# kanonski redosled gena u vektoru težina. Izveden iz DEFAULT_WEIGHTS
# (rečnici čuvaju redosled umetanja) tako da njih dva nikada ne mogu da se raziđu.
WEIGHT_FIELDS: tuple[str, ...] = tuple(DEFAULT_WEIGHTS)

# geni koji su razmera, ne apsolutna skala ocene
RATIO_FIELDS = frozenset({"defense"})
# kanonska skala: max(|apsolutni gen|) → SCALE_TARGET pri snimanju
SCALE_TARGET = 100_000.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights.json"


def weights_to_vector(weights):
    """Pretvara rečnik težina u uređeni numpy vektor (za genetski algoritam)."""
    return np.array([float(weights[field]) for field in WEIGHT_FIELDS], dtype=float)


def vector_to_weights(vector):
    """Inverzna funkcija od 'weights_to_vector'."""
    return {field: float(vector[i]) for i, field in enumerate(WEIGHT_FIELDS)}


def normalize_weights(weights):
    """Vraća kopiju težina sa max apsolutnim genom jednakim 'SCALE_TARGET'.

    Polja u 'RATIO_FIELDS' (npr. 'defense') ne skaliraju se. Ako su svi
    apsolutni geni nula, vraća ulaz nepromenjen.
    """
    out = {k: float(weights[k]) for k in WEIGHT_FIELDS}
    absolute = [abs(out[k]) for k in WEIGHT_FIELDS if k not in RATIO_FIELDS]
    peak = max(absolute) if absolute else 0.0
    if peak <= 0.0:
        return out
    scale = SCALE_TARGET / peak
    for key in WEIGHT_FIELDS:
        if key not in RATIO_FIELDS:
            out[key] *= scale
    return out


def load_weights(path=None):
    """Učitava evoluirane težine iz JSON-a, uz vraćanje na podrazumevane."""
    weights_path = Path(path) if path is not None else WEIGHTS_PATH
    if weights_path.exists():
        with open(weights_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw = {k: float(data.get(k, DEFAULT_WEIGHTS[k])) for k in WEIGHT_FIELDS}
        return normalize_weights(raw)
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights, path=None):
    """Trajno upisuje rečnik težina u JSON (normalizovan na 'SCALE_TARGET')."""
    weights_path = Path(path) if path is not None else WEIGHTS_PATH
    normalized = normalize_weights(weights)
    with open(weights_path, "w", encoding="utf-8") as handle:
        json.dump(
            {k: float(normalized[k]) for k in WEIGHT_FIELDS},
            handle,
            indent=2,
        )
    return weights_path
