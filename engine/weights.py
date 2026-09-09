"""Težine heurističke funkcije evaluacije.

Čuva podrazumevane vrednosti, redosled gena za genetski algoritam i I/O
prema ``weights.json``. Samo ocena pozicije ostaje u :mod:`engine.evaluation`.
"""

import json
from pathlib import Path

import numpy as np

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "weights.json"


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
