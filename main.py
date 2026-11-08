"""Ulazna tačka aplikacije: pokreće pygame igru čovek protiv AI-ja.

Pokretanje iz korena projekta::

    python main.py
"""

from play import Game


def main():
    """Pravi prozor igre i pokreće glavnu petlju."""
    Game().run()


if __name__ == "__main__":
    main()
