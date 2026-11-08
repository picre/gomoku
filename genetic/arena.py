"""Partija između dva agenta sa različitim skupovima težina (samoigra).

Koristi je genetski algoritam za merenje fitnesa: jedinka vredi onoliko koliko
partija može da dobije, pa se agenti sučeljavaju direktno. Isti mehanizam
koristi i eksperiment 'genetic.head_to_head'.
"""

from engine.ai import GomokuAI
from engine.board import GomokuBoard
from lib.constants import Stone


def play_game(black_weights, white_weights, depth=1, radius=1, size=15,
              max_moves=None, seed=None):
    """Odigrava jednu partiju; vraća pobednički 'Stone' ili 'None' za nerešeno.

    Podrazumevano se igra na punoj tabli 15x15; plitka pretraga ('depth=1')
    drži samoigru dovoljno brzom za veliki broj partija koje genetski
    algoritam zahteva. Knjiga otvaranja i VCF forsiranje su isključeni, kako
    bi ishod zavisio samo od težina funkcije evaluacije.
    """
    board = GomokuBoard(size)
    black = GomokuAI(
        seed=seed, depth=depth, radius=radius, weights=black_weights,
        use_opening_book=False, force_depth=0,
    )
    white = GomokuAI(
        seed=None if seed is None else seed + 1,
        depth=depth, radius=radius, weights=white_weights,
        use_opening_book=False, force_depth=0,
    )

    history = []
    limit = max_moves if max_moves is not None else size * size
    for _ in range(limit):
        agent = black if board.current_stone == Stone.BLACK else white
        move = agent.choose_move(board, history)
        if move is None:
            if not board.is_a_tie():
                raise RuntimeError("AI returned no move though the board is not full")
            break
        board.make_move(*move)
        history.append(move)
        if board.player_has_won():
            return board.stone_at(*board.last_move)
    return None


def outcome_for(result, stone):
    """Ishod partije iz ugla boje 'stone': +1 pobeda, -1 poraz, 0 nerešeno."""
    if result is None:
        return 0
    return 1 if result == stone else -1
