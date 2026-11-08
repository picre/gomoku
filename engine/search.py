"""Izbor poteza pomoću Minimax algoritma sa alfa-beta odsecanjem.

Pretraga radi nad običnim 'int' numpy nizom (za kodiranje videti
'engine.evaluation'), a ne nad tablom iz GUI-ja, tako da pozicije mogu da
se isprobavaju i poništavaju u mestu bez alociranja objekta table po čvoru.
Da bi faktor grananja ostao razuman na tabli 15x15, kao kandidati se razmatraju
samo prazna polja u okviru 'radius' od nekog postojećeg kamenčića.

Optimizacije koje ubrzavaju pretragu:

* **Inkrementalna evaluacija** ('engine.incremental.IncrementalEvaluator') —
  postavljanje ili uklanjanje kamenčića menja samo 4 pravca kroz to polje.
* **Transpoziciona tabela** (Zobrist heš) — kešira ocene pozicija.
* **Uređivanje poteza** — potezi se sortiraju po plitkoj oceni radi jačeg
  alfa-beta odsecanja.

Taktička analiza ('find_tactical_move') je odvojena od same pretrage: agent
('engine.ai.GomokuAI') je poziva pre 'search_best_move', pa se forsirani
potezi ne računaju dvaput.
"""

import numpy as np

from engine.board import DIRECTIONS, WIN_LENGTH
from engine.evaluation import EMPTY, PATTERN_CATEGORY_INDEX, opponent_of
from engine.incremental import IncrementalEvaluator

INF = float("inf")
WIN_SCORE = 10_000_000.0
VCF_MOVE_CAP = 20  # VCF razmatra samo prvih K poteza po plitkoj oceni

# oznake za transpozicionu tabelu (tačna vrednost / donja granica / gornja granica)
_EXACT, _LOWER, _UPPER = 0, 1, 2


def board_to_int(board):
    """Vraća nezavisnu 'int8' kopiju table iz 'engine.board.GomokuBoard'.

    Pretraga menja niz u mestu, pa se tabla iz igre nikada ne prosleđuje direktno.
    """
    return board.board.astype(np.int8, copy=True)


def _wins(arr, row, col):
    """Tačno ako upravo postavljeni kamenčić na (row, col) formira pet u nizu."""
    stone = arr[row, col]
    if stone == EMPTY:
        return False
    n = arr.shape[0]
    for dr, dc in DIRECTIONS:
        count = 1
        rr, cc = row + dr, col + dc
        while 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
            count += 1
            rr += dr
            cc += dc
        rr, cc = row - dr, col - dc
        while 0 <= rr < n and 0 <= cc < n and arr[rr, cc] == stone:
            count += 1
            rr -= dr
            cc -= dc
        if count >= WIN_LENGTH:
            return True
    return False


class _Searcher:
    """Minimax sa alfa-beta odsecanjem, transpozicionom tabelom i uređivanjem poteza."""

    def __init__(self, ev, ai_stone, radius):
        self.ev = ev
        self.ai = ai_stone
        self.radius = radius
        self.tt = {}

    def _order(self, moves, player, maximizing):
        """Sortira poteze po plitkoj (inkrementalnoj) oceni radi boljeg odsecanja."""
        scored = []
        for (r, c) in moves:
            self.ev.place(r, c, player)
            scored.append((self.ev.value(self.ai), (r, c)))
            self.ev.remove(r, c)
        scored.sort(key=lambda item: item[0], reverse=maximizing)
        return [move for _, move in scored]

    def _leaf_value(self, maximizing, player):
        """Ocena u listu: forsirani ishod preko brojača pretnji, inače heuristika.

        Na neparnoj dubini obična evaluacija ne vidi otvorenu četvorku: maksimizujući
        igrač brani jedan kraj, pretraga staje, a drugi kraj i dalje pobeđuje. Zato
        se u listu proverava da li igrač na potezu već ima četvorku / otvorenu četvorku.
        """
        four_i = PATTERN_CATEGORY_INDEX["four"]
        open_four_i = PATTERN_CATEGORY_INDEX["open_four"]
        mine = self.ev.total[player]
        if int(mine[open_four_i]) > 0 or int(mine[four_i]) > 0:
            return WIN_SCORE if maximizing else -WIN_SCORE
        other = opponent_of(player)
        if int(self.ev.total[other][open_four_i]) > 0:
            return -WIN_SCORE if maximizing else WIN_SCORE
        return self.ev.value(self.ai)

    def search(self, depth, alpha, beta, maximizing, player):
        """Rekurzivni alfa-beta minimax; vraća ocenu iz ugla 'self.ai'."""
        key = (self.ev.hash, maximizing)
        entry = self.tt.get(key)
        if entry is not None and entry[0] >= depth:
            flag, val = entry[1], entry[2]
            if flag == _EXACT:
                return val
            if flag == _LOWER and val > alpha:
                alpha = val
            elif flag == _UPPER and val < beta:
                beta = val
            if alpha >= beta:
                return val

        moves = self.ev.candidates(self.radius)
        if depth == 0 or not moves:
            return self._leaf_value(maximizing, player)

        alpha0, beta0 = alpha, beta
        other = opponent_of(player)
        if depth > 1:
            moves = self._order(moves, player, maximizing)

        if maximizing:
            best = -INF
            for (r, c) in moves:
                self.ev.place(r, c, player)
                if _wins(self.ev.arr, r, c):
                    val = WIN_SCORE + depth
                else:
                    val = self.search(depth - 1, alpha, beta, False, other)
                self.ev.remove(r, c)
                if val > best:
                    best = val
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        else:
            best = INF
            for (r, c) in moves:
                self.ev.place(r, c, player)
                if _wins(self.ev.arr, r, c):
                    val = -(WIN_SCORE + depth)
                else:
                    val = self.search(depth - 1, alpha, beta, True, other)
                self.ev.remove(r, c)
                if val < best:
                    best = val
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break

        if best <= alpha0:
            flag = _UPPER
        elif best >= beta0:
            flag = _LOWER
        else:
            flag = _EXACT
        self.tt[key] = (depth, flag, best)
        return best


def _cells_where_wins(ev, stone, moves):
    """Polja na kojima 'stone' odmah pravi pet u nizu."""
    cells = []
    for (r, c) in moves:
        ev.place(r, c, stone)
        won = _wins(ev.arr, r, c)
        ev.remove(r, c)
        if won:
            cells.append((r, c))
    return cells


def _top_moves_by_value(ev, player, moves, k, perspective):
    """Prvih 'k' poteza iz 'moves', rangiranih po plitkoj 'ev.value(perspective)'."""
    if k <= 0 or not moves:
        return []
    if len(moves) <= k:
        return list(moves)
    scored = []
    for (r, c) in moves:
        ev.place(r, c, player)
        scored.append((ev.value(perspective), (r, c)))
        ev.remove(r, c)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in scored[:k]]


def _find_forced_win(ev, attacker, radius, depth_left):
    """Prvi potez u nizu forsiranih pretnji do pobede, ili None.

    Forsiranje = pretnja „pobeda sledećim potezom“ (četvorka / otvorena
    četvorka). Protivnik mora da brani; ako ne može sve pretnje odjednom,
    ili posle odbrane napadač opet forsira do pobede — vraća se početni potez.
    Napadački potezi u VCF-u su ograničeni na prvih 'VCF_MOVE_CAP' po plitkoj oceni.
    """
    if depth_left <= 0:
        return None
    moves = ev.candidates(radius)
    if not moves:
        return None
    defender = opponent_of(attacker)
    four_i = PATTERN_CATEGORY_INDEX["four"]
    open_four_i = PATTERN_CATEGORY_INDEX["open_four"]

    # trenutna pobeda — preko svih kandidata (jeftino i kritično)
    wins = _cells_where_wins(ev, attacker, moves)
    if wins:
        return wins[0]

    # ako branilac već preti pobedom, napadač ne može da forsira
    if _cells_where_wins(ev, defender, moves):
        return None

    # VCF se razmatra samo na najboljim plitkim kandidatima
    attack_moves = _top_moves_by_value(
        ev, attacker, moves, VCF_MOVE_CAP, perspective=attacker,
    )
    for (r, c) in attack_moves:
        ev.place(r, c, attacker)
        if _wins(ev.arr, r, c):
            ev.remove(r, c)
            return (r, c)

        # bez nove četvorke nema forsiranja — štedi O(|kandidati|) po potezu
        if (
            int(ev.total[attacker][four_i]) == 0
            and int(ev.total[attacker][open_four_i]) == 0
        ):
            ev.remove(r, c)
            continue

        def_moves = ev.candidates(radius)
        threats = _cells_where_wins(ev, attacker, def_moves)
        if len(threats) >= 2:
            ev.remove(r, c)
            return (r, c)
        if len(threats) == 1:
            br, bc = threats[0]
            ev.place(br, bc, defender)
            if not _wins(ev.arr, br, bc):
                cont = _find_forced_win(ev, attacker, radius, depth_left - 1)
                if cont is not None:
                    ev.remove(br, bc)
                    ev.remove(r, c)
                    return (r, c)
            ev.remove(br, bc)
        ev.remove(r, c)
    return None


def _adaptive_force_depth(n_candidates, requested):
    """Blago smanjuje VCF dubinu na veoma širokim pozicijama.

    Glavna zaštita od eksplozije je 'VCF_MOVE_CAP'; ovde samo skraćuje dubinu.
    """
    if requested <= 0:
        return 0
    if n_candidates <= 40:
        return requested
    return min(requested, 3)


def _adaptive_search_depth(n_candidates, requested):
    """Smanjuje minimax dubinu na širim pozicijama da potez ostane interaktivan."""
    if n_candidates <= 20:
        return requested
    if n_candidates <= 35:
        return min(requested, 2)
    return min(requested, 1)


def find_tactical_move(ev, ai_stone, radius, force_depth=6):
    """Forsirani / hitni potez bez duboke pretrage, ili None.

    Prioritet:
    1. trenutna pobeda
    2. blokada protivničke pretnje „pobeda sledećim potezom“
    3. niz forsiranih pretnji do pobede (VCF nad prvih K poteza)
    4. pravljenje otvorene četvorke
    5. blokada već postojeće protivničke otvorene četvorke
    """
    moves = ev.candidates(radius)
    if not moves:
        return None
    opponent = opponent_of(ai_stone)
    open_four_i = PATTERN_CATEGORY_INDEX["open_four"]
    force_depth = _adaptive_force_depth(len(moves), force_depth)

    # 1. pobeda odmah
    wins = _cells_where_wins(ev, ai_stone, moves)
    if wins:
        return wins[0]

    # 2. mora da se brani pretnja pobede u jednom potezu
    blocks = _cells_where_wins(ev, opponent, moves)
    if blocks:
        return blocks[0]

    # 3. forsiran niz (VCF) — samo prvih K poteza po plitkoj oceni
    if force_depth > 0:
        forced = _find_forced_win(ev, ai_stone, radius, force_depth)
        if forced is not None:
            return forced

    # 4. otvorena četvorka — forsira odgovor
    for (r, c) in moves:
        ev.place(r, c, ai_stone)
        if int(ev.total[ai_stone][open_four_i]) > 0:
            ev.remove(r, c)
            return (r, c)
        ev.remove(r, c)

    # 5. protivnik već ima otvorenu četvorku — traži se blokada
    if int(ev.total[opponent][open_four_i]) > 0:
        for (r, c) in moves:
            ev.place(r, c, ai_stone)
            cleared = int(ev.total[opponent][open_four_i]) == 0
            ev.remove(r, c)
            if cleared:
                return (r, c)

    return None


def search_best_move(arr, ai_stone, weights, depth=3, radius=1, rng=None):
    """Bira najbolji potez za 'ai_stone' na 'arr' pomoću minimax-a.

    Vraća (row, col) torku, ili 'None' ako nema legalnog poteza. Izjednačeni
    potezi se biraju pomoću 'rng' kada je prosleđen, tako da partije nisu
    potpuno deterministične. Dubina pretrage se dodatno sužava na širokim
    pozicijama. Pretpostavlja se da je pozivalac prethodno proverio
    'find_tactical_move'.
    """
    opponent = opponent_of(ai_stone)
    ev = IncrementalEvaluator(arr, weights)
    moves = ev.candidates(radius)
    if not moves:
        if np.any(arr == EMPTY):
            raise RuntimeError("no candidate moves although the board is not full")
        return None

    depth = _adaptive_search_depth(len(moves), depth)

    # uređivanje poteza na korenu po plitkoj oceni
    scored = []
    for (r, c) in moves:
        ev.place(r, c, ai_stone)
        scored.append((ev.value(ai_stone), (r, c)))
        ev.remove(r, c)
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [move for _, move in scored]

    searcher = _Searcher(ev, ai_stone, radius)
    best, best_moves = -INF, []
    # Na korenu svaki kandidat ide sa punim prozorom (-inf, +inf).
    # Ako se alpha prenosi sa prethodnog poteza, fail-soft alfa-beta često
    # vraća istu granicu i za loše poteze, pa RNG izjednačava dobar i loš potez.
    for (r, c) in ordered:
        ev.place(r, c, ai_stone)
        if _wins(ev.arr, r, c):
            val = WIN_SCORE + depth
        else:
            val = searcher.search(depth - 1, -INF, INF, False, opponent)
        ev.remove(r, c)
        if val > best:
            best, best_moves = val, [(r, c)]
        elif val == best:
            best_moves.append((r, c))

    if rng is not None:
        return rng.choice(best_moves)
    return best_moves[0]
