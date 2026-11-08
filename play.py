"""Grafička aplikacija (pygame) u kojoj čovek igra protiv AI agenta.

Crni uvek igra prvi. Pre svake partije bira se ko igra prvi — čovek ili AI.
AI računa potez u pozadinskoj niti, pa prozor ostaje responzivan tokom pretrage.
Aplikacija se pokreće preko 'main.py'.

Kontrole:
    levi klik   — postavljanje kamenčića / izbor opcije u meniju
    1 / 2       — meni: čovek prvi / AI prvi
    R           — povratak na početni meni
    Esc / close — izlaz
"""

import copy
import sys
import threading

import pygame

from engine.ai import GomokuAI
from engine.board import GomokuBoard
from lib.constants import Stone

# --- raspored -------------------------------------------------------------
BOARD_SIZE = 15
CELL = 42                       # broj piksela između linija mreže
MARGIN = 44                     # broj piksela ivice oko igrive mreže
GRID = (BOARD_SIZE - 1) * CELL
PANEL = 56                      # broj piksela visine statusne trake na dnu
WIDTH = GRID + 2 * MARGIN
HEIGHT = GRID + 2 * MARGIN + PANEL
STONE_RADIUS = CELL // 2 - 3    # broj piksela poluprečnika kamenčića
FPS = 60

# --- boje (RGB) -------------------------------------------------------------
COLOR_WOOD = (222, 184, 135)          # pozadina table
COLOR_LINE = (60, 40, 20)             # linije mreže
COLOR_STONE_BLACK = (20, 20, 20)      # crni kamenčić
COLOR_STONE_WHITE = (240, 240, 240)   # beli kamenčić
COLOR_SHADOW = (0, 0, 0, 60)          # senka kamenčića (sa providnošću)
COLOR_MARKER = (200, 40, 40)          # oznaka poslednjeg odigranog poteza
COLOR_PANEL_BG = (34, 30, 26)         # pozadina statusne trake i menija
COLOR_TEXT = (235, 235, 235)          # tekst
COLOR_HINT = (150, 140, 130)          # pomoćni tekst u meniju
COLOR_BUTTON = (74, 64, 54)           # dugme u meniju
COLOR_BUTTON_HOVER = (104, 90, 76)    # dugme kada je miš iznad njega
STAR_POINTS = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]  # zvezdane tačke

PHASE_LABELS = {
    "opening": "opening (book)",
    "tactical": "tactical (threat)",
    "search": "search (mid game)",
}


def grid_to_px(row, col):
    """Pretvara poziciju na mreži (row, col) u (x, y) piksele na ekranu."""
    return MARGIN + col * CELL, MARGIN + row * CELL


def px_to_grid(x, y):
    """Pretvara piksele (x, y) u polje (row, col), ili None ako je klik promašen."""
    col = round((x - MARGIN) / CELL)
    row = round((y - MARGIN) / CELL)
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        # odbacuje klikove koji su daleko od preseka
        cx, cy = grid_to_px(row, col)
        if (x - cx) ** 2 + (y - cy) ** 2 <= (CELL // 2) ** 2:
            return row, col
    return None


class Game:
    """Poseduje pygame petlju i povezuje tablu sa AI agentom."""

    def __init__(self):
        """Inicijalizuje pygame, prozor, fontove, dugmad menija i praznu partiju."""
        pygame.init()
        pygame.display.set_caption("Gomoku")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 22)
        self.big_font = pygame.font.SysFont("segoeui", 40, bold=True)
        self.state = "menu"        # "menu" ili "play"
        # dva centrirana dugmeta na početnom ekranu
        button_w, button_h, gap = 260, 64, 24
        cx = WIDTH // 2
        cy = HEIGHT // 2
        self.btn_human = pygame.Rect(cx - button_w // 2, cy - button_h - gap // 2,
                                     button_w, button_h)
        self.btn_ai = pygame.Rect(cx - button_w // 2, cy + gap // 2, button_w, button_h)

        # stanje partije (postavlja ga 'start_game')
        self.ai = GomokuAI()
        self.board = GomokuBoard(BOARD_SIZE)
        self.history: list[tuple[int, int]] = []
        self.human = Stone.BLACK
        self.ai_stone = Stone.WHITE
        self.winner: Stone | None = None
        self.game_over = False
        self.ai_phase: str | None = None

        # komunikacija sa pozadinskom niti koja računa AI potez
        self.game_id = 0             # menja se sa svakom novom partijom
        self.ai_thinking = False
        self._ai_result = None       # (game_id, move, phase, error) iz niti

    # --- tok igre ---------------------------------------------------------
    def start_game(self, human_first):
        """Započinje novu partiju, dodeljuje boje i pušta AI da odigra ako igra prvi."""
        # crni igra prvi, pa prvi igrač dobija crne
        self.human = Stone.BLACK if human_first else Stone.WHITE
        self.ai_stone = self.human.opponent
        self.board = GomokuBoard(BOARD_SIZE)
        self.history = []
        self.winner = None
        self.game_over = False
        self.ai_phase = None
        self.state = "play"
        # novi agent i novi identifikator: rezultat niti iz prethodne partije se odbacuje
        self.ai = GomokuAI()
        self.game_id += 1
        self.ai_thinking = False
        if not human_first:
            self.request_ai_move()   # AI otvara kao crni

    def apply(self, row, col):
        """Odigrava potez na tabli i dodaje ga u istoriju partije."""
        self.board.make_move(row, col)
        self.history.append((row, col))

    def end_if_finished(self):
        """Proverava da li je partija gotova (pobeda ili nerešeno) i beleži pobednika."""
        if self.board.player_has_won():
            self.winner = self.board.stone_at(*self.board.last_move)
            self.game_over = True
        elif self.board.is_a_tie():
            self.game_over = True

    def human_move(self, row, col):
        """Obrađuje potez čoveka; ako je legalan, odigrava ga i prepušta potez AI-ju."""
        if self.game_over or self.ai_thinking or self.board.current_stone != self.human:
            return
        if not self.board.is_valid_move(row, col):
            return
        self.apply(row, col)
        print(f"Player: ({row}, {col})")
        self.end_if_finished()
        if not self.game_over:
            self.request_ai_move()

    def request_ai_move(self):
        """Pokreće računanje AI poteza u pozadinskoj niti.

        Nit dobija kopiju table i istorije, pa ne deli promenljivo stanje sa
        glavnom petljom; rezultat preuzima 'collect_ai_move'.
        """
        board_snapshot = copy.deepcopy(self.board)
        history_snapshot = list(self.history)
        game_id = self.game_id
        agent = self.ai
        self.ai_thinking = True

        def worker():
            try:
                move = agent.choose_move(board_snapshot, history_snapshot)
                self._ai_result = (game_id, move, agent.last_phase, None)
            except Exception as error:  # greška se prosleđuje glavnoj niti
                self._ai_result = (game_id, None, None, error)

        threading.Thread(target=worker, daemon=True).start()

    def collect_ai_move(self):
        """Ako je pozadinska nit završila, odigrava izračunati AI potez."""
        result = self._ai_result
        if result is None:
            return
        self._ai_result = None
        game_id, move, phase, error = result
        if game_id != self.game_id:
            return  # rezultat pripada napuštenoj partiji
        self.ai_thinking = False
        if error is not None:
            raise error
        if move is None:
            if not self.board.is_a_tie():
                raise RuntimeError("AI returned no move though the board is not full")
            self.game_over = True
            return
        self.ai_phase = phase
        self.apply(*move)
        print(f"AI [{PHASE_LABELS.get(phase, phase)}]: ({move[0]}, {move[1]})")
        self.end_if_finished()

    # --- iscrtavanje ------------------------------------------------------
    def draw(self):
        """Iscrtava trenutni ekran: meni, ili tablu sa kamenčićima i statusnom trakom."""
        if self.state == "menu":
            self._draw_menu()
            return
        self.screen.fill(COLOR_WOOD)
        self._draw_grid()
        self._draw_stones()
        self._draw_last_marker()
        self._draw_panel()
        self._draw_overlay()

    def _draw_menu(self):
        """Iscrtava početni meni: naslov, pitanje i dugmad za izbor ko igra prvi."""
        self.screen.fill(COLOR_PANEL_BG)
        title = self.big_font.render("Gomoku", True, COLOR_TEXT)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150)))
        subtitle = self.font.render("Who plays first?", True, COLOR_TEXT)
        self.screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100)))
        mouse = pygame.mouse.get_pos()
        self._draw_button(self.btn_human, "You first  (black)", mouse)
        self._draw_button(self.btn_ai, "AI first  (you: white)", mouse)
        hint = self.font.render("click, or press 1 / 2", True, COLOR_HINT)
        self.screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 130)))

    def _draw_button(self, rect, label, mouse):
        """Iscrtava jedno dugme sa natpisom, sa isticanjem kada je miš iznad njega."""
        hovered = rect.collidepoint(mouse)
        color = COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_LINE, rect, 2, border_radius=10)
        text = self.font.render(label, True, COLOR_TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_grid(self):
        """Iscrtava linije mreže table i zvezdane (hoši) tačke."""
        for i in range(BOARD_SIZE):
            x0, y0 = grid_to_px(i, 0)
            x1, y1 = grid_to_px(i, BOARD_SIZE - 1)
            pygame.draw.line(self.screen, COLOR_LINE, (x0, y0), (x1, y1), 1)
            x0, y0 = grid_to_px(0, i)
            x1, y1 = grid_to_px(BOARD_SIZE - 1, i)
            pygame.draw.line(self.screen, COLOR_LINE, (x0, y0), (x1, y1), 1)
        for r, c in STAR_POINTS:
            x, y = grid_to_px(r, c)
            pygame.draw.circle(self.screen, COLOR_LINE, (x, y), 4)

    def _draw_stones(self):
        """Iscrtava sve postavljene kamenčiće (sa senkom) na njihovim poljima."""
        shadow_size = STONE_RADIUS * 2 + 4
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = self.board.stone_at(r, c)
                if stone == Stone.EMPTY:
                    continue
                x, y = grid_to_px(r, c)
                shadow = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
                pygame.draw.circle(
                    shadow, COLOR_SHADOW, (STONE_RADIUS + 2, STONE_RADIUS + 4), STONE_RADIUS,
                )
                self.screen.blit(shadow, (x - STONE_RADIUS - 2, y - STONE_RADIUS - 2))
                color = COLOR_STONE_BLACK if stone == Stone.BLACK else COLOR_STONE_WHITE
                pygame.draw.circle(self.screen, color, (x, y), STONE_RADIUS)
                pygame.draw.circle(self.screen, COLOR_LINE, (x, y), STONE_RADIUS, 1)

    def _draw_last_marker(self):
        """Iscrtava crvenu oznaku na poslednjem odigranom potezu."""
        if self.board.last_move is None:
            return
        x, y = grid_to_px(*self.board.last_move)
        pygame.draw.circle(self.screen, COLOR_MARKER, (x, y), 5)

    def _draw_panel(self):
        """Iscrtava statusnu traku na dnu: čiji je potez, broj poteza ili ishod partije."""
        rect = pygame.Rect(0, HEIGHT - PANEL, WIDTH, PANEL)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect)
        if self.game_over:
            msg = self._outcome_text() + "   (R: menu)"
        else:
            if self.ai_thinking:
                turn = "AI thinking..."
            else:
                color = "black" if self.human == Stone.BLACK else "white"
                turn = f"Your turn ({color})"
                phase_ui = {
                    "opening": "early (book)",
                    "tactical": "tactical",
                    "search": "mid (search)",
                }.get(self.ai_phase)
                if phase_ui:
                    turn += f"  |  AI: {phase_ui}"
            msg = f"{turn}    moves: {len(self.history)}   (R: menu)"
        label = self.font.render(msg, True, COLOR_TEXT)
        self.screen.blit(label, (16, HEIGHT - PANEL + (PANEL - label.get_height()) // 2))

    def _outcome_text(self):
        """Vraća tekst ishoda partije iz ugla čoveka."""
        if self.winner == self.human:
            return "You win!"
        if self.winner == self.ai_stone:
            return "AI wins!"
        return "Tie game."

    def _draw_overlay(self):
        """Preko table iscrtava zatamnjenje i poruku o ishodu kada je partija gotova."""
        if not self.game_over:
            return
        overlay = pygame.Surface((WIDTH, HEIGHT - PANEL), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 90))
        self.screen.blit(overlay, (0, 0))
        text = self._outcome_text().rstrip("!.").upper()
        label = self.big_font.render(text, True, COLOR_STONE_WHITE)
        self.screen.blit(label, label.get_rect(center=(WIDTH // 2, (HEIGHT - PANEL) // 2)))

    # --- petlja -----------------------------------------------------------
    def run(self):
        """Glavna petlja igre: obrađuje događaje, preuzima AI potez i osvežava prikaz."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_click(event.pos)

            self.collect_ai_move()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

    def _on_key(self, key):
        """Obrađuje pritiske tastera: Esc (izlaz), R (meni), 1/2 (izbor u meniju)."""
        if key == pygame.K_ESCAPE:
            self.quit()
        elif key == pygame.K_r:
            self.state = "menu"
        elif self.state == "menu":
            if key == pygame.K_1:
                self.start_game(human_first=True)
            elif key == pygame.K_2:
                self.start_game(human_first=False)

    def _on_click(self, pos):
        """Obrađuje klik miša: dugmad u meniju ili postavljanje kamenčića na tabli."""
        if self.state == "menu":
            if self.btn_human.collidepoint(pos):
                self.start_game(human_first=True)
            elif self.btn_ai.collidepoint(pos):
                self.start_game(human_first=False)
        else:
            cell = px_to_grid(*pos)
            if cell:
                self.human_move(*cell)

    @staticmethod
    def quit():
        """Uredno zatvara pygame i izlazi iz programa."""
        pygame.quit()
        sys.exit()
