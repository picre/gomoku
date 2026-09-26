"""
Gomoku — pygame aplikacija. Igraj protiv AI-ja.

Crni uvek igra prvi. Pre svake partije se bira ko igra prvi - igrač ili AI.

Kontrole:
    Levi klik   - postavljanje kamenčića / izbor opcije u meniju
    1 / 2       - meni: Čovek prvi / AI prvi
    R           - povratak na početni meni
    Esc / close - izlaz
"""

import sys

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
STONE_R = CELL // 2 - 3          # broj piksela poluprečnika kamenčića

# --- boje -----------------------------------------------------------------
WOOD = (222, 184, 135)            # boja drveta
LINE = (60, 40, 20)               # boja linija mreže
BLACK = (20, 20, 20)              # boja crnog kamenčića
WHITE = (240, 240, 240)           # boja belog kamenčića
SHADOW = (0, 0, 0, 60)            # boja senke kamenčića
MARKER = (200, 40, 40)            # boja crvene oznake na poslednjem odigranom potezu
PANEL_BG = (34, 30, 26)           # boja pozadine statusne trake
TEXT = (235, 235, 235)            # boja teksta na dugmadima u meniju
BTN = (74, 64, 54)                # boja dugmeta u meniju
BTN_HOVER = (104, 90, 76)         # boja dugmeta u meniju kada je miš iznad njega
STAR_PTS = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]  # koordinate zvezdanih tačaka


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
    """Poseduje pygame petlju i povezuje tablu + AI agenta."""

    def __init__(self):
        """Inicijalizuje pygame, prozor, fontove, AI agenta, dugmad menija i tablu."""
        pygame.init()
        pygame.display.set_caption("Gomoku")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 22)
        self.big_font = pygame.font.SysFont("segoeui", 40, bold=True)
        self.ai = GomokuAI()
        self.state = "menu"        # "menu" ili "play"
        # dva centrirana dugmeta na početnom ekranu
        bw, bh, gap = 260, 64, 24
        cx = WIDTH // 2
        cy = HEIGHT // 2
        self.btn_human = pygame.Rect(cx - bw // 2, cy - bh - gap // 2, bw, bh)
        self.btn_ai = pygame.Rect(cx - bw // 2, cy + gap // 2, bw, bh)
        self.board = GomokuBoard(BOARD_SIZE)  # rezervisano dok partija ne počne

    # --- tok igre ---------------------------------------------------------
    def start_game(self, human_first):
        """Započinje novu partiju, dodeljuje boje i pušta AI da odigra ako igra prvi."""
        # crni igra prvi, pa prvi igrač dobija crne
        self.human = Stone.BLACK if human_first else Stone.WHITE
        self.ai_stone = Stone.WHITE if human_first else Stone.BLACK
        self.board = GomokuBoard(BOARD_SIZE)
        self.history = []
        self.winner = None
        self.game_over = False
        self.state = "play"
        if not human_first:
            self.ai_move()         # AI otvara kao crni

    def apply(self, row, col):
        """Odigra potez na tabli i dodaje ga u istoriju partije."""
        self.board.make_move(row, col)
        self.history.append((row, col))

    def end_if_finished(self):
        """Proverava da li je partija gotova (pobeda ili nerešeno) i beleži pobednika."""
        if self.board.player_has_won():
            self.winner = self.board.board[self.history[-1]]
            self.game_over = True
        elif self.board.is_a_tie():
            self.game_over = True

    def human_move(self, row, col):
        """Obrađuje potez čoveka; ako je legalan, odigrava ga i prepušta potez AI-ju."""
        if self.game_over or self.board.current_stone != self.human:
            return
        if not self.board.is_valid_move(row, col):
            return
        self.apply(row, col)
        print(f"Igrač: ({row}, {col})")
        self.end_if_finished()
        if not self.game_over:
            self.ai_move()

    def ai_move(self):
        """Traži AI potez i odigrava ga (uz kratku pauzu radi vizuelnog utiska)."""
        # odmah iscrtava da se prethodni kamenčić vidi pre nego što AI „razmišlja”
        self.draw()
        pygame.display.flip()
        pygame.time.wait(180)
        move = self.ai.choose_move(self.board)
        if move is None:
            if not self.board.is_a_tie():
                raise RuntimeError("AI nije vratio potez iako tabla nije puna")
            self.game_over = True
            return
        self.apply(*move)
        print(f"AI: ({move[0]}, {move[1]})")
        self.end_if_finished()

    # --- iscrtavanje ------------------------------------------------------
    def draw(self):
        """Iscrtava trenutni ekran: meni, ili tablu sa kamenčićima i statusnom trakom."""
        if self.state == "menu":
            self._draw_menu()
            return
        self.screen.fill(WOOD)
        self._draw_grid()
        self._draw_stones()
        self._draw_last_marker()
        self._draw_panel()

    def _draw_menu(self):
        """Iscrtava početni meni: naslov, pitanje i dugmad za izbor ko igra prvi."""
        self.screen.fill(PANEL_BG)
        title = self.big_font.render("Gomoku", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150)))
        sub = self.font.render("Who plays first?", True, TEXT)
        self.screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100)))
        mouse = pygame.mouse.get_pos()
        self._draw_button(self.btn_human, "You first  (black)", mouse)
        self._draw_button(self.btn_ai, "AI first  (white)", mouse)
        hint = self.font.render("click, or press 1 / 2", True, (150, 140, 130))
        self.screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 130)))

    def _draw_button(self, rect, label, mouse):
        """Iscrtava jedno dugme sa natpisom, sa isticanjem kada je miš iznad njega."""
        hovered = rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, BTN_HOVER if hovered else BTN, rect, border_radius=10)
        pygame.draw.rect(self.screen, LINE, rect, 2, border_radius=10)
        text = self.font.render(label, True, TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _draw_grid(self):
        """Iscrtava linije mreže table i zvezdane (hoshi) tačke."""
        for i in range(BOARD_SIZE):
            x0, y0 = grid_to_px(i, 0)
            x1, y1 = grid_to_px(i, BOARD_SIZE - 1)
            pygame.draw.line(self.screen, LINE, (x0, y0), (x1, y1), 1)
            x0, y0 = grid_to_px(0, i)
            x1, y1 = grid_to_px(BOARD_SIZE - 1, i)
            pygame.draw.line(self.screen, LINE, (x0, y0), (x1, y1), 1)
        for r, c in STAR_PTS:
            x, y = grid_to_px(r, c)
            pygame.draw.circle(self.screen, LINE, (x, y), 4)

    def _draw_stones(self):
        """Iscrtava sve postavljene kamenčiće (sa senkom) na njihovim poljima."""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = self.board.board[r][c]
                if stone == Stone.EMPTY:
                    continue
                x, y = grid_to_px(r, c)
                shadow = pygame.Surface((STONE_R * 2 + 4, STONE_R * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(shadow, SHADOW, (STONE_R + 2, STONE_R + 4), STONE_R)
                self.screen.blit(shadow, (x - STONE_R - 2, y - STONE_R - 2))
                color = BLACK if stone == Stone.BLACK else WHITE
                pygame.draw.circle(self.screen, color, (x, y), STONE_R)
                pygame.draw.circle(self.screen, LINE, (x, y), STONE_R, 1)

    def _draw_last_marker(self):
        """Iscrtava crvenu oznaku na poslednjem odigranom potezu."""
        if self.board.last_move is None:
            return
        x, y = grid_to_px(*self.board.last_move)
        pygame.draw.circle(self.screen, MARKER, (x, y), 5)

    def _draw_panel(self):
        """Iscrtava statusnu traku na dnu: čiji je potez, broj poteza ili ishod partije."""
        rect = pygame.Rect(0, HEIGHT - PANEL, WIDTH, PANEL)
        pygame.draw.rect(self.screen, PANEL_BG, rect)
        if self.game_over:
            if self.winner == self.human:
                msg = "You win!"
            elif self.winner == self.ai_stone:
                msg = "AI wins!"
            else:
                msg = "Tie game."
            msg += "   (R: menu)"
        else:
            your_turn = self.board.current_stone == self.human
            color = "black" if self.human == Stone.BLACK else "white"
            turn = f"Your turn ({color})" if your_turn else "AI thinking..."
            msg = f"{turn}    moves: {len(self.history)}   (R: menu)"
        label = self.font.render(msg, True, TEXT)
        self.screen.blit(label, (16, HEIGHT - PANEL + (PANEL - label.get_height()) // 2))

    def _draw_overlay(self):
        """Preko table iscrtava zatamnjenje i poruku o ishodu kada je partija gotova."""
        if self.state != "play" or not self.game_over:
            return
        overlay = pygame.Surface((WIDTH, HEIGHT - PANEL), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 90))
        self.screen.blit(overlay, (0, 0))
        if self.winner == self.human:
            text = "YOU WIN"
        elif self.winner == self.ai_stone:
            text = "AI WINS"
        else:
            text = "TIE"
        label = self.big_font.render(text, True, WHITE)
        rect = label.get_rect(center=(WIDTH // 2, (HEIGHT - PANEL) // 2))
        self.screen.blit(label, rect)

    # --- petlja -----------------------------------------------------------
    def run(self):
        """Glavna petlja igre: obrađuje događaje, iscrtava ekran i osvežava prikaz."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_click(event.pos)

            self.draw()
            self._draw_overlay()
            pygame.display.flip()
            self.clock.tick(60)

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

    def quit(self):
        """Uredno zatvara pygame i izlazi iz programa."""
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
