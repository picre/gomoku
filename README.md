# Gomoku — inteligentni protivnik (Minimax + alfa-beta + genetski algoritam)
Implementacija inteligentnog protivnika za igru **Gomoku** (pet u nizu) na tabli
15×15. Osnovu agenta čini **Minimax algoritam sa alfa-beta odsecanjem**, koji
bira potez na osnovu **heurističke funkcije evaluacije** zasnovane na
prepoznavanju karakterističnih obrazaca na tabli. Težine te funkcije optimizuje
**genetski algoritam** kroz veliki broj partija agenata jednih protiv drugih.
Uz to je priložen i funkcionalan prototip aplikacije (pygame) u kome čovek igra
protiv AI-ja.
---
## Sadržaj
- [Uvod](#uvod)
- [Kako se igra](#kako-se-igra)
- [Struktura projekta](#struktura-projekta)
- [Pregled arhitekture](#pregled-arhitekture)
- [Tabla i pravila (](#tabla-i-pravila-engineboardpy)`engine/board.py`[)](#tabla-i-pravila-engineboardpy)
- [Funkcija evaluacije (](#funkcija-evaluacije-engineevaluationpy)`engine/evaluation.py`[)](#funkcija-evaluacije-engineevaluationpy)
- [Minimax sa alfa-beta odsecanjem (](#minimax-sa-alfa-beta-odsecanjem-enginesearchpy)`engine/search.py`[)](#minimax-sa-alfa-beta-odsecanjem-enginesearchpy)
- [AI agent (](#ai-agent-engineaipy)`engine/ai.py`[)](#ai-agent-engineaipy)
- [Genetski algoritam (](#genetski-algoritam-genetic)`genetic/`[)](#genetski-algoritam-genetic)
- [Tok jednog poteza (end-to-end)](#tok-jednog-poteza-end-to-end)
- [Podešavanje i saveti](#podešavanje-i-saveti)
- [Optimizacije pretrage](#optimizacije-pretrage)
- [Moguća poboljšanja](#moguća-poboljšanja)
---
## Uvod
Potreban je Python 3.10+.
Na Windows-u je priložena skripta koja pravi virtuelno okruženje i instalira sve
zavisnosti:
```bat
setup_env.bat
```
Ručno (bilo koja platforma):
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```
Zavisnosti (`requirements.txt`): `numpy`, `tqdm`, `scipy`, `pygame`.
Pokretanje igre:
```bash
python play.py
```
Pokretanje treninga (genetska optimizacija težina):
```bash
python -m genetic.train --generations 15 --population 16
```
> **Važno o putanjama:** sve se pokreće **iz korena projekta** (`gomoku/`).
> Paketi `engine/` i `genetic/` rade kao *namespace* paketi (Python 3), pa uvozi
> tipa `from engine.board import GomokuBoard` rade kada je radni direktorijum
> koren projekta. `genetic/train.py` dodatno ubacuje koren u `sys.path`,
> tako da radi i kao `python -m genetic.train` i kao `python genetic/train.py`.
---
## Kako se igra
Cilj je postaviti **pet kamenčića u neprekinut niz** — horizontalno, vertikalno
ili dijagonalno — pre protivnika. Crni uvek igra prvi. Pre svake partije bira se
ko igra prvi, što određuje boju čoveka (prvi igrač dobija crne).
Kontrole u aplikaciji:
| Unos          | Radnja                                         |
| ------------- | ---------------------------------------------- |
| Levi klik     | postavljanje kamenčića / izbor opcije u meniju |
| `1` / `2`     | meni: čovek prvi / AI prvi                     |
| `R`           | povratak na početni meni                       |
| `Esc` / close | izlaz                                          |
---
## Struktura projekta
```
gomoku/
├── play.py                   # pygame aplikacija (GUI): čovek protiv AI-ja
├── main.py                   # ulazna tačka → pokreće play
├── requirements.txt          # zavisnosti
├── setup_env.bat             # pravljenje .venv i instalacija (Windows)
├── weights.json              # (opciono) evoluirane težine; pravi ih trening
├── data/
│   └── opening_book.json     # mapa sekvenca → sledeći potez (otvaranja)
├── lib/
│   └── constants.py          # enum Stone (BLACK / WHITE / EMPTY)
├── engine/
│   ├── board.py              # GomokuBoard: stanje table, potezi, provera pobede
│   ├── evaluation.py         # heuristička ocena pozicije (obrasci)
│   ├── weights.py            # podrazumevane težine, load/save, vektori gena
│   ├── incremental.py        # IncrementalEvaluator (delta ocena + Zobrist)
│   ├── search.py             # Minimax + alfa-beta + taktičke pretnje
│   ├── opening_book.py       # lookup u data/opening_book.json
│   └── ai.py                 # GomokuAI: knjiga → taktika → pretraga
└── genetic/
    ├── arena.py              # self-play: dve jedinke odigraju partiju
    ├── genetic_algorithm.py  # genetski algoritam (selekcija/ukrštanje/mutacija)
    └── train.py              # CLI ulazna tačka za trening
```
---
## Pregled arhitekture
Sistem je podeljen na tri sloja koji su međusobno labavo povezani:
1. **Model igre** (`lib/constants.py`, `engine/board.py`) — čisto stanje i
  pravila, bez ikakve „inteligencije".
2. **Agent** (`engine/evaluation.py`, `engine/weights.py`, `engine/incremental.py`,
  `engine/search.py`, `engine/ai.py`) — ocenjuje pozicije i pretragom bira
   najbolji potez.
3. **Optimizacija** (`genetic/`) — genetskim algoritmom traži težine heuristike
  koje daju najjaču igru, i snima ih u `weights.json`.
Tok podataka: `play.py` (GUI) drži `GomokuBoard` i istoriju poteza, pa poziva
`GomokuAI.choose_move(board, history)`. Agent redom: opening book → taktički
forsiran potez → `search_best_move` (minimax). Genetski algoritam iste
komponente koristi za self-play i podešavanje težina (bez opening book-a).
```
play.py ──> GomokuBoard ──┐
                          │  choose_move(board, history)
                          ▼
                     GomokuAI ──> opening_book / find_tactical_move / search_best_move
                                                        │
                                                        ▼
                                                    evaluate / IncrementalEvaluator
genetic.train ──> run_genetic_algorithm ──> arena.play_game ──> GomokuAI ──> weights.json
```
---
## Tabla i pravila (`engine/board.py`)
`Stone` (`lib/constants.py`) je enum sa tri vrednosti, kodirane kao `numpy.uint8`
radi kompaktnosti:
- `BLACK = 0`
- `WHITE = 1`
- `EMPTY = 2`
`GomokuBoard` čuva stanje kao 2D niz i prati čiji je potez:
- `board` — `numpy` niz `board_size × board_size` popunjen sa `Stone.EMPTY`.
- `current_stone` — ko je na potezu (počinje `BLACK`).
- `last_move` — poslednje odigrano polje (za brzu proveru pobede i crtanje).
Ključne metode:
- `make_move(row, col)` — postavlja kamenčić trenutnog igrača, pamti potez i
**prebacuje potez** na protivnika.
- `move_attempt(row, col)` — bezbedan potez: vraća `False` ako je nelegalan.
- `is_valid_move(row, col)` — polje je unutar table i prazno.
- `is_a_tie()` — nema više praznih polja.
- `player_has_won()` — razmatra samo okolinu `last_move`: u sva 4 pravca
(`horizontala`, `vertikala`, dve dijagonale) sabira niz kamenčića na obe strane
i proverava da li je dužina ≥ 5. Pomoćna `_count_in_direction` broji uzastopne
iste kamenčiće u jednom smeru.
> Napomena: pobeda se proverava kao **≥ 5** (dozvoljava i „overline" od 6+).
> Varijanta sa tačno pet u nizu zahteva izmenu uslova u `player_has_won`.
---
## Funkcija evaluacije (`engine/evaluation.py`)
Heuristička funkcija evaluacije dodeljuje numeričku ocenu poziciji na tabli.
Radi nad malim `int` numpy nizom (isto kodiranje kao `Stone`: 0/1/2), a ne nad
objektom `GomokuBoard`, jer pretraga mnogo puta menja pozicije i zahteva
jeftinu reprezentaciju.
### Prepoznavanje obrazaca
Svaki **pravac** table — vrsta, kolona ili dijagonala dužine ≥ 5 — pretvara se
u niz cifara iz ugla igrača koga ocenjujemo (relativno kodiranje, nije Stone):
- `1` — sopstveni kamenčić,
- `2` — protivnički kamenčić **ili zid** (ivica table),
- `0` — prazno polje.
Pravac se sa obe strane dopunjava sa `2` da bi ivice table brojale kao blokada
(npr. otvorena trojka uz ivicu više nije „otvorena").
**Suština je ista kao regex:** za svaki klizni prozor dužine 5 ili 6 ispituje se
da li se poklapa sa nekim obrascem iz `IMPORTANT_PATTERNS` (sa preklapanjem —
svaka početna pozicija). Umesto građenja niski i poziva regularnih izraza u toku
igre, to se radi **ternarnim kodiranjem i lookup-om**:
1. `_span_to_digits` mapira Stone vrednosti u `0/1/2` (sopstveni / protivnik / prazno)
  i dodaje zidove;
2. svaki prozor se sabije u jedan ceo broj u **bazi 3** (`_segment_codes`);
3. `_TABLE5` / `_TABLE6` unapred, za svaki mogući kod, drže koliko puta koji obrazac
  pogađa taj prozor — lookup umesto `re.search`.
Rezultat je brojač po kategorijama (`PATTERN_CATEGORIES`), **identičan** onome što
bi dao regex nad istim pravcem. Više o brzini u
[Optimizacije pretrage](#optimizacije-pretrage). Kategorije:
| Kategorija   | Značenje                                    | Primeri obrazaca                              |
| ------------ | ------------------------------------------- | --------------------------------------------- |
| `five`       | pet u nizu (pobeda)                         | `11111`                                       |
| `open_four`  | otvorena četvorka (neodbranjiva pretnja)    | `011110`                                      |
| `four`       | četvorka / prekinuta četvorka (1 do pobede) | `011112`, `211110`, `10111`, `11011`, `11101` |
| `open_three` | otvorena trojka (postaje otvorena četvorka) | `011100`, `001110`, `011010`, `010110`        |
| `three`      | zatvorena trojka                            | `211100`, `001112`, `211010`, ...             |
| `open_two`   | otvorena dvojka                             | `001100`, `011000`, `010100`, ...             |
| `two`        | dvojka                                      | `211000`, `210100`, `010010`, ...             |
`board_counts` prolazi kroz sve pravce i broji pojave svake kategorije za
zadatog igrača.
### Pozicioni faktori
- **Kontrola centra** (`center`): svaki kamenčić nosi poene srazmerno blizini
centru table (bliže centru = više). Ocenjuje se razlika (sopstveni − protivnički).
- **Povezanost** (`connectivity`): broj susednih parova sopstvenih kamenčića
(ortogonalno i dijagonalno), opet kao razlika u odnosu na protivnika.
### Forkovi (višestruke pretnje)
Fork = više istovremenih ozbiljnih pretnji odjednom (npr. dupla trojka). Meri se
brojem „pretnji":
```
threats = open_three + four + 2 * open_four
```
pa se dodaje `fork * max(0, threats − 1)` — bonus koji postoji tek kada postoji
**više od jedne** pretnje.
### Konačna formula
Neka su `own_pat` i `opponent_pat` zbirovi bodova za obrasce (uključujući fork bonus)
za igrača `own` i za protivnika. Ocena iz ugla `own` je:
```
value = own_pat
      − defense * opponent_pat
      + center * (razlika u kontroli centra)
      + connectivity * (razlika u povezanosti)
```
Pošto se protivnikovi obrasci **oduzimaju** (skalirani `defense` težinom),
pretraga prirodno favorizuje **blokadu** pretnji, a ne samo gradnju sopstvenih.
Pozitivna vrednost je dobra za igrača `own`, negativna za protivnika.
Numerički koeficijenti formule (`weights`) opisani su u odeljku [Genetski algoritam](#genetski-algoritam-genetic), zajedno sa njihovom optimizacijom.
---
## Minimax sa alfa-beta odsecanjem (`engine/search.py`)
Pretraga bira potez gledajući nekoliko poteza unapred, pod pretpostavkom da obe
strane igraju optimalno (strana na potezu maksimizuje ocenu, protivnik je
minimizuje).
Ključni delovi:
- `board_to_int(board)` — prevodi `GomokuBoard` (spor objektni niz enum-a) u
brzi `int8` niz nad kojim radi pretraga.
- `candidate_moves(arr, radius)` — **ograničava grananje**: umesto svih praznih
polja (do 225 na 15×15), razmatra samo prazna polja u okolini (`radius`)
postojećih kamenčića. Ako je tabla prazna, vraća centar.
- `_wins(arr, r, c)` — brza provera da li upravo postavljeni kamenčić pravi pet u
nizu (koristi se za rano prekidanje).
- `_minimax(...)` — standardni **alfa-beta** minimax. U listu (kada `depth <= 1`)
vraća `evaluate(...)`. Pobednička pozicija vraća `WIN_SCORE (10^7) + depth`
(`+depth` daje prednost **bržim** pobedama i sporijim porazima).
- `find_tactical_move(...)` — hitni / forsiran potez bez duboke pretrage:
  pobeda odmah, obavezna blokada, niz forsiranih pretnji (VCF-stil: pretnja
  koja mora da se brani, pa nastavak niza), otvorena četvorka / njena blokada.
- `search_best_move(...)` — ulazna tačka:
  1. prvo `find_tactical_move` (štedi vreme na očiglednim pretnjama);
  2. **sortira poteze** po plitkoj statičkoj oceni (bolje odsecanje);
  3. pokreće minimax za svaki potez i bira najbolji;
  4. izjednačene poteze bira nasumično preko prosleđenog `rng` (partije nisu
    potpuno deterministične).
Parametri: `depth` (dubina; podrazumevano 3 u igri, 1 u treningu radi brzine),
`radius` (širina razmatranja kandidata; podrazumevano 1).
---
## AI agent (`engine/ai.py`)
`GomokuAI` je tanak omotač koji spaja knjigu otvaranja, taktiku, pretragu i težine:
```python
GomokuAI(seed=None, depth=3, radius=1, weights=None, use_opening_book=True)
```
- `seed` — sopstveni RNG radi ponovljivosti (bez diranja globalnog `random`).
- `depth`, `radius` — prosleđuju se pretrazi.
- `weights` — ako se ne prosledi, poziva se `load_weights()` (učita
`weights.json` ako postoji, inače podrazumevane).
- `use_opening_book` — u GUI-ju uključeno; u areni treninga isključeno da se porede težine.
`choose_move(board, history=None)` bira potez ovim redosledom:
1. **otvaranje** — `opening_book.lookup(history)` iz `data/opening_book.json`
   (ako je knjiga uključena i sekvenca poznata);
2. **taktika** — `find_tactical_move` (forsirane pretnje / blokade);
3. **pretraga** — `search_best_move` (minimax + αβ) za mid-game.
Poslednji izvor poteza stoji u `last_phase` (`opening` / `tactical` / `search`).
`play.py` to prikazuje u konzoli i u statusnoj traci (npr. `AI: early (book)`
naspram `AI: mid (search)`).
`legal_moves(board)` je pomoćna statička metoda koja vraća sva prazna polja.
### Knjiga otvaranja
`data/opening_book.json` je mapa `sekvenca → sledeći potez` (prazan ključ =
početak partije; algebraic A–P bez I, red 1 = dno). Primer: `"" → F6`,
`"F6" → E5`, `"F6,E5" → H6`. Lookup je u `engine/opening_book.py`.
Ako tačan ključ ne postoji, a prvi potez je u zoni **±2** oko centra (`H8`),
knjiga pokušava **translaciju**: istorija se mapira na najbliži knjiški koren
u istoj zoni, pa se odgovor vrati istim pomerajem. Ako bi potez ispao van
table ili nema grane — mid-game pretraga.
---
## Genetski algoritam (`genetic/`)
Cilj genetskog algoritma jeste pronalazak kombinacije **težina** heurističke
funkcije koja daje najjaču igru. Populacija igra **round-robin** turnir
(svaki protiv svakog, obe boje); skor je +1 / −1 / 0. Najboljih nekoliko
jedinki proizvodi sledeću generaciju. Partije se paralelizuju.
### Težine (geni)
Koeficijenti formule evaluacije čuvaju se kao rečnik `weights`. Redosled gena
u vektoru za genetski algoritam definiše `WEIGHT_FIELDS` u `engine/weights.py`
(izveden iz `DEFAULT_WEIGHTS`, kako se polja ne bi razišla). Podrazumevane
vrednosti:
| Gen            | Podrazumevano | Uloga                                  |
| -------------- | ------------- | -------------------------------------- |
| `five`         | 100000        | pet u nizu (praktično presudno)        |
| `open_four`    | 10000         | otvorena četvorka                      |
| `four`         | 1000          | (prekinuta) četvorka                   |
| `open_three`   | 1000          | otvorena trojka                        |
| `three`        | 100           | zatvorena trojka                       |
| `open_two`     | 100           | otvorena dvojka                        |
| `two`          | 10            | dvojka                                 |
| `center`       | 3             | kontrola centra                        |
| `connectivity` | 5             | povezanost kamenčića                   |
| `fork`         | 2000          | bonus za višestruke pretnje            |
| `defense`      | 1.1           | težinski faktor protivničkih obrazaca  |
Pomoćne funkcije u `engine/weights.py`: `weights_to_vector` /
`vector_to_weights` (rečnik ↔ numpy vektor), `load_weights` i `save_weights`.
### Datoteka `weights.json`
Rezultat treninga nalazi se u korenu projekta (`WEIGHTS_PATH` u
`engine/weights.py`). Ako datoteka postoji, `GomokuAI` je učita pri pokretanju;
inače se koriste `DEFAULT_WEIGHTS`. Brisanje datoteke vraća podrazumevane
vrednosti. Pri snimanju i učitavanju apsolutni geni se **normalizuju** tako da
najveći iznosi `100000` (`SCALE_TARGET`); `defense` ostaje neskaliran jer je
relativni faktor. Za rangiranje poteza bitni su **odnosi** gena, ne njihov zbir.
Primer:
```json
{
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
  "defense": 1.1
}
```
### `arena.py` — self-play
- `play_game(black_weights, white_weights, depth, radius, size, ...)` — odigrava
jednu partiju između dva agenta i vraća pobednički `Stone` ili `None` za
nerešeno. Opening book je isključen; `force_depth=0` radi brzine treninga.
- `match_score(a, b, ...)` — meč od dve partije (svaka boja po jednom); koristi
se u head-to-head alatima.
### `genetic_algorithm.py` — evolucija
Glavna petlja `run_genetic_algorithm(...)` radi **round-robin** turnir:
1. **Inicijalizacija** — polazni genom iz `weights.json` (ili podrazumevane
  težine) + mutirana deca do veličine populacije.
2. **Fitnes** — svaki protiv svakog, obe boje; skor pobeda **+1**, poraz **−1**,
  nerešeno **0**. Partije idu na više procesa.
3. **Prolaz** — najboljih `elite` (podrazumevano 2) prelazi nepromenjeno.
4. **Deca** — ostatak populacije: ukrštanje elitnih roditelja + mutacija.
5. Najbolji genom se snima u `weights.json` pri poboljšanju i na kraju.
Za populaciju 10: \(10\times9 = 90\) partija po generaciji.
### `train.py` — pokretanje
Ulazna tačka iz komandne linije. Argumenti (sa podrazumevanim vrednostima):
| Argument          | Podr. | Opis                                        |
| ----------------- | ----- | ------------------------------------------- |
| `--population`    | 10    | broj jedinki (round-robin)                  |
| `--generations`   | 100   | broj generacija                             |
| `--depth`         | 2     | dubina minimax pretrage tokom self-play-a   |
| `--radius`        | 1     | radijus kandidat-poteza                     |
| `--size`          | 15    | veličina table za self-play partije         |
| `--elite`         | 2     | najboljih koji prolaze i proizvode decu     |
| `--mutation-rate` | 0.2   | verovatnoća mutacije po genu                |
| `--workers`       | CPU   | broj procesa za paralelne partije           |
| `--seed`          | —     | seme RNG-a radi ponovljivosti               |
| `--out`           | —     | izlazna JSON putanja (podr. `weights.json`) |
Primeri:
```bash
# brz probni trening (manja tabla)
python -m genetic.train --population 4 --generations 1 --size 9 --seed 42 --workers 2
# pun trening
python -m genetic.train --workers 12 --seed 42
```
---
## Tok jednog poteza (end-to-end)
1. Igrač klikne polje → `Game._on_click` → `Game.human_move`.
2. `human_move` odigrava potez (`board.make_move`) i proverava kraj partije.
3. Ako partija nije gotova, poziva se `Game.ai_move`.
4. `ai_move` → `GomokuAI.choose_move(board, history)`.
5. `choose_move`: knjiga otvaranja → taktika → inače `search_best_move`.
6. U listovima minimax-a ocena dolazi iz inkrementalne evaluacije / obrazaca.
7. Najbolji potez se vraća i odigrava; faza (`last_phase`) ide u konzolu i GUI.
---
## Podešavanje i saveti
- **Jačina vs. brzina:** veći `depth` daje jaču ali sporiju igru. Na 15×15 je
`depth=3` podrazumevani kompromis; dublje zahteva bolje odsecanje/ograničenje
kandidata (`radius`).
- **`radius`:** `radius=1` je najbrži i obično dovoljan; `radius=2` razmatra više
poteza (šire kombinacije) ali je sporiji.
- **Trening** podrazumevano ide na 15×15 i koristi više procesa (`--workers`).
Za brzu proveru može se smanjiti tabla (`--size 9`) ili broj radnika.
- **Ponovljivost:** zadaje se `--seed` za trening i `seed` u `GomokuAI` za partije.
---
## Optimizacije pretrage
Da bi pretraga bila upotrebljiva na 15×15 (i trening izvodljiv), evaluacija i
minimax koriste nekoliko optimizacija — u `engine/evaluation.py`,
`engine/incremental.py` i `engine/search.py`:
- **Unapred izračunate tabele obrazaca.** Logika je ista kao regex nad pravcem
(klizni prozori + poklapanje sa `IMPORTANT_PATTERNS`), samo bez niski i
`re` u toku igre: svaki prozor dužine 5/6 kodira se u bazi 3, a odgovor
„koji obrasci pogađaju?“ čita se iz `_TABLE5` / `_TABLE6`. Rezultati su
**identični** regex pristupu; razlika je samo brzina.
- **Koordinate pravaca** (`get_spans` / `get_cell_spans`,
`_SPANS` / `_CELL_SPANS`) — sve vrste, kolone i dijagonale, kao i
mapa polje→pravac, računaju se jednom po veličini table.
- **Inkrementalna (delta) evaluacija** (`IncrementalEvaluator`) — postavljanje
ili uklanjanje kamenčića menja samo **4 pravca** kroz to polje.
- **Transpoziciona tabela sa Zobrist hešom** — kešira ocene pozicija.
- **Uređivanje poteza** — sortiranje po plitkoj oceni radi jačeg alfa-beta
odsecanja.
- **Taktički presek** — pre dubokog minimax-a, forsiran niz pretnji / blokada.
- **Paralelni fitnes** — mečevi u genetskom treningu raspodeljeni na procese.
Orijentaciono, na praznijoj sredini table jedan potez na `depth=3` je i dalje
interaktivan; `depth=2` ubrzava igru uz pliću pretragu.
---
## Moguća poboljšanja
- **Širi VCT** (pobeda kroz trojke, ne samo četvorke) za duže forsiranja.
- **Ko-evolucija / turnir svih protiv svih** za stabilniji fitnes.
- **Proširenje knjige otvaranja** — više grana u `data/opening_book.json`.
