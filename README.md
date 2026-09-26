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

- [Brzi start](#brzi-start)
- [Kako se igra](#kako-se-igra)
- [Struktura projekta](#struktura-projekta)
- [Pregled arhitekture](#pregled-arhitekture)
- [Tabla i pravila (`engine/board.py`)](#tabla-i-pravila-engineboardpy)
- [Funkcija evaluacije (`engine/evaluation.py`)](#funkcija-evaluacije-engineevaluationpy)
- [Minimax sa alfa-beta odsecanjem (`engine/search.py`)](#minimax-sa-alfa-beta-odsecanjem-enginesearchpy)
- [AI agent (`engine/ai.py`)](#ai-agent-engineaipy)
- [Genetski algoritam (`genetic/`)](#genetski-algoritam-genetic)
- [Datoteka `weights.json`](#datoteka-weightsjson)
- [Tok jednog poteza (end-to-end)](#tok-jednog-poteza-end-to-end)
- [Podešavanje i saveti](#podešavanje-i-saveti)
- [Optimizacije pretrage](#optimizacije-pretrage)
- [Moguća poboljšanja](#moguća-poboljšanja)

---

## Brzi start

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
> koren projekta. `genetic/train.py` dodatno sam ubacuje koren u `sys.path`,
> tako da radi i kao `python -m genetic.train` i kao `python genetic/train.py`.

---

## Kako se igra

Cilj je postaviti **pet kamenčića u neprekinut niz** — horizontalno, vertikalno
ili dijagonalno — pre protivnika. Crni uvek igra prvi. Pre svake partije biraš ko
igra prvi, što određuje tvoju boju (prvi igrač dobija crne).

Kontrole u aplikaciji:

| Unos          | Radnja                                        |
| ------------- | --------------------------------------------- |
| Levi klik     | postavljanje kamenčića / izbor opcije u meniju |
| `1` / `2`     | meni: ti prvi / AI prvi                       |
| `R`           | povratak na početni meni                      |
| `Esc` / close | izlaz                                         |

---

## Struktura projekta

```
gomoku/
├── play.py                 # pygame aplikacija (GUI): čovek protiv AI-ja
├── main.py                 # ulazna tačka → pokreće play
├── requirements.txt        # zavisnosti
├── setup_env.bat           # pravljenje .venv i instalacija (Windows)
├── weights.json            # (opciono) evoluirane težine; pravi ih trening
├── lib/
│   └── constants.py        # enum Stone (BLACK / WHITE / EMPTY)
├── engine/
│   ├── board.py            # GomokuBoard: stanje table, potezi, provera pobede
│   ├── evaluation.py       # heuristička ocena pozicije (obrasci)
│   ├── weights.py          # podrazumevane težine, load/save, GA vektori
│   ├── incremental.py      # IncrementalEvaluator (delta ocena + Zobrist)
│   ├── search.py           # Minimax + alfa-beta odsecanje
│   └── ai.py               # GomokuAI: bira potez koristeći pretragu
└── genetic/
    ├── arena.py            # self-play: dve jedinke odigraju partiju
    ├── ga.py               # genetski algoritam (selekcija/ukrštanje/mutacija)
    └── train.py            # CLI ulazna tačka za trening
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

Tok podataka: `play.py` (GUI) drži `GomokuBoard` i poziva `GomokuAI.choose_move`.
`GomokuAI` prevodi tablu u lagani `int` niz i poziva `search_best_move`
(minimax), koji u listovima stabla poziva `evaluate` (heuristika sa težinama).
Genetski algoritam iste te komponente koristi za self-play i podešavanje težina.

```
play.py ──> GomokuBoard ──┐
                          │  choose_move(board)
                          ▼
                     GomokuAI ──> board_to_int ──> search_best_move (minimax/αβ)
                                                        │
                                                        ▼
                                                    evaluate(arr, own, weights)

genetic.train ──> run_ga ──> arena.play_game ──> GomokuAI (razne težine) ──> weights.json
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
- `player_has_won()` — gleda samo oko `last_move`: u sva 4 pravca (`horizontala`,
  `vertikala`, dve dijagonale) sabira niz kamenčića na obe strane i proverava da
  li je dužina ≥ 5. Pomoćna `_count_in_direction` broji uzastopne iste kamenčiće
  u jednom smeru.

> Napomena: pobeda se proverava kao **≥ 5** (dozvoljava i „overline" od 6+); ako
> ti treba tačno 5, promeni uslov u `player_has_won`.

---

## Funkcija evaluacije (`engine/evaluation.py`)

Ovo je „mozak" statičke ocene pozicije. Umesto na `GomokuBoard`, radi na malom
`int` numpy nizu (isto kodiranje kao `Stone`: 0/1/2), jer pretraga mnogo puta
kopira i menja pozicije, pa je to znatno brže.

### Prepoznavanje obrazaca

Svaki **span** (pravac) table — vrsta, kolona ili dijagonala dužine ≥ 5 — pretvara se
u nisku iz ugla igrača koga ocenjujemo:

- `'1'` — sopstveni kamenčić,
- `'2'` — protivnički kamenčić **ili zid** (ivica table),
- `'0'` — prazno polje.

Span se sa obe strane dopunjava sa `'2'` da bi ivice table brojale kao blokada
(npr. otvorena trojka uz ivicu više nije „otvorena").

Zatim se u svakom span-u broje obrasci iz kataloga `IMPORTANT_PATTERNS`, klizanjem
**linija** dužine 5 i prozora dužine 6 preko **unapred izračunatih tabela**
(`_TABLE5`, `_TABLE6`); obrasci se broje sa **preklapanjem** (svaka početna
pozicija). Više o samoj tehnici u odeljku [Optimizacije pretrage](#optimizacije-pretrage). Kategorije:

| Kategorija    | Značenje                                   | Primeri obrazaca                        |
| ------------- | ------------------------------------------ | --------------------------------------- |
| `five`        | pet u nizu (pobeda)                        | `11111`                                 |
| `open_four`   | otvorena četvorka (neodbranjiva pretnja)   | `011110`                                |
| `four`        | četvorka / prekinuta četvorka (1 do pobede)| `011112`, `211110`, `10111`, `11011`, `11101` |
| `open_three`  | otvorena trojka (postaje otvorena četvorka)| `011100`, `001110`, `011010`, `010110`  |
| `three`       | zatvorena trojka                           | `211100`, `001112`, `211010`, ...       |
| `open_two`    | otvorena dvojka                            | `001100`, `011000`, `010100`, ...       |
| `two`         | dvojka                                     | `211000`, `210100`, `010010`, ...       |

`board_counts` prolazi kroz sve span-ove i broji pojave svake kategorije za
zadatog igrača.

### Pozicioni faktori

- **Kontrola centra** (`center`): svaki kamenčić nosi poene srazmerno blizini
  centru table (bliže centru = više). Ocenjuje se razlika (moji − protivnikovi).
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

Neka su `own_pat` i `opp_pat` zbirovi bodova za obrasce (uključujući fork bonus)
za mene i za protivnika. Ocena iz mog ugla je:

```
value = own_pat
      − defense * opp_pat
      + center * (razlika u kontroli centra)
      + connectivity * (razlika u povezanosti)
```

Pošto se protivnikovi obrasci **oduzimaju** (skalirani `defense` težinom),
agent prirodno uči da **blokira** pretnje, a ne samo da gradi svoje. Pozitivna
vrednost je dobra za igrača `own`, negativna za protivnika.

### Težine (geni)

Sve gore navedeno određeno je rečnikom `weights`. Redosled gena za genetski
algoritam definiše `WEIGHT_FIELDS` u `engine/weights.py` (izveden iz
`DEFAULT_WEIGHTS` da se ne mogu razići). Podrazumevane (ručno podešene)
vrednosti `DEFAULT_WEIGHTS`:

| Gen            | Podrazumevano | Uloga                                    |
| -------------- | ------------- | ---------------------------------------- |
| `five`         | 100000        | pet u nizu (praktično presudno)          |
| `open_four`    | 10000         | otvorena četvorka                        |
| `four`         | 1000          | (prekinuta) četvorka                     |
| `open_three`   | 1000          | otvorena trojka                          |
| `three`        | 100           | zatvorena trojka                         |
| `open_two`     | 100           | otvorena dvojka                          |
| `two`          | 10            | dvojka                                   |
| `center`       | 3             | kontrola centra                          |
| `connectivity` | 5             | povezanost kamenčića                     |
| `fork`         | 2000          | bonus za višestruke pretnje              |
| `defense`      | 1.1           | koliko ozbiljno shvatamo protivnika      |

Pomoćne funkcije (u `engine/weights.py`): `weights_to_vector` /
`vector_to_weights` (rečnik ↔ numpy vektor za GA), `load_weights` (učita
`weights.json` ako postoji, inače podrazumevane) i `save_weights` (upiše JSON).

---

## Minimax sa alfa-beta odsecanjem (`engine/search.py`)

Pretraga bira potez gledajući nekoliko poteza unapred, pod pretpostavkom da obe
strane igraju optimalno (ja maksimizujem ocenu, protivnik je minimizuje).

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
- `search_best_move(...)` — ulazna tačka:
  1. odmah vraća potez koji **momentalno pobeđuje** ako postoji;
  2. **sortira poteze** po plitkoj statičkoj oceni (bolje odsecanje);
  3. pokreće minimax za svaki potez i bira najbolji;
  4. izjednačene poteze bira nasumično preko prosleđenog `rng` (partije nisu
     potpuno deterministične).

Parametri: `depth` (dubina; podrazumevano 3 u igri, 1 u treningu radi brzine),
`radius` (širina razmatranja kandidata; podrazumevano 1).

---

## AI agent (`engine/ai.py`)

`GomokuAI` je tanak omotač koji spaja tablu, pretragu i težine:

```python
GomokuAI(seed=None, depth=3, radius=1, weights=None)
```

- `seed` — sopstveni RNG radi ponovljivosti (bez diranja globalnog `random`).
- `depth`, `radius` — prosleđuju se pretrazi.
- `weights` — ako se ne prosledi, poziva se `load_weights()` (učita
  `weights.json` ako postoji, inače podrazumevane).

`choose_move(board)`:

1. `board_to_int(board)` → `int` niz,
2. određuje ko je na potezu (`current_stone`),
3. `search_best_move(...)` vraća najbolji potez,
4. ako pretraga ništa ne vraća (krajnji slučaj), bira nasumičan legalan potez.

`legal_moves(board)` je pomoćna statička metoda koja vraća sva prazna polja.

> `play.py` konstruiše `GomokuAI()` bez argumenata, pa GUI automatski koristi
> `weights.json` ako postoji. Interfejs (`GomokuAI()`, `choose_move`,
> `legal_moves`) je namerno zadržan tako da GUI ne mora da se menja.

---

## Genetski algoritam (`genetic/`)

Cilj: pronaći kombinaciju **težina** iz funkcije evaluacije koja igra
najjači Gomoku. Svaka **jedinka** je vektor težina; njena **sposobnost (fitnes)**
meri se brojem partija koje osvoji protiv drugih jedinki.

### `arena.py` — self-play

- `play_game(black_weights, white_weights, depth, radius, size, ...)` — odigrava
  jednu partiju između dva agenta (svaki sa svojim težinama) i vraća pobednički
  `Stone` ili `None` za nerešeno. Trening podrazumevano koristi punu tablu
  (`size=15`) i plitku pretragu (`depth=1`) radi brzine.
- `match_score(a, b, ...)` — meč od **dve partije** (svaka strana jednom igra
  crnim, jednom belim, čime se poništava prednost prvog poteza). Vraća poene za
  `a` u opsegu `[0, 2]` (pobeda 1.0, nerešeno 0.5, poraz 0.0).

### `ga.py` — evolucija

Glavna petlja `run_ga(...)` radi sledeće:

1. **Inicijalizacija populacije** — `random_individual` uzima podrazumevane
   težine i množi ih nasumičnim faktorom u opsegu `1 ± spread` (podrazumevano
   ±0.5), uz odsecanje na nenegativne vrednosti.
2. **Evaluacija (fitnes)** — `evaluate_population`: svaka jedinka odigrava
   `opponents` mečeva protiv nasumično izabranih drugih jedinki; poeni se sabiraju.
3. **Selekcija** — `tournament_select`: bira se najbolja od `k` (=3) nasumičnih
   kandidatkinja.
4. **Ukrštanje** — `crossover`: **uniformno**, svaki gen nasumično dolazi od
   jednog od dva roditelja.
5. **Mutacija** — `mutate`: **multiplikativna Gaussova** (`gen *= 1 + N(0, scale)`)
   sa verovatnoćom `rate` (=0.2) po genu, uz odsecanje na nenegativno.
6. **Elitizam** — `elite` (=2) najboljih jedinki prelazi nepromenjeno u sledeću
   generaciju.
7. Najbolje težine se **snimaju u `weights.json`** pri svakom poboljšanju
   (checkpoint) i na kraju treninga.

`run_ga` vraća `(najbolje_težine, najbolji_fitnes)`.

### `train.py` — pokretanje

Ulazna tačka iz komandne linije. Argumenti (sa podrazumevanim vrednostima):

| Argument          | Podr. | Opis                                            |
| ----------------- | ----- | ----------------------------------------------- |
| `--population`    | 12    | broj jedinki po generaciji                      |
| `--generations`   | 10    | broj generacija                                 |
| `--opponents`     | 3     | mečeva po jedinki u svakoj generaciji           |
| `--depth`         | 1     | dubina minimax pretrage tokom self-play-a       |
| `--radius`        | 1     | radijus kandidat-poteza                         |
| `--size`          | 15    | veličina table za self-play partije             |
| `--elite`         | 2     | jedinke koje se prenose nepromenjene            |
| `--mutation-rate` | 0.2   | verovatnoća mutacije po genu                    |
| `--seed`          | —     | seme RNG-a radi ponovljivosti                   |
| `--out`           | —     | izlazna JSON putanja (podr. `weights.json`)     |

Primeri:

```bash
# brz probni trening (manja tabla samo za proveru da sve radi)
python -m genetic.train --population 6 --generations 3 --size 9 --seed 42

# pun trening na 15x15 (podrazumevana veličina)
python -m genetic.train --population 20 --generations 30 --opponents 5 --depth 1
```

---

## Datoteka `weights.json`

Rezultat treninga. Sadrži jedan rečnik težina (ista polja kao `WEIGHT_FIELDS`).
Nalazi se u **korenu projekta** (putanju definiše `WEIGHTS_PATH` u
`engine/weights.py`). Ako postoji, `GomokuAI` je automatski učita pri
pokretanju; ako ne postoji, koriste se `DEFAULT_WEIGHTS`. Slobodno je obriši da
se vratiš na ručno podešene vrednosti.

Primer sadržaja:

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

---

## Tok jednog poteza (end-to-end)

1. Igrač klikne polje → `Game._on_click` → `Game.human_move`.
2. `human_move` odigrava potez (`board.make_move`) i proverava kraj partije.
3. Ako partija nije gotova, poziva se `Game.ai_move`.
4. `ai_move` → `GomokuAI.choose_move(board)`.
5. `choose_move` prevodi tablu u `int` niz i poziva `search_best_move`.
6. `search_best_move` (minimax + αβ) u listovima poziva `evaluate`, koja prebrojava
   obrasce i vraća ocenu prema `weights`.
7. Najbolji potez se vraća i odigrava; ekran se ponovo iscrtava.

---

## Podešavanje i saveti

- **Jačina vs. brzina:** veći `depth` daje jaču ali sporiju igru. Na 15×15 je
  `depth=3` podrazumevani kompromis; dublje zahteva bolje odsecanje/ograničenje
  kandidata (`radius`).
- **`radius`:** `radius=1` je najbrži i obično dovoljan; `radius=2` razmatra više
  poteza (šire kombinacije) ali je sporiji.
- **Trening podrazumevano ide na 15×15** i podrazumeva mnogo partija, pa traje.
  Za brzu proveru da sve radi može se privremeno smanjiti tabla (`--size 9`); za
  jače težine povećavaju se populacija, broj generacija i `--opponents`.
- **Ponovljivost:** zadaje se `--seed` za trening i `seed` u `GomokuAI` za partije.

---

## Optimizacije pretrage

Da bi pretraga bila upotrebljiva na 15×15 (i trening izvodljiv), evaluacija i
minimax koriste nekoliko optimizacija — u `engine/evaluation.py`,
`engine/incremental.py` i `engine/search.py`:

- **Unapred izračunate tabele obrazaca.** Umesto regularnih izraza, skor svakog
  mogućeg prozora dužine 5 i 6 (kodiranog u bazi 3) izračunat je jednom u tabele
  (`_TABLE5`, `_TABLE6`). Prebrojavanje span-a je tada klizanje linija/prozora +
  sabiranje vektora iz tabele — bez građenja niski i bez regexa. Rezultati su
  **identični** ranijem regex pristupu (iste kategorije i brojevi).
- **Koordinate span-ova** (`get_spans` / `get_cell_spans`,
  `_SPANS` / `_CELL_SPANS`) — sve vrste, kolone i dijagonale, kao i
  mapa polje→span, računaju se jednom po veličini table.
- **Inkrementalna (delta) evaluacija** (`IncrementalEvaluator` u
  `engine/incremental.py`) — postavljanje
  ili uklanjanje kamenčića menja samo **4 span-a** kroz to polje, pa se tekuća
  ocena (brojači obrazaca, kontrola centra, povezanost) ažurira u O(4) span-ova
  umesto ponovnog skeniranja cele table u svakom čvoru. Puna
  `evaluate(...)` je zadržana kao referenca i za GA, i daje isti rezultat kao
  inkrementalna `value(...)`.
- **Transpoziciona tabela sa Zobrist hešom** — heš pozicije se održava
  inkrementalno (XOR pri svakom potezu), a ocene se keširaju (sa oznakama
  tačno/donja/gornja granica), tako da se ista pozicija dostignuta različitim
  redosledom poteza ne pretražuje ponovo.
- **Uređivanje poteza** — na korenu i u dubljim čvorovima potezi se sortiraju po
  plitkoj (inkrementalnoj) oceni radi jačeg alfa-beta odsecanja.

Orijentaciono, na praznijoj sredini table jedan potez na `depth=3` je i dalje
interaktivan; `depth=2` je brži ako treba ubrzati igru.

---

## Moguća poboljšanja

- **Iterativno produbljivanje** (uz postojeću transpozicionu tabelu) za bolju
  raspodelu vremena i još jače uređivanje poteza.
- **Detekcija forsiranih pobeda** kroz neprekidne pretnje (VCT/VCF).
- **Ko-evolucija / turnir svih protiv svih** za stabilniji fitnes u GA.
- **Otvaranja** (biblioteka poznatih poteza) za jaču ranu fazu igre.
