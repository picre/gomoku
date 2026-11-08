# Inteligentni agent za igru Gomoku zasnovan na Minimax pretrazi i genetskoj optimizaciji heuristike

**Seminarski rad**

| | |
| --- | --- |
| Predmet | [naziv predmeta] |
| Autor | [ime i prezime, broj indeksa] |
| Mentor | [ime i prezime] |
| Datum | [datum predaje] |

---

## Sažetak

U radu je opisana implementacija veštačkog agenta za igru Gomoku (pet u nizu)
na tabli 15×15. Agent bira potez kombinacijom tri tehnike: knjige otvaranja za
početak partije, taktičke analize forsiranih nizova pretnji i Minimax pretrage sa
alfa-beta odsecanjem za središnji deo partije. Listovi stabla pretrage ocenjuju
se heurističkom funkcijom zasnovanom na prepoznavanju karakterističnih obrazaca
kamenčića (otvorene i zatvorene dvojke, trojke i četvorke), uz pozicione faktore
kontrole centra i povezanosti. Prepoznavanje obrazaca svedeno je na unapred
izračunate tabele nad kodovima u bazi 3, a pretraga koristi inkrementalnu
evaluaciju, transpozicionu tabelu sa Zobrist heširanjem i uređivanje poteza.
Težine heurističke funkcije optimizovane su genetskim algoritmom u kome se
jedinke porede turnirom svako-protiv-svakog. U eksperimentu od 200 partija evoluirane težine
ostvarile su 50 % poena protiv početnih, što pokazuje da trening nije doveo do
statistički značajnog poboljšanja; mogući uzroci razmotreni su u poglavljima 5 i 6. Uz agenta je
razvijena i grafička aplikacija u kojoj čovek igra protiv računara.

**Ključne reči:** Gomoku, Minimax, alfa-beta odsecanje, heuristička funkcija
evaluacije, transpoziciona tabela, genetski algoritam, samoigra

---

## Sadržaj

1. [Uvod](#1-uvod)
2. [Teorijske osnove](#2-teorijske-osnove)
3. [Arhitektura sistema](#3-arhitektura-sistema)
4. [Implementacija](#4-implementacija)
5. [Eksperimentalna evaluacija](#5-eksperimentalna-evaluacija)
6. [Ograničenja i moguća poboljšanja](#6-ograničenja-i-moguća-poboljšanja)
7. [Zaključak](#7-zaključak)
8. [Literatura](#8-literatura)
9. [Prilog A: Instalacija i pokretanje](#prilog-a-instalacija-i-pokretanje)
10. [Prilog B: Parametri komandne linije](#prilog-b-parametri-komandne-linije)

---

## 1. Uvod

Društvene igre sa potpunom informacijom dugo su jedan od osnovnih poligona za
proučavanje veštačke inteligencije, jer imaju jasno definisana pravila i
merljiv ishod, a prostor stanja im je dovoljno veliki da iscrpna pretraga nije
moguća [3], [8]. Gomoku pripada toj grupi: dva igrača naizmenično postavljaju
kamenčiće na tablu 15×15, a pobeđuje onaj ko prvi formira neprekinut niz od pet
kamenčića horizontalno, vertikalno ili dijagonalno. Iako su pravila jednostavna,
faktor grananja na početku partije iznosi do 225 poteza, a partija može trajati
i preko stotinu poteza.

Cilj rada je razvoj agenta koji igra Gomoku na nivou koji je izazovan za
čoveka, uz dva dodatna zahteva:

1. vreme odlučivanja mora ostati interaktivno (reda veličine sekunde po potezu);
2. parametri heurističke funkcije ne treba da budu isključivo ručno podešeni,
   već optimizovani automatski, na osnovu ishoda partija.

Prvi zahtev rešava se kombinacijom ograničene Minimax pretrage, alfa-beta
odsecanja i niza optimizacija opisanih u poglavlju 4. Drugi zahtev rešava se
genetskim algoritmom [6], [7] u kome jedinke (skupovi težina) igraju jedna
protiv druge (samoigra), slično pristupu u radu [9].

Rad je organizovan na sledeći način. Poglavlje 2 daje teorijske osnove
korišćenih tehnika. Poglavlje 3 opisuje arhitekturu sistema, a poglavlje 4
implementaciju pojedinačnih komponenti. U poglavlju 5 prikazani su rezultati
eksperimenta, a poglavlje 6 navodi ograničenja i pravce daljeg razvoja.
Poglavlje 7 daje zaključak.

---

## 2. Teorijske osnove

### 2.1 Igra Gomoku

U radu se koristi varijanta **slobodnog gomokua** (engl. *freestyle*): pobeđuje
niz od pet *ili više* kamenčića, a crni igra prvi. Allis je pokazao da u ovoj
varijanti na tabli 15×15 prvi igrač ima dobitnu strategiju [1], [2]. Taj
rezultat ne umanjuje praktični značaj igre, jer dobitna strategija zahteva
pretragu ogromnog prostora pretnji i nije primenljiva u realnom vremenu.

Za opis pozicija koriste se sledeći standardni pojmovi:

- **petorka** — pet kamenčića u nizu (pobeda);
- **otvorena četvorka** — četiri kamenčića sa oba slobodna kraja; protivnik
  ne može da blokira oba kraja, pa je pobeda neizbežna;
- **četvorka** — četiri kamenčića (uključujući i prekinute oblike, npr. `11011`)
  kojima nedostaje jedan potez do pobede, sa samo jednim mogućim mestom dovršenja;
- **otvorena trojka** — tri kamenčića koja jednim potezom postaju otvorena
  četvorka;
- **višestruka pretnja** (engl. *fork*) — istovremeno postojanje dve ili više
  ozbiljnih pretnji, od kojih protivnik može da blokira najviše jednu.

### 2.2 Minimax i alfa-beta odsecanje

Minimax algoritam [3], [8] pretražuje stablo igre do zadate dubine pod
pretpostavkom da oba igrača igraju optimalno: igrač na potezu (MAX) bira potez
koji maksimizuje ocenu, a protivnik (MIN) potez koji je minimizuje. Pošto se
stablo ne može pretražiti do kraja partije, listovi se ocenjuju
**heurističkom funkcijom evaluacije**.

**Alfa-beta odsecanje** [4] održava interval `[α, β]` vrednosti koje još mogu
uticati na izbor u korenu i odbacuje grane izvan tog intervala. Rezultat je
identičan Minimax pretrazi, a broj posećenih čvorova se u najboljem slučaju
smanjuje sa `O(b^d)` na `O(b^(d/2))`, gde je `b` faktor grananja, a `d` dubina.
Najbolji slučaj se postiže kada se najbolji potezi razmatraju prvi, zbog čega je
**uređivanje poteza** ključno za efikasnost.

### 2.3 Transpoziciona tabela i Zobrist heširanje

Ista pozicija se u stablu može dostići različitim redosledima poteza
(transpozicijama). **Transpoziciona tabela** pamti već izračunate vrednosti
pozicija, zajedno sa dubinom pretrage i tipom granice (tačna vrednost, donja ili
gornja granica). Ključ pozicije računa se **Zobrist heširanjem** [5]: svakom
paru (polje, boja) dodeljuje se slučajan broj, a heš pozicije je XOR brojeva
svih zauzetih polja. Postavljanje i uklanjanje kamenčića ažurira heš jednom XOR
operacijom.

### 2.4 Forsirani nizovi pretnji

Allis je rešio Gomoku pretragom **prostora pretnji** (engl. *threat-space
search*) [1], [2]: umesto svih poteza razmatraju se samo potezi koji stvaraju
pretnju na koju protivnik mora da odgovori. Poseban slučaj je **VCF** (engl.
*victory by continuous fours*) — niz četvorki, od kojih svaka ostavlja
protivniku tačno jedan odgovor, a koji se završava dvostrukom pretnjom ili
petorkom. Zbog malog faktora grananja ovakvi nizovi mogu se pretraživati znatno
dublje od opšte Minimax pretrage.

### 2.5 Genetski algoritmi

Genetski algoritam [6], [7] je metaheuristika optimizacije inspirisana
prirodnom evolucijom. Populacija kandidat-rešenja (jedinki) iterativno se
poboljšava kroz **selekciju** boljih jedinki prema funkciji **prilagođenosti**
(engl. *fitness*), **ukrštanje** (kombinovanje gena roditelja) i **mutaciju**
(slučajne izmene gena). Pristup je pogodan kada funkcija cilja nije
diferencijabilna niti eksplicitno zadata, kao u slučaju jačine igre agenta, koja
se može meriti samo kroz odigrane partije [9].

---

## 3. Arhitektura sistema

Sistem je podeljen na tri labavo povezana sloja i grafičku aplikaciju:

1. **Model igre** (`lib/constants.py`, `engine/board.py`) — stanje table i
   pravila, bez ikakve „inteligencije".
2. **Agent** (`engine/evaluation.py`, `engine/incremental.py`,
   `engine/search.py`, `engine/opening_book.py`, `engine/weights.py`,
   `engine/ai.py`) — ocena pozicija i izbor poteza.
3. **Optimizacija** (`genetic/`) — genetski algoritam koji kroz samoigru
   podešava težine heurističke funkcije i snima ih u `weights.json`.
4. **Grafička aplikacija** (`play.py`, `main.py`) — igra čoveka protiv agenta.

### 3.1 Struktura projekta

```
gomoku/
├── main.py                   # ulazna tačka aplikacije
├── play.py                   # grafička aplikacija (pygame)
├── pyproject.toml            # metapodaci projekta, podešavanja pytest-a i ruff-a
├── requirements.txt          # zavisnosti
├── setup_env.bat             # pravljenje Conda okruženja (Windows)
├── weights.json              # evoluirane težine (rezultat treninga)
├── weights_initial.json      # početne, ručno izabrane težine (za eksperiment)
├── data/
│   └── opening_book.json     # knjiga otvaranja: sekvenca poteza → sledeći potez
├── lib/
│   └── constants.py          # enum Stone (BLACK / WHITE / EMPTY)
├── engine/
│   ├── board.py              # GomokuBoard: stanje table, potezi, provera pobede
│   ├── evaluation.py         # heuristička ocena pozicije (obrasci)
│   ├── incremental.py        # inkrementalna ocena i Zobrist heš
│   ├── search.py             # Minimax + alfa-beta + taktička analiza
│   ├── opening_book.py       # pretraživanje knjige otvaranja
│   ├── weights.py            # podrazumevane težine, normalizacija, čitanje/upis
│   └── ai.py                 # GomokuAI: knjiga → taktika → pretraga
├── genetic/
│   ├── arena.py              # partija između dva agenta (samoigra)
│   ├── genetic_algorithm.py  # selekcija, ukrštanje, mutacija, turnir
│   ├── train.py              # pokretanje treninga iz komandne linije
│   └── head_to_head.py       # eksperiment: početne protiv evoluiranih težina
└── tests/                    # automatski testovi (pytest)
```

### 3.2 Tok podataka

```
main.py ──> play.Game ──> GomokuBoard
                │
                │  choose_move(board, history)   [pozadinska nit]
                ▼
            GomokuAI ──> opening_book.lookup
                     ──> find_tactical_move
                     ──> search_best_move ──> IncrementalEvaluator ──> evaluation

genetic.train ──> run_genetic_algorithm ──> arena.play_game ──> GomokuAI
                                       └──> weights.json
```

Grafička aplikacija drži tablu i istoriju poteza i od agenta traži potez u
pozadinskoj niti, tako da prozor ostaje responzivan tokom pretrage. Genetski
algoritam koristi iste komponente agenta, ali bez knjige otvaranja i bez VCF
analize, kako bi ishod partija zavisio isključivo od težina heuristike.

---

## 4. Implementacija

Implementacija je napisana u jeziku Python 3.10+, uz biblioteke NumPy
(numerika), pygame (grafika) i tqdm (prikaz napretka).

### 4.1 Predstavljanje table

`Stone` je `IntEnum` sa vrednostima `BLACK = 0`, `WHITE = 1` i `EMPTY = 2`.
`GomokuBoard` čuva tablu kao NumPy niz tipa `int8`, igrača na potezu
(`current_stone`) i poslednji potez (`last_move`). Provera pobede razmatra samo
okolinu poslednjeg poteza: u svakom od četiri smera (horizontala, vertikala i dve
dijagonale) broje se uzastopni kamenčići iste boje na obe strane, pa trajanje
provere ne zavisi od broja kamenčića na tabli.

### 4.2 Funkcija evaluacije

Funkcija evaluacije (`engine/evaluation.py`) dodeljuje poziciji numeričku ocenu
iz ugla zadatog igrača.

**Pravci i relativno kodiranje.** Tabla 15×15 ima 72 **pravca** na kojima je
moguća pobeda: 15 vrsta, 15 kolona i 2 × 21 dijagonala dužine najmanje pet.
Svaki pravac se prevodi u niz cifara iz ugla igrača koji se ocenjuje:
`1` — sopstveni kamenčić, `2` — protivnički kamenčić ili ivica table,
`0` — prazno polje. Pravac se sa obe strane dopunjava cifrom `2`, pa ivica table
deluje kao blokada (npr. trojka uz ivicu nije otvorena).

**Obrasci.** Katalog obrazaca (`IMPORTANT_PATTERNS`) grupisan je u sedam
kategorija:

| Kategorija   | Značenje                              | Primeri obrazaca                              |
| ------------ | ------------------------------------- | --------------------------------------------- |
| `five`       | petorka (pobeda)                      | `11111`                                       |
| `open_four`  | otvorena četvorka                     | `011110`                                      |
| `four`       | četvorka, uključujući prekinute       | `011112`, `211110`, `10111`, `11011`, `11101` |
| `open_three` | otvorena trojka                       | `011100`, `001110`, `011010`, `010110`        |
| `three`      | zatvorena trojka                      | `211100`, `001112`, `211010`, …               |
| `open_two`   | otvorena dvojka                       | `001100`, `011000`, `010100`, …               |
| `two`        | zatvorena dvojka                      | `211000`, `210100`, `010010`, …               |

**Tabele pretraživanja umesto regularnih izraza.** Prirodna implementacija bi
nad tekstualnim zapisom pravca primenila regularne izraze za svaki obrazac. Umesto
toga, svaki klizni prozor dužine 5 ili 6 kodira se kao ceo broj u bazi 3, a za
svih `3^5 = 243` i `3^6 = 729` mogućih kodova unapred se izračunava vektor broja
pogodaka po kategorijama (`_TABLE5`, `_TABLE6`). Prebrojavanje obrazaca na
pravcu svodi se na vektorsko računanje kodova i sabiranje redova tabele.
Automatski test upoređuje rezultat sa referentnom implementacijom zasnovanom na
regularnim izrazima i potvrđuje da su rezultati identični (odeljak 5.4).

**Pozicioni faktori.**

- *kontrola centra*: svaki kamenčić donosi `n − (|r − c₀| + |c − c₀|)` poena,
  gde je `c₀` središte table; ocenjuje se razlika između igrača;
- *povezanost*: broj parova susednih sopstvenih kamenčića (ortogonalno i
  dijagonalno), takođe kao razlika između igrača.

**Višestruke pretnje.** Broj pretnji računa se kao
`threats = open_three + four + 2 · open_four`, a bonus iznosi
`fork · max(0, threats − 1)`, tj. postoji samo kada istovremeno postoji više od
jedne pretnje.

**Konačna formula.** Neka `P(x)` označava ponderisani zbir obrazaca igrača `x`
(uključujući bonus za višestruke pretnje). Ocena iz ugla igrača `own` je:

```
value = P(own) − defense · P(opponent)
      + center · Δcentar + connectivity · Δpovezanost
```

Pošto se protivnički obrasci oduzimaju, a težina `defense` je veća od 1,
pretraga prirodno favorizuje blokiranje protivničkih pretnji, a ne samo gradnju
sopstvenih.

### 4.3 Inkrementalna evaluacija

Ponovno skeniranje svih 72 pravca u svakom čvoru stabla bilo bi skupo. Klasa
`IncrementalEvaluator` (`engine/incremental.py`) zato održava brojače obrazaca
po pravcima i po bojama, zbirove kontrole centra i povezanosti, skup zauzetih
polja i Zobrist heš. Postavljanje ili uklanjanje kamenčića menja samo četiri
pravca kroz to polje, pa se preračunavaju samo oni. Konačna ocena se dobija iz
iste funkcije (`score_from_counts`) kao i u punoj evaluaciji, čime je
zagarantovano da su rezultati identični; i ovo je potvrđeno automatskim testom.

### 4.4 Minimax pretraga

Pretraga (`engine/search.py`) radi nad NumPy nizom, a potezi se postavljaju i
poništavaju u mestu, bez kopiranja table po čvoru. Glavni elementi su:

- **Ograničenje kandidata.** Razmatraju se samo prazna polja na udaljenosti
  najviše `radius` (podrazumevano 1) od nekog postojećeg kamenčića, što faktor
  grananja smanjuje sa 225 na tipično 10–40 poteza.
- **Uređivanje poteza.** Kandidati se sortiraju po plitkoj (inkrementalnoj)
  oceni posle odigravanja, čime se povećava efikasnost alfa-beta odsecanja.
- **Transpoziciona tabela.** Ključ je par (Zobrist heš, indikator MAX/MIN), a
  vrednost trojka (dubina, tip granice, ocena).
- **Ocena pobede.** Potez koji pravi petorku dobija `WIN_SCORE + depth`
  (`WIN_SCORE = 10^7`), pa agent preferira brže pobede i sporije poraze.
- **Ocena u listu.** Na dubini 0 proverava se da li igrač na potezu već ima
  četvorku ili otvorenu četvorku (sledećim potezom pobeđuje) i da li protivnik ima
  otvorenu četvorku (nezaustavljiva pretnja); u suprotnom se vraća heuristička
  ocena. Bez ove provere pretraga na neparnoj dubini ne bi uočila otvorenu
  četvorku čiji je jedan kraj upravo blokiran.
- **Prilagodljiva dubina.** Na širokim pozicijama dubina se smanjuje
  (do 20 kandidata — zadata dubina, do 35 — najviše 2, inače 1), da bi vreme
  odlučivanja ostalo interaktivno.
- **Pun prozor u korenu.** Svaki potez u korenu pretražuje se sa intervalom
  `(−∞, +∞)`. Prenošenje granice α između poteza u korenu dovodilo bi do toga da
  pretraga sa „mekim" odsecanjem (engl. *fail-soft*) vrati istu granicu i za
  lošije poteze, pa bi slučajni izbor među izjednačenim potezima mogao da
  izabere lošiji.

Izjednačeni najbolji potezi biraju se slučajno, pomoću generatora slučajnih
brojeva samog agenta, pa partije nisu potpuno determinističke, a ipak su
ponovljive za zadato seme.

### 4.5 Taktička analiza

Pre Minimax pretrage agent poziva `find_tactical_move`, koji redom proverava:

1. potez koji odmah pobeđuje;
2. blokadu protivničke pretnje pobedom u jednom potezu;
3. forsiran niz četvorki do pobede (VCF), ograničen na 20 najbolje ocenjenih
   napadačkih poteza po nivou i dubinu do 6 (na širokim pozicijama do 3);
4. potez koji pravi otvorenu četvorku;
5. blokadu postojeće protivničke otvorene četvorke.

Ako nijedan uslov nije ispunjen, potez se bira Minimax pretragom.

### 4.6 Knjiga otvaranja

Knjiga (`data/opening_book.json`, 54 unosa) je mapa koja niz poteza u
algebarskoj notaciji (kolone A–P bez I, red 1 na dnu table) preslikava u sledeći
potez. Ako tačan niz ne postoji, a prvi potez je u zoni ±2 polja oko centra,
pokušava se **translacija**: istorija se pomera tako da prvi potez padne na
najbliži koren iz knjige, a pronađeni odgovor se vraća istim pomerajem. Knjiga
se koristi samo na tabli 15×15.

### 4.7 Agent

`GomokuAI` (`engine/ai.py`) objedinjuje prethodne komponente redosledom
knjiga otvaranja → taktička analiza → Minimax pretraga i u atributu
`last_phase` beleži koji je izvor dao potez. Agent ne poseduje stanje table:
čita prosleđenu tablu i vraća potez, a primena poteza je odgovornost petlje igre.

### 4.8 Genetski algoritam

**Predstavljanje jedinke.** Jedinka je vektor od 11 realnih gena — težina
heurističke funkcije:

| Gen            | Početna vrednost | Uloga                                      |
| -------------- | ---------------: | ------------------------------------------ |
| `five`         |           100000 | petorka                                    |
| `open_four`    |            10000 | otvorena četvorka                          |
| `four`         |             1000 | četvorka                                   |
| `open_three`   |             1000 | otvorena trojka                            |
| `three`        |              100 | zatvorena trojka                           |
| `open_two`     |              100 | otvorena dvojka                            |
| `two`          |               10 | zatvorena dvojka                           |
| `center`       |                3 | kontrola centra                            |
| `connectivity` |                5 | povezanost kamenčića                       |
| `fork`         |             2000 | bonus za višestruke pretnje                |
| `defense`      |              1.1 | faktor ponderisanja protivničkih obrazaca  |

Za rangiranje poteza bitni su samo odnosi težina, pa se pri snimanju svi geni
osim `defense` skaliraju tako da najveći iznosi 100000; `defense` je relativni
faktor i ne skalira se.

**Inicijalizacija.** Početna populacija sadrži polazni vektor (iz
`weights.json`, ili podrazumevane težine) i njegove slučajne varijacije, u kojima
je svaki gen pomnožen faktorom iz intervala `[0,5; 1,5]`.

**Prilagođenost.** U svakoj generaciji jedinke igraju turnir svako-protiv-svakog,
po jednu partiju svakom bojom (za populaciju veličine `N` to je `N(N − 1)`
partija). Pobeda donosi +1, poraz −1, nerešeno 0. Tokom samoigre koristi se
dubina pretrage 2, a knjiga otvaranja i VCF analiza su isključeni. Partije se
izvršavaju paralelno, na više procesa.

**Selekcija i reprodukcija.** Primenjuje se elitizam: najboljih `elite`
(podrazumevano 2) jedinki prelazi u sledeću generaciju nepromenjeno, a ostatak
populacije nastaje **uniformnim ukrštanjem** nasumično izabranih elitnih
roditelja i **multiplikativnom Gausovom mutacijom** (svaki gen se sa
verovatnoćom 0,2 množi faktorom `1 + N(0; 0,2²)`, uz ograničenje na nenegativne
vrednosti).

**Izbor rezultata.** Skor iz turnira je relativan: meri jedinku samo u odnosu na
ostale članove iste generacije, pa skorovi iz različitih generacija nisu
uporedivi. Zato se kao rezultat uzima šampion poslednje generacije, koji se snima
posle svake generacije. Zahvaljujući elitizmu, šampion svake generacije igrao je
u istom turniru sa šampionom prethodne i bio bolji ili jednak njemu.

### 4.9 Grafička aplikacija

Aplikacija (`play.py`) je napisana u biblioteci pygame. Pre partije korisnik bira
ko igra prvi, a prvi igrač dobija crne kamenčiće. Statusna traka prikazuje igrača
na potezu, broj poteza i izvor poslednjeg AI poteza (knjiga, taktika ili
pretraga). Potez agenta računa se u pozadinskoj niti nad kopijom table, pa
interfejs ne blokira tokom pretrage; rezultat koji pripada napuštenoj partiji
(npr. posle povratka u meni) se odbacuje.

---

## 5. Eksperimentalna evaluacija

### 5.1 Postavka eksperimenta

Cilj eksperimenta je da se utvrdi da li težine dobijene genetskim algoritmom
(`weights.json`) daju jaču igru od početnih, ručno izabranih težina
(`weights_initial.json`). Eksperiment se izvodi skriptom `genetic.head_to_head`:

```bash
python -m genetic.head_to_head --rounds 100 --seed 42 --workers 2
```

- 100 rundi, svaka sa dve partije u kojima agenti menjaju boje, ukupno
  200 partija; zamena boja poništava prednost prvog poteza;
- tabla 15×15, dubina pretrage 2, `radius = 1`;
- knjiga otvaranja i VCF analiza isključeni (kao i tokom treninga);
- seme 42 radi ponovljivosti.

Evoluirane težine dobijene su treningom sa sledećim parametrima:
[upisati parametre treninga kojim je dobijen `weights.json`: veličina populacije,
broj generacija, dubina, seme].

### 5.2 Rezultati

| Ishod                         | Broj partija | Udeo |
| ----------------------------- | -----------: | ---: |
| pobeda evoluiranih težina     |          100 | 50,0 % |
| pobeda početnih težina        |          100 | 50,0 % |
| nerešeno                      |            0 |  0,0 % |
| **ukupno**                    |          200 | 100 % |

Evoluirane težine ostvarile su **50,0 %** mogućih poena (pobeda = 1, nerešeno = ½). Wilsonov 95 % interval poverenja za taj udeo iznosi [43,1 %; 56,9 %]. Interval obuhvata 50 %, pa razlika u jačini **nije statistički značajna**: na ovom uzorku ne može se tvrditi da su evoluirane težine jače ili slabije od početnih. Za pouzdaniji zaključak potrebno je više partija ili duži trening (videti poglavlje 6).

### 5.3 Analiza evoluiranih težina

Tabela poredi početne i evoluirane težine. Pošto je bitan samo odnos težina,
pored apsolutnih vrednosti dati su i odnosi susednih kategorija.

| Gen            | Početna | Evoluirana |
| -------------- | ------: | ---------: |
| `five`         |  100000 |     100000 |
| `open_four`    |   10000 |     1479,8 |
| `four`         |    1000 |      711,7 |
| `open_three`   |    1000 |      137,7 |
| `three`        |     100 |       50,5 |
| `open_two`     |     100 |       80,3 |
| `two`          |      10 |        7,8 |
| `center`       |       3 |        1,8 |
| `connectivity` |       5 |        3,0 |
| `fork`         |    2000 |      462,2 |
| `defense`      |     1,1 |       1,01 |

| Odnos                    | Početni | Evoluirani |
| ------------------------ | ------: | ---------: |
| `four / open_four`       |    0,10 |       0,48 |
| `open_three / four`      |    1,00 |       0,19 |
| `three / open_three`     |    0,10 |       0,37 |
| `open_two / open_three`  |    0,10 |       0,58 |
| `fork / four`            |    2,00 |       0,65 |

Uočava se da je evolucija znatno smanjila razliku između „otvorenih" i
„zatvorenih" oblika: zatvorena četvorka vredi skoro polovinu otvorene (umesto
desetine), a otvorena trojka je postala znatno manje vredna od četvorke.
Istovremeno je smanjen bonus za višestruke pretnje i faktor odbrane se približio
jedinici. Jedno moguće objašnjenje je da na dubini pretrage 2 sama pretraga i
taktička provera u listu već otkrivaju neposredne posledice otvorenih oblika, pa
heuristika ne mora dodatno da ih naglašava. Ovo tumačenje je hipoteza i
zahtevalo bi dodatne eksperimente sa različitim dubinama.

### 5.4 Verifikacija ispravnosti

Projekat sadrži 39 automatskih testova (`tests/`, okvir pytest) koji proveravaju:

- pravila igre: pobedu u sva četiri smera, niz duži od pet, nerešen ishod i
  smenu igrača;
- da se rezultati prebrojavanja obrazaca preko tabela pretraživanja **poklapaju**
  sa referentnom implementacijom pomoću regularnih izraza, na 30 nasumičnih
  pozicija (svih 72 pravca, obe boje);
- da je inkrementalna ocena jednaka punoj oceni posle niza od 40 nasumičnih
  postavljanja, kao i da se Zobrist heš vraća na početnu vrednost posle
  poništavanja poteza;
- taktiku (uzimanje pobede, blokada četvorke), legalnost izabranog poteza i
  ponovljivost agenta za zadato seme;
- knjigu otvaranja (konverziju notacije, tačno pretraživanje i translaciju);
- genetske operatore, normalizaciju težina i to da je turnir igra sa nultom sumom.

---

## 6. Ograničenja i moguća poboljšanja

- **Dubina pretrage.** Na širokim pozicijama dubina se smanjuje na 1 ili 2, što
  ograničava strateško planiranje. Iterativno produbljivanje sa vremenskim
  ograničenjem i potezi-ubice (engl. *killer moves*) omogućili bi dublju pretragu
  u istom vremenu.
- **Forsirani nizovi.** VCF analiza razmatra samo nizove četvorki. Proširenje na
  nizove koji uključuju i otvorene trojke (VCT, engl. *victory by continuous
  threats*) [1] otkrivalo bi duže kombinacije.
- **Pouzdanost fitnesa.** Turnir u maloj populaciji, sa po jednom partijom po
  boji, daje skor sa velikom varijansom. Više partija po paru ili poređenje sa
  fiksnim referentnim protivnikom dalo bi stabilniji signal.
- **Uslovi treninga.** Težine su optimizovane na dubini 2 bez taktičke analize,
  dok aplikacija koristi dubinu 3 sa taktičkom analizom; težine optimalne za
  jedan režim ne moraju biti optimalne za drugi.
- **Knjiga otvaranja.** Knjiga je mala (54 unosa) i mogla bi se proširiti
  analizom partija jakih igrača.
- **Pravila.** Implementiran je slobodni gomoku, u kome prvi igrač ima dobitnu
  strategiju [1]. Varijante sa ograničenjima za crnog (npr. Renju) bile bi
  ravnopravnije.

---

## 7. Zaključak

U radu je razvijen agent za igru Gomoku koji kombinuje knjigu otvaranja,
analizu forsiranih nizova pretnji i Minimax pretragu sa alfa-beta odsecanjem.
Pokazano je da se prepoznavanje obrazaca može svesti na unapred izračunate
tabele nad kodovima u bazi 3, bez gubitka tačnosti u odnosu na pristup sa
regularnim izrazima, i da inkrementalna evaluacija, transpoziciona tabela i
uređivanje poteza čine pretragu dovoljno brzom za interaktivnu igru.

Težine heurističke funkcije optimizovane su genetskim algoritmom kroz samoigru.
U 200 partija protiv početnih težina evoluirane težine ostvarile su 50,0 % poena, što se statistički ne razlikuje od 50 %; genetski algoritam je pronašao težine bitno drugačijih odnosa, čija se jačina u ovoj postavci statistički ne razlikuje od početnih. Dalji rad bi se mogao usmeriti na dublju pretragu (iterativno
produbljivanje), širu analizu pretnji (VCT) i pouzdaniju funkciju
prilagođenosti u genetskom algoritmu.

---

## 8. Literatura

1. Allis, L. V. (1994). *Searching for Solutions in Games and Artificial
   Intelligence* (doktorska disertacija). University of Limburg, Maastricht.
2. Allis, L. V., van den Herik, H. J., & Huntjens, M. P. H. (1996). Go-Moku
   solved by new search techniques. *Computational Intelligence, 12*(1), 7–23.
3. Shannon, C. E. (1950). Programming a computer for playing chess.
   *Philosophical Magazine, 41*(314), 256–275.
4. Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning.
   *Artificial Intelligence, 6*(4), 293–326.
5. Zobrist, A. L. (1970). *A new hashing method with application for game
   playing* (Technical Report 88). University of Wisconsin, Madison.
6. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
   University of Michigan Press.
7. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and
   Machine Learning*. Addison-Wesley.
8. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern
   Approach* (4th ed.). Pearson.
9. Chellapilla, K., & Fogel, D. B. (1999). Evolving neural networks to play
   checkers without relying on expert knowledge. *IEEE Transactions on Neural
   Networks, 10*(6), 1382–1391.

---

## Prilog A: Instalacija i pokretanje

Potreban je Python 3.10 ili noviji. Sve komande se pokreću iz korena projekta.

**Windows (Conda).** Skripta `setup_env.bat` pronalazi Minicondu/Anacondu, pravi
okruženje `gomoku` (Python 3.12) i instalira zavisnosti:

```bat
setup_env.bat
```

**Bilo koja platforma (venv).**

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

**Pokretanje.**

```bash
python main.py                                   # igra protiv AI-ja
python -m genetic.train --generations 15         # trening težina
python -m genetic.head_to_head --rounds 100      # eksperiment iz poglavlja 5
python -m pytest                                 # automatski testovi
```

Kontrole u aplikaciji:

| Unos          | Radnja                                         |
| ------------- | ---------------------------------------------- |
| levi klik     | postavljanje kamenčića / izbor opcije u meniju |
| `1` / `2`     | meni: čovek prvi / AI prvi                     |
| `R`           | povratak na početni meni                       |
| `Esc` / close | izlaz                                          |

---

## Prilog B: Parametri komandne linije

**`python -m genetic.train`**

| Argument          | Podrazumevano | Opis                                        |
| ----------------- | ------------- | ------------------------------------------- |
| `--population`    | 10            | broj jedinki u generaciji                   |
| `--generations`   | 100           | broj generacija                             |
| `--depth`         | 2             | dubina Minimax pretrage tokom samoigre      |
| `--radius`        | 1             | radijus kandidat-poteza                     |
| `--size`          | 15            | veličina table                              |
| `--elite`         | 2             | broj najboljih jedinki koje se prenose      |
| `--mutation-rate` | 0.2           | verovatnoća mutacije po genu                |
| `--workers`       | broj jezgara  | broj procesa za paralelne partije           |
| `--seed`          | —             | seme generatora slučajnih brojeva           |
| `--out`           | `weights.json`| izlazna putanja                             |

**`python -m genetic.head_to_head`**

| Argument     | Podrazumevano          | Opis                                   |
| ------------ | ---------------------- | -------------------------------------- |
| `--initial`  | `weights_initial.json` | početne težine                         |
| `--evolved`  | `weights.json`         | evoluirane težine                      |
| `--rounds`   | 200                    | broj rundi (po dve partije)            |
| `--depth`    | 2                      | dubina pretrage                        |
| `--radius`   | 1                      | radijus kandidat-poteza                |
| `--size`     | 15                     | veličina table                         |
| `--workers`  | broj jezgara           | broj procesa                           |
| `--seed`     | —                      | seme generatora slučajnih brojeva      |
