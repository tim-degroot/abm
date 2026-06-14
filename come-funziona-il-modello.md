# Come funziona il modello ABM del mercato immobiliare

Questo documento spiega in modo chiaro come è strutturato e come gira il modello.

---

## Panoramica

Il modello simula un **mercato immobiliare** in cui famiglie e istituzioni comprano, vendono e affittano immobili nel tempo. Ogni step corrisponde a **un mese**. Il default è 6000 step = 500 anni, ma si può cambiare in `config.toml`.

L'obiettivo scientifico è osservare come emergono dinamiche di prezzo, ownership e affitto a partire da regole semplici di comportamento individuale.

---

## 1. Gli agenti

### Famiglie (`HouseholdAgent`)
Ogni famiglia ha:
- **Reddito mensile** (distribuito log-normale, mediana ~£35.000/anno)
- **Liquidità** (cash), proporzionale al reddito
- **Avversione al rischio** (log-normale)
- **Aspettative** sui prezzi e sugli affitti futuri (adattive)

Il ruolo di una famiglia non è fisso, ma **emerge dallo stato**:

| Stato | Significato |
|---|---|
| `owner-occupier` | Vive in un immobile di sua proprietà |
| `renter` | Vive in un immobile affittato da altri |
| `landlord` | Possiede immobili che affitta a terzi |
| `owner-occupier+landlord` | Vive nel proprio immobile e ne affitta altri |
| `unhoused` | Tra una casa e l'altra (stato transitorio) |

### Istituzioni (`InstitutionalAgent`)
5 investitori istituzionali con grande liquidità (£5–20 milioni). Comprano immobili solo come investimento (affitto), con un **rendimento minimo richiesto** (`inst_min_yield = 5%` annuo lordo). Non vivono in nessun immobile.

---

## 2. La struttura spaziale

Il territorio è diviso in **zone** disposte su una **griglia toroidale** (10×10 = 100 zone di default). "Toroidale" significa che i bordi si collegano: non esistono angoli o bordi — ogni zona ha esattamente 4 vicini (su/giù/sinistra/destra).

Ogni agente cerca immobili solo nella **propria zona + le 4 zone adiacenti** (5 zone totali). Questo simula la ricerca localizzata nel mercato reale.

Le proprietà hanno una **qualità** che dipende dalla zona (le zone hanno qualità media diversa) più una componente casuale individuale. La qualità determina il prezzo ancora iniziale:

```
prezzo_ancora = 200.000 + 50.000 × qualità_normalizzata
```

---

## 3. L'inizializzazione

All'avvio (step 0), il modello costruisce tutto da zero in modo da garantire che i bilanci tornino per costruzione:

1. Genera il **parco immobiliare** (120 proprietà distribuite tra le zone)
2. Crea le **famiglie** (reddito, ricchezza totale, avversione al rischio)
3. Abbina famiglie a immobili per reddito (i più ricchi ottengono le case migliori)
4. Decide chi diventa **proprietario** in modo emergente: diventa proprietario chi può permettersi il deposito e supera il test reddito/mutuo (DTI); gli altri diventano affittuari
5. Assegna a una quota di proprietari un **portafoglio di immobili extra** da affittare (landlord privati, ~10% dei proprietari)
6. Le istituzioni ricevono una tranche separata di immobili (~10% del totale)
7. I residui affittuari vengono piazzati nello stock disponibile

Il bilancio di ogni proprietario è derivato dall'allocazione:
```
deposito = (1 - LTV) × prezzo
mutuo = LTV × prezzo
liquidità = ricchezza_totale - deposito
```

---

## 4. Il loop per step (un mese)

Ogni step esegue questo ciclo in ordine fisso:

```
1. Avanza lo stato macro (Boom / Neutral / Recession)
2. Evoluzione redditi delle famiglie
3. Rivalutazione mark-to-market degli immobili
4. Scadenza contratti d'affitto (lease turnover)
5. Pianifica vendite forzate (distress sales)
6. Ogni agente sceglie l'azione: buy / rent / hold / sell / rent_out
7. I venditori listano gli immobili sul mercato
8. Gli acquirenti e affittuari presentano le offerte
9. Clearing del mercato di proprietà (asta Vickrey)
10. Clearing del mercato degli affitti (asta Vickrey)
11. Applica le transazioni + aggiorna i bilanci
12. Servizio mutui (pagamento rate mensili)
13. Aggiorna le aspettative di ogni agente
14. Raccolta dati
```

---

## 5. Come gli agenti prendono decisioni

### Scelta dell'azione (logit multinomiale — Stage 1)
Ogni agente calcola uno **score** per ogni azione possibile e sceglie in modo probabilistico (non deterministico):

| Azione | Score basato su |
|---|---|
| `buy` | WTP massimo sugli immobili accessibili nell'area |
| `rent` | Carico dell'affitto rispetto al reddito (negativo) |
| `hold` | Aspettativa di crescita dei prezzi |
| `sell` | Crescita attesa negativa + piccola spinta a vendere |
| `rent_out` | Rendimento atteso da affitto sull'immobile occupato |

### Scelta dell'immobile (logit — Stage 2)
Tra i candidati nella propria area di ricerca, l'agente sceglie proporzionalmente alla propria WTP per ciascuno.

### Formazione dell'offerta (WTP — Stage 3)

**Famiglie (acquisto per abitare):**
```
WTP = (valore_qualità + guadagno_atteso - alternativa_affitto) / (tasso_mutuo × LTV)
```
Poi cappata da: (a) vincolo creditizio, (b) max prezzo/reddito (tetto fondamentale).

**Investitori (landlord e istituzioni):**
```
WTP = (affitto_netto + guadagno_atteso) / (tasso_finanziamento × LTV)
```
Le istituzioni hanno un tasso di finanziamento più basso dei landlord privati → WTP strutturalmente più alta a parità di immobile.

---

## 6. Il mercato: aste Vickrey

Entrambi i mercati (proprietà e affitti) usano un'**asta a secondo prezzo (Vickrey)**:
- Il vincitore è chi offre di più
- **Paga il secondo prezzo più alto** (o il prezzo di riserva del venditore, se superiore)

Questo meccanismo incentiva le offerte veritiere (non conviene bluffare).

Il venditore fissa un **prezzo di riserva**:
- Famiglie e landlord: 95% del valore stimato dell'immobile
- Istituzioni: 97% (meno disposte a svendere)

---

## 7. Il credito

La banca è modellata come vincolo esterno (nessun agente banca):

| Vincolo | Valore default |
|---|---|
| LTV massimo | 85% (deposito minimo 15%) |
| DTI massimo | 35% del reddito mensile per rata |
| Durata mutuo | 300 mesi (25 anni) |
| Tasso mutuo | ~5% annuo (0.4167%/mese) |
| Tasso BTL (landlord) | ~6% annuo |

Un acquisto è **fattibile** solo se soddisfa entrambi i vincoli (deposito E reddito).

---

## 8. Gli stati macro (Markov chain)

Il modello alterna tra tre stati economici con una **catena di Markov**:

| Stato | Crescita redditi | Volatilità |
|---|---|---|
| **Boom** | +0.25%/mese | bassa |
| **Neutral** | +0.08%/mese | media |
| **Recession** | -0.17%/mese | alta |

Le probabilità di transizione (configurabili) sono quasi diagonali: ogni stato tende a persistere (es. Boom→Boom = 98%). I redditi delle famiglie evolvono con shock log-normali centrati sul tasso del macro-stato corrente, con mean-reversion al reddito base dell'agente.

---

## 9. Le aspettative adattive

Ogni agente forma aspettative su prezzi e affitti futuri in modo adattivo:

```
E_t = 0.7 × E_{t-1} + 0.3 × Segnale_t
```

Il segnale è calcolato dalla crescita media degli ultimi 60 mesi (finestra scorrevole), più un po' di rumore casuale individuale. Le aspettative entrano direttamente nella WTP: più un agente si aspetta che i prezzi salgano, più è disposto a pagare oggi.

Per evitare esplosioni di prezzo, la crescita attesa è **cappata** (modalità `fixed_level`: guadagno atteso fisso a £167/mese, indipendente dal prezzo).

---

## 10. Turnover degli affitti

I contratti d'affitto hanno una durata minima di **12 mesi**. Dopo il minimo:
- Ogni mese c'è una probabilità del 2.78% che il contratto scada (~3 anni di durata media)
- Prima del minimo: probabilità bassissima di uscita anticipata (0.3%)
- L'affittuario già alloggiato può anche ricercare spontaneamente (~0.6%/mese)

Quando un contratto scade, l'immobile torna sul mercato degli affitti e l'ex-inquilino è "senza casa" e ri-cerca attivamente.

---

## 11. Cosa viene misurato

Ad ogni step il datacollector raccoglie:

- Prezzo medio transazioni (proprietà e affitti)
- Quota proprietà famiglie vs istituzioni
- Famiglie senza casa (`unhoused`)
- Cash totale, debito ipotecario, equity delle famiglie
- Price-to-rent ratio e price-to-income ratio
- LTV medio
- Vacancy rate
- Stato macro corrente
- Tasso di "bid ceiling" (quante offerte vengono cappate dal tetto fondamentale)
- Contatori di debug: bid presentati, filtrati, media offerte

---

## 12. Come eseguirlo

```bash
# Simulazione base (usa config.toml)
python run.py baseline

# Simulazione con shock creditizio
python run.py experiment

# Genera report/grafici
python run.py report

# Dashboard interattiva (con sliders)
solara run viz.py
```

Per cambiare i parametri: modifica `code/config.toml` — non serve toccare il codice Python.

---

## Riepilogo del flusso

```
config.toml
    ↓ (caricato e validato da config.py)
HousingModel.__init__()
    → crea immobili, agenti, alloca ownership/affitti iniziali
    ↓
HousingModel.step() × N mesi
    → macro state → redditi → mark-to-market → scadenze affitti
    → ogni agente: sceglie azione → lista o offerta
    → mercati: asta Vickrey → transazioni → bilanci
    → mutui → aspettative → datacollector
    ↓
DataFrame risultati → run.py report / viz.py
```
