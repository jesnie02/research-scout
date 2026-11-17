# Research Scout Agent

En AI-agent bygget med [PydanticAI](https://pydantic.dev/docs/ai/), som finder en forskningsartikel ud
fra et emne og krav til udgivelsesår og antal citationer, og returnerer et struktureret, dokumenteret
svar. Agenten søger i [OpenAlex](https://openalex.org) og bruger en gratis LLM (Cohere Command A+).

Agenten bygger aldrig på modellens egen viden om artikler. Titel, forfattere, år, citationstal og link
kommer altid fra OpenAlex. Modellen fortolker forespørgslen, vælger blandt de hentede kandidater og
skriver forklaringen.

Compulsory Assignment #2, Machine Learning (E2026).

## Indhold

- [Eksempel](#eksempel)
- [Opsætning](#opsætning)
- [Kør agenten](#kør-agenten)
- [Workflow: hvad gør LLM'en, og hvad gør koden?](#workflow-hvad-gør-llmen-og-hvad-gør-koden)
- [Værktøjet: OpenAlex](#værktøjet-openalex)
- [Fortolkning af krav](#fortolkning-af-krav)
- [Valg af model](#valg-af-model)
- [Evaluering](#evaluering)
- [Resultater](#resultater)
- [Refleksion](REFLECTION.md)
- [Gruppemedlemmernes bidrag](#gruppemedlemmernes-bidrag)
- [Projektstruktur og udvikling](#projektstruktur-og-udvikling)

## Eksempel

```bash
uv run python main.py "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
```

```text
Outcome: match
Found a paper that meets the constraints.

Interpreted topic: retrieval-augmented generation
Constraints: year min=None max=2020, citations min=501 max=None

Title:      Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Authors:    Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin et al.
Year:       2020
Citations:  10269
Source:     OpenAlex cited_by_count for W3027879771, retrieved 2026-11-17 13:02 UTC: https://openalex.org/W3027879771
Link:       https://doi.org/10.48550/arxiv.2005.11401

The Retrieval-Augmented Generation (RAG) paper introduces a framework that combines ... (7 sætninger)

[2 search(es), 2 tool call(s), 4 model request(s)]
```

## Opsætning

Kræver [uv](https://docs.astral.sh/uv/) og Python 3.12 eller nyere.

```bash
git clone https://github.com/jesnie02/research-scout.git
cd research-scout
uv sync                  # installerer alle pakker fra uv.lock
cp .env.example .env     # udfyld derefter nøglerne; .env er i .gitignore
uv run pytest            # testene kører uden netværk og uden nøgler
```

### API-nøgler

Alle nøgler ligger i `.env`, som aldrig committes. `.env.example` viser variablerne.

| Variabel | Bruges til | Hvor fås den |
|---|---|---|
| `LLM_BASE_URL` | Modellens API. Til Cohere: `https://api.cohere.com` | – |
| `LLM_MODEL` | Model-ID. Til Cohere: `command-a-plus-05-2026` | – |
| `LLM_API_KEY` | Nøgle til modellen | Gratis prøvenøgle via [cohere.com](https://cohere.com) |
| `OPENALEX_API_KEY` | Søgning i OpenAlex | Gratis konto på [openalex.org](https://openalex.org) |
| `COHERE_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` | Kun til modelmålingerne i `scripts/` | Udbydernes egne sider |

En Cohere-adresse vælger automatisk PydanticAI's indbyggede Cohere-klient (se [Valg af model](#valg-af-model)).
Andre adresser bruger den OpenAI-kompatible klient, som i kursets labs.

## Kør agenten

```bash
uv run python main.py "Find a recent paper about AI agents using tools and explain whether it would be useful for someone building autonomous software agents."
uv run python main.py "..." --json     # hele det strukturerede resultat som JSON
```

Svaret indeholder titel, forfattere, år, citationstal, kilden til citationstallet (OpenAlex-ID og
tidspunkt), link til artiklen og en forklaring. Når agenten ikke finder noget, eller forespørgslen
ikke kan opfyldes, siger den det i stedet for at gætte.

## Workflow: hvad gør LLM'en, og hvad gør koden?

```text
Brugerens forespørgsel
 1. LLM   Agent A læser forespørgslen: emne, krav til år og citationer, "recent",
          og hvad svaret skal indeholde (fx resumé på 5-7 sætninger).
    KODE  Mangler emnet → agenten stiller et opklarende spørgsmål og søger ikke.
 2. KODE  Kravene omsættes til faste intervaller. Modstridende krav (fx før 2010
          og efter 2020) afvises med en forklaring, uden søgning.
 3. LLM   Agent B kalder search_papers med søgeord, læser kandidaterne og søger
          evt. igen med andre ord (højst 3 søgninger). Intervallerne ligger i
          koden, så modellen kan ikke løsne dem.
    KODE  Værktøjet søger i OpenAlex, kontrollerer hvert resultat og gemmer
          kandidaterne (se næste afsnit).
 4. LLM   Agent B vælger én kandidat ud fra dens OpenAlex-ID (eller null) og
          skriver svarteksten ud fra titel og abstract.
 5. KODE  Svaret sendes tilbage til modellen (højst 2 gange), hvis ID'et ikke er
          en af kandidaterne, hvis den ikke har søgt, hvis antallet af sætninger
          er forkert, eller hvis teksten nævner tal, som ikke findes i artiklens data.
 6. KODE  Det endelige svar bygges kun af de gemte OpenAlex-data. Til sidst
          tjekkes år og citationer igen, og koden afgør udfaldet.
```

Trin 3-4 er et ReAct-forløb (ræsonnér, handl, observér) som i uge 46.

**Udfald**, som altid bestemmes af koden:

| Udfald | Betyder |
|---|---|
| `match` | En artikel opfylder kravene |
| `no_match` | Søgningerne lykkedes, men ingen artikel opfyldte kravene. Det betyder kun "ikke fundet i disse søgninger", ikke "findes ikke" |
| `rejected_on_relevance` | Der var kandidater, men modellen vurderede, at ingen var relevante |
| `needs_clarification` | Forespørgslen har intet emne, der kan søges på |
| `conflicting_constraints` | Kravene kan ikke opfyldes samtidig |
| `search_error` | Mindst én søgning fejlede, så agenten kan ikke sige, om artiklen findes. Bliver aldrig til `no_match` |
| `agent_error` | Ugyldigt svar efter genforsøg, opbrugt budget eller tidsgrænse |

**Budget pr. forespørgsel:** Agent A højst 2 modelkald, agent B højst 8 modelkald og 3 søgninger,
og 180 sekunder i alt. Ved for mange kald (HTTP 429) venter Cohere-forbindelsen 5, 15 og 40 sekunder
(eller den tid, API'et beder om). OpenAlex-kald prøves igen 2 gange.

## Værktøjet: OpenAlex

`search_papers` (i `scout/openalex.py`) er agentens eneste værktøj:

- Henter op til 25 resultater med `search=` og filtre på `publication_year` og `cited_by_count`.
- **Kontrollerer hvert resultat igen lokalt** mod intervallerne, så et forkert API-svar ikke slipper igennem.
- Beholder kun forskning (`article`, `preprint`, `conference-paper`, `review`, `book-chapter`).
  Boganmeldelser, datasæt og software sorteres fra.
- Fjerner dubletter på OpenAlex-ID **og** titel (samme artikel findes ofte som preprint og konferenceversion).
- Gemmer kandidaterne i én liste pr. forespørgsel. En senere søgning kan aldrig overskrive et tidligere fund.
- Returnerer pr. kandidat: ID, titel, forfattere, år, citationstal, type og abstract (højst 1.000 tegn,
  genskabt fra OpenAlex' `abstract_inverted_index`). Mangler abstractet, får modellen det at vide.
- Et manglende citationstal forbliver ukendt og kan aldrig opfylde et krav om citationer.
- Fejler søgningen, meldes en fejl. Den returnerer aldrig en tom liste.

OpenAlex' data er CC0. Citationstallene kan afvige fra fx Google Scholar.

## Fortolkning af krav

Alle intervaller er inklusive. Fortolkningen af "cirka" og "recent" vises i svaret.

| Krav | Interval |
|---|---|
| før år X | år ≤ X − 1 |
| efter år X | år ≥ X + 1 |
| i år X | år = X |
| fra / siden år X | år ≥ X |
| til og med år X | år ≤ X |
| over N citationer | citationer ≥ N + 1 |
| mindst N | citationer ≥ N |
| under N | citationer ≤ N − 1 |
| højst N | citationer ≤ N |
| cirka N | N ± 20 %, afrundet indad |
| recent / nyere | de seneste 3 kalenderår (fx 2024-2026) |

Flere krav på samme felt lægges sammen til det strammeste interval.

## Valg af model

Opgaven kræver en gratis model, der kan kalde værktøjer. Vi målte kandidaterne med
`scripts/check_model.py`: 10 kørsler pr. model, hvor modellen skal kalde et værktøj og vælge den
eneste kandidat, der opfylder kravene, eller svare `null`, når ingen gør (2 af de 10 kørsler).
Python tjekker svaret. Målt 17/11-2026:

| Model | Resultat |
|---|---|
| Mistral | Ikke længere gratis, ikke målt |
| Groq | Nøglen blev afvist (HTTP 401), ikke målt |
| Cohere Command A+ via OpenAI-kompatibelt endpoint | 0/10: API'et afviser `tool_choice` (HTTP 400) |
| **Cohere Command A+ via PydanticAI's `CohereModel`** | **9/10**, begge afvisninger korrekte, 2 modelkald pr. kørsel, median 2,5 s. Den ene fejl var HTTP 429 |
| Gemini 3.8 Flash | Svarede ikke (timeout på selv et minimalt kald) |
| Gemini 3.5 Flash-Lite | 6/10. Fejlede begge afvisninger (blev ved med at søge), 2 × HTTP 429 |

**Valg: Cohere Command A+** (`command-a-plus-05-2026`, gratis prøvenøgle). Afgørende var, at den kan
afvise kandidater, hvilket opgaven kræver.

Begrænsninger og forbehold:

- Prøvenøglen giver 20 kald i minuttet og **1.000 kald om måneden i alt**
  ([Cohere: rate limits](https://docs.cohere.com/docs/rate-limits)). En forespørgsel bruger typisk 4-5 kald.
- Coheres OpenAI-kompatible API understøtter hverken `tool_choice` eller `parallel_tool_calls`. Derfor
  bruges `PromptedOutput` (JSON i tekstsvaret, som i uge 44), så der aldrig tvinges et værktøjskald.
- Sammenligningen er skæv: Flash-Lite er en meget mindre model end Command A+, og kun 10 kørsler pr. model.

## Evaluering

### Metode

Metoden bygger på tre kilder om evaluering af agenter:

- **Anthropic, "Demystifying evals for AI agents" (2026)**
  ([link](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)): Kombinér
  kodebaserede, modelbaserede og menneskelige bedømmere. Bedøm resultatet frem for vejen dertil.
  Kør flere forsøg, fordi modellens svar varierer. Skriv entydige opgaver med en referenceløsning.
- **Yao m.fl., "τ-bench" (2024)** ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)): Måler
  pålidelighed over gentagne forsøg med pass^k, hvor alle k forsøg skal lykkes.
- **Yehudai m.fl., "Survey on Evaluation of LLM-based Agents" (2025)**
  ([arXiv:2503.16416](https://arxiv.org/abs/2503.16416)): Evaluering af bl.a. planlægning og
  værktøjsbrug, og efterlyser finkornede metoder og vurdering af omkostninger.

Det har vi gjort:

- **14 testprompts** i `eval/cases.json`, skrevet **før** agenten blev kørt. De dækker brede og smalle
  emner, før/efter/præcist år, krav til citationer, kombinerede krav, "recent", forespørgsler uden
  passende artikel, modstridende krav og tvetydige forespørgsler. Hver prompt angiver de forventede
  intervaller, krav til svaret og de udfald, der godkendes.
- **3 kørsler pr. prompt.** Vi rapporterer både enkeltkørsler og pass^3 (alle 3 kørsler består).
- **To led**, så en misforstået bestilling kan skelnes fra en dårlig besvarelse:

| Led | Kriterium | Bedømmes |
|---|---|---|
| A: forstod agenten bestillingen? | Intervaller som forventet | Automatisk |
| | Krav til svaret som forventet | Automatisk |
| | Emnet forstået | Manuelt |
| B: opfyldte svaret den? | Forventet udfald | Automatisk |
| | År og citationer inden for de **forventede** intervaller | Automatisk |
| | Gyldig kilde: ID'et slås op igen i OpenAlex (samme titel og år) | Automatisk |
| | Metadata identisk med de hentede data | Automatisk |
| | Antal sætninger | Automatisk |
| | Relevant artikel eller berettiget afvisning | Manuelt |
| | Nyttig forklaring | Manuelt |
| | Ingen påstande uden dækning i abstractet | Manuelt |

**Bestået-regel:** En kørsel består, når alle kriterier, der gælder for den, består. En korrekt
afvisning tæller som bestået. `search_error` og `agent_error` vises for sig og tæller som ikke bestået.
Hver kørsel gemmes med de rå OpenAlex-svar og abstracts, så bedømmelsen kan efterprøves.

### Kør evalueringen

```bash
uv run python -m eval.check_cases                          # tjek cases.json
uv run python -m eval.run_eval run --out eval/results/final
# udfyld de manuelle felter i eval/results/final/grading.json ("pass", "fail" eller "n/a")
uv run python -m eval.run_eval report eval/results/final   # skriver RESULTS.md igen
```

En afbrudt kørsel fortsætter, hvor den slap, når den får samme `--out`.

## Resultater

### Evalueringssættet: 14 prompts × 3 kørsler

Kørt 17/11-2026 med Cohere Command A+ og OpenAlex. Den fulde tabel med alle automatiske og manuelle
bedømmelser og noter står i [`eval/results/final/RESULTS.md`](eval/results/final/RESULTS.md), og hver
kørsel er gemt som JSON med de rå OpenAlex-svar.

| Mål | Resultat |
|---|---|
| Kørsler bestået | **30 af 42 (71 %)** |
| Søge- eller agentfejl | 0 |
| Forventet udfald | 42 af 42 |
| pass^3 (alle 3 kørsler af en prompt bestået) | 8 af 14 prompts |
| År og citationer inden for de forventede intervaller | 33 af 33 fundne artikler |
| Gyldig kilde (ID'et slået op igen i OpenAlex) | 33 af 33 |
| Metadata identisk med OpenAlex-data | 33 af 33 |
| Krav til antal sætninger overholdt | 6 af 6 |
| Krav korrekt fortolket (led A, intervaller) | 41 af 42 |
| Modelkald | 183 i alt; 4,9 pr. kørsel, der søger, og 1-2, når agenten stopper før søgning |
| Søgninger | 2,7 pr. kørsel, der søger (3 søgninger i 25 af 36) |

Afvisninger, opklarende spørgsmål og modstridende krav (C11, C12, C13) blev håndteret korrekt i alle 9 kørsler.

**De 12 kørsler, der ikke bestod:**

| Årsag | Kørsler |
|---|---|
| Svarteksten indeholdt påstande, der ikke står i artiklens abstract | 8 (C02 × 3, C04, C05, C09 × 2, C14) |
| Tvivlsom relevans: en artikel om keyword-søgning på RDF-data valgt til "query optimization in graph databases" | 3 (C08) |
| Agenten fortolkede "a few years ago" som en årsgrænse (≤ 2022), som forespørgslen ikke nævner | 1 (C14) |

### Opgavens tre eksempler

Kørt under udviklingen (17/11-2026). Alle gav `match` med en relevant artikel, der opfyldte kravene,
fx Lewis m.fl. 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (10.269
citationer, resumé på 7 sætninger) til RAG-eksemplet. Hver brugte 2-3 søgninger og 4-5 modelkald.

## Refleksion

Se [REFLECTION.md](REFLECTION.md).

## Gruppemedlemmernes bidrag

Projektet er lavet af begge gruppemedlemmer i fællesskab. Vi har arbejdet sammen om alle dele:
søgeværktøjet, agenten, evalueringen og dokumentationen. Det gjorde vi, så vi begge fik en bedre forståelse
for faget og for, hvordan man bruger AI-agenter med værktøjer i praksis.

## Projektstruktur og udvikling

```text
scout/
  agent.py        agent A og B, output-validator, udfald
  openalex.py     søgeværktøjet
  constraints.py  krav → intervaller, OpenAlex-filtre, sætningstælling
  models.py       datamodeller (Pydantic)
  config.py       indstillinger fra .env
  llm.py          forbindelse til modellen, vent ved 429
main.py           kommandolinje
eval/             testprompts, bedømmelse og kørsel af evalueringen
scripts/          målinger af model og OpenAlex (trin 0)
tests/            tests (kører uden netværk)
```

```bash
uv run pytest                              # tests, kræver ≥ 95 % dækning af scout/
uv run mypy --strict                       # typetjek
uv run ruff format . && uv run ruff check .
```

Modelmålingerne kan køres igen med `scripts/`. Resultaterne gemmes i `scripts/results/`, som ikke
committes:

```bash
uv run python -m scripts.check_openalex
uv run python -m scripts.check_model --preset cohere
uv run python -m scripts.check_model --preset gemini --model gemini-3.5-flash-lite
uv run python -m scripts.check_model --from-env --runs 1     # agentens egen LLM_*-opsætning
```
