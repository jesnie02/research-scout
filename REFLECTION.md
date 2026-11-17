# Refleksion

Refleksionen bygger på evalueringen (14 prompts × 3 kørsler, se
[`eval/results/final/RESULTS.md`](eval/results/final/RESULTS.md)), modelmålingen og det, vi lærte,
mens vi byggede agenten. Tallene står også i [README.md](README.md#resultater).

## Hvad virkede godt?

- **De dele, koden står for, var fejlfri.** I alle 33 kørsler, der fandt en artikel, lå år og
  citationer inden for de forventede intervaller, ID'et kunne slås op igen i OpenAlex, og metadata var
  identisk med de hentede data. Der var ingen søge- eller agentfejl i de 42 kørsler.
- **Agenten forstod bestillingen.** Intervallerne blev læst korrekt i 41 af 42 kørsler, også for
  "around 300" (240-360), "recent" (2024-2026) og "fewer than 50" (≤ 49). Kravene til svarets længde
  blev overholdt i alle 6 kørsler med et sætningskrav.
- **Agenten kunne sige nej.** Den afviste korrekt alle kandidater til "large language models før 2000"
  i alle 3 kørsler, stillede et opklarende spørgsmål til "Find me a good research paper" og afviste
  modstridende årskrav uden at søge. Det var 9 af 9 kørsler.
- **At dele arbejdet mellem LLM og kode.** Modellen fortolker og vurderer relevans. Koden håndhæver alt,
  der kan kontrolleres. Derfor havnede fejlene kun i den del, modellen selv skriver: svarteksten.
- **At måle før vi valgte.** Modelmålingen afslørede problemer, som dokumentationen ikke nævnte. Coheres
  OpenAI-kompatible API afviser fx `tool_choice`, og Gemini 3.8 Flash svarede slet ikke.
- **Tests uden netværk.** Med PydanticAI's `FunctionModel` testede vi alle syv udfald og hvert genforsøg
  uden at bruge kald fra den gratis kvote.

## Hvad fejlede eller var upålideligt?

- **Påstande i svarteksten uden dækning (8 af 33 svar).** Modellen fyldte huller med egen viden:
  - Den beskrev en artikel om federated learning, som om den træner "without sharing raw patient data".
    Det er lærebogsdefinitionen af federated learning, men det står ikke i abstractet, som tværtimod
    nævner overførsel af MRI-scanninger. Det skete i alle 3 kørsler.
  - Den nævnte GANs, LSTMs og Transformers i en artikel, hvor intet af det står i abstractet.
  - Den kaldte YOLO "state-of-the-art", selvom abstractet skriver, at YOLO laver flere fejl end
    state-of-the-art.
  - I 2 tilfælde beskrev den eksperimentelle resultater, som lå i den del af abstractet, vi havde skåret
    af ved 1.000 tegn. Påstanden er sandsynligvis sand, men modellen kunne ikke vide det fra dataene.
- **Relevans under stramme krav (C08).** Til "query optimization in graph databases with around 300
  citations" valgte agenten i alle 3 kørsler en artikel om keyword-søgning på RDF-data. Forklaringen
  strakte koblingen via en bisætning om "database optimization techniques". Agenten foretrak et svagt
  match frem for at afvise.
- **Vage tidsangivelser.** "A few years ago" blev i 1 af 3 kørsler fortolket som "til og med 2022".
  Forespørgslen nævner ikke et år, og fortolkningen stod ikke i `uncertainty`.
- **Stabilitet.** Kun 8 af 14 prompts bestod i alle 3 kørsler (pass^3). Agenten valgte samme artikel i
  alle kørsler for 9 af 14 prompts. Ved de andre skiftede den mellem flere relevante kandidater.
- **De gratis modeller.** Mistral var ikke længere gratis, og Groq-nøglen blev afvist. Coheres
  OpenAI-kompatible API afviste alle kald, og Gemini 3.8 Flash svarede ikke. Begge udbydere svarede med
  HTTP 429 ved mange kald i træk, så vi skrev en forbindelse, der venter. Prøvenøglen har et loft på
  1.000 kald om måneden, og evalueringen alene brugte 183.

## Hvor mange værktøjskald brugte agenten?

- I de 36 kørsler, der søgte: i gennemsnit 2,7 søgninger og 4,9 modelkald. 25 af dem brugte alle 3
  tilladte søgninger. Agenten søgte typisk først bredt og derefter med mere præcise ord.
- Når agenten kunne stoppe uden at søge (modstridende krav, intet emne), brugte den 1-2 modelkald og 0
  værktøjskald. Det sparer kvote og er det rigtige at gøre.
- Agenten svarede aldrig uden at søge, når den skulle finde en artikel.
- Evalueringen brugte 183 modelkald for 42 kørsler.

## Hallucinerede LLM'en?

- **Metadata: nej.** Titel, forfattere, år, citationstal og link var korrekte i alle 33 svar, fordi de
  kopieres fra OpenAlex og aldrig fra modellens tekst. Ingen opdigtede artikler eller ID'er.
- **Svarteksten: ja, i 8 af 33 svar.** Påstandene var plausible og ofte sande om feltet generelt, men de
  stod ikke i de data, agenten havde hentet. Det er den farligste form for hallucination, fordi den er
  svær at opdage uden at læse abstractet ved siden af.
- Tal-tjekket virkede: ingen af de 33 svar indeholdt tal, der ikke fandtes i dataene. Men det fanger kun
  tal, ikke påstande i ord.

## Hvordan forhindrede vi forkerte svar?

- **Krav håndhæves i koden.** Intervallerne beregnes i Python, sendes med i OpenAlex-søgningen og
  kontrolleres igen lokalt for hvert resultat. Modellen kan ikke løsne dem.
- **Modellen vælger kun et ID.** Svaret bygges af de gemte data, og år og citationer tjekkes en sidste
  gang, før det returneres.
- **Output-validatoren** sender svaret tilbage, hvis ID'et ikke er en kandidat, hvis modellen ikke har
  søgt, hvis antallet af sætninger er forkert, eller hvis teksten nævner tal, som ikke findes i dataene.
- **Ærlige udfald.** En fejlet søgning bliver `search_error`, aldrig "ingen artikel". Modstridende krav
  afvises, før der søges, og et manglende emne giver et spørgsmål i stedet for et gæt.
- **Grænser** på søgninger, modelkald og tid pr. forespørgsel.
- **Evaluering mod forventninger skrevet på forhånd**, både automatisk og manuelt, så fejlene i
  svarteksten blev opdaget og ikke skjult af, at alle de automatiske tjek bestod.

## Hvad ville vi forbedre med mere tid?

- **Et review-trin til svarteksten** (Reflection som i uge 46): Et ekstra modelkald markerer hver påstand
  som dokumenteret eller udokumenteret ud fra abstractet, og udokumenterede påstande fjernes. Det ville
  have ramt 8 af de 12 fejl.
- **Hele abstractet** i stedet for de første 1.000 tegn, så modellen ikke mangler de dele, den alligevel
  beskriver.
- **Strengere relevanskrav:** Bede modellen om at afvise, når den bedste kandidat kun delvist handler om
  emnet, som i C08, og skrive det i `uncertainty`.
- **Vage tidsangivelser:** Lade agent A markere dem i `note`, så fortolkningen altid vises.
- **Flere og større målinger:** Flere prompts og kørsler, og en sammenligning med en model af samme
  størrelse, fx via Groq med en gyldig nøgle.
- **Kvalitetssignaler** som tidsskrift eller konference og fagfællebedømmelse, så valget mellem relevante
  kandidater ikke kun hviler på en blød regel.
