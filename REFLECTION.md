# Refleksion

Refleksionen bygger på modelmålingen (10 kørsler pr. model), kørslerne af opgavens tre eksempler og
det, vi lærte undervejs, mens vi byggede og testede agenten. Se [README.md](README.md) for tallene.

## Hvad virkede godt?

- **At dele arbejdet mellem LLM og kode.** Modellen fortolker forespørgslen, vurderer relevans og
  skriver forklaringen. Koden står for alt, der kan kontrolleres: intervaller for år og citationer,
  filtrering, hvilke artikler der er kandidater, og hvad der står i svarets metadata. Derfor kunne
  modellen ikke give et svar med et forkert år eller citationstal. Den slags fejl blev fanget i koden.
- **OpenAlex som datakilde.** Filtrene på år og citationer blev overholdt i alle søgninger, og
  `search=` gav relevante resultater til de tre eksempler. Søgningerne var hurtige (under 1 sekund).
- **Afvisning af resultater.** Cohere Command A+ svarede korrekt `null` i begge målingens opgaver, hvor
  ingen kandidat passede. Det er det, opgaven kræver, og det var her, den anden model fejlede.
- **Tests uden netværk.** Med PydanticAI's `FunctionModel` kunne vi fremprovokere alle syv udfald og
  hvert genforsøg uden at bruge kald fra den gratis kvote. Testene fandt fejl, før de nåede den rigtige
  model.
- **At måle før vi valgte.** Målingen afslørede problemer, som dokumentationen ikke nævnte. Coheres
  OpenAI-kompatible API afviser fx `tool_choice`, og Gemini 3.8 Flash svarede slet ikke.

## Hvad fejlede eller var upålideligt?

- **De gratis modeller.** Mistral var ikke længere gratis, og vores Groq-nøgle blev afvist (HTTP 401).
  Coheres OpenAI-kompatible API afviste alle kald (`tool_choice is not supported`), så vi måtte skifte til
  PydanticAI's indbyggede Cohere-model og bruge `PromptedOutput`. Gemini 3.8 Flash svarede ikke inden
  for tidsgrænsen. Gemini 3.5 Flash-Lite blev ved med at søge i stedet for at afvise og fik 6/10.
- **Grænser for antal kald.** Både Cohere (20 kald i minuttet) og Gemini svarede med HTTP 429, når vi
  kørte mange kald i træk. Cohere-SDK'et prøvede igen med det samme og gav op, så vi skrev selv en
  forbindelse, der venter (5, 15 og 40 sekunder). Prøvenøglen har også et loft på 1.000 kald om måneden.
- **OpenAlex' anonyme søgning** var sat på pause (HTTP 503), så en gratis nøgle var nødvendig.
- **Støj og dubletter i søgeresultaterne.** Søgningen på "retrieval-augmented generation" gav også en
  boganmeldelse og en artikel om satellitmålinger af nedbør, og RAG-artiklen fandtes to gange (som
  preprint og som konferenceartikel). Dubletter og ikke-forskning sorteres nu fra i koden. Relevans er
  stadig modellens vurdering.
- **Kvaliteten ved "recent".** Uden krav til citationer valgte modellen første gang en preprint med én
  forfatter og 0 citationer. Den var relevant, men forskningsmæssigt svag.
- **Modellen fulgte ikke altid instruksen.** Den gentog titel, år og citationstal i svarteksten, selvom
  instruksen sagde, at det ikke var nødvendigt.

## Hvor mange værktøjskald brugte agenten?

- I modelmålingen: 1 værktøjskald og 2 modelkald pr. kørsel (et kald, der vælger værktøjet, og et, der
  svarer). Én kørsel brugte 3 modelkald.
- Med den rigtige agent: 2-3 søgninger og 4-5 modelkald pr. forespørgsel. Det ene modelkald er agent A,
  der læser forespørgslen. Resten er agent B, der søger og svarer. Agenten søgte ofte igen med andre ord,
  før den valgte, og brugte i flere tilfælde alle 3 tilladte søgninger.
- Et værktøjskald er altså normen, ikke undtagelsen. Agenten svarede aldrig uden at søge. Hvis den havde
  prøvet, ville koden have sendt svaret tilbage.

## Hallucinerede LLM'en?

- **Metadata: nej.** Det kan ikke ske i vores design, fordi titel, forfattere, år, citationstal og link
  kopieres fra de hentede OpenAlex-data og aldrig fra modellens tekst. Evalueringen tjekker det
  automatisk, og ID'et slås op igen i OpenAlex.
- **Svarteksten: én gang i mindre omfang.** Modellen skrev, at OpenAlex' citationstal kommer fra "automatic
  citation tracking which aggregates citation data from academic databases and repositories". Det står
  ikke i artiklens data, og det er modellens egen viden. Påstanden er ikke forkert, men den er ikke
  dokumenteret af værktøjet. Tallene, den gentog (år og citationstal), var korrekte.
- Vi så ingen opdigtede artikler eller ID'er. Et ukendt ID ville blive afvist af koden.

## Hvordan forhindrede vi forkerte svar?

- **Krav håndhæves i koden.** Intervallerne beregnes i Python, sendes med i OpenAlex-søgningen og
  kontrolleres igen lokalt for hvert resultat. Modellen kan ikke løsne dem.
- **Modellen vælger kun et ID.** Svaret bygges af de gemte data, og år og citationer tjekkes en sidste
  gang, før svaret returneres.
- **Output-validatoren** sender svaret tilbage, hvis ID'et ikke er en kandidat, hvis modellen ikke har
  søgt, hvis antallet af sætninger er forkert, eller hvis teksten nævner tal, der ikke findes i artiklens
  titel, abstract, år, citationstal eller i forespørgslen.
- **Ærlige udfald.** En fejlet søgning bliver `search_error` og aldrig "ingen artikel fundet". `no_match`
  betyder kun "ikke fundet i disse søgninger". Modstridende krav afvises, før der søges.
- **Grænser.** Højst 3 søgninger, et fast antal modelkald og 180 sekunder pr. forespørgsel.
- **Evaluering mod forventninger skrevet på forhånd**, så vi måler, om agenten forstod bestillingen, og
  ikke kun om den holdt sig til sin egen fortolkning.

## Hvad ville vi forbedre med mere tid?

- **Tjek af påstande uden tal.** Tal-tjekket fanger kun tal. Et ekstra review-trin (Reflection som i uge
  46) kunne markere hver påstand i svaret som dokumenteret eller udokumenteret ud fra abstractet.
- **Kvalitetssignaler.** Vise modellen tidsskrift eller konference og om artiklen er fagfællebedømt, så
  valget ved fx "recent" ikke kun hviler på en blød regel.
- **Flere og større målinger.** Flere kørsler pr. model og en sammenligning med en model af samme
  størrelse som Command A+, fx Groq med en gyldig nøgle.
- **Bedre søgning.** Flere resultater pr. søgning eller semantisk søgning i OpenAlex, så færre relevante
  artikler bliver overset.
- **Opfølgende spørgsmål** ved forespørgsler, der kan tolkes på flere måder. I dag vælger agenten en
  fortolkning og skriver den i `uncertainty`.
- **Mindre afhængighed af én gratis udbyder**, fx automatisk skift til en anden model ved gentagne 429-svar.
