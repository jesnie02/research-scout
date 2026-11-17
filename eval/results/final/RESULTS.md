# Evaluation results

- Runs: 42. Pass: 30, fail: 12, error (search/agent): 0, pending: 0.
- Pass rate (errors count as not passed): 71%.
- Outcomes: {'match': 33, 'rejected_on_relevance': 3, 'conflicting_constraints': 3, 'needs_clarification': 3}.
- Model requests: 183 in total, 4.4 per run; tool calls 2.4 per run.
- pass^3 (every run of a case passed): 8 of 14 cases.

Per case: all runs passed / same paper in every run:

- C01: yes / yes
- C02: no / yes
- C03: yes / no
- C04: no / no
- C05: no / yes
- C06: yes / no
- C07: yes / yes
- C08: no / yes
- C09: no / no
- C10: yes / yes
- C11: yes / yes
- C12: yes / yes
- C13: yes / yes
- C14: no / no

✓ pass, ✗ fail, – not applicable, ? not graded yet. A = understood the request, B = fulfilled it. Year/citations are checked against the expected intervals.

| Case | Run | Category | Outcome | Paper | A: intervals | A: requirements | A: topic | B: outcome | B: year | B: citations | B: source | B: metadata | B: sentences | B: relevance | B: explanation | B: grounded | Verdict | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | 1 | broad_topic | match | Progress in Machine Translation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C01 | 2 | broad_topic | match | Progress in Machine Translation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C01 | 3 | broad_topic | match | Progress in Machine Translation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C02 | 1 | narrow_topic | match | Federated Learning-Based ResNet-18 Model for Brain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✗ | fail | Påstår træning 'without sharing raw patient data'; står ikke i abstract, som tværtimod nævner overførsel af MRI-scanninger til medicinske centre. Relevant emne, men svag artikel (3 citationer). |
| C02 | 2 | narrow_topic | match | Federated Learning-Based ResNet-18 Model for Brain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✗ | fail | Påstår træning 'without sharing raw patient data'; står ikke i abstract, som tværtimod nævner overførsel af MRI-scanninger til medicinske centre. Relevant emne, men svag artikel (3 citationer). |
| C02 | 3 | narrow_topic | match | Federated Learning-Based ResNet-18 Model for Brain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✗ | fail | Påstår træning 'without sharing raw patient data'; står ikke i abstract, som tværtimod nævner overførsel af MRI-scanninger til medicinske centre. Relevant emne, men svag artikel (3 citationer). |
| C03 | 1 | year_before | match | Image Classification Using Convolutional Neural Ne | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C03 | 2 | year_before | match | Image Classification Using Convolutional Neural Ne | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C03 | 3 | year_before | match | Flexible, high performance convolutional neural ne | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C04 | 1 | year_after | match | LayoutDiffusion: Controllable Diffusion Model for  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | pass | 2 sentences |
| C04 | 2 | year_after | match | LayoutDiffusion: Controllable Diffusion Model for  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | pass | 2 sentences |
| C04 | 3 | year_after | match | Comprehensive exploration of diffusion models in i | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | fail | 3 sentences; Påstår at surveyen dækker metoder 'published after 2021' og er 'authoritative'; står ikke i abstract. |
| C05 | 1 | year_exact | match | You Only Look Once: Unified, Real-Time Object Dete | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C05 | 2 | year_exact | match | You Only Look Once: Unified, Real-Time Object Dete | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C05 | 3 | year_exact | match | You Only Look Once: Unified, Real-Time Object Dete | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✗ | fail | Påstår 'state-of-the-art performance'; abstract siger, at YOLO laver flere lokaliseringsfejl end state-of-the-art. |
| C06 | 1 | citations_min | match | Safe Learning in Robotics: From Learning-Based Con | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C06 | 2 | citations_min | match | Safe Learning in Robotics: From Learning-Based Con | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C06 | 3 | citations_min | match | Learning agile and dynamic motor skills for legged | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C07 | 1 | citations_max | match | Crop Yield Prediction Using Machine Learning Algor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C07 | 2 | citations_max | match | Crop Yield Prediction Using Machine Learning Algor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C07 | 3 | citations_max | match | Crop Yield Prediction Using Machine Learning Algor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C08 | 1 | citations_about | match | Top-k Exploration of Query Candidates for Efficien | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✗ | ✗ | ✓ | fail | Keyword-søgning på RDF-grafdata, ikke query optimization i grafdatabaser. Forklaringen strækker koblingen via 'database optimization techniques' og kalder den et direkte match. |
| C08 | 2 | citations_about | match | Top-k Exploration of Query Candidates for Efficien | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✗ | ✗ | ✓ | fail | Keyword-søgning på RDF-grafdata, ikke query optimization i grafdatabaser. Forklaringen strækker koblingen via 'database optimization techniques' og kalder den et direkte match. |
| C08 | 3 | citations_about | match | Top-k Exploration of Query Candidates for Efficien | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✗ | ✗ | ✓ | fail | Keyword-søgning på RDF-grafdata, ikke query optimization i grafdatabaser. Forklaringen strækker koblingen via 'database optimization techniques' og kalder den et direkte match. |
| C09 | 1 | combined | match | ETSformer: Exponential Smoothing Transformers for  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | fail | 5 sentences; Påstår eksperimentelle resultater (bedre langtidsforecasting); eksperimenterne står ikke i den del af abstractet, modellen fik (afskåret ved 1.000 tegn). |
| C09 | 2 | combined | match | ETSformer: Exponential Smoothing Transformers for  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | fail | 5 sentences; Påstår at ETSformer slår baselines i eksperimenter; står ikke i den del af abstractet, modellen fik (afskåret ved 1.000 tegn). |
| C09 | 3 | combined | match | Are Transformers Effective for Time Series Forecas | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | pass | 4 sentences |
| C10 | 1 | recent | match | Less Is More: Engineering Challenges of On-Device  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C10 | 2 | recent | match | Less Is More: Engineering Challenges of On-Device  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C10 | 3 | recent | match | Less Is More: Engineering Challenges of On-Device  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C11 | 1 | no_result | rejected_on_relevance |  | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – | ✓ | – | – | pass | Berettiget afvisning: ingen kandidat handler om store sprogmodeller. |
| C11 | 2 | no_result | rejected_on_relevance |  | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – | ✓ | – | – | pass | Berettiget afvisning: ingen kandidat handler om store sprogmodeller. |
| C11 | 3 | no_result | rejected_on_relevance |  | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – | ✓ | – | – | pass | Berettiget afvisning: ingen kandidat handler om store sprogmodeller. |
| C12 | 1 | conflicting | conflicting_constraints |  | – | – | ✓ | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C12 | 2 | conflicting | conflicting_constraints |  | – | – | ✓ | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C12 | 3 | conflicting | conflicting_constraints |  | – | – | ✓ | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C13 | 1 | ambiguous | needs_clarification |  | – | – | – | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C13 | 2 | ambiguous | needs_clarification |  | – | – | – | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C13 | 3 | ambiguous | needs_clarification |  | – | – | – | ✓ | – | – | – | – | – | – | – | – | pass |  |
| C14 | 1 | ambiguous | match | A systematic review of artificial intelligence-bas | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✗ | fail | Nævner GANs, LSTMs, Transformers og anvendelser i gaming og sundhed; intet af det står i abstract. |
| C14 | 2 | ambiguous | match | A Survey of AI Music Generation Tools and Models | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | pass |  |
| C14 | 3 | ambiguous | match | An Empirical Study on How People Perceive AI-gener | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | fail |  |
