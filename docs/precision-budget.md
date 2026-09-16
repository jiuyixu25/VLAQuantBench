# Activation-bit budget, libero_spatial, 4-bit weights throughout
# budget = lowest activation width within 5 points of baseline

| model | baseline | component | A4 | A6 | A8 | none | budget |
|---|---|---|---|---|---|---|---|
| openvla_oft | 95.0 | ve | 0.0 | — | 95.0 | 95.5 | **A8** |
| openvla_oft | 95.0 | llm | 0.0 | — | 95.5 | 92.5 | **A8** |
| openvla_oft | 95.0 | ah | 0.0 | 16.5 | 20.5 | 94.0 | **none** |
| pi0 | 61.5 | ve | 61.5 | — | 61.5 | 61.5 | **A4** |
| pi0 | 61.5 | llm | 58.0 | — | 61.5 | 62.0 | **A4** |
| pi0 | 61.5 | ah | 44.0 | 55.0 | 58.0 | 53.0 | **A8** |
| pi05 | 87.5 | ve | 27.0 | — | 88.0 | 86.5 | **A8** |
| pi05 | 87.5 | llm | 0.0 | 79.5 | 84.0 | 86.5 | **A8** |
| pi05 | 87.5 | ah | 49.5 | 77.0 | 87.0 | 87.0 | **A8** |
| xvla | 96.8 | ve | 67.9 | — | 97.4 | 97.4 | **A8** |
| xvla | 96.8 | llm | 96.3 | — | 96.3 | 95.8 | **A4** |
| xvla | 96.8 | ah | 0.0 | 93.0<sub>n=171</sub> | 96.8 | 97.4 | **A6** |

`none` = the component tolerates 4-bit weights but no activation quantization at any width measured. A budget is a floor, not a guarantee: read it with the full row, since the columns are not always monotone.
