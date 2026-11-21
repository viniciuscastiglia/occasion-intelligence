# Occasion Intelligence Pizza + Guaraná

## Try it

## The question and why it is framed this way

## Architecture

## Data and target

<!-- BEGIN:contagens -->
| Count | 2008-2009 | 2017-2018 |
|---|---|---|
| Person-days | 66,903 | 85,018 |
| Cells (person-day x time band x place) | 669,030 | 850,180 |
| Days with pizza and soda in the same hour | 651 | 796 |
| Days with pizza and guaraná soda in the same hour | 101 | 180 |
| Soda occasions | 19,182 | 16,112 |

| Person-days by weekday | 2008-2009 | 2017-2018 |
|---|---|---|
| Monday | 5,013 | 13,179 |
| Tuesday | 14,699 | 14,668 |
| Wednesday | 8,686 | 14,281 |
| Thursday | 17,383 | 14,799 |
| Friday | 10,314 | 7,644 |
| Saturday | 6,430 | 3,862 |
| Sunday | 4,378 | 16,585 |

Counts of the modelling bases (cells and soda occasions) after exclusions, unweighted. In 2017-2018 a few soda occasions are dropped for undetermined place, so the totals differ slightly from the parser reference table. Days with pizza and no soda are not listed: counting them would require reading the 2017-2018 person-day base outside the gated loader.
<!-- END:contagens -->

## Method

## Results

<!-- BEGIN:hipoteses -->
| ID | Part | Quantity | Estimate [95% CI] | Verdict |
|---|---|---|---|---|
| H1 | occasion | log-loss gain of logistic (B) over rate table (A) | 0.0005431 [0.0003421, 0.0007538] | supported |
| H1 | preference | log-loss gain of logistic (B) over rate table (A) | 0.004924 [0.0004795, 0.009413] | supported |
| H2 | occasion | log-loss gain of curves and interactions (C) over logistic (B) | 0.00005515 [0.000008558, 0.0001018] | supported |
| H2 | preference | log-loss gain of curves and interactions (C) over logistic (B) | 0.002859 [0.0004035, 0.005265] | supported |
| H3 | occasion | log-loss gain of LightGBM (D) over best logistic | 0.00002391 [-0.00006402, 0.0001138] | refuted |
| H3 | preference | log-loss gain of LightGBM (D) over best logistic | -0.003015 [-0.006464, 0.0006158] | refuted |
| H4 | preference | pizza coefficient, log-odds | -0.3441 [-0.6212, -0.05415] | supported |
| H5 | occasion | log-loss gain of unweighted over weighted training | -0.00002390 [-0.0001291, 0.00008281] | no evidence against the default |
| H5 | preference | log-loss gain of unweighted over weighted training | 0.002408 [-0.0002882, 0.005259] | no evidence against the default |
| H6 | - | log-loss gain of direct model over product | -0.00006572 [-0.0001361, -0.00001569] | no evidence against the default |
| H7 | - | log-loss gain of corrected over raw product | -0.000005365 [-0.00001630, 0.000003415] | refuted |
| H8 | - | PR-AUC gain of recipe over rate-table product, 2017-2018 | -0.00003392 [-0.0009391, 0.0007684] | refuted |
| H9 | occasion | log-loss gain of recent-only over combined training | 0.00002312 [-0.00005178, 0.00009611] | no evidence against the default |
| H9 | preference | log-loss gain of recent-only over combined training | -0.0002741 [-0.004465, 0.003926] | no evidence against the default |
| H9 | preference (pizza only) | log-loss gain of recent-only over combined training | 0.01037 [-0.02056, 0.04505] | no evidence against the default |
| H10 | - | rank correlation, search interest vs. POF share | - | not run |
<!-- END:hipoteses -->

<!-- BEGIN:teste_final -->
| Target | PR-AUC recipe | PR-AUC reference | Gain [95% CI] | Verdict | Reference |
|---|---|---|---|---|---|
| H8: score (guaraná soda with pizza) | 0.001877 | 0.001911 | -0.00003392 [-0.0009391, 0.0007684] | refuted | product of step-A rate tables |
| occasion | 0.01304 | 0.008841 | 0.004199 [0.002116, 0.008206] | supported | step-A rate table |
| preference (all soda occasions) | 0.3187 | 0.3067 | 0.01208 [-0.004224, 0.02856] | refuted | step-A rate table |
| preference (pizza only) | 0.2398 | 0.2329 | 0.006865 [-0.05160, 0.05794] | refuted | step-A rate table |

A part is 'supported' only if the 95% interval of the PR-AUC gain excludes zero in favor of the recipe; otherwise it is 'refuted'. Evaluated on 2017-2018 cells, weighted.
<!-- END:teste_final -->

<!-- BEGIN:segmentos_teste -->
| Dimension | Level | Cells | Positives | PR-AUC | Observed / expected |
|---|---|---|---|---|---|
| region | Center-West | 103,800 | 25 | 0.002369 | 1.361 |
| region | North | 128,070 | 20 | 0.001519 | 1.672 |
| region | Northeast | 296,900 | 66 | 0.002663 | 1.659 |
| region | South | 110,530 | 27 | 0.003253 | 1.203 |
| region | Southeast | 210,880 | 42 | 0.001737 | 1.958 |
| place | at home | 425,090 | 122 | 0.002428 | 1.948 |
| place | away from home | 425,090 | 58 | 0.0009988 | 1.106 |
| sex | female | 455,890 | 100 | 0.002268 | 1.572 |
| sex | male | 394,290 | 80 | 0.001605 | 1.733 |

Score of the recipe on 2017-2018, weighted; observed / expected is the ratio of mean outcome to mean score.
<!-- END:segmentos_teste -->

<!-- BEGIN:metricas_2008 -->
| Model | Rows | Positives | Prevalence | PR-AUC | ROC-AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| Occasion (cells) | 669,030 | 654 | 0.001303 | 0.02377 | 0.9177 | 0.007770 | 0.001285 |
| Preference (all soda occasions) | 19,182 | 4,684 | 0.2345 | 0.3189 | 0.6084 | 0.5310 | 0.1746 |
| Preference (pizza only) | 655 | 101 | 0.1392 | 0.1644 | 0.5269 | 0.4134 | 0.1226 |
| Score (product) | 669,030 | 101 | 0.0001815 | 0.002627 | 0.9145 | 0.001462 | 0.0001813 |

Out-of-fold predictions on 2008-2009; prevalence, metrics and the score are weighted by the survey weight, rows and positives are unweighted.
<!-- END:metricas_2008 -->

## What the negative result means

## Decision Engine

## Explainability and LLM briefing

## Limitations

## Reproduce everything

## Repository map

## Next steps

## How this was built

## A note on language

## License

