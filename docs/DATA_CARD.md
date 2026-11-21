# Data Card

## Counts

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

## Out-of-fold metrics, 2008-2009

<!-- BEGIN:metricas_2008 -->
| Model | Rows | Positives | Prevalence | PR-AUC | ROC-AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| Occasion (cells) | 669,030 | 654 | 0.001303 | 0.02377 | 0.9177 | 0.007770 | 0.001285 |
| Preference (all soda occasions) | 19,182 | 4,684 | 0.2345 | 0.3189 | 0.6084 | 0.5310 | 0.1746 |
| Preference (pizza only) | 655 | 101 | 0.1392 | 0.1644 | 0.5269 | 0.4134 | 0.1226 |
| Score (product) | 669,030 | 101 | 0.0001815 | 0.002627 | 0.9145 | 0.001462 | 0.0001813 |

Out-of-fold predictions on 2008-2009; prevalence, metrics and the score are weighted by the survey weight, rows and positives are unweighted.
<!-- END:metricas_2008 -->
