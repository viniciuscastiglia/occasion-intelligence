# References

Literature behind the method choices in this project. Each entry was checked on 2026-11-21 (the data-source entry on 2026-11-22) against the publisher page, the journal's own site or a Crossref record (authors, year, title, venue, volume and pages); DOI links were resolved. A few publisher sites (JAMIA, BMJ) refuse automated requests, so for those the metadata was confirmed through Crossref. "Supports" says which project decision the reference informs; none of these works is evidence about the project's own results.

## Temporal validation and model selection

- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed., chapter 7 ("Model Assessment and Selection"). Springer. https://doi.org/10.1007/978-0-387-84858-7 (free PDF: https://hastie.su.domains/ElemStatLearn/)
  - Supports: keeping the 2017-2018 test edition out of every selection step, so the test measures the chosen model and does not take part in choosing it.
  - Verified: 2026-11-21.
- Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107. https://jmlr.org/papers/v11/cawley10a.html
  - Supports: choosing between the model steps (tables, logistic, LightGBM) on 2008-2009 folds only, with the rule fixed before the test.
  - Verified: 2026-11-21.
- Kapoor, S. and Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
  - Supports: the per-variable leakage check, the declared prediction instant and the time-ordered train/test split.
  - Verified: 2026-11-21.
- Lones, M. A. (2024). Avoiding common machine learning pitfalls. *Patterns*, 5(10), 101046. https://doi.org/10.1016/j.patter.2024.101046 (preprint: https://arxiv.org/abs/2108.02497, titled "How to avoid machine learning pitfalls: a guide for academic researchers")
  - Supports: declaring experiments before running them and recording every comparison made.
  - Verified: 2026-11-21.
- Wieczorek, J., Guerin, C. and McMahon, T. (2022). K-fold cross-validation for complex sample surveys. *Stat*, 11(1), e454. https://doi.org/10.1002/sta4.454
  - Supports: building cross-validation folds that follow the survey design (grouping by primary sampling unit, stratifying by stratum).
  - Verified: 2026-11-21.

## Final model refit and reporting

- Steyerberg, E. W. and Harrell, F. E. (2016). Prediction models need appropriate internal, internal-external, and external validation. *Journal of Clinical Epidemiology*, 69, 245-247. https://doi.org/10.1016/j.jclinepi.2015.04.005
  - Supports: validating the recipe (the procedure), then refitting that same recipe on all available data for the served model.
  - Verified: 2026-11-21.
- Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B. et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385, e078378. https://doi.org/10.1136/bmj-2023-078378
  - Supports: the reporting structure of the README and Data Card (data, predictors, validation, performance). Used as a reference for reporting, not as a checklist.
  - Verified: 2026-11-21.

## Sample size and events per variable

- van der Ploeg, T., Austin, P. C. and Steyerberg, E. W. (2014). Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints. *BMC Medical Research Methodology*, 14, 137. https://doi.org/10.1186/1471-2288-14-137
  - Supports: counting events, not rows, when judging how complex a model the rare pizza-with-guaraná outcome can support.
  - Verified: 2026-11-21.
- Riley, R. D., Snell, K. I. E., Ensor, J., Burke, D. L., Harrell, F. E., Moons, K. G. M. and Collins, G. S. (2019). Minimum sample size for developing a multivariable prediction model: Part II, binary and time-to-event outcomes. *Statistics in Medicine*, 38(7), 1276-1296. https://doi.org/10.1002/sim.7992
  - Supports: the sample-size calculation used to bound the number of model parameters.
  - Verified: 2026-11-21.
- Riley, R. D., Ensor, J., Snell, K. I. E., Harrell, F. E., Martin, G. P., Reitsma, J. B. et al. (2020). Calculating the sample size required for developing a clinical prediction model. *BMJ*, 368, m441. https://doi.org/10.1136/bmj.m441
  - Supports: the same sample-size criteria in their practical form.
  - Verified: 2026-11-21.
- Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y. and Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*, 110, 12-22. https://doi.org/10.1016/j.jclinepi.2019.02.004
  - Supports: treating a regularized logistic model with curves and interactions as a serious candidate against a higher-capacity model.
  - Verified: 2026-11-21.

## Learning curves

- Perlich, C., Provost, F. and Simonoff, J. S. (2003). Tree induction vs. logistic regression: a learning-curve analysis. *Journal of Machine Learning Research*, 4, 211-255. https://www.jmlr.org/papers/v4/perlich03a.html
  - Supports: the expectation that a flexible model can overtake logistic regression as the sample grows, which motivated the exploratory learning curve.
  - Verified: 2026-11-21.
- Figueroa, R. L., Zeng-Treitler, Q., Kandula, S. and Ngo, L. H. (2012). Predicting sample size required for classification performance. *BMC Medical Informatics and Decision Making*, 12, 8. https://doi.org/10.1186/1472-6947-12-8
  - Supports: fitting a learning curve on subsamples to project performance at a larger sample size.
  - Verified: 2026-11-21.
- Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y. and Zhou, Y. (2017). Deep learning scaling is predictable, empirically. arXiv:1712.00409. https://arxiv.org/abs/1712.00409
  - Supports: the general idea of extrapolating learning curves; the project used a straight line in log(n), not a power law (preprint; not peer reviewed).
  - Verified: 2026-11-21.

## Class imbalance and calibration

- van den Goorbergh, R., van Smeden, M., Timmerman, D. and Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*, 29(9), 1525-1534. https://doi.org/10.1093/jamia/ocac093
  - Supports: not resampling or reweighting classes to fix the rarity of the event, and keeping probabilities interpretable.
  - Verified: 2026-11-21.
- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L. and Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230. https://doi.org/10.1186/s12916-019-1466-7
  - Supports: reporting calibration at three levels (mean, slope, curve) alongside discrimination.
  - Verified: 2026-11-21.

## Data source

- Rodrigues, R. M., De Carli, E., Araújo, M. C., Verly Junior, E., Marchioni, D. M. L., Bezerra, I. N., Souza, A. M., Yokoo, E. M., Pereira, R. A. and Sichieri, R. (2021). Limitations in the comparison of the Brazilian National Dietary Surveys of 2008–2009 and 2017–2018. *Revista de Saúde Pública*, 55(Suppl 1). https://doi.org/10.11606/s1518-8787.2021055003365 (full text: https://www.scielo.br/j/rsp/a/d7YbhsbV6sjf9Z4CLwLWp5f/?lang=en)
  - Supports: the Data Card's statement that 2008-2009 used a food record and 2017-2018 a 24-hour recall, and that the food and beverage intake database had 1,503 items in 2008-2009 and 1,832 in 2017-2018 (stated in the article's text and in its comparison box).
  - Verified: 2026-11-22, on the SciELO full text and the journal's article page (authors, year, title, venue, DOI).

## Not included

The decision log also lists a PR-AUC reference and a cluster-bootstrap reference only as design rationale, without a specific citation. They are left out here rather than cited from memory.
