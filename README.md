# Freight Rate Prediction

Predicts the posted rate for a truckload shipment from lane, equipment, weight,
distance, date and a market index. Built for the Spotter ML assessment.

Training data covers **January - October 2025** (48,000 loads). Predictions are
required for **November - December 2025** (12,000 loads), plus a fixed
single-lane scenario for every day of December.

---

## Headline result

Rolling-origin validation (train on everything up to month *m*, predict month
*m+1*), averaged over six folds. Test folds keep their outliers, so these are
not flattered:

| Model | MAE | MAPE | R² |
| --- | --- | --- | --- |
| Global mean $/mi × distance | $263.88 | 11.91% | 0.7891 |
| Lane mean $/mi × distance | $187.36 | 8.24% | 0.8070 |
| Plain LightGBM | $129.78 | — | — |
| Ridge on log $/mi | $105.80 | 4.96% | 0.8213 |
| **Hybrid (linear level + LightGBM residual)** | **$106.18** | **4.60%** | **0.8302** |

Those aggregates are dominated by a thin tail of anomalous loads. On typical
freight the model is far tighter — **median APE 1.40%**, and $50.80 MAE once the
~1% of loads priced far from anything their features imply are set aside. Both
numbers appear below; neither is the whole story on its own.

---

## Quick start

```bash
python -m pip install -r requirements.txt
```

```bash
python -m src.predict
```

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

To reproduce every validation experiment (this is the slow one, ~5 minutes):

```bash
python -m src.evaluate
```

Outputs:

| Path | Contents |
| --- | --- |
| `validation_predictions.csv` | 12,000 rows, `load_id,predicted_rate` — the submission file |
| `data/december_chart_inputs.csv` | fixed December scenario with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | chart produced by the provided `score.py` |
| `reports/validation_results.md` | all five validation experiments |
| `reports/feature_importance.md` | gain breakdown for the residual model |
| `reports/Freight_Rate_Prediction_Report.pdf` | the write-up (also as `.docx`) |

---

## What the data turned out to be

Three findings drove every subsequent decision.

### 1. `quote_signal` is a trap, and it is dropped

The column is a rate-per-mile quote. For ~50% of training rows it equals the
actual rate-per-mile to within ±2%. That makes it look like the single most
valuable feature in the dataset — under a random split.

It is not stable. Regressing it against load characteristics month by month
reveals three distinct regimes:

| Months | corr(`quote_signal`, log distance) | Equipment ordering | Regime |
| --- | --- | --- | --- |
| Jan, Feb, Mar, Jun, Sep | **−0.80** | Reefer > Flatbed > Dry Van ✓ | genuine quote |
| Apr, May, Jul, Oct | **+0.81** (sign inverted) | Dry Van > Flatbed > Reefer ✗ | systematically corrupted |
| **Aug, Nov, Dec** | **≈ 0.00** | all equal at ~2.05, std 0.22 | **pure noise** |

The real relationship is unambiguous — shorter hauls cost more per mile, and
reefer costs more than flatbed, which costs more than dry van. Months where the
sign flips are inverted copies; months where the structure vanishes entirely
carry no information at all.

**November and December — every row requiring a prediction — are in the third
regime.** Within August, the one labelled month that shares it,
corr(`quote_signal`, actual $/mi) = −0.014.

The cost of getting this wrong is measurable (experiment E1):

| Split | with `quote_signal` | without |
| --- | --- | --- |
| Random 80/20 | MAE **$90.94** | $97.96 |
| Forward, → October | MAE **$126.84** | $159.10 |
| Forward, → **August** (matches Nov/Dec regime) | MAE $149.40 | **$111.10** |

A random split says keep it. So does an October holdout — October is one of the
*sign-inverted* months, where a booster can learn the inversion and exploit it.
Only August, the single labelled month in the same dead regime as November and
December, tells the truth: there the feature inflates error by 34%. It is
excluded from the final feature set.

### 2. The rate level is a market effect *plus* a time trend

Aggregating to daily level and controlling for load mix:

```
daily level = 0.143 × market_index + 0.00021 × t
```

R² = 0.826 on daily aggregates. Neither term suffices alone — `market_index`
gives 0.478, time gives 0.261. The trend works out to **+0.64% per month**, and
it is not a repackaging of the market index: September and October have a *low*
market index (0.89, 0.96, comparable to January's 0.93) but sit ~6% higher in
rate.

This is the central modelling problem. Predictions are needed 30-90 days past
the end of training, and **a tree cannot extrapolate a trend** — it flattens at
the last bucket it ever saw. Left to its own devices, LightGBM used
`market_index` to memorise individual day levels, which is why removing that
feature entirely *improved* its forward score by $26 MAE (experiment E4).

### 3. Rates ramp through the month

After controlling for market level, equipment and distance, the day-of-month
residual runs from **−1.1% on the 1st to +1.5% on the 29th** — a real
end-of-month tightening worth ~2.5%. Day-of-week is negligible by comparison
(~0.7% total spread). Position within the month is used; absolute calendar
position is not.

---

## Model

**A linear backbone that extrapolates, plus a gradient-boosted residual for
cross-sectional structure.**

```
log(rate / distance)  =  linear( market_index, t, log_distance, equipment )
                       + LightGBM( lane, geography, weight, calendar position, … )
```

The split is deliberate. The backbone owns the two quantities that must project
past the training window and whose relationship is smooth — the market index and
the drift — so they keep extrapolating linearly instead of flattening.
Everything else is cross-sectional and stationary, so LightGBM fits it on what
the backbone leaves behind. `market_index` and `trend_days` are **withheld** from
the booster to stop it re-memorising day levels.

Fitted backbone coefficients (log $/mi):

| Term | Coefficient | Reading |
| --- | --- | --- |
| `market_index` | +0.14313 | a 0.1 rise in the index adds ~1.4% to rate |
| `trend_days`/100 | +0.02122 | +0.64% per month |
| `log_distance` | −0.12594 | doubling haul length cuts $/mi by ~8.4% |
| Reefer | +0.11901 | +12.6% over dry van |
| Flatbed | +0.07799 | +8.1% over dry van |

(Coefficients are on the log scale; the percentages are `exp(β) − 1`.)

### Why the target is log rate-per-mile

Rate is close to multiplicative in distance (corr = 0.91), so dividing distance
out leaves the model to learn the part that is genuinely uncertain. Logs make
the loss proportional: a $200 miss on a 3,000-mile load is not the same mistake
as a $200 miss on a 300-mile load. Predictions are converted back with Duan's
smearing estimate (1.00106 here) so exponentiating a mean-unbiased log
prediction is not biased low — this reduced residual bias by roughly a third at
no cost to MAE.

### Trend damping

Projecting a +0.64%/month drift into a window with no labels is the riskiest
assumption in the model, so the amount projected is a swept parameter rather
than an assumption (experiment E5):

| Damping | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| --- | --- | --- | --- | --- | --- |
| Mean MAE | **$110.19** | $110.64 | $111.19 | $111.83 | $112.54 |

`0.0` — hold the level at the last observed day — wins, and agrees with the raw
series, where the level is flat across August-October (+4.3%, +6.1%, +6.3%
against January). The spread is narrow and folds disagree in detail, so this is
a mild preference rather than a strong result; it is also the conservative
choice, which matters more at December's two-month horizon than at the
one-month horizon the sweep could actually test.

---

## Validation design

**Rolling-origin, forward in time.** Train on every load up to month *m*, predict
month *m+1*, for *m* = April … September. Six folds, each one a miniature of the
real task.

Two rules keep the folds honest, and both matter more than they sound:

- **The outlier filter is a training decision only.** Each fold's test half
  keeps every extreme load, so the reported error includes the loads the model
  cannot get right. Filtering globally before splitting would have removed ~1%
  of each test month and cut the reported MAE roughly in half.
- **Target encodings are computed inside each fold's training window.** Encoding
  once over the whole period and then slicing would let a January training row
  carry lane statistics drawn from the month it is being asked to predict.

A random split was rejected on evidence, not principle — see finding 1. It
overstates accuracy (MAE $90.94 vs $106.18 for the same booster) and, more
damagingly, selects the wrong feature set.

August is treated as the **dress-rehearsal fold**: it is the only labelled month
whose `quote_signal` regime matches November and December.

| Train through | Test | Rows | Plain LightGBM | Hybrid |
| --- | --- | --- | --- | --- |
| April | May | 4,913 | $96.26 | **$86.16** |
| May | June | 4,783 | $138.88 | **$118.98** |
| June | July | 4,912 | **$102.27** | $120.23 |
| July | August | 4,759 | $111.10 | **$104.66** |
| August | September | 4,670 | $171.08 | **$107.57** |
| September | October | 4,853 | $159.10 | **$99.50** |
| **mean** | | | $129.78 | **$106.18** |

The hybrid wins five of six folds. July is the exception — June was the market
peak and the level fell afterwards, so any model projecting forward from it
overshoots.

### Where the error actually is

On the October fold the model is tight for almost every load — **median APE
1.40%** — but the mean is dragged by a small set of loads priced far from
anything their features imply:

| October fold | Rows | MAE | MAPE | median APE |
| --- | --- | --- | --- | --- |
| All rows | 4,853 | $99.50 | 4.69% | 1.40% |
| Excluding $/mi outliers | 4,793 | $50.80 | 2.24% | 1.40% |

Sixty rows — 1.2% of the fold — account for the entire gap between the two. The
**0.42% of rows with APE above 25% carry 95.8% of the total squared error**,
which is why RMSE sits an order of magnitude above MAE. They are not
identifiable in advance, and no adjustment helps: the conditional mean is
already the right prediction for them.

Accuracy is even across segments — median APE is 1.37% (Dry Van), 1.37%
(Reefer), 1.46% (Flatbed), and 1.31–1.48% across every distance band — so there
is no segment the model quietly fails on. Calibration drifts slightly negative
on the most expensive loads (mean bias −$16 in the top predicted-rate quintile,
about −0.4%).

---

## Data quality

| Check | Count | Action |
| --- | --- | --- |
| Rows read | 48,000 | — |
| Duplicate `load_id` | 0 | — |
| Fully duplicated rows | 0 | — |
| Missing `weight` | 300 | median impute + missing flag |
| Missing `market_index` | 374 | **same-day mean** impute + missing flag |
| Non-positive `posted_rate` | 0 | — |
| $/mi outside [0.75, 6.0] | 513 (1.07%) | dropped from training only |
| Rows used for training | 47,487 | — |

`market_index` is a market-wide daily signal — its between-day standard
deviation (0.165) is 6.6× its mean within-day standard deviation (0.025) — so a
same-day mean is a far better fill than a global constant. `weight` has no such
structure and gets a median.

Outlier bounds come from the empirical distribution: the 0.1% tails sit at 0.44
and 10.1 $/mi against a median of 2.15. They are applied to **training only** —
test folds keep their outliers, so reported metrics are not flattered.

The filter earns its place rather than being assumed. Training with the
outliers left in, and scoring on the same untouched test folds:

| Test month | Filtered training | Unfiltered training |
| --- | --- | --- |
| September | MAE **$107.57** | $115.63 |
| October | MAE **$99.50** | $120.30 |

RMSE is unchanged either way (±$1), so the filter buys typical-load accuracy
without trading away tail performance.

**Eight cities appear only in the validation set** (Allentown, Charlotte,
Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego). The model never keys
on city identity: geography enters as latitude/longitude, midpoint, great-circle
distance, bearing and circuity, and lane/city target encodings are
prior-smoothed so unseen categories fall back to the global prior exactly.
Coordinates were verified internally consistent — each of the 72 cities has
exactly one lat/lon, identical whether it appears as origin or destination.

---

## The fixed December chart

`data/december_chart_inputs.csv` ships with seven columns and **no
`market_index`** — the strongest single driver of rate level.

Every one of its 31 dates is present in `validation.csv`, and `market_index` is
a market-wide daily quantity rather than a per-load one. The daily mean across
the ~200 validation loads on each December date is therefore the correct value
for the fixed scenario, and it uses no information the model would not already
have on those dates. This is done in `data.december_market_index`.

The resulting curve for Lexington → Fort Wayne (360 mi, Dry Van, 32,000 lb)
spans **$823.85 - $869.51** (5.5%), shaped by the daily market index and the
end-of-month ramp described above.

---

## Layout

```
src/config.py      paths, constants, the two validated hyperparameters
src/data.py        loading, cleaning, imputation, quality report
src/features.py    feature construction, smoothed target encoding
src/model.py       target transform, metrics, HybridModel
src/evaluate.py    validation experiments E1-E5
src/predict.py     fits the final model, writes both prediction files
```

Reproducibility: `RANDOM_STATE = 42` throughout; LightGBM is seeded and
single-config (no random search), so `python -m src.predict` reproduces the
submitted files exactly.

### Notes and honest limits

- The +0.64%/month drift is fitted over ten months. Ten months cannot separate
  trend from annual seasonality, so it is damped to zero beyond the training
  window rather than projected. If the true process is seasonal and December is
  a seasonal peak (which freight typically is), the December level is
  understated.
- The trend-damping sweep could only test a one-month horizon; December is two
  months out.
- `market_index` is taken as given for November and December. If it were not
  supplied, it would have to be forecast, and the error bars would be
  materially wider.
