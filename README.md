# EvalGuard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passing-3fb950.svg)](tests/test_checks.py)

Upload a labelled dataset and its train/test split. EvalGuard runs four checks for the
mistakes that make an accuracy number wrong, and explains each one in plain English —
what it found, why it matters, and what to do about it.

Built with Flask, pandas and scikit-learn. Runs on Replit with no configuration.

## Why I built this

I trained an identity-document classifier on MIDV-500 — ResNet50 features, linear probe,
three classes. Split the video frames randomly into train and test, and it scored **97.25%**.

That number was garbage, and it took me a while to see why. MIDV-500 has four nested levels
of near-duplication: class → document type → video clip → frame. Frames 11 and 12 come from
the same three-second video of the same physical card. A random frame split puts one in train
and the other in test, so the model gets to look at almost exactly the same image twice and
I score it on recall dressed up as generalisation.

The part that surprised me was how far I had to go to fix it. Splitting by video clip felt
rigorous — no clip appears on both sides. It scored **97.10%**. Almost no change, because all
ten clips of `19_esp_drvlic` show *the same laminated card*, and the model was memorising that
specific card's texture rather than learning what a driving licence looks like. Only when I held
out entire document *designs* did the number move: **82.57%**.

So a 15-point drop was hiding behind a split that looked careful.

Then I checked whether 82.57% was even stable. Re-running with five different training seeds
moved it by 0.30 points — reassuringly tight, and exactly the kind of number people quote as
evidence of robustness. Re-running with five different *held-out design sets* moved it by
**11.55 points**, ranging from 57.1% to 85.3%. One document type (`09_chn_id`, which the model
gets 0% right) was single-handedly swinging the headline by 16 points depending on whether it
landed in test.

Both mistakes were invisible to every metric I computed after the split. That is what EvalGuard
checks for.

## The four checks

**1. Near-duplicate / leakage detection.** Encodes the feature columns, then for each held-out
row finds its nearest training row by cosine similarity. Anything above the threshold is
reported with the offending pairs. If a `prediction` column is present it also reports what
accuracy looks like with the leaked rows removed.

**2. Class imbalance vs. naive baseline.** Computes the majority-class accuracy — what you would
score by ignoring the input and always guessing the most common class — and compares your model
against it. Also flags a model that has collapsed to predicting one class, and reports macro-F1
and a Wilson confidence interval.

**3. Variance across seeds and splits.** Takes scores from repeated runs and separates two kinds
of noise: re-running with a new training seed (optimisation noise) and re-drawing the split
(data dependence). When the second is much larger than the first, a tight seed spread is being
mistaken for robustness.

**4. Split integrity.** Ids appearing in more than one split, groups straddling the boundary,
classes present in test but absent from train, label-distribution drift, and held-out sets too
small for the result to mean anything.

## Try it

Three datasets are bundled. The first two are the real thing:

| Dataset | What it shows |
|---|---|
| **ID documents — random frame split** | 4,600 real ResNet50 feature vectors, split by shuffling frames. Scores 97.2%. Every check that can fire, fires. |
| **ID documents — document-design split** | The same frames, re-split so entire designs are held out. Leakage gone, accuracy 82.6% — and the variance check shows even that swings by 11 points. |
| **Fraud detection — imbalanced classes** | Synthetic. 94.9% accuracy against a 94.5% baseline. |

## Input format

A CSV where one row is one evaluation sample.

| Column | Required | Meaning |
|---|---|---|
| `split` | **yes** | `train` / `test` (`val` optional). Aliases: `partition`, `fold`, `subset`. |
| `label` | **yes** | Ground-truth class. Aliases: `target`, `y`, `class`, `ground_truth`. |
| `id` | no | Unique sample identifier. Enables duplicate-id detection. |
| `prediction` | no | Model output. Unlocks the accuracy-vs-baseline comparison. |
| group column | no | The real-world entity a row belongs to — document, patient, session. Set it under **Advanced settings**. |
| everything else | — | Treated as features and used for similarity. |

Column names are matched case-insensitively against common aliases, so you usually do not have
to rename anything.

## Running it

On Replit: import the repo and press Run. The `.replit` file already binds Flask to
`0.0.0.0:5000`.

Locally:

```bash
pip install -r requirements.txt
python main.py
```

Then open http://localhost:5000.

```bash
python -m pytest tests/ -q
```

## Where the sample data comes from

The two identity-document CSVs are derived from
[MIDV-500](https://arxiv.org/abs/1807.05786), a public academic dataset of identity documents
belonging to fictional people. No real personal data is involved.

Two things are worth stating plainly, because the numbers only mean something if you know how
they were produced:

- **The predictions are real.** `scripts/make_samples.py` imports the classifier project's own
  `train.py` and trains the same linear probe on the same 2048-dimensional ResNet50 features.
  It reproduces 0.9725 and 0.8257, matching that project's committed reports exactly — the
  script asserts this on every run.
- **The feature columns are reduced.** 2048 dimensions per row will not fit in a CSV that
  belongs in a git repository, so the columns written out are 64 principal components (85.7% of
  variance). The similarity structure the leakage check relies on survives this; frames from the
  same clip stay near-identical. Accuracy is computed before the reduction, not after.

The variance figures in the UI are pasted from that project's `reports/stability.txt`.

## Honest limitations

- **Near-duplicate detection needs a reasonably high-dimensional feature space.** In a handful of
  dimensions, a few thousand rows pack the space densely enough that every held-out row has a
  close neighbour by chance — a cosine of 0.99 stops meaning "same sample" and starts meaning
  "small space". Below 16 encoded dimensions the check reports exact duplicates only and says so
  rather than inventing a contamination rate. The fraud sample deliberately triggers this.
- **The similarity threshold is calibrated, not absolute.** EvalGuard samples random pairs to
  learn what an ordinary similarity looks like in your feature space and raises the threshold if
  0.95 would be meaningless there. It reports both numbers.
- **It checks the split you give it.** It cannot know that `doc_type` is the right grouping level
  rather than `clip` — that judgement is yours, and getting it wrong is exactly the mistake that
  cost me 15 points. What EvalGuard can do is show you the consequences of the choice you made.
- **Check 3 needs you to have done the runs.** It cannot estimate variance from a single result.
- **Datasets are capped** at 100,000 rows, and each split is subsampled to 8,000 rows for the
  nearest-neighbour search. When that happens the report says so and the counts are a lower bound.

## Project layout

```
main.py                     Flask app; upload, sample selection, /api/analyze
evalguard/
  loader.py                 CSV ingest, column detection, split normalisation
  features.py               numeric / categorical / text encoding into one matrix
  report.py                 runs all checks, assembles the verdict
  checks/
    base.py                 CheckResult - the shape every check returns
    leakage.py              check 1
    baseline.py             check 2
    variance.py             check 3
    integrity.py            check 4
templates/index.html        single page
static/style.css, app.js    UI; results rendered client-side from JSON
samples/                    bundled datasets + index.json
scripts/make_samples.py     regenerates them from the classifier project
tests/test_checks.py        17 tests, one fault per test
```

## License

MIT
