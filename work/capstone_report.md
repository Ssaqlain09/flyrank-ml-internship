# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Saqlain Shahbaz
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Ssaqlain09/flyrank-ml-internship
- **Date:** 2026-09-28

## 0. Abstract

Which existing pages should a content reviewer look at first, given limited weekly review capacity? This
project scores 16,726 visible pages from FlyRank's starter search-performance dataset, first with a
transparent hand-written rule (CTR-vs-position plus staleness), then with a random forest trained on 28
pre-decision features and validated with 20 repeated client-holdout splits. The model beats the baseline's
discrimination on 90% of splits (mean AUC 0.642 vs 0.573) and its top-of-queue precision on 75% of splits
(Precision@50 0.785 vs 0.717), though the gain is real but modest and the top-of-queue metric is noisy
because of how unevenly the 32 clients split across folds. The output is a ranked review queue with reason
codes, confidence labels, and a concrete finding that the deployed queue needs a per-client cap before it
reaches a reviewer, since its unadjusted top 50 rows all come from a single client.

## 1. Problem framing

**Decision:** which pages enter a content team's weekly review queue, and in what order, when the team can
only act on a limited number of pages per cycle. **Unit of analysis:** one content page (`content_id` +
`client_id`). **Output:** a ranked queue with a score, a reason code, an action label, and a confidence
level. **Action a human takes:** `review_title_snippet` (rewrite a weak title/meta for its position), or
`monitor` (no action yet). **Cost of a wrong call:** a false positive wastes a reviewer's limited hours on a
page that did not need it, at the opportunity cost of a page further down the queue that did; a false
negative lets a genuinely declining, high-demand page keep losing visibility unnoticed. Because review
capacity is scarce, this is fundamentally a **ranking problem** — precision at the top of the list matters
far more than accuracy across every row — which is why a fixed rule, however reasonable, can miss patterns
that several weak signals only reveal together (worked through in `w02_ml_task_framing.ipynb`).

## 2. Data safety

**Source:** the starter dataset shipped in this repo, `data/raw/content_refresh_anonymized.csv` (30,000
pseudonymized rows, 32 clients) — the lane's own default dataset per the lane guide. All IDs
(`content_id`, `client_id`) are already scrambled by FlyRank before release and are used only for grouping
and validation splits, never as model features.

**Deliberately excluded:** `trend_direction` and `trend_pct` (they define the proxy label itself — using
them as features would let the model read the answer), and the four 30-day trend columns
(`impressions_last_30d`, `impressions_prev_30d`, `clicks_last_30d`, `clicks_prev_30d`,
`sessions_last_30d`, `sessions_prev_30d`) since they are the mechanical inputs to that same label. A
notebook assertion (`w03`, `w04`, and this capstone notebook, section 8) confirms none of these ever enters
the feature matrix. No client name, domain, URL, or raw query text appears anywhere in this repo or this
paper.

## 3. Baseline

Unchanged from `w04_baseline_score.ipynb`: `score = 100 × (0.8 × ctr_gap + 0.2 × stale_90) × demand_weight`
on visible pages (≥500 impressions in 90 days, a real average position), with reason code
`low_ctr_for_position` firing when a page's CTR is under half its position tier's median CTR. The weights
were set by hand after two signal checks: CTR-vs-position was **CONFIRMED** (decline rate falls step by
step from 0.682 to 0.518 as relative CTR rises, n printed per bucket) and staleness was **MIXED** (a real
but small, partly client-driven effect — +0.096 pooled, +0.053 inside clients). This is a fair baseline
precisely because it is transparent, was checked against real evidence before being written, and is scored
on the exact same client-holdout folds as the model below.

## 4. Model / analysis

**Method:** random forest (200–300 trees, max depth 8, min 20 samples per leaf, class-balanced), compared
against logistic regression and the baseline rule. Random forest fits this lane because the signal checks
already showed CTR-vs-position and staleness interact with content type, position tier, and client in ways
a single linear rule undersells.

**Feature list (28 features, all pre-decision):** the five baseline inputs (`impressions_90d`,
`avg_position`, `ctr`, `position_tier`, `days_since_last_update`) plus `content_age_days`,
`days_with_impressions`, `word_count`, `char_count`, `search_volume`, `competition`, `cpc`,
`engagement_rate`, `scroll_rate`, one-hot `content_type` and `main_intent`, and a `*_missing` indicator for
every feature with real gaps (word/char count, keyword-market fields, scroll rate) so "unknown" stays
distinguishable from "measured zero". **Excluded on purpose:** every banned column above, plus
`impressions_prev_30d`-style engineering of any kind.

**Target/proxy, in one sentence:** `is_declining = 1` when `trend_direction == "down"`, a same-window bucket
FlyRank precomputes — a proxy for a real future-window decline, not an observed later outcome, so every
result below is read as directional, never causal.

## 5. Evaluation

**Split:** `GroupShuffleSplit` grouped by `client_id`, 70/30, so whole clients are held out and never split
across train and test — pages from one client can share a template, CMS, or tracking setup that a plain
random split would let the model partly memorize instead of learning a real page-level pattern. **Repeated
20 times** with different random seeds, because the 32 clients vary a lot in size and decline rate, so a
single split's test-fold base rate can swing widely (seen directly: one seed alone gave a test base rate of
0.801 against a 20-split average of 0.576).

| Metric (mean ± std over 20 client-grouped splits) | Baseline | Logistic regression | Random forest |
|---|---|---|---|
| ROC-AUC | 0.573 ± 0.031 | 0.610 ± 0.028 | **0.642 ± 0.032** |
| Precision@50 | 0.717 ± 0.089 | 0.752 ± 0.101 | **0.785 ± 0.140** |
| Beats baseline AUC | — | 80% of splits | 90% of splits |
| Beats baseline P@50 | — | — | 75% of splits |

**Error analysis, not just a metric table:** on the held-out seed-0 fold, the random forest's top 20 by
score contained 11 truly declining pages out of 20 and only 4 distinct clients (one client held 14 of the
20). This matches the client-concentration pattern already seen in the baseline's own top-10 review in
`w04`, and it recurs at deploy scale (section 9 of the notebook): the final, all-data queue's unadjusted top
50 rows are **all one client**.

## 6. Interpretation

Random forest feature importances and logistic regression coefficients agree on the core story: `ctr` and
`avg_position` matter most and point the same direction the CONFIRMED signal check found — lower relative
CTR and worse position push toward "declining". `content_age_days` and `impressions_90d` are strong too:
older, lower-impression pages are flagged more, agreeing across both models. `days_since_last_update` carries
a small positive weight, matching the MIXED staleness verdict — real, but a minor contributor.

**A genuine negative/nuanced result:** `position_tier_page_3_5` and `position_tier_striking` carry some of
the largest positive coefficients in the linear model — pages already sitting deep on the results page are
simply more likely to keep declining almost regardless of other signals, which reads more like "where the
page already is" than "a fixable content problem", and is worth saying plainly rather than folding into a
single score.

## 7. Recommendation

- **High confidence + `low_ctr_for_position`**: `review_title_snippet` first — both the transparent rule and
  the model agree, and the CTR-vs-position signal check backs it.
- **High model score, but no baseline reason code**: route to `monitor` with a note — the model sees a
  broader pattern (age, low impressions for position, weak keyword-market context) the single-signal rule
  cannot name as specifically.
- **Cap per-client concentration before the queue reaches a reviewer.** The deployed queue's unadjusted top
  50 rows are all one client; without a cap, a reviewer would work through one client fifty times before
  reaching a second. This is a policy choice for whoever owns review capacity, not a modeling fix.
- **Low confidence**: leave as `monitor`; do not spend limited review time here.
- **Confidence, stated plainly:** the AUC gap (0.642 vs 0.573) is a real, modest edge, not a confident
  diagnosis of any single page. Precision@50's wide spread across splits (±0.140) means a single number from
  one run should not be quoted on its own.

## 8. Reproducibility

To re-run from a fresh clone:

```bash
git clone https://github.com/Ssaqlain09/flyrank-ml-internship
cd flyrank-ml-internship
pip install -r requirements.txt scikit-learn
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

All random state is fixed: `GroupShuffleSplit(random_state=seed)` for `seed in range(20)`, and
`RandomForestClassifier(random_state=42)` / class-balanced `LogisticRegression` for every fit. The notebook
regenerates `work/outputs/capstone_ranked_queue.csv` (16,726 rows, gitignored by design — it is a data file)
on every run; the numbers in this report were produced by exactly that run and are checkable by re-running
the notebook, not taken on faith. `work/outputs/baseline_metrics.json` (from `w04`) and this report are the
committed receipts.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset. [flyrank.ai](https://flyrank.ai)

---

> **Claims checklist:** observed / measured / directional / decision-support language throughout · base rate
> reported next to every precision@K (0.576 mean test base rate, 20-split average) · no causal claims · no
> "predicted Google's algorithm" claim · no client-identifying details anywhere · numbers here match the
> committed notebook's actual output, not a rounded or aspirational version of it.
