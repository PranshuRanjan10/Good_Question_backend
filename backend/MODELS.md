# IronSense models: what we train, why, and how to run it

Read this before running any training command. It covers every model in the backend, which
algorithm it uses, why that algorithm, and what a good result looks like.

## How many models are we actually training?

**Three trained models across the whole backend, and every one of them is supervised, trained
on labelled data.** No clustering, no Isolation Forest, no unsupervised method anywhere.
Everything else is rules or statistics, which is deliberate: a rule can be explained to an
operator and defended to a judge, so we only reach for a model where a rule genuinely can't
do the job.

| # | Model | Owner | Algorithm | Trained? |
|---|---|---|---|---|
| 1 | Task time estimator | **Pranshu** | Ridge regression + LightGBM quantile regression | Yes |
| 2 | Readiness Score weights | **Pranshu** | Logistic regression (5 inputs) | Yes, weights only |
| 3 | Anomaly detector | **Teammate** | LightGBM classifier + rule engine | Yes |
| — | Operator skill profile | **Pranshu** | Rolling percentile statistics | No |
| — | Operator baselines | **Pranshu** | Medians per operator | No |
| — | Safety rules engine | **Teammate** | Thresholds from the spec | No |
| — | Training recommender | **Teammate** | Trigger mapping | No |

So: **I train 2, my teammate trains 1.** I also build 2 statistical artifacts that her
anomaly explanations depend on.

---

## 1. Task time estimator (mine)

**Purpose.** Answer "how long will this actually take?" better than the planner's fixed
estimate, and say *why* in words an operator trusts: "45 min planned → 52 min: rain +7%,
beginner +30%".

**What it predicts.** Not minutes. It predicts `log(actual ÷ estimated)`, a **correction
multiplier on the existing plan**. This is the single most important design decision:

- It works with tiny data. The organizers only gave us 5 task rows.
- It generalises across task types: a 40% overrun means the same thing for a 30-minute load
  and a 90-minute demolition.
- It stays explainable, because each factor is a percentage you can read off.

**Algorithms, and why both.**

| Part | Algorithm | Why |
|---|---|---|
| Explanations | **Ridge regression** on one-hot weather / skill / ground / light / task type | Linear coefficients convert straight into "rain +7%". A tree model can't give you that sentence. Ridge over plain least squares because the one-hot columns are correlated. |
| The numbers | **LightGBM, quantile objective**, three models at α = 0.10 / 0.50 / 0.90 | Gradient boosting handles interactions (a beginner *in the rain* is worse than either alone). Quantile regression gives a **range**, not a fake-precise single number, which is what the cab should show. |
| Band calibration | Empirical offset fitted on a held-out month | Quantile models are over-confident on their own training data. Without this the p10–p90 band covered ~71% of tasks instead of the ~80% it claims. |

**Honesty about the data.** 5 real rows can't separate skill from weather, so the model is
trained on synthetic history calibrated to match those 5 rows, with the real rows weighted
20×. Say this to the panel before they ask; it reads as rigour, not as a gap.

**Run it:**

```bash
.venv/bin/python backend/training/train_time.py
```

**What good looks like:** MAE around 6 min against the planner's ~14 min, p10–p90 coverage
near 80%, and the recovered factors close to reality (Storm ≈ +28%, Beginner ≈ +30%,
Expert ≈ −18%). The biggest gain is on beginners, where the planner is worst.

---

## 2. Readiness Score (mine)

**Purpose.** One 0–100 number in the cab, with five parts behind it: seatbelt, proximity,
behaviour, fatigue, conditions.

**The design choice.** The five sub-scores are **hand-written rules**, not a model, so we can
always answer "why did my score drop?". Only the **five weights** are learned. A neural
network here would be worse *and* unexplainable.

**Algorithm.** Logistic regression with balanced class weights, on the five sub-score
deficits, predicting whether an incident follows. Two horizons are fitted: an incident *in
this window* (transient risk, e.g. someone in the danger zone) and one *within 30 minutes*
(persistent risk, e.g. fatigue and weather).

**A judgment call worth defending.** The raw fitted weights nearly zero out seatbelt
compliance, because being unbuckled rarely causes an incident in the next half hour. Shipping
that would mean a Readiness Score that ignores seatbelts, which is both bad product sense and
an easy thing for a panel to attack. So final weights are **40% hand priors, 30% concurrent
fit, 30% forward fit**. The script prints both the raw coefficients and the shipped weights,
so the choice is visible rather than hidden.

**Run it** (after the datasets exist; needs no other model):

```bash
.venv/bin/python backend/training/train_readiness.py
```

**What good looks like:** AUC around 0.65 for concurrent risk and 0.60 for the 30-minute
horizon, and the worst 5% of windows carrying roughly 2.5× the incident rate of the rest.
These are modest numbers, and that is the truthful result: incidents are rare and partly
random, so a score that claimed AUC 0.95 would mean we'd leaked the answer into the inputs.

---

## 3. Operator skill profile + baselines (mine, no training)

**Purpose.** This is the closed loop that turns five separate features into one product:
telemetry measures how someone operates → the profile drives personalised training → better
behaviour raises the profile → the time estimator gets more accurate for that operator.

**Method.** Four components, each a percentile rank against the whole fleet: efficiency (task
overrun), compliance (seatbelt and idle), safety (incidents and alerts), smoothness (harsh
braking, fast swings, over-revving). Weighted into a 0–100 score per week, with recent weeks
counting more, plus a trend.

**No model, on purpose.** It's rolling statistics, so it updates the moment behaviour changes
and every number can be traced to its source rows.

**It also writes `operator_baselines.json`**, which my teammate's anomaly explanations read to
say "idle 55 min vs your usual 22 min".

**Run it:**

```bash
.venv/bin/python backend/training/build_profiles.py
```

**What good looks like:** the score ordering matches the generator's hidden operator traits
(which are never an input), and the two operators the generator made improve show up as
`trend: improving`. That's our validation that the profile measures something real.

---

## 4. Anomaly detector (teammate's)

**Purpose.** Spot excessive idling and unsafe patterns, and always say why.

**Algorithms.** A **rule engine** for the known patterns with thresholds from the spec, plus a
**supervised LightGBM classifier** over 5-minute window features to catch combinations the
rules miss. Binary on `is_anomaly` first, then multiclass on `anomaly_label` so the model
**names** the anomaly rather than only flagging that something is odd.

**Why supervised, not Isolation Forest.** The project rule is labelled data and supervised
learning only. That works here because the dataset carries `is_anomaly` and `anomaly_label`
for all 18 anomaly types, and it is arguably the better choice anyway: an unsupervised
detector can only say "this window is unusual", while a classifier says "this is excessive
idling" and gives a clean per-type precision/recall table.

**The one question the panel may ask:** on a real site nobody hands you labelled anomalies.
Our answer: ours come from the simulator, and in production they would come from confirmed
incident reports and supervisor review of flagged windows, which is how fleet systems build
labels in practice.

**Class balance matters.** Only ~4.9% of windows are anomalous, so `class_weight="balanced"`
(or `scale_pos_weight`), and judge on precision and recall, never accuracy: a model that
predicted "normal" every time would be 95% accurate and useless.

Her command, once she's written the script:

```bash
.venv/bin/python backend/training/train_anomaly.py
```

**What good looks like:** precision and recall **per label**, reported separately for rules
only, model only, and both combined. Per-label matters because `fatigue` is by far the
largest class and would otherwise flatter the overall number.

## Order to run things

1. `data/generate_data.py` — already done; only rerun if a column needs to change.
2. `backend/training/build_profiles.py` — baselines feed the anomaly explanations.
3. `backend/training/train_time.py`
4. `backend/training/train_readiness.py`
5. `backend/training/train_anomaly.py` (teammate)

All of it is tabular and CPU-only: minutes on an M4 Air, **no cloud GPU needed**. A GPU would
only matter if we added camera vision, and proximity is simulated, so we don't.

Everything lands in `backend/artifacts/` as `.joblib` / `.json`, which the FastAPI app loads
at startup.
