# Model Drift Detection + Auto-Retrain Service — FastAPI + CI/CD to EC2

A complete, beginner-friendly project with a **full closed loop**:

1. Serves a **linear regression** model through a **FastAPI** backend.
2. **Detects data drift** using two lightweight methods — the **KS test** and
   **PSI** — with only `scipy` + `numpy`.
3. When drift is found, it **retrains** a new model, **validates** that the new
   model is actually better, and only then **redeploys** it to **AWS EC2** —
   all automatically through **GitHub Actions CI/CD**.

Everything uses **only free tools**.

```
   new data ──▶ /drift detects drift ──▶ retrain a new model
                                              │
                                              ▼
                                     is the new model better?
                                       │              │
                                     yes             no
                                       │              │
                                       ▼              ▼
                            commit + deploy to EC2   keep the old model
```

---

## Part 1 — What is drift detection, and how do we do it here?

A model is trained on data with a certain "shape" (distribution). Over time the
*new* data coming in can change shape — people buy bigger houses, prices shift, a
sensor gets recalibrated. That change is **data drift**, and it quietly makes
your model less accurate.

**Drift detection = comparing the new data against the original training data to
see if its shape has changed.** That is why, when we train the model, we also
save a snapshot of the training data (the "reference"). Every drift check
compares fresh data against that reference. We use two methods and flag a feature
if *either* fires.

### Method 1 — The KS test (Kolmogorov–Smirnov)

It answers: *"Do these two samples come from the same distribution?"* and outputs
a **p-value**. A **small** p-value means "very unlikely they're the same" →
drift. A **large** p-value means "they look the same" → no drift.

One real lesson baked into this project: the KS test is **overly sensitive on
large samples** — it shouts "drift!" at tiny meaningless wiggles. So we only
trust it when it's *extremely* confident (p < 0.001). The comments in
`app/drift.py` explain this.

### Method 2 — PSI (Population Stability Index)

PSI is a single number for how far the new data's distribution moved from the
reference:

| PSI value        | Meaning                          |
|------------------|----------------------------------|
| less than 0.10   | no real change                   |
| 0.10 to 0.25     | moderate change — keep watching  |
| 0.25 or more     | significant change — **drift!**  |

How it works: slice the reference into 10 buckets, see what fraction of reference
values land in each, then check what fraction of the *new* values land in those
same buckets. Big differences → big PSI. Unlike the KS test, PSI is based on the
*size* of the change, not sample count, so it's more stable — that's why it
carries the main signal here.

---

## Part 2 — The full project layout

```
drift-detection-fastapi/
├── train.py                      # Trains the FIRST model + saves the reference
├── retrain.py                    # Retrains + validates a NEW model (champion vs challenger)
├── monitor.py                    # Checks drift, and triggers the retrain pipeline
├── app/
│   ├── main.py                   # FastAPI app + endpoints
│   ├── drift.py                  # Drift logic: KS test + PSI  (the core)
│   ├── schemas.py                # Request/response shapes
│   └── __init__.py
├── tests/
│   └── test_app.py               # Tests the CI stage runs before any deploy
├── reference/                    # The "model registry" kept in git
│   ├── model.pkl                 #   the live model (the "champion")
│   ├── reference_data.csv        #   the baseline data drift is measured against
│   └── archive/                  #   old models, kept so you can roll back
├── requirements.txt
├── gunicorn_config.py            # Production server (ASGI / uvicorn worker)
├── myapp.service                 # systemd unit -> keeps the app alive on EC2
├── .github/workflows/
│   ├── ci-cd.yml                 # Pipeline 1: deploy on CODE push
│   ├── retrain.yml               # Pipeline 2: on DRIFT, retrain + redeploy   <-- NEW
│   └── deploy-to-ec2.yml         # Reusable deploy steps, called by both
└── .gitignore
```

---

## Part 3 — The API endpoints

Interactive docs are auto-generated at `/docs`.

| Method & path | What it does |
|---------------|--------------|
| `GET /`        | Welcome info and endpoint list |
| `GET /health`  | Health check — returns 200 and whether the model is loaded |
| `POST /predict`| Predict one house price |
| `POST /drift`  | Check a **batch** of recent data for drift; reports + logs it |

`POST /predict` body: `{ "size_sqft": 1800, "num_rooms": 4, "age_years": 10 }` →
`{ "predicted_price": 389441.16 }`.

`POST /drift` takes three equal-length lists (a batch). Drift is about
*distributions*, so use a real batch (50+ rows), not a handful. It returns
`drift_detected`, which features drifted, and the KS/PSI numbers per feature.

---

## Part 4 — THE NEW FLOW: from "drift detected" to "new model deployed"

This is the part you asked for. Before, the service only detected and logged
drift; the model was never actually replaced. Now there is a complete loop. It
has four moving pieces.

### Piece 1 — Detection (`app/drift.py`, `/drift`)

The API detects drift on incoming data, as described above. Detection on its own
changes nothing — it just produces a result that says "drift: yes/no".

### Piece 2 — The trigger (`monitor.py` → GitHub `repository_dispatch`)

Something has to turn "drift: yes" into "start the pipeline". That's `monitor.py`.
It calls `/drift`, and if drift is found, it sends GitHub a `repository_dispatch`
event of type `drift-detected`, which starts the retrain pipeline.

Why a separate script and not the API itself? Because triggering GitHub needs a
secret token, and we never want secret tokens living inside the public-facing web
app. The monitor runs wherever your monitoring lives (a cron job, a small box,
your laptop). It needs two environment variables (never committed):

```bash
export GITHUB_TOKEN=ghp_your_personal_access_token   # needs repo/workflow scope
export GITHUB_REPO=your-username/drift-detection-fastapi
python monitor.py --url http://YOUR_EC2_IP:5000
```

There are actually **three ways** the retrain pipeline can start, all free:

1. **Automatically** on detected drift — `monitor.py` fires `repository_dispatch`.
2. **On a schedule** — the `cron` line in `retrain.yml` (e.g. weekly).
3. **Manually** — the "Run workflow" button on the GitHub Actions tab.

### Piece 3 — Retrain + validate (`retrain.py`)

This is the safety gate, and the most important new idea. We never blindly deploy
a freshly trained model — a new model can easily be *worse*. So we use the
**champion vs challenger** pattern:

- **Champion** = the model currently live (`reference/model.pkl`).
- **Challenger** = a brand-new model trained on fresh, recent data.

`retrain.py` trains the challenger on fresh data, then tests **both** models on
the same fresh hold-out data using **RMSE** (average dollar error — lower is
better). It promotes the challenger **only if it beats the champion**:

- If the challenger wins: the old champion is **archived** (for rollback), the
  challenger becomes the new `model.pkl`, and the reference baseline is **moved
  forward** to the fresh data (more on that below). It signals `promoted=true`.
- If the challenger loses or ties: nothing changes, `promoted=false`, and no
  deployment happens.

You can see this work locally:

```bash
python train.py     # train the original "champion"
python retrain.py    # train a challenger on a CHANGED world and compare
```

In the demo, `train.py` builds the original world and `retrain.py`'s
`make_fresh_data()` simulates a changed market (bigger houses *and* a higher price
per sqft — the relationship itself changed). The stale champion scores a huge
error on the new data while the challenger scores a tiny one, so it's promoted.
Sample output:

```
Champion RMSE:   219323.05
Challenger RMSE: 11677.17
RESULT: PROMOTED. New model + updated reference saved.
```

> Why move the reference forward after a real retrain? Because once you've
> accepted that the world genuinely changed and retrained for it, the new data IS
> the new "normal". If you kept the old reference, you'd detect drift forever.
> So a promoted retrain updates both the model and the baseline together.

### Piece 4 — Commit + redeploy (`retrain.yml` → `deploy-to-ec2.yml`)

If (and only if) the challenger was promoted, the retrain pipeline:

1. **Commits** the new `model.pkl`, updated `reference_data.csv`, and the archived
   old model back to the repo. The repo acts as a tiny, free "model registry" and
   gives you full version history.
2. **Deploys** by calling the reusable `deploy-to-ec2.yml`, which SSHes into EC2,
   does a `git pull` (which brings the new model down), and restarts the app.

The result: a better model is live on EC2, with zero manual steps, and you can
trace exactly which model is running and roll back if needed.

---

## Part 5 — How the three workflow files fit together

There are three workflow files, each with one clear job:

- **`ci-cd.yml`** — runs on every **code push**. CI tests the code; if it passes,
  it deploys. This is for shipping code changes (features, bug fixes).
- **`retrain.yml`** — runs on **drift / schedule / manual**. It retrains,
  validates, and (only if better) commits + deploys the new model. This is for
  shipping *model* changes.
- **`deploy-to-ec2.yml`** — a **reusable workflow**. It holds the actual EC2
  deploy steps once, and both pipelines above "call" it with
  `uses: ./.github/workflows/deploy-to-ec2.yml`. This avoids copy-pasting the SSH
  deploy logic in two places.

One subtle but important detail: a workflow that pushes a commit using GitHub's
built-in token does **not** automatically trigger other workflows (GitHub blocks
this to prevent infinite loops). That's exactly why `retrain.yml` deploys the new
model *itself* (by calling the reusable workflow) instead of relying on its own
commit to wake up `ci-cd.yml`. The `[skip ci]` in its commit message makes this
explicit.

---

## Part 6 — Run it all locally first

```bash
pip install -r requirements.txt
python train.py                                  # creates the reference/ folder
uvicorn app.main:app --reload --port 5000
```

Open `http://localhost:5000/docs`. Try `/predict`, then `/drift` with normal vs
huge houses. Then in another terminal, see the retrain gate:

```bash
python retrain.py
pytest -v        # 11 tests, including the promotion-logic tests
```

---

## Part 7 — Deploy to EC2

You need a free **GitHub account** and **AWS account**.

### Step 1: Launch a free EC2 server
1. AWS Console → **EC2** → **Launch instance**.
2. **AMI**: Ubuntu (free-tier). **Type**: t2.micro / t3.micro.
3. **Key pair**: create one (RSA, `.pem`), download it.
4. **Security group**: allow **SSH (port 22)** and **Custom TCP port 5000**
   (source `0.0.0.0/0`).
5. Launch; copy the **Public IPv4 address**.

### Step 2: Prepare the server
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### Step 3: Install the systemd service
```bash
sudo nano /etc/systemd/system/driftapp.service     # paste myapp.service contents
sudo systemctl daemon-reload
sudo systemctl enable driftapp
exit
```
(It fails to start until the code arrives — that's fine.)

### Step 4: Push the code to GitHub
Create an empty repo `drift-detection-fastapi`, then locally:
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/drift-detection-fastapi.git
git push -u origin main
```
Replace `YOUR_USERNAME` in `deploy-to-ec2.yml` (the `git clone` line) with your
real username.

### Step 5: Add GitHub Secrets
Repo → **Settings** → **Secrets and variables** → **Actions**. Add:

| Secret name   | Value |
|---------------|-------|
| `EC2_HOST`    | Your EC2 public IP |
| `EC2_USER`    | `ubuntu` |
| `EC2_SSH_KEY` | The **entire** contents of your `.pem` file |

### Step 6: Watch the two pipelines
- Push code → **Actions** tab shows `CI-CD` run: test, then deploy.
- Trigger a retrain three ways: click **Run workflow** on the `Retrain` workflow,
  wait for the weekly schedule, or run `monitor.py` to fire it on detected drift.
  When promoted, you'll see it commit the new model and deploy it.

Then open `http://YOUR_EC2_PUBLIC_IP:5000/docs` to use the live service.

---

## Part 8 — Rolling back a bad model

Every promotion archives the previous model to `reference/archive/`. To roll back,
restore an archived model over the live one and redeploy. On the server:

```bash
cd /home/ubuntu/drift-detection-fastapi
cp reference/archive/model_<timestamp>.pkl reference/model.pkl
sudo systemctl restart driftapp
```

(Or, better, restore it in git and push so the repo stays the source of truth.)

---

## Part 9 — Troubleshooting

| Problem | Fix |
|---------|-----|
| `ci-cd` fails at "Run tests" | A test failed — read the log, fix, push. Working as intended. |
| Deploy step can't connect | Check `EC2_HOST`, that port 22 is open, and `EC2_SSH_KEY` is the full `.pem`. |
| `/docs` won't load | Port **5000** must be open in the EC2 security group. Check `sudo systemctl status driftapp`. |
| `/health` shows `model_loaded: false` | The model isn't on the server. SSH in, activate venv, `python train.py`, restart. |
| Retrain runs but never deploys | That's correct if the challenger wasn't better (`promoted=false`). Check the run's "Retrain & validate" log. |
| `monitor.py` errors on token | Set `GITHUB_TOKEN` and `GITHUB_REPO` env vars; the token needs repo/workflow scope. |
| `/drift` always flags tiny batches | PSI is unstable on very few rows. Send 50+ rows. |

Handy server commands:
```bash
sudo systemctl status driftapp        # is it running?
sudo journalctl -u driftapp -n 50     # recent app logs
cat drift.log                          # history of drift events
git log --oneline -- reference/model.pkl   # history of model changes
```

---

## Part 10 — Where to go next

- Use a real **model registry** (e.g. MLflow) instead of committing `.pkl` files.
- Add **prediction drift** (watch the distribution of outputs, often the earliest
  warning sign).
- Add a **manual approval** step before the retrain deploys (turns automatic
  Continuous Deployment into Continuous Delivery).
- Add a **canary / A-B rollout**: send the new model a small % of traffic first.
- Send a Slack/email alert when a model is promoted or when drift is logged.

---

## Quick glossary

- **Reference data**: saved snapshot of the training data; the baseline drift is
  measured against. Moves forward after a real retrain.
- **Champion / challenger**: the live model vs a candidate new model; you only
  promote the challenger if it validates as better.
- **RMSE**: average prediction error (in dollars here); lower is better.
- **repository_dispatch**: a GitHub event you can fire via an API call to start a
  workflow from outside (this is how drift detection triggers retraining).
- **Reusable workflow**: a workflow other workflows can `uses:` so deploy steps
  aren't duplicated.
- **KS test / PSI**: the two drift-detection methods.
