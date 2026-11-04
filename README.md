# Titanic Disaster: Survival Prediction (Python + R in Docker)

This repo trains a **logistic regression** model to predict which passengers survived the
Titanic disaster (April 15, 1912). The same pipeline is implemented twice, once in
**Python** and once in **R**, and each runs in its own **Docker container**, so you don't
need Python or R installed on your machine.

Each script prints every step to the terminal: loading the data, exploring it, each
cleaning/feature change, model training, training accuracy and test-set predictions.

---

## Repository structure

```
titanic-disaster/
├── .gitignore              # ignores src/data/, *.csv, .DS_Store, virtual envs, R history
├── README.md
└── src/
    ├── data/               # <- YOU put train.csv and test.csv here (not tracked by git)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic.R
```

---

## Prerequisites

1. **Docker Desktop** installed and **running**: https://www.docker.com/products/docker-desktop/
   Check it works with `docker --version`.
2. **Git**: https://git-scm.com/downloads
3. A free **Kaggle account** (needed to download the data).

---

## Step 1: Clone the repo

```bash
git clone https://github.com/<your-username>/titanic-disaster.git
cd titanic-disaster
```

> All commands below are run from this top-level `titanic-disaster` folder.

---

## Step 2: Download the data

The dataset is **not** included in this repo. Download it from Kaggle:

1. Go to https://www.kaggle.com/competitions/titanic/data
2. Sign in, click **"Join Competition"** / accept the rules (required before downloading).
3. Click **"Download All"**. This gives you `titanic.zip`.
4. Unzip it and move **`train.csv`** and **`test.csv`** into `src/data/`.

**Using the terminal instead (macOS / Linux):**
```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip -d src/data
```

**Or with the Kaggle CLI** (if you have an API token set up in `~/.kaggle/kaggle.json`):
```bash
pip install kaggle
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

Your folder should now look like this:
```
src/data/
├── train.csv
└── test.csv
```

---

## Step 3: Run the Python model

**Build the image** (the `.` at the end matters: it tells Docker to use the repo root as the build context):
```bash
docker build -t titanic-python -f src/python/Dockerfile .
```

**Run the container.** The data folder is mounted into the container so the dataset is never baked into the image:

macOS / Linux / Git Bash:
```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python
```

Windows PowerShell:
```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-python
```

Windows Command Prompt:
```cmd
docker run --rm -v "%cd%/src/data:/app/src/data" titanic-python
```

Predictions are saved to `src/data/predictions_python.csv`.

---

## Step 4: Run the R model

**Build the image** (the first build takes a few minutes while R packages install):
```bash
docker build -t titanic-r -f src/r/Dockerfile .
```

**Run the container:**

macOS / Linux / Git Bash:
```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r
```

Windows PowerShell:
```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-r
```

Windows Command Prompt:
```cmd
docker run --rm -v "%cd%/src/data:/app/src/data" titanic-r
```

Predictions are saved to `src/data/predictions_r.csv`.

---

## What the scripts do

Both scripts follow the same seven printed sections:

1. **Load** `train.csv` and show its shape and first rows.
2. **Explore**: missing values per column, overall survival rate, survival by sex and by class.
3. **Adjust the data**:
   - Extract `Title` (Mr, Mrs, Miss, Master, Rare) from `Name`.
   - Fill missing `Age` with the median age for that passenger's title.
   - Fill missing `Fare` with the median fare, and missing `Embarked` with the most common port.
   - Add `FamilySize` (SibSp + Parch + 1) and `IsAlone`.
   - Encode `Sex` as 0/1 and add `LogFare` = log(1 + Fare).
   - One-hot encode `Pclass`, `Embarked`, `Title`.
   - Drop `PassengerId`, `Name`, `Ticket`, `Cabin` (mostly missing) and raw `Fare`.
   - All fill values are learned from the training data only, then reused on the test data.
4. **Train** a logistic regression on 80% of `train.csv`, holding out 20% for validation.
5. **Training accuracy**, plus accuracy on the held-out 20%.
6. **Load `test.csv`**, apply the identical transformations, refit on all of `train.csv`, and predict.
7. **Test accuracy**: Kaggle's `test.csv` does **not** contain the `Survived` column, so true
   test accuracy can't be computed locally. The script reports the held-out validation accuracy
   as the estimate, and writes a Kaggle-ready predictions file you can submit for the official score.
   (If a `test.csv` with a `Survived` column is provided, the script computes real test accuracy automatically.)

---

## Package versions

| Python (`python:3.11-slim`) | R (`rocker/r-ver:4.4.1`) |
|---|---|
| numpy 1.26.4 | readr (pinned via the image's dated CRAN snapshot) |
| pandas 2.2.2 | dplyr (pinned via the image's dated CRAN snapshot) |
| scikit-learn 1.5.1 | `glm()` from base R `stats` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: .../train.csv not found` | The data isn't in `src/data/`, or you ran `docker run` from the wrong folder. `cd` to the repo root and check `ls src/data`. |
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running. Open it and wait until it says "running". |
| `COPY failed: file not found` during build | You forgot the trailing `.` or ran `docker build` from inside `src/`. Run it from the repo root. |
| Volume mount errors on Windows | Use the PowerShell or Command Prompt version of the command shown above. |
| Apple Silicon (M1/M2/M3) build issues | Add `--platform linux/amd64` to the `docker build` and `docker run` commands. |
