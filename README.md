# Automated Detection System of Human Emotion Using CNN

Undergraduate thesis web application for real-time facial-expression
classification with Flask, TensorFlow/Keras, OpenCV, MySQL, Jinja, JavaScript,
and Waitress.

## Authoritative five-class contract

Model outputs always use this exact index order:

```text
0 Happy
1 Angry
2 Sad
3 Neutral
4 Surprise
```

The shared definition lives in `ml_config.py`. Flask, training, evaluation,
and dataset utilities import it rather than declaring independent lists.
Although the class label is `Surprise`, the user-facing sentence is
`<name> is Surprised`.

The deployed legacy model is:

```text
emotion_recognition_model_5class.h5
```

## User and administrator workflows

Users can register, log in, open a dashboard, run live webcam detection, view
saved history and simple analytics, change browser camera settings, contact
support, and log out.

Administrators have a separate login and can review users, global detection
statistics and records, and support conversations. Image/video upload APIs
remain for backward compatibility, but the user interface intentionally
focuses on live webcam detection and contains no upload controls, probability
bars, or emotion-distribution section.

## Local setup on Windows

Requirements:

- Python 3.10
- MySQL 8
- A database named `emosense`
- A webcam for live-browser validation

Create and activate the environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a least-privilege database login:

1. Copy `database/create_app_user.sql.example` to
   `database/create_app_user.local.sql`.
2. Replace both password placeholders.
3. Run the local SQL file as a MySQL administrator.

Then copy `.env.example` to `.env` and use the same database password:

```powershell
Copy-Item .env.example .env
```

Generate a strong Flask secret:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

Put the generated value in `.env` as `SECRET_KEY`. The `.env` file is ignored
by Git.

Start the server:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:5000/ui
```

Waitress should keep the terminal occupied. Stop it with `Ctrl+C`. If the
prompt returns while the health endpoint still responds, another background
process is serving the application.

## Important configuration

Configuration is read from environment variables and, when `python-dotenv` is
installed, from `.env`.

| Variable | Development default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Set `production` to enforce production secrets and DB credentials |
| `SECRET_KEY` | Random process-local key | Must be at least 32 characters in production |
| `DB_HOST` / `DB_PORT` | `localhost` / `3306` | MySQL location |
| `DB_NAME` | `emosense` | Database |
| `DB_USER` / `DB_PASSWORD` | Legacy local fallback | Required and non-root in production |
| `MODEL_PATH` | `emotion_recognition_model_5class.h5` | Deployed model |
| `MODEL_PREPROCESSING` | Metadata or `legacy` | `basic`, `clahe`, or `legacy` |
| `MAX_REQUEST_MB` | `32` | Whole-request upload limit |
| `OUTPUT_DIR` | `instance/outputs` | Private annotated-video storage |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Waitress bind address |

See `.env.example` for the complete development configuration.

## Verification

Run the fast automated suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Compile all application scripts:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
```

Run the read-only dataset audit:

```powershell
.\.venv\Scripts\python.exe audit_dataset.py `
  --output artifacts\dataset_audit.json
```

The audit reports class counts, invalid images, dimensions, exact duplicates,
cross-split leakage, and conflicting labels without changing the dataset.

## Dataset splitting

`prepare_test_split.py` is dry-run by default. It plans a reproducible,
exact-deduplicated train/validation/test split in the canonical class order:

```powershell
.\.venv\Scripts\python.exe prepare_test_split.py
```

After reviewing the plan, `--apply` copies files into a new protected output
root. It never deletes or modifies the current dataset:

```powershell
.\.venv\Scripts\python.exe prepare_test_split.py `
  --output-root dataset\split_v2 `
  --apply
```

Do not train against `split_v2` until its manifest and counts have been
reviewed. Keep the test split untouched until final model selection.

## Training controlled experiments

Validate folders, preprocessing, generators, and model shapes without
training:

```powershell
.\.venv\Scripts\python.exe train.py --dry-run
```

Run a short pipeline smoke test:

```powershell
.\.venv\Scripts\python.exe train.py --smoke-test --preprocessing basic
```

Recommended experiments:

```powershell
.\.venv\Scripts\python.exe train.py `
  --preprocessing basic `
  --experiment-name baseline_basic

.\.venv\Scripts\python.exe train.py `
  --preprocessing clahe `
  --experiment-name comparison_clahe
```

Training never overwrites the deployed model. Each run receives a timestamped
folder under `artifacts/experiments` containing best/last models, class and
preprocessing metadata, configuration, and CSV history. Class weights,
checkpointing, early stopping, learning-rate reduction, realistic
augmentation, and reproducible seeds are enabled.

The old `train_model.py` and `train_model_3class.py` names are compatibility
wrappers around the canonical five-class trainer.

## Evaluation

Legacy deployed model, using its known preprocessing assumption:

```powershell
.\.venv\Scripts\python.exe evaluate_model.py `
  --model emotion_recognition_model_5class.h5 `
  --dataset dataset\split_v2\test `
  --preprocessing legacy `
  --allow-missing-metadata
```

This legacy-model result is diagnostic only: the old model may previously have
seen some source images that were later assigned to `split_v2\test`. The split
is genuinely untouched only for new models trained exclusively on
`split_v2\train`.

New candidate models should have metadata, so only model and split are needed:

```powershell
.\.venv\Scripts\python.exe evaluate_model.py `
  --model artifacts\experiments\<run>\emotion_recognition_model_5class_best.h5 `
  --dataset dataset\test
```

Evaluation writes overall and balanced accuracy, per-class precision/recall/F1,
macro and weighted metrics, class counts, predictions, CSV confusion matrices,
and a thesis-usable SVG confusion matrix under `artifacts/evaluations`.

Deploy a candidate only if it improves untouched-test macro F1 and acceptable
per-class recall, not merely confidence or training accuracy.

## Database notes

`database/preflight.sql` contains read-only integrity checks. Existing
historical detections assigned to the old hard-coded `input_id=1` cannot be
re-attributed reliably and must not be silently rewritten.

New detections create a user-owned input and output record. History and
analytics queries filter through the authenticated user's input records.
Database engine/constraint changes are intentionally not applied
automatically; back up MySQL before running any migration.

## Privacy and security

- CSRF protection covers state-changing forms and AJAX requests.
- Session roles are mutually exclusive and API auth failures return JSON.
- Login, support, and inference requests are rate-limited locally.
- Upload sizes, extensions, frame counts, and numeric controls are bounded.
- Support messages are rendered as text, preventing stored-script execution.
- Generated annotated videos are stored outside `static` and served only to
  their owning session.
- Legacy `/static/outputs` URLs are blocked, but old files are not deleted
  automatically.
- Production refuses a missing/short secret, missing DB credentials, or a root
  DB user.

Use HTTPS and set `SESSION_COOKIE_SECURE=true` in production. The built-in
in-memory limiter is suitable for the thesis workstation; multi-process public
deployment should use a shared rate-limit store.
