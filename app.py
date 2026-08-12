from collections import defaultdict, deque
from datetime import timedelta
import functools
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid

import cv2
import mysql.connector
import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:  # Keeps source-check tooling usable before dependencies install.
    load_dotenv = None
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_bcrypt import Bcrypt
from mysql.connector import pooling
from waitress import serve
from ml_config import (
    DEPLOYED_MODEL_PATH,
    EMOTIONS,
    NUM_CLASSES,
    read_model_metadata,
    validate_model_contract,
    validate_model_metadata,
)
from ml_preprocessing import (
    DEFAULT_DEPLOYMENT_PREPROCESSING,
    preprocess_face,
    validate_preprocessing_mode,
)
from auth_helpers import (
    auth_admin_required,
    auth_authenticate_admin,
    auth_get_all_users,
    auth_get_user_by_email,
    auth_get_user_by_id,
    auth_request_wants_json,
    auth_set_admin_session,
    auth_set_user_session,
    auth_login_required,
    auth_user_or_guest_required,
)
from support import (
    support_get_admin_chats,
    support_get_admin_messages,
    support_get_chat_users,
    support_get_messages_for_user,
    support_insert_message,
    support_ensure_guest_schema,
    support_get_guest_conversations,
    support_get_guest_messages,
    support_insert_guest_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default, minimum=None, maximum=None):
    raw_value = os.environ.get(name)
    try:
        value = int(raw_value) if raw_value is not None else int(default)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be at most {maximum}.")
    return value


APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
if APP_ENV not in {"development", "testing", "production"}:
    raise RuntimeError("APP_ENV must be development, testing, or production.")
IS_PRODUCTION = APP_ENV == "production"


def resolve_secret_key():
    configured = os.environ.get("SECRET_KEY")
    if configured:
        if IS_PRODUCTION and len(configured) < 32:
            raise RuntimeError(
                "SECRET_KEY must contain at least 32 characters in production."
            )
        return configured
    if IS_PRODUCTION:
        raise RuntimeError("SECRET_KEY is required when APP_ENV=production.")
    logger.warning(
        "SECRET_KEY is not configured. Using a random process-local key for %s; "
        "sessions will be invalidated whenever the application restarts.",
        APP_ENV,
    )
    return secrets.token_urlsafe(48)


app = Flask(__name__)
app.config.update(
    SECRET_KEY=resolve_secret_key(),
    SESSION_COOKIE_NAME="emotion_recognition_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=env_bool("SESSION_COOKIE_SECURE", IS_PRODUCTION),
    PERMANENT_SESSION_LIFETIME=timedelta(
        hours=env_int("SESSION_HOURS", 8, minimum=1, maximum=168)
    ),
    MAX_CONTENT_LENGTH=env_int(
        "MAX_REQUEST_MB", 32, minimum=1, maximum=512
    ) * 1024 * 1024,
)
bcrypt = Bcrypt(app)
DUMMY_PASSWORD_HASH = bcrypt.generate_password_hash(
    secrets.token_urlsafe(32)
).decode("utf-8")
 
# ── Database ──────────────────────────────────────────────────
# Database credentials are environment-driven. Legacy local access remains
# available only outside production so existing thesis workstations keep working.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = env_int("DB_PORT", 3306, minimum=1, maximum=65535)
DB_NAME = os.environ.get("DB_NAME", "emosense")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_POOL_SIZE = env_int("DB_POOL_SIZE", 5, minimum=1, maximum=32)
DB_CONNECT_TIMEOUT = env_int("DB_CONNECT_TIMEOUT", 5, minimum=1, maximum=60)

if IS_PRODUCTION:
    if not DB_USER or not DB_PASSWORD or not DB_PASSWORD.strip():
        raise RuntimeError(
            "DB_USER and DB_PASSWORD are required when APP_ENV=production."
        )
    if DB_USER.strip().lower() == "root":
        raise RuntimeError("The production application must not use the MySQL root account.")
elif not DB_USER:
    DB_USER = "root"
    logger.warning(
        "DB_USER is not configured. Falling back to the legacy local root account. "
        "Create a least-privilege MySQL user before deployment."
    )

_db_pool = None
_db_pool_lock = threading.Lock()


def _create_db_pool():
    return pooling.MySQLConnectionPool(
        pool_name=f"emotion_app_{os.getpid()}",
        pool_size=DB_POOL_SIZE,
        pool_reset_session=True,
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        connection_timeout=DB_CONNECT_TIMEOUT,
    )


def get_db():
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = _create_db_pool()
    connection = _db_pool.get_connection()
    try:
        connection.ping(reconnect=True, attempts=1, delay=0)
    except Exception:
        connection.close()
        raise
    return connection

def query_db(query, params=None, dictionary=True):
    db = get_db()
    cur = None
    try:
        cur = db.cursor(dictionary=dictionary)
        cur.execute(query, params if params is not None else ())
        return cur.fetchall()
    finally:
        if cur is not None:
            cur.close()
        db.close()


def query_db_one(query, params=None, dictionary=True):
    rows = query_db(query, params, dictionary=dictionary)
    return rows[0] if rows else None


def execute_db(query, params=None):
    db = get_db()
    cur = None
    try:
        cur = db.cursor()
        cur.execute(query, params if params is not None else ())
        db.commit()
        return cur.lastrowid
    except Exception:
        db.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        db.close()


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CSRF_SESSION_KEY = "_csrf_token"
MAX_SUPPORT_MESSAGE_CHARS = env_int(
    "MAX_SUPPORT_MESSAGE_CHARS", 2000, minimum=100, maximum=10000
)
MAX_IMAGE_FILES = env_int("MAX_IMAGE_FILES", 5, minimum=1, maximum=20)
MAX_IMAGE_BYTES = env_int(
    "MAX_IMAGE_MB", 5, minimum=1, maximum=50
) * 1024 * 1024
MAX_IMAGE_PIXELS = env_int(
    "MAX_IMAGE_PIXELS", 12_000_000, minimum=48 * 48, maximum=100_000_000
)
MAX_VIDEO_FRAMES = env_int("MAX_VIDEO_FRAMES", 120, minimum=1, maximum=1000)
MAX_FRAME_SKIP = env_int("MAX_FRAME_SKIP", 60, minimum=1, maximum=1000)
MAX_VIDEO_READ_FRAMES = env_int(
    "MAX_VIDEO_READ_FRAMES", 3000, minimum=1, maximum=100_000
)
MAX_VIDEO_PIXELS = env_int(
    "MAX_VIDEO_PIXELS", 8_294_400, minimum=48 * 48, maximum=100_000_000
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}
OUTPUT_DIR = os.path.abspath(
    os.environ.get("OUTPUT_DIR", os.path.join(app.instance_path, "outputs"))
)


class InMemoryRateLimiter:
    """Small local limiter suitable for the thesis workstation.

    A shared Redis-backed limiter should be configured if the application is
    later deployed with multiple processes or hosts.
    """

    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = threading.Lock()
        self._checks = 0

    def check(self, scope, identity, maximum, window_seconds):
        now = time.monotonic()
        key = (scope, identity)
        with self._lock:
            events = self._events[key]
            cutoff = now - window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= maximum:
                retry_after = max(1, int(window_seconds - (now - events[0])) + 1)
                return False, retry_after
            events.append(now)
            self._checks += 1
            if self._checks % 500 == 0:
                stale_keys = [
                    stored_key
                    for stored_key, values in self._events.items()
                    if not values or values[-1] < now - 3600
                ]
                for stored_key in stale_keys:
                    self._events.pop(stored_key, None)
            return True, 0


rate_limiter = InMemoryRateLimiter()


def request_identity():
    if session.get("role") == "user" and session.get("user_id"):
        return f"user:{session['user_id']}"
    if session.get("role") == "admin" and session.get("admin_id"):
        return f"admin:{session['admin_id']}"
    return f"ip:{request.remote_addr or 'unknown'}"


def limit_requests(maximum, window_seconds, methods=None):
    limited_methods = {method.upper() for method in methods} if methods else None

    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            if limited_methods is not None and request.method not in limited_methods:
                return view(*args, **kwargs)
            allowed, retry_after = rate_limiter.check(
                request.endpoint or view.__name__,
                request_identity(),
                maximum,
                window_seconds,
            )
            if allowed:
                return view(*args, **kwargs)
            if auth_request_wants_json():
                response = jsonify({
                    "ok": False,
                    "error": "Too many requests. Please try again shortly.",
                })
            else:
                response = app.make_response(
                    ("Too many requests. Please try again shortly.", 429)
                )
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        return wrapped

    return decorator


def csrf_token():
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def block_public_legacy_outputs():
    # Older builds stored biometric-derived annotated videos below Flask's
    # public static tree without ownership metadata. Keep the files untouched,
    # but do not serve them.
    if request.path.startswith("/static/outputs/"):
        return "Not found.", 404
    return None


def _defer_unsafe_unauthenticated_request_to_auth():
    path = request.path
    if path == "/logout" and session.get("role") != "user":
        return True
    if path.startswith("/api/admin/"):
        return session.get("role") != "admin"
    if (
        path.startswith("/admin/")
        and path != "/admin/login"
        and session.get("role") != "admin"
    ):
        return True
    if path == "/predict" and session.get("role") not in {"user", "guest"}:
        return True
    if (
        (
            (path.startswith("/predict") and path != "/predict")
            or (path.startswith("/api/") and not path.startswith("/api/admin/"))
            or path.startswith("/support")
        )
        and session.get("role") != "user"
    ):
        return True
    return False


@app.before_request
def enforce_csrf():
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    if _defer_unsafe_unauthenticated_request_to_auth():
        return None

    submitted = request.headers.get("X-CSRF-Token", "")
    if not submitted:
        submitted = request.form.get("csrf_token", "")
    expected = session.get(CSRF_SESSION_KEY, "")

    if expected and submitted and secrets.compare_digest(expected, submitted):
        return None

    logger.warning("Rejected request with a missing or invalid CSRF token: %s", request.path)
    if auth_request_wants_json():
        return jsonify({"ok": False, "error": "Invalid or missing CSRF token."}), 400
    return "Invalid or missing CSRF token. Please reload the page and try again.", 400


@app.after_request
def set_security_headers(response):
    content_security_policy = os.environ.get(
        "CONTENT_SECURITY_POLICY",
        (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'"
        ),
    )
    response.headers.setdefault("Content-Security-Policy", content_security_policy)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(self), microphone=(), geolocation=(), payment=()",
    )
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if request.endpoint != "static":
        response.headers.setdefault("Cache-Control", "no-store")
    if app.config["SESSION_COOKIE_SECURE"]:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response


@app.errorhandler(413)
def request_too_large(_error):
    message = (
        f"Request is too large. Maximum size is "
        f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
    )
    if auth_request_wants_json():
        return jsonify({"ok": False, "error": message}), 413
    return message, 413


_output_columns = None
_output_columns_lock = threading.Lock()


def get_output_columns():
    global _output_columns
    if _output_columns is None:
        with _output_columns_lock:
            if _output_columns is None:
                rows = query_db(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'output'
                    """,
                    (DB_NAME,),
                )
                _output_columns = {row["COLUMN_NAME"] for row in rows}
    return _output_columns


def save_detection_result(user_id, emotion_number, emotion, confidence, faces_detected):
    """Create the user input and its output together using one DB transaction."""
    has_faces_detected = "faces_detected" in get_output_columns()
    db = get_db()
    cur = None
    try:
        db.start_transaction()
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO input (user_id, status)
            VALUES (%s, 'incomplete')
            """,
            (user_id,),
        )
        input_id = cur.lastrowid
        if has_faces_detected:
            cur.execute(
                """
                INSERT INTO output
                    (input_id, number, name, confidence, faces_detected)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    input_id,
                    emotion_number,
                    emotion,
                    confidence,
                    faces_detected,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO output (input_id, number, name, confidence)
                VALUES (%s, %s, %s, %s)
                """,
                (input_id, emotion_number, emotion, confidence),
            )
        output_id = cur.lastrowid
        cur.execute(
            "UPDATE input SET status = 'complete' WHERE input_id = %s",
            (input_id,),
        )
        db.commit()
        return {"input_id": int(input_id), "output_id": int(output_id)}
    except Exception:
        db.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        db.close()


def serialize_detection_record(row):
    detected_at = row.get("detected_at")
    return {
        "output_id": int(row["output_id"]),
        "emotion": row["emotion"],
        "confidence": float(row["confidence"] or 0),
        "faces_detected": (
            int(row["faces_detected"])
            if row.get("faces_detected") is not None
            else None
        ),
        "detected_at": (
            detected_at.isoformat(timespec="seconds")
            if detected_at is not None
            else None
        ),
    }


def parse_bounded_int(raw_value, default, minimum, maximum, field_name):
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}."
        )
    return value


def validate_uploaded_file(file, allowed_extensions, kind):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Unsupported {kind} file type. Allowed: {allowed}.")
    mimetype = (file.mimetype or "").lower()
    expected_prefix = "image/" if kind == "image" else "video/"
    if mimetype and mimetype != "application/octet-stream" and not mimetype.startswith(expected_prefix):
        raise ValueError(f"The uploaded file is not identified as a valid {kind}.")
    return extension




def serialize_support_message(message, include_user_name=False):
    result = {
        "sender": message["sender"],
        "message": message["message"],
        "sent_at": message["sent_at"].strftime("%b %d, %H:%M") if message["sent_at"] else "",
    }
    if include_user_name:
        result["user_name"] = message.get("user_name", "")
    return result


def get_uploaded_files(field_name="file"):
    return [f for f in request.files.getlist(field_name) if f and f.filename]


def save_uploaded_temp_file(file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        path = temp_video.name
    try:
        file.save(path)
        return path
    except Exception:
        cleanup_temp_file(path)
        raise


def read_uploaded_image(file):
    data = file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Each image must be no larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    if not data:
        raise ValueError("An uploaded image is empty.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("An uploaded file could not be decoded as an image.")
    if int(image.shape[0]) * int(image.shape[1]) > MAX_IMAGE_PIXELS:
        raise ValueError("The uploaded image dimensions are too large.")
    return image


def cleanup_temp_file(path):
    if path and os.path.exists(path):
        for _ in range(5):
            try:
                os.remove(path)
                return
            except PermissionError:
                time.sleep(0.2)
            except OSError:
                logger.warning("Could not remove temporary file %s.", path)
                return
        logger.warning("Temporary file remained locked and could not be removed: %s.", path)

# ── Model ─────────────────────────────────────────────────────
from tensorflow.keras.models import load_model


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(DEPLOYED_MODEL_PATH),
)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")
model_metadata = read_model_metadata(MODEL_PATH)
configured_preprocessing = os.environ.get("MODEL_PREPROCESSING")
if model_metadata is not None and configured_preprocessing is None:
    MODEL_PREPROCESSING = validate_preprocessing_mode(
        model_metadata.get("preprocessing", "")
    )
else:
    MODEL_PREPROCESSING = validate_preprocessing_mode(
        configured_preprocessing or DEFAULT_DEPLOYMENT_PREPROCESSING
    )

logger.info("Loading model from %s ...", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
logger.info("Model loaded.")
logger.info("Model output shape: %s", model.output_shape)
validate_model_contract(model)
if model_metadata is not None:
    validate_model_metadata(
        model_metadata,
        expected_preprocessing=MODEL_PREPROCESSING,
    )
else:
    logger.warning(
        "Model metadata is not available for %s. Using explicitly configured "
        "preprocessing mode %s.",
        MODEL_PATH,
        MODEL_PREPROCESSING,
    )
model_predict_lock = threading.Lock()
  
FACE_CASCADE  = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
SMILE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml"))
if FACE_CASCADE.empty() or SMILE_CASCADE.empty():
    raise RuntimeError("Required OpenCV Haar cascade files could not be loaded.")
 
# ── Preprocessing ─────────────────────────────────────────────
def subset_predict(face_tensor):
    with model_predict_lock:
        scores = model.predict(face_tensor, verbose=0)[0].astype(np.float32)

    total = float(scores.sum())
    if total > 0:
        scores = scores / total

    max_i = int(np.argmax(scores))

    probabilities = {
        EMOTIONS[i]: float(scores[i])
        for i in range(len(EMOTIONS))
    }

    return (
        EMOTIONS[max_i],
        probabilities,
        float(scores[max_i]),
        scores,
    )
def summarize_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float32)

    total = float(scores.sum())
    if total > 0:
        scores = scores / total

    max_i = int(np.argmax(scores))

    probabilities = {
        EMOTIONS[i]: float(scores[i])
        for i in range(len(EMOTIONS))
    }

    return EMOTIONS[max_i], float(scores[max_i]), probabilities
 
def smile_detected(gray_face_2d):
    if gray_face_2d is None or gray_face_2d.size == 0:
        return False
    H, W = gray_face_2d.shape[:2]
    lower_face = gray_face_2d[int(H * 0.45):H, :]
    smiles = SMILE_CASCADE.detectMultiScale(
        lower_face, scaleFactor=1.7, minNeighbors=18, minSize=(25, 25))
    return len(smiles) > 0
 
def analyze_gray_frame(image):
    faces = FACE_CASCADE.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    results = []
 
    if len(faces) == 0:
        return [], None
 
    H, W = image.shape[:2]
    for (x, y, w, h) in faces:
        pad = int(0.25 * max(w, h))
        x1  = max(0, x - pad);   y1 = max(0, y - pad)
        x2  = min(W, x + w + pad); y2 = min(H, y + h + pad)
        shift = int(0.10 * (y2 - y1))
        y1    = min(H - 1, y1 + shift)
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        face_tensor = preprocess_face(face_crop, mode=MODEL_PREPROCESSING)
        emotion, probabilities, confidence, subset_vec = subset_predict(face_tensor)
 
        results.append({
            "box": {"x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1)},
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "_vec": subset_vec
        })
 
    if not results:
        return [], None
 
    best = max(results,
               key=lambda r: 0 if r.get("box") is None else r["box"]["w"] * r["box"]["h"])
    return results, best
 
 
# ════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════
 
@app.route('/')
def index():
    return redirect(url_for('login'))
 
@app.route('/register', methods=['GET', 'POST'])
@limit_requests(5, 60 * 60, methods={"POST"})
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
 
        if not name or not email or not password:
            return render_template("register.html", error="All fields are required.")
        if len(name) < 2 or len(name) > 100:
            return render_template(
                "register.html",
                error="Name must contain between 2 and 100 characters.",
            )
        if len(email) > 100 or not EMAIL_RE.fullmatch(email):
            return render_template(
                "register.html",
                error="Enter a valid email address.",
            )
        password_bytes = len(password.encode("utf-8"))
        if len(password) < 8 or password_bytes > 72:
            return render_template(
                "register.html",
                error="Password must be at least 8 characters and no more than 72 bytes.",
            )

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        try:
            execute_db(
                "INSERT INTO user (name, email, password) VALUES (%s, %s, %s)",
                (name, email, hashed),
            )
            return redirect(url_for("login", success="Account created! Please log in."))
        except mysql.connector.IntegrityError:
            return render_template("register.html", error="Email already registered.")
        except Exception:
            logger.exception("Registration database operation failed.")
            return render_template(
                "register.html",
                error="Account creation is temporarily unavailable. Please try again.",
            ), 503

    return render_template("register.html")
 
@app.route('/login', methods=['GET', 'POST'])
@limit_requests(10, 5 * 60, methods={"POST"})
def login():
    success = request.args.get("success", "")[:200]
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(email) > 100 or len(password.encode("utf-8")) > 72:
            return render_template("login.html", error="Invalid email or password.")
        try:
            user = auth_get_user_by_email(query_db_one, email)
            password_hash = user["password"] if user else DUMMY_PASSWORD_HASH
            password_matches = bcrypt.check_password_hash(password_hash, password)
            if user and password_matches:
                auth_set_user_session(user)
                return redirect(url_for("dashboard"))
            return render_template("login.html", error="Invalid email or password.")
        except Exception:
            logger.exception("User login database operation failed.")
            return render_template(
                "login.html",
                error="Login is temporarily unavailable. Please try again.",
            ), 503
    return render_template("login.html", success=success)
 
@app.route('/logout', methods=['POST'])
@auth_login_required
def logout():
    session.clear()
    return redirect(url_for('login'))
 
@app.route('/dashboard')
@auth_login_required
def dashboard():
    return render_template('index.html', name=session.get('user_name'), guest_mode=False)


@app.route("/guest")
def guest_dashboard():
    """Start a browser-only guest session without collecting account details."""
    guest_fresh = session.get("role") != "guest"
    if session.get("role") != "guest":
        session.clear()
        session.permanent = True
        session["role"] = "guest"
        session["guest_mode"] = True
        session["guest_support_token"] = secrets.token_urlsafe(24)
    return render_template("index.html", name="Guest", guest_mode=True, guest_fresh=guest_fresh)


def guest_support_token():
    if session.get("role") != "guest":
        return None
    token = session.get("guest_support_token")
    if not token:
        token = secrets.token_urlsafe(24)
        session["guest_support_token"] = token
    return token


@app.route("/guest/exit", methods=["POST"])
def guest_exit():
    if session.get("role") != "guest":
        return redirect(url_for("login"))
    session.clear()
    return redirect(url_for("login"))
 
@app.route("/support")
@auth_login_required
def support():
    try:
        messages = support_get_messages_for_user(query_db, session["user_id"])
        load_error = None
    except Exception:
        logger.exception("Failed to load support messages for user %s.", session["user_id"])
        messages = []
        load_error = "Support messages could not be loaded. Please try again."
    return render_template(
        "support.html",
        messages=messages,
        name=session.get("user_name"),
        load_error=load_error,
    )
 
@app.route('/support/send', methods=['POST'])
@auth_login_required
@limit_requests(30, 60)
def support_send():
    message = request.form.get("message", "").strip()
    if not message or len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return redirect(url_for("support", error="invalid-message"))
    try:
        support_insert_message(execute_db, session["user_id"], message, "user")
    except Exception:
        logger.exception("Failed to save support message for user %s.", session["user_id"])
        return redirect(url_for("support", error="send-failed"))
    return redirect(url_for("support"))
 
 
@app.route("/support/messages")
@auth_login_required
@limit_requests(60, 60)
def support_messages():
    try:
        messages = support_get_messages_for_user(query_db, session["user_id"])
        return jsonify({
            "ok": True,
            "messages": [serialize_support_message(m) for m in messages],
        })
    except Exception:
        logger.exception("Failed to load support messages for user %s.", session["user_id"])
        return jsonify({
            "ok": False,
            "error": "Support messages could not be loaded.",
        }), 503
 
@app.route("/support/send-ajax", methods=["POST"])
@auth_login_required
@limit_requests(30, 60)
def support_send_ajax():
    data = request.get_json(silent=True)
    message = (data.get("message", "") if data else "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400
    if len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return jsonify({
            "ok": False,
            "error": f"Message must not exceed {MAX_SUPPORT_MESSAGE_CHARS} characters.",
        }), 400
    try:
        support_insert_message(execute_db, session["user_id"], message, "user")
        return jsonify({"ok": True})
    except Exception:
        logger.exception("Failed to save support message for user %s.", session["user_id"])
        return jsonify({
            "ok": False,
            "error": "Support message could not be sent.",
        }), 503


@app.route("/guest/support")
def guest_support():
    token = guest_support_token()
    if not token:
        return redirect(url_for("guest_dashboard"))
    try:
        support_ensure_guest_schema(execute_db)
        messages = support_get_guest_messages(query_db, token)
        load_error = None
    except Exception:
        logger.exception("Failed to load guest support messages.")
        messages = []
        load_error = "Guest support is temporarily unavailable. Please try again."
    return render_template("support.html", messages=messages, name="Guest", guest_mode=True, load_error=load_error)


@app.route("/guest/support/messages")
@limit_requests(60, 60)
def guest_support_messages():
    token = guest_support_token()
    if not token:
        return jsonify({"ok": False, "error": "Guest session required."}), 401
    try:
        support_ensure_guest_schema(execute_db)
        messages = support_get_guest_messages(query_db, token)
        return jsonify({"ok": True, "messages": [serialize_support_message(message) for message in messages]})
    except Exception:
        logger.exception("Failed to refresh guest support messages.")
        return jsonify({"ok": False, "error": "Guest support messages could not be loaded."}), 503


@app.route("/guest/support/send-ajax", methods=["POST"])
@limit_requests(30, 60)
def guest_support_send_ajax():
    token = guest_support_token()
    if not token:
        return jsonify({"ok": False, "error": "Guest session required."}), 401
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    guest_name = " ".join(str(data.get("guest_name", "")).split())[:100]
    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400
    if len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return jsonify({"ok": False, "error": f"Message must not exceed {MAX_SUPPORT_MESSAGE_CHARS} characters."}), 400
    try:
        support_ensure_guest_schema(execute_db)
        support_insert_guest_message(execute_db, token, guest_name, message, "guest")
        return jsonify({"ok": True})
    except Exception:
        logger.exception("Failed to save guest support message.")
        return jsonify({"ok": False, "error": "Guest support message could not be sent."}), 503
 
@app.route("/admin/support/send-ajax", methods=["POST"])
@auth_admin_required
@limit_requests(60, 60)
def admin_send_ajax():
    data = request.get_json(silent=True)
    user_id = data.get("user_id") if data else None
    message = (data.get("message", "") if data else "").strip()
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "A valid user is required."}), 400
    if not message:
        return jsonify({"ok": False, "error": "Message is required."}), 400
    if len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return jsonify({
            "ok": False,
            "error": f"Message must not exceed {MAX_SUPPORT_MESSAGE_CHARS} characters.",
        }), 400
    try:
        if not auth_get_user_by_id(query_db_one, user_id):
            return jsonify({"ok": False, "error": "User not found."}), 404
        support_insert_message(execute_db, user_id, message, "admin")
        return jsonify({"ok": True})
    except Exception:
        logger.exception("Administrator failed to send a support reply.")
        return jsonify({
            "ok": False,
            "error": "Support reply could not be sent.",
        }), 503
 
@app.route("/admin/support/messages/<int:user_id>")
@auth_admin_required
@limit_requests(60, 60)
def admin_support_messages(user_id):
    try:
        if not auth_get_user_by_id(query_db_one, user_id):
            return jsonify({"ok": False, "error": "User not found."}), 404
        messages = support_get_admin_messages(query_db, user_id)
        return jsonify({
            "ok": True,
            "messages": [
                serialize_support_message(m, include_user_name=True)
                for m in messages
            ],
        })
    except Exception:
        logger.exception("Failed to load support messages for user %s.", user_id)
        return jsonify({
            "ok": False,
            "error": "Support messages could not be loaded.",
        }), 503
# ── Admin ─────────────────────────────────────────────────────
@app.route('/admin/login', methods=['GET', 'POST'])
@limit_requests(10, 5 * 60, methods={"POST"})
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            return render_template('admin_login.html', error='Username and password are required.')
        if len(username) > 50 or len(password.encode("utf-8")) > 72:
            return render_template('admin_login.html', error='Invalid credentials.')

        try:
            admin = auth_authenticate_admin(
                query_db_one,
                bcrypt,
                username,
                password,
                dummy_password_hash=DUMMY_PASSWORD_HASH,
            )
            if admin:
                auth_set_admin_session(admin)
                return redirect(url_for('admin_dashboard'))

            return render_template('admin_login.html', error='Invalid credentials.')

        except Exception:
            logger.exception("Administrator login database operation failed.")
            return render_template(
                'admin_login.html',
                error='Administrator login is temporarily unavailable.',
            ), 503

    return render_template('admin_login.html')
 
@app.route('/admin/logout', methods=['POST'])
@auth_admin_required
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))
 
@app.route('/admin/dashboard')
@auth_admin_required
def admin_dashboard():
    try:
        users = auth_get_all_users(query_db)

        detection_count = query_db_one(
            "SELECT COUNT(*) AS total FROM output"
        )

        total_detections = (
            detection_count["total"]
            if detection_count
            else 0
        )

    except Exception as e:
        logger.exception("Failed to load admin dashboard: %s", e)

        users = []
        total_detections = 0

    return render_template(
        "admin_dashboard.html",
        name=session.get("admin_name"),
        user_count=len(users),
        total_detections=total_detections,
    )


@app.route('/admin/users')
@auth_admin_required
def admin_users():
    try:
        users = auth_get_all_users(query_db)
    except Exception:
        logger.exception("Failed to load administrator user management page.")
        users = []
    return render_template(
        'admin_users.html',
        name=session.get('admin_name'),
        users=users,
    )


@app.route('/admin/detection-records')
@auth_admin_required
def admin_detection_records():
    return render_template(
        'admin_detection_records.html',
        name=session.get('admin_name'),
    )
@app.route("/admin/dashboard/stats")
@auth_admin_required
@limit_requests(60, 60)
def admin_dashboard_stats():
    try:
        detection_count = query_db_one(
            "SELECT COUNT(*) AS total FROM output"
        )

        total_detections = (
            int(detection_count["total"])
            if detection_count
            else 0
        )

        rows = query_db(
            """
            SELECT name, COUNT(*) AS count
            FROM output
            GROUP BY name
            """
        )

        emotion_counts = {emotion: 0 for emotion in EMOTIONS}

        for row in rows:
            emotion_name = row.get("name")

            if emotion_name in emotion_counts:
                emotion_counts[emotion_name] = int(row["count"])

        return jsonify({
            "ok": True,
            "total_detections": total_detections,
            "emotion_counts": emotion_counts,
        })

    except Exception as e:
        logger.exception("Failed to load dashboard statistics: %s", e)

        return jsonify({
            "ok": False,
            "error": "Could not load dashboard statistics.",
        }), 500

@app.route('/admin/support')
@auth_admin_required
def admin_support():
    try:
        support_ensure_guest_schema(execute_db)
        all_chats = support_get_admin_chats(query_db)
        chat_users = support_get_chat_users(query_db)
        guest_chats = support_get_guest_conversations(query_db)
        load_error = None
    except Exception:
        logger.exception("Failed to load administrator support dashboard.")
        all_chats = []
        chat_users = []
        guest_chats = []
        load_error = "Support conversations could not be loaded."
    return render_template(
        'admin_support.html',
        name=session.get('admin_name'),
        all_chats=all_chats,
        chat_users=chat_users,
        guest_chats=guest_chats,
        load_error=load_error,
    )
 
@app.route('/admin/support/reply', methods=['POST'])
@auth_admin_required
@limit_requests(60, 60)
def admin_reply():
    user_id = request.form.get('user_id')
    message = request.form.get('message', '').strip()
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return redirect(url_for('admin_support', error='invalid-user'))
    if not message or len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return redirect(url_for('admin_support', error='invalid-message'))
    try:
        if not auth_get_user_by_id(query_db_one, user_id):
            return redirect(url_for('admin_support', error='user-not-found'))
        support_insert_message(execute_db, user_id, message, 'admin')
    except Exception:
        logger.exception("Administrator failed to save a support reply.")
        return redirect(url_for('admin_support', error='send-failed'))
    return redirect(url_for('admin_support'))


@app.route("/admin/guest-support/<guest_token>")
@auth_admin_required
def admin_guest_support(guest_token):
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", guest_token):
        return "Not found.", 404
    # Keep legacy/bookmarked guest-chat URLs in the unified administrator
    # support workspace rather than rendering a separate guest-only page.
    return redirect(url_for("admin_support", guest=guest_token))


@app.route("/admin/guest-support/<guest_token>/messages")
@auth_admin_required
@limit_requests(60, 60)
def admin_guest_support_messages(guest_token):
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", guest_token):
        return jsonify({"ok": False, "error": "Guest conversation not found."}), 404
    try:
        support_ensure_guest_schema(execute_db)
        messages = support_get_guest_messages(query_db, guest_token)
        return jsonify({"ok": True, "messages": [serialize_support_message(message) for message in messages]})
    except Exception:
        logger.exception("Failed to refresh guest support conversation.")
        return jsonify({"ok": False, "error": "Guest messages could not be loaded."}), 503


@app.route("/admin/guest-support/<guest_token>/reply-ajax", methods=["POST"])
@auth_admin_required
@limit_requests(30, 60)
def admin_guest_support_reply(guest_token):
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", guest_token):
        return jsonify({"ok": False, "error": "Guest conversation not found."}), 404
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message or len(message) > MAX_SUPPORT_MESSAGE_CHARS:
        return jsonify({"ok": False, "error": "Enter a valid reply of up to 2,000 characters."}), 400
    try:
        support_ensure_guest_schema(execute_db)
        support_insert_guest_message(execute_db, guest_token, None, message, "admin")
        return jsonify({"ok": True})
    except Exception:
        logger.exception("Failed to save guest support reply.")
        return jsonify({"ok": False, "error": "Guest reply could not be sent."}), 503
 
 
# ════════════════════════════════════════════════════════════════
#  ORIGINAL ROUTES (unchanged)
# ════════════════════════════════════════════════════════════════
 
@app.route("/ui")
def ui():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if session.get("role") == "guest":
        return redirect(url_for("guest_dashboard"))
    return redirect(url_for('login'))
 
@app.route("/health")
@limit_requests(30, 60)
def health():
    output_parent = OUTPUT_DIR
    while not os.path.exists(output_parent):
        next_parent = os.path.dirname(output_parent)
        if next_parent == output_parent:
            break
        output_parent = next_parent
    checks = {
        "model": bool(model is not None and int(model.output_shape[-1]) == NUM_CLASSES),
        "face_detector": not FACE_CASCADE.empty(),
        "database": False,
        "private_output_storage": bool(
            os.path.isdir(output_parent) and os.access(output_parent, os.W_OK)
        ),
    }
    try:
        row = query_db_one("SELECT 1 AS healthy")
        checks["database"] = bool(row and row["healthy"] == 1)
    except Exception:
        logger.exception("Database health check failed.")

    healthy = all(checks.values())
    payload = {"status": "ok" if healthy else "degraded"}
    if env_bool("HEALTH_DETAILS", False) or session.get("role") == "admin":
        payload["checks"] = checks
    return jsonify(payload), 200 if healthy else 503


def detection_select_fields():
    faces_field = (
        "o.faces_detected"
        if "faces_detected" in get_output_columns()
        else "NULL"
    )
    return (
        "o.output_id, o.name AS emotion, o.confidence, "
        f"{faces_field} AS faces_detected, o.date AS detected_at"
    )


@app.route("/api/history")
@auth_login_required
@limit_requests(120, 60)
def api_history():
    try:
        limit = parse_bounded_int(
            request.args.get("limit"),
            default=100,
            minimum=1,
            maximum=200,
            field_name="limit",
        )
        rows = query_db(
            f"""
            SELECT {detection_select_fields()}
            FROM output o
            JOIN input i ON i.input_id = o.input_id
            WHERE i.user_id = %s
            ORDER BY o.date DESC, o.output_id DESC
            LIMIT %s
            """,
            (session["user_id"], limit),
        )
        return jsonify({
            "ok": True,
            "records": [serialize_detection_record(row) for row in rows],
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        logger.exception("Failed to load detection history for user %s.", session["user_id"])
        return jsonify({
            "ok": False,
            "error": "Detection history could not be loaded.",
        }), 503


@app.route("/api/analytics")
@auth_login_required
@limit_requests(120, 60)
def api_analytics():
    try:
        summary = query_db_one(
            """
            SELECT COUNT(*) AS total, AVG(o.confidence) AS average_confidence
            FROM output o
            JOIN input i ON i.input_id = o.input_id
            WHERE i.user_id = %s
            """,
            (session["user_id"],),
        )
        counts = query_db(
            """
            SELECT o.name AS emotion, COUNT(*) AS count
            FROM output o
            JOIN input i ON i.input_id = o.input_id
            WHERE i.user_id = %s
            GROUP BY o.name
            """,
            (session["user_id"],),
        )
        latest_row = query_db_one(
            f"""
            SELECT {detection_select_fields()}
            FROM output o
            JOIN input i ON i.input_id = o.input_id
            WHERE i.user_id = %s
            ORDER BY o.date DESC, o.output_id DESC
            LIMIT 1
            """,
            (session["user_id"],),
        )
        emotion_counts = {emotion: 0 for emotion in EMOTIONS}
        for row in counts:
            if row["emotion"] in emotion_counts:
                emotion_counts[row["emotion"]] = int(row["count"])
        return jsonify({
            "ok": True,
            "total": int(summary["total"] if summary else 0),
            "average_confidence": (
                float(summary["average_confidence"])
                if summary and summary["average_confidence"] is not None
                else None
            ),
            "emotion_counts": emotion_counts,
            "latest": (
                serialize_detection_record(latest_row)
                if latest_row
                else None
            ),
        })
    except Exception:
        logger.exception("Failed to load detection analytics for user %s.", session["user_id"])
        return jsonify({
            "ok": False,
            "error": "Detection analytics could not be loaded.",
        }), 503


@app.route("/api/admin/detections")
@auth_admin_required
@limit_requests(120, 60)
def api_admin_detections():
    try:
        limit = parse_bounded_int(
            request.args.get("limit"),
            default=100,
            minimum=1,
            maximum=200,
            field_name="limit",
        )
        rows = query_db(
            f"""
            SELECT {detection_select_fields()},
                   u.user_id, u.name AS user_name, u.email AS user_email
            FROM output o
            JOIN input i ON i.input_id = o.input_id
            JOIN user u ON u.user_id = i.user_id
            ORDER BY o.date DESC, o.output_id DESC
            LIMIT %s
            """,
            (limit,),
        )
        records = []
        for row in rows:
            record = serialize_detection_record(row)
            record["user"] = {
                "user_id": int(row["user_id"]),
                "name": row["user_name"],
                "email": row["user_email"],
            }
            records.append(record)
        return jsonify({"ok": True, "records": records})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        logger.exception("Failed to load administrator detection records.")
        return jsonify({
            "ok": False,
            "error": "Detection records could not be loaded.",
        }), 503
 
@app.route("/predict", methods=["POST"])
@auth_user_or_guest_required
@limit_requests(60, 60)
def predict():
    files = get_uploaded_files()
    if not files:
        return jsonify({
            "ok": False,
            "error": "Select at least one image using the 'file' field.",
        }), 400
    if len(files) > MAX_IMAGE_FILES:
        return jsonify({
            "ok": False,
            "error": f"A maximum of {MAX_IMAGE_FILES} images is allowed per request.",
        }), 400
 
    prob_accum = np.zeros((len(EMOTIONS),), dtype=np.float32)
    used = 0
    best_faces_from_last_frame = []
    best_summary_from_last_frame = None
 
    try:
        for file in files:
            validate_uploaded_file(file, IMAGE_EXTENSIONS, "image")
            image = read_uploaded_image(file)
            results, best = analyze_gray_frame(image)
            if best is None:
                continue
            prob_accum += best["_vec"]
            used += 1
            best_faces_from_last_frame = [
                {key: value for key, value in result.items() if key != "_vec"}
                for result in results
            ]
            best_summary_from_last_frame = {
                key: value for key, value in best.items() if key != "_vec"
            }
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
 
    if used == 0:
        return jsonify({
            "ok": False,
            "error": "No face could be detected in the submitted image frames.",
        }), 422
 
    avg_probs = prob_accum / float(used)

    top_emotion, confidence, probabilities = summarize_probabilities(
        avg_probs
    )

    record = None
    if session.get("role") == "user":
        try:
            emotion_number = int(np.argmax(avg_probs))
            record = save_detection_result(
                user_id=int(session["user_id"]),
                emotion_number=emotion_number,
                emotion=top_emotion,
                confidence=confidence,
                faces_detected=len(best_faces_from_last_frame),
            )
            logger.info(
                "Detection saved: user_id=%s, input_id=%s, output_id=%s, "
                "number=%s, emotion=%s, confidence=%.4f",
                session["user_id"],
                record["input_id"],
                record["output_id"],
                emotion_number,
                top_emotion,
                confidence,
            )
        except Exception:
            logger.exception("Failed to save detection for user %s.", session["user_id"])
            return jsonify({
                "ok": False,
                "error": "The detection succeeded but could not be saved. Please try again.",
            }), 503
    else:
        logger.info("Guest detection completed without server-side persistence.")

    return jsonify({
        "ok": True,
        "record_id": record["output_id"] if record else None,
        "saved_to_account": bool(record),
        "emotion":      top_emotion,
        "confidence":   confidence,
        "frames_used":  int(used),
        "faces_detected": int(len(best_faces_from_last_frame)),
        "faces":        best_faces_from_last_frame,
        "best_face":    best_summary_from_last_frame,
        "probabilities": probabilities,
    })
 
@app.route("/predict-video", methods=["POST"])
@auth_login_required
@limit_requests(5, 60)
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No video file. Send multipart/form-data with key 'file'."}), 400
 
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected video file."}), 400

    try:
        suffix = validate_uploaded_file(file, VIDEO_EXTENSIONS, "video")
        max_frames_to_process = parse_bounded_int(
            request.form.get("max_frames"),
            default=min(30, MAX_VIDEO_FRAMES),
            minimum=1,
            maximum=MAX_VIDEO_FRAMES,
            field_name="max_frames",
        )
        frame_skip = parse_bounded_int(
            request.form.get("frame_skip"),
            default=min(10, MAX_FRAME_SKIP),
            minimum=1,
            maximum=MAX_FRAME_SKIP,
            field_name="frame_skip",
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
 
    temp_path = None
    cap       = None
 
    try:
        temp_path = save_uploaded_temp_file(file, suffix)
 
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open uploaded video."}), 400
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            width <= 0
            or height <= 0
            or width * height > MAX_VIDEO_PIXELS
        ):
            return jsonify({"error": "Uploaded video dimensions are not supported."}), 400
 
        prob_accum       = np.zeros((len(EMOTIONS),), dtype=np.float32)
        used_frames      = 0
        total_frames_read = 0
        frame_results    = []
        frame_read_limit = min(
            max_frames_to_process * frame_skip,
            MAX_VIDEO_READ_FRAMES,
        )
 
        while total_frames_read < frame_read_limit:
            success, frame = cap.read()
            if not success: break
            total_frames_read += 1
            if total_frames_read % frame_skip != 0: continue
 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results, best = analyze_gray_frame(gray)
            if best is None: continue
 
            prob_accum += best["_vec"]
            used_frames += 1
            frame_results.append({
                "frame_number": int(total_frames_read),
                "emotion":      best["emotion"],
                "confidence":   float(best["confidence"]),
                "probabilities": best["probabilities"],
                "faces_detected": len(results),
                "faces": [{k: v for k, v in r.items() if k != "_vec"} for r in results]
            })
            if used_frames >= max_frames_to_process: break
 
        if used_frames == 0:
            return jsonify({"error": "No usable frames were processed from the video."}), 400
 
        avg_probs = prob_accum / float(used_frames)

        top_emotion, confidence, probabilities = summarize_probabilities(
            avg_probs
        )

        emotion_number = int(np.argmax(avg_probs))
        latest_faces = (
            int(frame_results[-1]["faces_detected"])
            if frame_results
            else 0
        )
        record = save_detection_result(
            user_id=int(session["user_id"]),
            emotion_number=emotion_number,
            emotion=top_emotion,
            confidence=confidence,
            faces_detected=latest_faces,
        )
 
        return jsonify({
            "ok":                   True,
            "record_id":            record["output_id"],
            "type":                 "video",
            "emotion":              top_emotion,
            "confidence":           confidence,
            "frames_used":          int(used_frames),
            "total_frames_read":    int(total_frames_read),
            "frame_skip":           int(frame_skip),
            "max_frames_requested": int(max_frames_to_process),
            "probabilities":        probabilities,
            "frame_results":        frame_results
        })
 
    except Exception:
        logger.exception("Video prediction failed for user %s.", session["user_id"])
        return jsonify({
            "ok": False,
            "error": "Video prediction failed. Please verify the file and try again.",
        }), 500
 
    finally:
        if cap is not None: cap.release()
        cleanup_temp_file(temp_path)
 
@app.route("/predict-video-annotated", methods=["POST"])
@auth_login_required
@limit_requests(2, 60)
def predict_video_annotated():
    if "file" not in request.files:
        return jsonify({"error": "No video file. Send multipart/form-data with key 'file'."}), 400
 
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected video file."}), 400

    try:
        suffix = validate_uploaded_file(file, VIDEO_EXTENSIONS, "video")
        max_frames_to_process = parse_bounded_int(
            request.form.get("max_frames"),
            default=MAX_VIDEO_FRAMES,
            minimum=1,
            maximum=MAX_VIDEO_FRAMES,
            field_name="max_frames",
        )
        frame_skip = parse_bounded_int(
            request.form.get("frame_skip"),
            default=min(5, MAX_FRAME_SKIP),
            minimum=1,
            maximum=MAX_FRAME_SKIP,
            field_name="frame_skip",
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
 
    temp_path = None
    cap       = None
    writer    = None
    output_path = None
    keep_output = False
 
    try:
        temp_path = save_uploaded_temp_file(file, suffix)
 
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open uploaded video."}), 400
 
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24)
        if not np.isfinite(fps) or fps <= 0 or fps > 120:
            fps = 24.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return jsonify({"error": "Uploaded video has invalid dimensions."}), 400
        if width * height > MAX_VIDEO_PIXELS:
            return jsonify({"error": "Uploaded video dimensions are too large."}), 400
 
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = (
            f"user_{int(session['user_id'])}_annotated_{uuid.uuid4().hex}.mp4"
        )
        output_path = os.path.join(OUTPUT_DIR, output_filename)
 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("The annotated video writer could not be initialized.")
 
        prob_accum        = np.zeros((len(EMOTIONS),), dtype=np.float32)
        used_frames       = 0
        total_frames_read = 0
        last_faces        = []
        frame_read_limit = min(
            max_frames_to_process * frame_skip,
            MAX_VIDEO_READ_FRAMES,
        )
 
        while total_frames_read < frame_read_limit:
            success, frame = cap.read()
            if not success: break
            total_frames_read += 1
            should_predict = total_frames_read % frame_skip == 0
 
            if should_predict and used_frames < max_frames_to_process:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                results, best = analyze_gray_frame(gray)
                if best is not None:
                    prob_accum += best["_vec"]
                    used_frames += 1
                    last_faces = results
 
            for face in last_faces:
                box        = face.get("box")
                emotion    = face.get("emotion", "Unknown")
                confidence = float(face.get("confidence", 0)) * 100
                label      = f"{emotion} ({confidence:.1f}%)"
 
                if box is not None:
                    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.putText(frame, label, (x, max(30, y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, label, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
 
            writer.write(frame)
            if used_frames >= max_frames_to_process: break
 
        if used_frames == 0:
            return jsonify({"error": "No usable frames were processed from the video."}), 400
 
        avg_probs     = prob_accum / float(used_frames)

        top_emotion, confidence, probabilities = summarize_probabilities(
            avg_probs
        )

        emotion_number = int(np.argmax(avg_probs))
        record = save_detection_result(
            user_id=int(session["user_id"]),
            emotion_number=emotion_number,
            emotion=top_emotion,
            confidence=confidence,
            faces_detected=len(last_faces),
        )
        keep_output = True
 
        return jsonify({
            "ok":                True,
            "record_id":         record["output_id"],
            "type":              "annotated_video",
            "emotion":           top_emotion,
            "confidence":        confidence,
            "frames_used":       int(used_frames),
            "total_frames_read": int(total_frames_read),
            "probabilities":     probabilities,
            "video_url":         url_for(
                "annotated_video_file",
                filename=output_filename,
            ),
        })
 
    except Exception:
        logger.exception(
            "Annotated video prediction failed for user %s.",
            session["user_id"],
        )
        return jsonify({
            "ok": False,
            "error": "Annotated video prediction failed. Please try again.",
        }), 500
 
    finally:
        if cap is not None:    cap.release()
        if writer is not None: writer.release()
        cleanup_temp_file(temp_path)
        if output_path and not keep_output:
            cleanup_temp_file(output_path)


@app.route("/media/annotated/<path:filename>")
@auth_login_required
def annotated_video_file(filename):
    expected_prefix = f"user_{int(session['user_id'])}_annotated_"
    if (
        not filename.startswith(expected_prefix)
        or not re.fullmatch(
            rf"{re.escape(expected_prefix)}[0-9a-f]{{32}}\.mp4",
            filename,
        )
    ):
        return jsonify({"ok": False, "error": "Video not found."}), 404
    if not os.path.isfile(os.path.join(OUTPUT_DIR, filename)):
        return jsonify({"ok": False, "error": "Video not found."}), 404
    response = send_from_directory(
        OUTPUT_DIR,
        filename,
        conditional=True,
        mimetype="video/mp4",
    )
    response.headers["Cache-Control"] = "private, no-store"
    return response
 
 
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = env_int("PORT", 5000, minimum=1, maximum=65535)
    waitress_threads = env_int("WAITRESS_THREADS", 4, minimum=1, maximum=32)
    logger.info(
        "Starting server on http://%s:%s (UI at /ui, threads=%s)",
        host,
        port,
        waitress_threads,
    )
    serve(app, host=host, port=port, threads=waitress_threads)
