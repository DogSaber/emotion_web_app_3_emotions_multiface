import functools
from flask import jsonify, redirect, request, session, url_for


def auth_get_user_by_email(query_fn, email):
    return query_fn(
        """
        SELECT user_id, name, email, password, created_at
        FROM user
        WHERE email = %s
        """,
        (email,),
        dictionary=True,
    )


def auth_get_user_by_id(query_fn, user_id):
    return query_fn(
        """
        SELECT user_id, name, email, created_at
        FROM user
        WHERE user_id = %s
        """,
        (user_id,),
        dictionary=True,
    )


def auth_get_admin_by_username(query_fn, username):
    return query_fn(
        """
        SELECT admin_id, username, password, name
        FROM admin
        WHERE username = %s
        """,
        (username,),
        dictionary=True,
    )


def auth_get_all_users(query_fn):
    # Never pass password hashes into a template.
    return query_fn(
        """
        SELECT user_id, name, email, created_at
        FROM user
        ORDER BY created_at DESC
        """,
        dictionary=True,
    )


def auth_authenticate_admin(
    query_fn,
    bcrypt,
    username,
    password,
    dummy_password_hash=None,
):
    admin = auth_get_admin_by_username(query_fn, username)
    password_hash = (
        admin["password"]
        if admin
        else dummy_password_hash
    )
    password_matches = bool(
        password_hash
        and bcrypt.check_password_hash(password_hash, password)
    )
    if admin and password_matches:
        return admin
    return None


def auth_set_user_session(user):
    # Clearing the signed session prevents a previous role from surviving login.
    session.clear()
    session.permanent = True
    session["user_id"] = user["user_id"]
    session["user_name"] = user["name"]
    session["role"] = "user"


def auth_set_admin_session(admin):
    session.clear()
    session.permanent = True
    session["admin_id"] = admin["admin_id"]
    session["admin_name"] = admin["name"]
    session["role"] = "admin"


def auth_request_wants_json():
    """Return True for API-style requests that must not receive HTML redirects."""
    path = request.path
    return (
        request.is_json
        or path.startswith("/api/")
        or path.startswith("/predict")
        or path.endswith("-ajax")
        or path.endswith("/messages")
        or "/messages/" in path
        or path == "/admin/dashboard/stats"
        or request.accept_mimetypes.best == "application/json"
    )


def _unauthorized(login_endpoint, message):
    if auth_request_wants_json():
        return jsonify({"ok": False, "error": message}), 401
    return redirect(url_for(login_endpoint, next=request.full_path.rstrip("?")))


def auth_login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "user":
            return _unauthorized("login", "User authentication required.")
        return f(*args, **kwargs)

    return decorated


def auth_user_or_guest_required(f):
    """Allow the browser-only guest workflow only where explicitly applied."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") not in {"user", "guest"}:
            return _unauthorized("login", "User or guest access is required.")
        return f(*args, **kwargs)

    return decorated


def auth_admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "admin_id" not in session or session.get("role") != "admin":
            return _unauthorized("admin_login", "Administrator authentication required.")
        return f(*args, **kwargs)

    return decorated
