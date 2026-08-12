"""Focused, non-mutating tests for backend security and ownership behavior."""

from datetime import datetime
import io
import os
import unittest
from unittest.mock import patch


os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("SECRET_KEY", "backend-test-secret-key-that-is-long-enough")

import app as application  # noqa: E402


class _FakeCursor:
    def __init__(self):
        self.executions = []
        self.lastrowid = None

    def execute(self, statement, params=()):
        normalized = " ".join(statement.split()).lower()
        self.executions.append((normalized, params))
        if normalized.startswith("insert into input"):
            self.lastrowid = 101
        elif normalized.startswith("insert into output"):
            self.lastrowid = 202

    def close(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.cursor_value = _FakeCursor()
        self.started = False
        self.committed = False
        self.rolled_back = False

    def start_transaction(self):
        self.started = True

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        return None


class BackendSecurityTests(unittest.TestCase):
    def setUp(self):
        application.app.config.update(TESTING=True)
        application.rate_limiter._events.clear()
        self.client = application.app.test_client()

    def _set_user_session(self, csrf="test-csrf-token"):
        with self.client.session_transaction() as flask_session:
            flask_session["user_id"] = 42
            flask_session["user_name"] = "Ellen"
            flask_session["role"] = "user"
            flask_session[application.CSRF_SESSION_KEY] = csrf

    def _set_guest_session(self, csrf="guest-csrf-token"):
        with self.client.session_transaction() as flask_session:
            flask_session["role"] = "guest"
            flask_session["guest_mode"] = True
            flask_session[application.CSRF_SESSION_KEY] = csrf

    def test_api_auth_failure_is_json(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["ok"], False)

    def test_guest_dashboard_and_prediction_do_not_persist_to_database(self):
        guest_page = self.client.get("/guest")
        self.assertEqual(guest_page.status_code, 200)
        self.assertIn(b"Guest access", guest_page.data)

        self._set_guest_session()
        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(b"image"), "live_frame.jpg", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("CSRF", response.get_json()["error"])

        face = {
            "box": {"x": 1, "y": 2, "w": 20, "h": 20},
            "emotion": "Happy",
            "confidence": 0.9,
            "probabilities": {"Happy": 0.9},
            "_vec": application.np.array([0.9, 0.025, 0.025, 0.025, 0.025]),
        }
        with (
            patch.object(application, "validate_uploaded_file", return_value=".jpg"),
            patch.object(application, "read_uploaded_image", return_value=object()),
            patch.object(application, "analyze_gray_frame", return_value=([face], face)),
            patch.object(
                application,
                "summarize_probabilities",
                return_value=("Happy", 0.9, {"Happy": 0.9}),
            ),
            patch.object(application, "save_detection_result") as save_result,
        ):
            response = self.client.post(
                "/predict",
                data={"file": (io.BytesIO(b"image"), "live_frame.jpg", "image/jpeg")},
                headers={"X-CSRF-Token": "guest-csrf-token"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["saved_to_account"])
        self.assertIsNone(payload["record_id"])
        save_result.assert_not_called()

    def test_role_guards_redirect_pages_but_not_apis(self):
        page = self.client.get("/dashboard")
        self.assertEqual(page.status_code, 302)
        self.assertIn("/login", page.headers["Location"])

        admin_page = self.client.get("/admin/dashboard")
        self.assertEqual(admin_page.status_code, 302)
        self.assertIn("/admin/login", admin_page.headers["Location"])

        api = self.client.get("/api/history")
        self.assertEqual(api.status_code, 401)
        self.assertEqual(api.get_json()["ok"], False)

    def test_authenticated_post_requires_csrf_header(self):
        self._set_user_session()
        response = self.client.post(
            "/support/send-ajax",
            json={"message": "Hello"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("CSRF", response.get_json()["error"])

        with patch.object(application, "support_insert_message") as insert_message:
            response = self.client.post(
                "/support/send-ajax",
                json={"message": "Hello"},
                headers={"X-CSRF-Token": "test-csrf-token"},
            )
        self.assertEqual(response.status_code, 200)
        insert_message.assert_called_once()

    def test_logout_is_post_only_and_csrf_protected(self):
        self._set_user_session()
        self.assertEqual(self.client.get("/logout").status_code, 405)
        self.assertEqual(self.client.post("/logout").status_code, 400)
        response = self.client.post(
            "/logout",
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        self.assertEqual(response.status_code, 302)

    def test_history_is_filtered_and_has_stable_shape(self):
        self._set_user_session()
        row = {
            "output_id": 7,
            "emotion": "Happy",
            "confidence": 0.75,
            "faces_detected": None,
            "detected_at": datetime(2026, 7, 30, 8, 15, 0),
        }
        with (
            patch.object(application, "get_output_columns", return_value=set()),
            patch.object(application, "query_db", return_value=[row]) as query,
        ):
            response = self.client.get("/api/history")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["records"][0]["emotion"], "Happy")
        self.assertEqual(payload["records"][0]["confidence"], 0.75)
        self.assertEqual(query.call_args.args[1][0], 42)

    def test_detection_persistence_uses_new_owned_input(self):
        fake_db = _FakeConnection()
        with (
            patch.object(application, "get_output_columns", return_value=set()),
            patch.object(application, "get_db", return_value=fake_db),
        ):
            result = application.save_detection_result(
                user_id=42,
                emotion_number=0,
                emotion="Happy",
                confidence=0.8,
                faces_detected=1,
            )

        self.assertTrue(fake_db.started)
        self.assertTrue(fake_db.committed)
        self.assertFalse(fake_db.rolled_back)
        self.assertEqual(result, {"input_id": 101, "output_id": 202})
        input_insert = fake_db.cursor_value.executions[0]
        output_insert = fake_db.cursor_value.executions[1]
        self.assertEqual(input_insert[1], (42,))
        self.assertEqual(output_insert[1][0], 101)

    def test_security_headers_and_legacy_video_block(self):
        response = self.client.get("/login")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertEqual(
            self.client.get("/static/outputs/legacy.mp4").status_code,
            404,
        )

    def test_bounded_integer_validation(self):
        self.assertEqual(
            application.parse_bounded_int("30", 10, 1, 120, "max_frames"),
            30,
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 120"):
            application.parse_bounded_int("9999", 10, 1, 120, "max_frames")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            application.parse_bounded_int("not-a-number", 10, 1, 120, "max_frames")

    def test_login_rate_limit_is_enforced(self):
        self.client.get("/login")
        with self.client.session_transaction() as flask_session:
            token = flask_session[application.CSRF_SESSION_KEY]
        form = {
            "email": "nobody@example.com",
            "password": "not-the-password",
            "csrf_token": token,
        }
        with (
            patch.object(application, "auth_get_user_by_email", return_value=None),
            patch.object(application.bcrypt, "check_password_hash", return_value=False),
        ):
            responses = [self.client.post("/login", data=form) for _ in range(11)]
        self.assertTrue(all(response.status_code == 200 for response in responses[:10]))
        self.assertEqual(responses[-1].status_code, 429)
        self.assertIn("Retry-After", responses[-1].headers)

    def test_private_video_route_enforces_user_ownership(self):
        self._set_user_session()
        own_filename = "user_42_annotated_" + ("a" * 32) + ".mp4"
        other_filename = "user_99_annotated_" + ("b" * 32) + ".mp4"

        self.assertEqual(
            self.client.get(f"/media/annotated/{other_filename}").status_code,
            404,
        )
        with (
            patch.object(application.os.path, "isfile", return_value=True),
            patch.object(
                application,
                "send_from_directory",
                return_value=application.app.make_response(b"video"),
            ) as send_file,
        ):
            response = self.client.get(f"/media/annotated/{own_filename}")
        self.assertEqual(response.status_code, 200)
        send_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
