"""
End-to-end backend tests for Guest Mode, session expiration, permission enforcement,
and registered user regression checks. Uses an isolated in-memory SQLite database
and a zero-dependency ASGI harness against the real FastAPI app.
"""

import asyncio
from datetime import timedelta
import json
import os
import sys
from typing import Any, Optional
from urllib.parse import urlsplit

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-guest-mode-tests")

from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from auth.utils import create_guest_token
from db.config import get_session
from db.models import User, UserAuth
from main import app


class AsgiResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


def asgi_request(
    method: str,
    url: str,
    json_body: Optional[dict] = None,
    headers: Optional[dict[str, str]] = None,
) -> AsgiResponse:
    parts = urlsplit(url)
    body_bytes = json.dumps(json_body).encode("utf-8") if json_body is not None else b""

    raw_headers = [(b"host", b"testserver")]
    if json_body is not None:
        raw_headers.append((b"content-type", b"application/json"))
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": parts.path,
        "raw_path": parts.path.encode("latin-1"),
        "query_string": parts.query.encode("latin-1"),
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    sent_body = False
    status_code = 500
    response_chunks: list[bytes] = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_chunks.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return AsgiResponse(status_code, b"".join(response_chunks))


def setup_isolated_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_get_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    return engine


def run_all_tests():
    engine = setup_isolated_db()

    # 1. Guest session creation does NOT create any DB account
    res = asgi_request("POST", "/api/auth/guest")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    guest_data = res.json()
    assert guest_data["role"] == "guest"
    assert guest_data["guest_id"].startswith("guest_")
    assert guest_data["access_token"]
    assert guest_data["expires_at"]
    guest_headers = {"Authorization": f"Bearer {guest_data['access_token']}"}

    with Session(engine) as session:
        assert len(session.exec(select(User)).all()) == 0, "Guest must not create a User row"
        assert len(session.exec(select(UserAuth)).all()) == 0, "Guest must not create a UserAuth row"
    print("PASS: Guest token issued without creating DB account")

    # 2. Session validation (valid guest, expired guest, forged token, missing token)
    res = asgi_request("GET", "/api/auth/session", headers=guest_headers)
    assert res.status_code == 200 and res.json()["role"] == "guest"

    res = asgi_request("GET", "/api/auth/session")
    assert res.status_code == 401

    expired_token, _, _ = create_guest_token(expires_delta=timedelta(seconds=-10))
    res = asgi_request(
        "GET", "/api/auth/session", headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert res.status_code == 401, "Expired guest token must be rejected with 401"

    res = asgi_request(
        "GET", "/api/auth/session", headers={"Authorization": "Bearer forged.jwt.token"}
    )
    assert res.status_code == 401, "Forged token must be rejected with 401"
    print("PASS: Session validation & expiration enforcement")

    # 3. Permitted interactive endpoints work for guests
    res = asgi_request("POST", "/evaluate", json_body={"expression": "a&b"}, headers=guest_headers)
    assert res.status_code == 200 and "headers" in res.json()

    res = asgi_request(
        "POST",
        "/is_equivalent",
        json_body={"expression1": "a&b", "expression2": "b&a"},
        headers=guest_headers,
    )
    assert res.status_code == 200 and res.json()["equivalent"] is True

    res = asgi_request(
        "POST",
        "/cluster",
        json_body={
            "points": [[0, 0], [0.5, 0.5], [5, 5], [5.5, 5.5]],
            "algorithm": "kmeans",
            "k": 2,
        },
        headers=guest_headers,
    )
    assert res.status_code == 200 and len(res.json()["labels"]) == 4
    print("PASS: Guests can access permitted interactive APIs")

    # 4. Restricted endpoints are rejected with 403 for guests and 401 for unauthenticated
    restricted_calls = [
        ("POST", "/api/saved_visual?user_id=1", {"saved_visual": {"nodes": [], "edges": []}, "type": "g1"}),
        ("GET", "/api/saved_visuals/user/1", None),
        ("GET", "/api/saved_visuals", None),
        ("DELETE", "/api/saved_visual/1", None),
        ("POST", "/api/label_correction?user_id=1", {"image_path": "a.png", "corrections": {}, "predicted_labels": {}}),
        ("GET", "/api/user/1", None),
        ("PUT", "/api/user/1", {"username": "hacked"}),
        ("DELETE", "/api/user/1", None),
        ("POST", "/api/auth/change-password?user_id=1", {"current_password": "old", "new_password": "newpassword"}),
        ("GET", "/api/user_posts/user/1", None),
    ]
    for method, url, payload in restricted_calls:
        guest_res = asgi_request(method, url, json_body=payload, headers=guest_headers)
        assert guest_res.status_code == 403, f"Guest {method} {url} expected 403, got {guest_res.status_code}: {guest_res.text}"

        unauth_res = asgi_request(method, url, json_body=payload)
        assert unauth_res.status_code == 401, f"Unauth {method} {url} expected 401, got {unauth_res.status_code}: {unauth_res.text}"
    print("PASS: All account-specific endpoints reject guests (403) and unauthenticated callers (401)")

    # 5. Registered user regression & cross-user data isolation
    reg_a = asgi_request(
        "POST",
        "/api/auth/register",
        json_body={
            "username": "alice",
            "email": "alice@example.com",
            "password": "password123",
            "preferred_theme": "amber",
        },
    )
    assert reg_a.status_code == 200, f"Register A failed: {reg_a.text}"
    alice = reg_a.json()
    assert alice["role"] == "user"
    alice_headers = {"Authorization": f"Bearer {alice['access_token']}"}

    login_a = asgi_request(
        "POST",
        "/api/auth/login",
        json_body={"email": "alice@example.com", "password": "password123"},
    )
    assert login_a.status_code == 200 and login_a.json()["role"] == "user"

    save_res = asgi_request(
        "POST",
        f"/api/saved_visual?user_id={alice['user_id']}",
        json_body={"saved_visual": {"nodes": [{"id": 1, "label": "A", "pos": [0, 0], "neighbors": []}], "edges": []}, "type": "MyGraph"},
        headers=alice_headers,
    )
    assert save_res.status_code == 200, f"Alice save failed: {save_res.text}"
    visual_id = save_res.json()["id"]

    list_res = asgi_request(
        "GET",
        f"/api/saved_visuals/user/{alice['user_id']}",
        headers=alice_headers,
    )
    assert list_res.status_code == 200 and len(list_res.json()) == 1

    # Register Bob and verify Bob CANNOT access or delete Alice's saved visuals
    reg_b = asgi_request(
        "POST",
        "/api/auth/register",
        json_body={
            "username": "bob",
            "email": "bob@example.com",
            "password": "password123",
            "preferred_theme": "blue",
        },
    )
    assert reg_b.status_code == 200
    bob = reg_b.json()
    bob_headers = {"Authorization": f"Bearer {bob['access_token']}"}

    assert asgi_request("GET", f"/api/saved_visuals/user/{alice['user_id']}", headers=bob_headers).status_code == 403
    assert asgi_request("DELETE", f"/api/saved_visual/{visual_id}", headers=bob_headers).status_code == 403
    assert (
        asgi_request(
            "POST",
            f"/api/auth/change-password?user_id={alice['user_id']}",
            json_body={"current_password": "password123", "new_password": "hacked123"},
            headers=bob_headers,
        ).status_code
        == 403
    )

    assert asgi_request("DELETE", f"/api/saved_visual/{visual_id}", headers=alice_headers).status_code == 200
    print("PASS: Registered user flows & cross-user data isolation")

    app.dependency_overrides.clear()
    print("\nALL BACKEND GUEST MODE TESTS PASSED!")


if __name__ == "__main__":
    run_all_tests()
