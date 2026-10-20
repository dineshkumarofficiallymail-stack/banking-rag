import pytest

from tests.conftest import auth, integration

pytestmark = integration


async def test_register_login_me(client):
    r = await client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "Passw0rd!xyz"})
    assert r.status_code == 201 and r.json()["role"] == "analyst"
    r = await client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "Passw0rd!xyz"})
    assert r.status_code == 409
    r = await client.post("/api/v1/auth/login", data={"username": "a@example.com", "password": "wrong"})
    assert r.status_code == 401
    r = await client.post("/api/v1/auth/login", data={"username": "a@example.com", "password": "Passw0rd!xyz"})
    token = r.json()["access_token"]
    r = await client.get("/api/v1/auth/me", headers=auth(token))
    assert r.status_code == 200 and r.json()["email"] == "a@example.com"


async def test_protected_routes_reject_bad_tokens(client):
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers=auth("garbage"))).status_code == 401


async def test_ready_endpoint(client):
    r = await client.get("/api/v1/ready")
    assert r.status_code == 200 and r.json()["checks"]["postgres"] == "ok"


@pytest.mark.parametrize("pw", ["short", "x" * 73])
async def test_password_validation(client, pw):
    r = await client.post("/api/v1/auth/register", json={"email": "b@example.com", "password": pw})
    assert r.status_code == 422
