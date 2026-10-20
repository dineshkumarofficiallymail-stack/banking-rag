from tests.conftest import auth, integration

pytestmark = integration


async def test_agent_calls_search_tool_and_cites(client, user_token, seeded_corpus):
    r = await client.post(
        "/api/v1/chat/agent", json={"question": "How often must KYC be updated for low risk customers?"}, headers=auth(user_token)
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "search_regulations" in body["tool_calls"]
    assert body["grounded"] and body["citations"]
    assert "[1]" in body["answer"]


async def test_agent_handles_empty_corpus(client, user_token):
    r = await client.post("/api/v1/chat/agent", json={"question": "What is the NACH return window?"}, headers=auth(user_token))
    assert r.status_code == 200 and r.json()["answer"].startswith("I could not find this")
