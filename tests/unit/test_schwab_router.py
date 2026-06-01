from maverick_mcp.api.routers.schwab import _extract_code


def test_extract_code_accepts_raw_code():
    assert _extract_code("abc123") == "abc123"


def test_extract_code_accepts_callback_url():
    url = "https://127.0.0.1:8765/callback?code=abc123&state=unused"
    assert _extract_code(url) == "abc123"
