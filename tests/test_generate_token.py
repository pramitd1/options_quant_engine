from __future__ import annotations

from config.generate_token import _extract_request_token


def test_extract_request_token_from_redirect_url():
    redirect_url = (
        "http://127.0.0.1:8000/?action=login&type=login&status=success"
        "&request_token=UfbpmemAwoexZza0yhog5VIaVvwBnJ47"
    )

    assert _extract_request_token(redirect_url) == "UfbpmemAwoexZza0yhog5VIaVvwBnJ47"


def test_extract_request_token_returns_none_when_missing():
    redirect_url = "http://127.0.0.1:8000/?action=login&type=login&status=success"

    assert _extract_request_token(redirect_url) is None