"""
Module: generate_token.py

Purpose:
    Define configuration values used by generate token.

Role in the System:
    Part of the configuration layer that centralizes policy defaults, thresholds, and governance controls.

Key Outputs:
    Configuration objects and threshold bundles consumed by runtime and research workflows.

Downstream Usage:
    Consumed by analytics, signal generation, strategy, risk overlays, tuning, and backtests.
"""
import argparse
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect


def _resolve_value(cli_value: str | None, env_key: str, default: str) -> str:
    if cli_value:
        return cli_value.strip()
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        return env_val
    return default


def _extract_request_token(redirect_url: str | None) -> str | None:
    if not redirect_url:
        return None

    parsed = urlparse(redirect_url.strip())
    request_tokens = parse_qs(parsed.query).get("request_token")
    if not request_tokens:
        return None

    request_token = request_tokens[0].strip()
    return request_token or None


def _upsert_env_key(env_path: Path, key: str, value: str) -> None:
    """Insert or update KEY=VALUE in a dotenv file while preserving other lines."""
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    target_prefix = f"{key}="
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(target_prefix):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Zerodha access token from request token")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--api-secret", dest="api_secret", default=None)
    parser.add_argument("--request-token", dest="request_token", default=None)
    parser.add_argument(
        "--redirect-url",
        dest="redirect_url",
        default=None,
        help="Full Zerodha login redirect URL containing request_token",
    )
    parser.add_argument(
        "--env-file",
        dest="env_file",
        default=None,
        help="Path to .env file to update (default: project_root/.env)",
    )
    args = parser.parse_args()

    api_key = _resolve_value(args.api_key, "ZERODHA_API_KEY", "YOUR_API_KEY")
    api_secret = _resolve_value(args.api_secret, "ZERODHA_API_SECRET", "YOUR_API_SECRET")
    request_token = args.request_token.strip() if args.request_token else None
    if not request_token and args.redirect_url:
        request_token = _extract_request_token(args.redirect_url)
        if not request_token:
            print("Redirect URL does not contain a valid request_token query parameter.")
            return 1

    request_token = _resolve_value(request_token, "ZERODHA_REQUEST_TOKEN", "PASTE_REQUEST_TOKEN")

    if api_key.startswith("YOUR_") or not api_key:
        print("Missing API key. Set ZERODHA_API_KEY or pass --api-key.")
        return 1
    if api_secret.startswith("YOUR_") or not api_secret:
        print("Missing API secret. Set ZERODHA_API_SECRET or pass --api-secret.")
        return 1
    if request_token in {"PASTE_REQUEST_TOKEN", ""}:
        print("Missing request token. Set ZERODHA_REQUEST_TOKEN or pass --request-token or --redirect-url.")
        return 1

    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        if "invalid" in lowered or "expired" in lowered:
            print("Request token is invalid/expired. Generate a fresh request_token and retry immediately.")
        else:
            print(f"Token generation failed: {msg}")
        return 1

    print("ACCESS TOKEN:")
    access_token = str(data["access_token"])
    print(access_token)

    default_env = Path(__file__).resolve().parents[1] / ".env"
    env_path = Path(args.env_file).expanduser().resolve() if args.env_file else default_env
    try:
        _upsert_env_key(env_path, "ZERODHA_ACCESS_TOKEN", access_token)
        print(f"Updated {env_path} with ZERODHA_ACCESS_TOKEN")
    except Exception as exc:
        print(f"Warning: token generated but failed to update env file ({env_path}): {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())