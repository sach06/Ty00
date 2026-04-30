import json
import time
from datetime import datetime, timezone

from capital_client import CapitalClient

STATE_FILE = "bot_state.json"
SYNC_SECS = 5


def iso_age_hours(created_at: str | None) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - created).total_seconds() / 3600, 2)
    except ValueError:
        return 0.0


def fetch_live_positions(client: CapitalClient) -> list[dict]:
    payload = client.get_positions() or {}
    output = []
    for entry in payload.get("positions", []):
        position = entry.get("position", {})
        market = entry.get("market", {})
        created_at = position.get("createdDateUTC")
        output.append({
            "deal_id": position.get("dealId"),
            "epic": market.get("epic") or market.get("symbol") or "UNKNOWN",
            "direction": position.get("direction", "—"),
            "entry": float(position.get("level") or 0.0),
            "stop": None,
            "tp": None,
            "size": float(position.get("size") or 0.0),
            "opened_at": created_at,
            "age_hours": iso_age_hours(created_at),
            "dry_run": False,
            "source": "capital-account",
            "upl": float(position.get("upl") or 0.0),
        })
    return output


def main():
    client = CapitalClient()
    if not client.authenticate():
        raise SystemExit("Capital auth failed")

    while True:
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            time.sleep(SYNC_SECS)
            continue

        if not state.get("open_positions"):
            positions = fetch_live_positions(client)
            if positions:
                state["open_positions"] = positions
                state["account_open_positions"] = positions
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)

        time.sleep(SYNC_SECS)


if __name__ == "__main__":
    main()
