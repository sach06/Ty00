import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import server


class StateFreshnessTests(unittest.TestCase):
    def test_stale_state_is_offline_without_broker_lookup(self):
        stale_state = {
            "last_updated": "2026-05-13T07:46:54.535033+00:00",
            "dry_run": False,
        }

        with patch.object(server, "_load_json_file", return_value=stale_state), patch.object(
            server, "_fetch_live_positions",
        ) as fetch_positions:
            response = server.app.test_client().get("/api/state")

        self.assertEqual(response.status_code, 503)
        self.assertIn("stale", response.get_json()["error"])
        fetch_positions.assert_not_called()

    def test_fresh_state_is_served(self):
        fresh_state = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
            "open_positions": [],
        }

        with patch.object(server, "_load_json_file", return_value=fresh_state), patch.object(
            server, "_fetch_live_positions", return_value=[],
        ):
            response = server.app.test_client().get("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["dry_run"])


if __name__ == "__main__":
    unittest.main()