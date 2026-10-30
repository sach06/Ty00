import unittest

from main import _sync_positions


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def get_positions(self):
        return self.payload


class PositionReconciliationTests(unittest.TestCase):
    def test_unavailable_snapshot_retains_tracked_position_and_fails_closed(self):
        tracked = {"EURUSD": {"deal_id": "deal-1", "dry_run": False}}

        updated, closed, open_epics, synced = _sync_positions(StubClient(None), tracked)

        self.assertIs(updated, tracked)
        self.assertEqual(closed, [])
        self.assertEqual(open_epics, set())
        self.assertFalse(synced)

    def test_valid_snapshot_reports_open_epics_and_keeps_matching_position(self):
        tracked = {"EURUSD": {"deal_id": "deal-1", "dry_run": False}}
        payload = {
            "positions": [
                {"position": {"dealId": "deal-1"}, "market": {"epic": "EURUSD"}},
                {"position": {"dealId": "external-2"}, "market": {"epic": "GBPUSD"}},
            ],
        }

        updated, closed, open_epics, synced = _sync_positions(StubClient(payload), tracked)

        self.assertIn("EURUSD", updated)
        self.assertEqual(closed, [])
        self.assertEqual(open_epics, {"EURUSD", "GBPUSD"})
        self.assertTrue(synced)

    def test_confirmed_empty_snapshot_closes_missing_live_position(self):
        tracked = {"EURUSD": {"deal_id": "deal-1", "dry_run": False}}

        updated, closed, open_epics, synced = _sync_positions(
            StubClient({"positions": []}), tracked,
        )

        self.assertEqual(updated, {})
        self.assertEqual(closed, [("EURUSD", {"deal_id": "deal-1", "dry_run": False})])
        self.assertEqual(open_epics, set())
        self.assertTrue(synced)


if __name__ == "__main__":
    unittest.main()