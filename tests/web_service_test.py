import unittest

from web import clamp_int, trim_series_json


class WebServiceHelpersTest(unittest.TestCase):
    def test_clamp_int_matches_original_bounds(self):
        self.assertEqual(clamp_int(None, 200, 10, 1000), 200)
        self.assertEqual(clamp_int("3", 200, 10, 1000), 10)
        self.assertEqual(clamp_int("5000", 200, 10, 1000), 1000)
        self.assertEqual(clamp_int("bad", 200, 10, 1000), 200)

    def test_trim_series_json_keeps_non_series_fields(self):
        data = {"time": [1, 2, 3], "yaw": [4, 5, 6], "metadata": {"mode": "AUTO_AIM"}}
        self.assertEqual(
            trim_series_json(data, 2),
            {"time": [2, 3], "yaw": [5, 6], "metadata": {"mode": "AUTO_AIM"}},
        )


if __name__ == "__main__":
    unittest.main()
