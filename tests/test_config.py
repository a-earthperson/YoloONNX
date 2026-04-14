import unittest

from yolo_frigate.config import (
    DEFAULT_EXPORT_CALIBRATION_MAX_SAMPLES,
    MAX_EXPORT_CALIBRATION_MAX_SAMPLES,
    MIN_EXPORT_CALIBRATION_MAX_SAMPLES,
    parse_args,
)


class TestConfig(unittest.TestCase):
    def test_parse_args_defaults_calibration_sample_cap(self):
        config = parse_args(["--model_file", "model.pt"])

        self.assertEqual(
            config.export_calibration_max_samples,
            DEFAULT_EXPORT_CALIBRATION_MAX_SAMPLES,
        )

    def test_parse_args_accepts_calibration_sample_cap_override(self):
        config = parse_args(
            [
                "--model_file",
                "model.pt",
                "--export_calibration_max_samples",
                "1024",
            ]
        )

        self.assertEqual(config.export_calibration_max_samples, 1024)

    def test_parse_args_rejects_out_of_range_calibration_sample_cap(self):
        for invalid in (
            MIN_EXPORT_CALIBRATION_MAX_SAMPLES - 1,
            MAX_EXPORT_CALIBRATION_MAX_SAMPLES + 1,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SystemExit):
                    parse_args(
                        [
                            "--model_file",
                            "model.pt",
                            "--export_calibration_max_samples",
                            str(invalid),
                        ]
                    )
