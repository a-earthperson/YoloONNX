import os
import unittest

from yolo_frigate.label import parse_labels

TEST_DIR = os.path.dirname(__file__)


class TestParseLabels(unittest.TestCase):
    def test_parse_labels_yaml(self):
        labels_dict = parse_labels(os.path.join(TEST_DIR, "labelmap.yml"))
        self.assertEqual(len(labels_dict), 3)
        self.assertEqual(labels_dict[1], "Label One")
        self.assertEqual(labels_dict[2], "Label Two")
        self.assertEqual(labels_dict[3], "Label Three")

    def test_parse_labels_text(self):
        labels_dict = parse_labels(os.path.join(TEST_DIR, "labelmap.txt"))
        self.assertEqual(len(labels_dict), 3)
        self.assertEqual(labels_dict[1], "Label One")
        self.assertEqual(labels_dict[2], "Label Two")
        self.assertEqual(labels_dict[3], "Label Three")


if __name__ == "__main__":
    unittest.main()
