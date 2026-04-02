import unittest

from yolo_frigate.confidence_evaluator import ConfidenceEvaluator


class TestConfidenceEvaluator(unittest.TestCase):
    def test_create(self):
        evaluator = ConfidenceEvaluator("deer:0.75,person:0.60-0.75,0.80")
        self.assertEqual(evaluator.rules["deer"], (0.75, 1))
        self.assertEqual(evaluator.rules["person"], (0.60, 0.75))
        self.assertEqual(evaluator.rules["default"], 0.80)

    def test_evaluate(self):
        evaluator = ConfidenceEvaluator("deer:0.75,person:0.60-0.75,0.80")
        self.assertEqual(evaluator.evaluate("deer", 0.75), True)
        self.assertEqual(evaluator.evaluate("deer", 1), True)
        self.assertEqual(evaluator.evaluate("person", 0.59), False)
        self.assertEqual(evaluator.evaluate("person", 0.60), True)
        self.assertEqual(evaluator.evaluate("person", 0.65), True)
        self.assertEqual(evaluator.evaluate("person", 0.75), True)
        self.assertEqual(evaluator.evaluate("person", 0.80), False)
        self.assertEqual(evaluator.evaluate("unknown", 0.80), True)
        self.assertEqual(evaluator.evaluate("unknown", 0.50), False)


if __name__ == "__main__":
    unittest.main()
