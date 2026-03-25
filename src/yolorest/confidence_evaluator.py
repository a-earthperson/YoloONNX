from __future__ import annotations


class ConfidenceEvaluator:
    def __init__(self, expression: str):
        self.rules = self._parse_expression(expression)

    def _parse_expression(
        self, expression: str
    ) -> dict[str, tuple[float, float] | float]:
        rules = {}
        for rule in expression.split(","):
            if ":" in rule:
                label, confidence_range = rule.split(":")
                if "-" in confidence_range:
                    min_conf, max_conf = map(float, confidence_range.split("-"))
                    rules[label] = (min_conf, max_conf)
                else:
                    rules[label] = (float(confidence_range), 1)
            else:
                rules["default"] = float(rule)
        return rules

    def evaluate(self, label: str, confidence: float) -> bool:
        if label in self.rules:
            min_conf, max_conf = self.rules[label]
            return min_conf <= confidence <= max_conf
        elif "default" in self.rules:
            return confidence >= self.rules["default"]
        return False
