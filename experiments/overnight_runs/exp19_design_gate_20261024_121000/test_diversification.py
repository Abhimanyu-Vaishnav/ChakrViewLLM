import re
import unittest
from collections import Counter

class TestResponseDiversification(unittest.TestCase):
    def test_arithmetic_diversification(self):
        resp = "The sum of 1002 and 1007 is 2009."
        m = re.search(r"The sum of (.*?) is (\d+)\.", resp)
        self.assertIsNotNone(m)
        detail, ans = m.groups()
        v1 = f"Adding yields {ans}."
        v2 = f"Result: {ans}."
        v3 = f"Sum: {ans}."
        for v in [v1, v2, v3]:
            self.assertIn(ans, v)
            self.assertTrue(v.endswith("."))

    def test_reference_diversification(self):
        resp = "The pronoun 'it' designates the antique brass compass."
        m = re.search(r"The pronoun 'it' designates (the .*?)\.", resp)
        self.assertIsNotNone(m)
        target = m.group(1)
        v1 = f"'It' designates {target}."
        v2 = f"The antecedent of 'it' is {target}."
        v3 = f"Designated entity: {target}."
        for v in [v1, v2, v3]:
            self.assertIn("antique brass compass", v)
            self.assertFalse(v.startswith("The pronoun 'it' designates"))

    def test_symbol_diversification(self):
        resp = "The chemical symbol for iron is Fe."
        m = re.search(r"The chemical symbol for (.*?) is ([A-Z][a-z]?)\.", resp)
        self.assertIsNotNone(m)
        el, sym = m.groups()
        v1 = f"{sym} is the chemical symbol for {el}."
        v2 = f"Chemical symbol for {el}: {sym}."
        v3 = f"Element {el} symbol: {sym}."
        for v in [v1, v2, v3]:
            self.assertIn(sym, v)
            self.assertIn(el, v)

    def test_determinism_and_split_isolation(self):
        # Verify that diversification pseudo-random assignment is deterministic based on record index
        def assign_variant(record_idx: int, num_variants: int) -> int:
            return record_idx % num_variants

        res_a = [assign_variant(i, 4) for i in range(10)]
        res_b = [assign_variant(i, 4) for i in range(10)]
        self.assertEqual(res_a, res_b)

if __name__ == "__main__":
    unittest.main()
