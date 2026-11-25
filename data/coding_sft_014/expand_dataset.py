"""
ChakrView Comprehensive Algorithmic Code Generator (Expansion to 600+ verified tasks).
Mission CV-GENESIS-NR-014 Phase 3.
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

def test_code(code: str, test: str) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"{code}\n\n{test}\n"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        return proc.returncode == 0
    except Exception:
        return False

def build_comprehensive_dataset():
    items = []
    idx = 0

    # 1. Math sequences and formulas
    # Area of shapes: circle, rectangle, triangle, square, trapezoid, cylinder, sphere
    shapes = [
        ("area_rectangle", "def area_rectangle(w: float, h: float) -> float:\n    return w * h\n", "assert area_rectangle(4, 5) == 20"),
        ("perimeter_rectangle", "def perimeter_rectangle(w: float, h: float) -> float:\n    return 2 * (w + h)\n", "assert perimeter_rectangle(4, 5) == 18"),
        ("area_triangle", "def area_triangle(b: float, h: float) -> float:\n    return 0.5 * b * h\n", "assert area_triangle(6, 4) == 12.0"),
        ("area_square", "def area_square(side: float) -> float:\n    return side * side\n", "assert area_square(7) == 49"),
        ("perimeter_square", "def perimeter_square(side: float) -> float:\n    return 4 * side\n", "assert perimeter_square(5) == 20"),
        ("area_trapezoid", "def area_trapezoid(a: float, b: float, h: float) -> float:\n    return 0.5 * (a + b) * h\n", "assert area_trapezoid(3, 5, 4) == 16.0"),
        ("volume_cube", "def volume_cube(s: float) -> float:\n    return s ** 3\n", "assert volume_cube(3) == 27"),
        ("volume_cuboid", "def volume_cuboid(l: float, w: float, h: float) -> float:\n    return l * w * h\n", "assert volume_cuboid(2, 3, 4) == 24"),
    ]
    for fn, code, test in shapes:
        if test_code(code, test):
            items.append({"topic": "geometry", "fn": fn, "prompt": f"Write a Python function `{fn}` to calculate geometric properties.", "code": code, "test": test})

    # 2. Financial & Business calculations
    finance = [
        ("simple_interest", "def simple_interest(p: float, r: float, t: float) -> float:\n    return (p * r * t) / 100.0\n", "assert simple_interest(1000, 5, 2) == 100.0"),
        ("total_with_interest", "def total_with_interest(p: float, r: float, t: float) -> float:\n    return p + (p * r * t) / 100.0\n", "assert total_with_interest(1000, 5, 2) == 1100.0"),
        ("calculate_discount", "def calculate_discount(price: float, discount_pct: float) -> float:\n    return price * (1 - discount_pct / 100.0)\n", "assert calculate_discount(200, 15) == 170.0"),
        ("calculate_tax", "def calculate_tax(amount: float, tax_rate: float) -> float:\n    return amount * (tax_rate / 100.0)\n", "assert calculate_tax(500, 18) == 90.0"),
        ("net_salary", "def net_salary(gross: float, deductions: float) -> float:\n    return max(0.0, gross - deductions)\n", "assert net_salary(50000, 8000) == 42000.0"),
    ]
    for fn, code, test in finance:
        if test_code(code, test):
            items.append({"topic": "finance", "fn": fn, "prompt": f"Write a Python function `{fn}` as specified.", "code": code, "test": test})

    # 3. Unit conversions
    units = [
        ("km_to_miles", "def km_to_miles(km: float) -> float:\n    return km * 0.621371\n", "assert abs(km_to_miles(10) - 6.21371) < 1e-4"),
        ("miles_to_km", "def miles_to_km(miles: float) -> float:\n    return miles * 1.60934\n", "assert abs(miles_to_km(5) - 8.0467) < 1e-4"),
        ("kg_to_pounds", "def kg_to_pounds(kg: float) -> float:\n    return kg * 2.20462\n", "assert abs(kg_to_pounds(10) - 22.0462) < 1e-4"),
        ("meters_to_feet", "def meters_to_feet(m: float) -> float:\n    return m * 3.28084\n", "assert abs(meters_to_feet(2) - 6.56168) < 1e-4"),
        ("hours_to_minutes", "def hours_to_minutes(h: float) -> float:\n    return h * 60.0\n", "assert hours_to_minutes(2.5) == 150.0"),
        ("minutes_to_seconds", "def minutes_to_seconds(m: float) -> float:\n    return m * 60.0\n", "assert minutes_to_seconds(3) == 180.0"),
        ("days_to_hours", "def days_to_hours(d: float) -> float:\n    return d * 24.0\n", "assert days_to_hours(3) == 72.0"),
    ]
    for fn, code, test in units:
        if test_code(code, test):
            items.append({"topic": "unit_conversion", "fn": fn, "prompt": f"Write a Python function `{fn}` to perform unit conversion.", "code": code, "test": test})

    # 4. Search and selection
    search_ops = [
        ("find_second_largest", "def find_second_largest(nums: list[int]) -> int:\n    unique = sorted(list(set(nums)), reverse=True)\n    if len(unique) < 2: raise ValueError()\n    return unique[1]\n", "assert find_second_largest([10, 20, 4, 45, 99]) == 45"),
        ("find_second_smallest", "def find_second_smallest(nums: list[int]) -> int:\n    unique = sorted(list(set(nums)))\n    if len(unique) < 2: raise ValueError()\n    return unique[1]\n", "assert find_second_smallest([10, 20, 4, 45, 99]) == 10"),
        ("index_of_max", "def index_of_max(arr: list[int | float]) -> int:\n    if not arr: raise ValueError()\n    return arr.index(max(arr))\n", "assert index_of_max([3, 15, 2, 8]) == 1"),
        ("index_of_min", "def index_of_min(arr: list[int | float]) -> int:\n    if not arr: raise ValueError()\n    return arr.index(min(arr))\n", "assert index_of_min([3, 15, 2, 8]) == 2"),
        ("count_above_threshold", "def count_above_threshold(nums: list[float], thresh: float) -> int:\n    return sum(1 for x in nums if x > thresh)\n", "assert count_above_threshold([1, 5, 10, 15], 6) == 2"),
        ("count_below_threshold", "def count_below_threshold(nums: list[float], thresh: float) -> int:\n    return sum(1 for x in nums if x < thresh)\n", "assert count_below_threshold([1, 5, 10, 15], 6) == 2"),
    ]
    for fn, code, test in search_ops:
        if test_code(code, test):
            items.append({"topic": "selection", "fn": fn, "prompt": f"Write a Python function `{fn}`.", "code": code, "test": test})

    # Combine with previous dataset
    prev_train = Path("data/coding_sft_014/coding_train_014.jsonl")
    combined = []
    if prev_train.exists():
        with open(prev_train, "r", encoding="utf-8") as f:
            for l in f:
                combined.append(json.loads(l))

    # Add the new templates with variations
    idx = len(combined)
    for it in items:
        # Prompt 1
        combined.append({
            "id": f"code_014_{idx:04d}",
            "topic": it["topic"],
            "prompt": f"User: {it['prompt']}\nAssistant:",
            "response": f"```python\n{it['code']}```",
            "code": it["code"],
            "test": it["test"]
        })
        idx += 1
        # Prompt 2
        combined.append({
            "id": f"code_014_{idx:04d}",
            "topic": it["topic"],
            "prompt": f"User: Implement the Python function `{it['fn']}`.\nAssistant:",
            "response": f"```python\n{it['code']}```",
            "code": it["code"],
            "test": it["test"]
        })
        idx += 1
        # Prompt 3
        combined.append({
            "id": f"code_014_{idx:04d}",
            "topic": it["topic"],
            "prompt": f"User: Provide a Python implementation of `{it['fn']}`.\nAssistant:",
            "response": f"Here is the function definition:\n\n```python\n{it['code']}```",
            "code": it["code"],
            "test": it["test"]
        })
        idx += 1

    # Save expanded training set
    with open("data/coding_sft_014/coding_train_expanded.jsonl", "w", encoding="utf-8") as f:
        for s in combined:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Total expanded verified coding samples in pool: {len(combined)}")

if __name__ == "__main__":
    build_comprehensive_dataset()
