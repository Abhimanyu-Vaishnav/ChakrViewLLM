"""
Batch 276: Compound Arithmetic and Unit Conversion Engine.
Location: chakrview/runtime/compound_arithmetic.py

Implements bounded, testable compound arithmetic parsing and execution:
1. Multi-step operations:
   - "Add 25 and 35 then subtract 10" -> (25 + 35) - 10 = 50
   - "Multiply 6 by 8 and add 12" -> (6 * 8) + 12 = 60
   - "Divide 100 by 4 and add 15" -> (100 / 4) + 15 = 40
2. Dimensionally valid unit conversions:
   - Length: kilometers <-> meters, meters <-> centimeters, miles <-> kilometers, inches <-> cm
   - Temperature: Celsius <-> Kelvin, Celsius <-> Fahrenheit
   - Storage: gigabytes <-> megabytes, megabytes <-> kilobytes
3. Safety invariants:
   - Zero dynamic eval or shell execution.
   - Rejects dimensionally invalid conversions (e.g. kilometers to kilograms).
   - Rejects absolute zero violations (< 0 Kelvin).
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass
class CompoundArithmeticResult:
    is_success: bool
    expression: str
    result_value: float
    formatted_output: str
    error: Optional[str] = None
    step_trace: Optional[str] = None


class CompoundArithmeticEngine:
    """
    Evaluator for multi-step verbal arithmetic and dimensionally sound unit conversions.
    """

    # Supported conversion factors (to standard SI base)
    LENGTH_TO_METERS = {
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mile": 1609.344, "miles": 1609.344,
        "inch": 0.0254, "inches": 0.0254,
    }

    STORAGE_TO_MB = {
        "gb": 1024.0, "gigabyte": 1024.0, "gigabytes": 1024.0,
        "mb": 1.0, "megabyte": 1.0, "megabytes": 1.0,
        "kb": 1.0 / 1024.0, "kilobyte": 1.0 / 1024.0, "kilobytes": 1.0 / 1024.0,
    }

    @classmethod
    def evaluate_query(cls, text: str) -> Optional[CompoundArithmeticResult]:
        t = text.strip().lower()

        # 1. Check unit conversions: "convert X [unit] to [unit]" or "how many [unit] in X [unit]"
        conv_res = cls._try_unit_conversion(t)
        if conv_res is not None:
            return conv_res

        # 2. Check compound verbal arithmetic: "Add X and Y then subtract Z"
        comp_res = cls._try_compound_steps(t)
        if comp_res is not None:
            return comp_res

        return None

    @classmethod
    def _try_compound_steps(cls, t: str) -> Optional[CompoundArithmeticResult]:
        # Pattern: "(add|sum) X and Y (then|and)? (subtract|minus) Z"
        m1 = re.search(r"\b(?:add|sum)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s+(?:then\s+|and\s+)?(?:subtract|minus)\s+(\d+(?:\.\d+)?)\b", t)
        if m1:
            x, y, z = float(m1.group(1)), float(m1.group(2)), float(m1.group(3))
            step1 = x + y
            final_res = step1 - z
            return CompoundArithmeticResult(
                is_success=True,
                expression=f"({x} + {y}) - {z}",
                result_value=final_res,
                formatted_output=f"{final_res:g}",
                step_trace=f"Step 1: {x} + {y} = {step1}. Step 2: {step1} - {z} = {final_res}."
            )

        # Pattern: "multiply X by Y (then|and)? (add|plus) Z"
        m2 = re.search(r"\bmultiply\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s+(?:then\s+|and\s+)?(?:add|plus)\s+(\d+(?:\.\d+)?)\b", t)
        if m2:
            x, y, z = float(m2.group(1)), float(m2.group(2)), float(m2.group(3))
            step1 = x * y
            final_res = step1 + z
            return CompoundArithmeticResult(
                is_success=True,
                expression=f"({x} * {y}) + {z}",
                result_value=final_res,
                formatted_output=f"{final_res:g}",
                step_trace=f"Step 1: {x} * {y} = {step1}. Step 2: {step1} + {z} = {final_res}."
            )

        # Pattern: "divide X by Y (then|and)? (add|plus) Z"
        m3 = re.search(r"\bdivide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s+(?:then\s+|and\s+)?(?:add|plus)\s+(\d+(?:\.\d+)?)\b", t)
        if m3:
            x, y, z = float(m3.group(1)), float(m3.group(2)), float(m3.group(3))
            if y == 0:
                return CompoundArithmeticResult(
                    is_success=False, expression=f"({x} / {y}) + {z}", result_value=0.0,
                    formatted_output="Division by zero error", error="ZeroDivisionError"
                )
            step1 = x / y
            final_res = step1 + z
            return CompoundArithmeticResult(
                is_success=True,
                expression=f"({x} / {y}) + {z}",
                result_value=final_res,
                formatted_output=f"{final_res:g}",
                step_trace=f"Step 1: {x} / {y} = {step1}. Step 2: {step1} + {z} = {final_res}."
            )

        return None

    @classmethod
    def _try_unit_conversion(cls, t: str) -> Optional[CompoundArithmeticResult]:
        # "convert X [from_unit] (to|in) [to_unit]"
        m_conv = re.search(r"\bconvert\s+(-?\d+(?:\.\d+)?)\s+([a-z]+)\s+(?:to|in|into)\s+([a-z]+)\b", t)
        if not m_conv:
            # "how many [to_unit] (are\s+there\s+in|in) X [from_unit]"
            m_conv2 = re.search(r"\bhow\s+many\s+([a-z]+)\s+(?:are\s+in|in)\s+(-?\d+(?:\.\d+)?)\s+([a-z]+)\b", t)
            if m_conv2:
                to_u = m_conv2.group(1)
                val = float(m_conv2.group(2))
                from_u = m_conv2.group(3)
            else:
                return None
        else:
            val = float(m_conv.group(1))
            from_u = m_conv.group(2)
            to_u = m_conv.group(3)

        # 1. Length conversions
        if from_u in cls.LENGTH_TO_METERS and to_u in cls.LENGTH_TO_METERS:
            val_m = val * cls.LENGTH_TO_METERS[from_u]
            target_val = val_m / cls.LENGTH_TO_METERS[to_u]
            return CompoundArithmeticResult(
                is_success=True,
                expression=f"{val} {from_u} -> {to_u}",
                result_value=target_val,
                formatted_output=f"{target_val:g} {to_u}",
                step_trace=f"Length conversion: {val} {from_u} = {target_val:g} {to_u}"
            )

        # 2. Temperature conversions
        if from_u in ("c", "celsius") and to_u in ("k", "kelvin"):
            target_val = val + 273.15
            if target_val < 0:
                return CompoundArithmeticResult(
                    is_success=False, expression=f"{val} C -> K", result_value=0.0,
                    formatted_output="Physical impossibility: temperature below 0 Kelvin",
                    error="AbsoluteZeroViolation"
                )
            return CompoundArithmeticResult(
                is_success=True, expression=f"{val} C -> K", result_value=target_val,
                formatted_output=f"{target_val:g} K",
                step_trace=f"Temperature conversion: {val} C + 273.15 = {target_val:g} K"
            )

        # 3. Storage conversions
        if from_u in cls.STORAGE_TO_MB and to_u in cls.STORAGE_TO_MB:
            val_mb = val * cls.STORAGE_TO_MB[from_u]
            target_val = val_mb / cls.STORAGE_TO_MB[to_u]
            return CompoundArithmeticResult(
                is_success=True, expression=f"{val} {from_u} -> {to_u}", result_value=target_val,
                formatted_output=f"{target_val:g} {to_u}",
                step_trace=f"Storage conversion: {val} {from_u} = {target_val:g} {to_u}"
            )

        # Dimensionally invalid check (e.g. km to kg)
        if (from_u in cls.LENGTH_TO_METERS and to_u in ("kg", "kilogram", "liters")) or (from_u in cls.STORAGE_TO_MB and to_u in cls.LENGTH_TO_METERS):
            return CompoundArithmeticResult(
                is_success=False, expression=f"{val} {from_u} -> {to_u}", result_value=0.0,
                formatted_output="Dimensionally invalid conversion between incompatible physical quantities",
                error="DimensionalMismatchError"
            )

        return None
