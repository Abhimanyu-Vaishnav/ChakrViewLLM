"""
Empirical Performance Benchmark for Step 17: Sovereign Capability & Device Abstraction.

Measures:
1. Registration throughput and overhead
2. Registry lookup latency
3. Request schema and parameter validation overhead
4. CapabilityGate authorization latency
5. Mock capability execution latency across categories:
   - Compute (Calculator)
   - Utility (Clock)
   - Sensor (Mock Sensor)
   - Actuator (Mock Motor)
6. Complete end-to-end governed capability pipeline latency
7. Scale benchmark: 10, 100, 1,000 registered capabilities

Outputs structured JSON to docs/STEP_17_BENCHMARK_RESULTS.json.
"""

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List

from chakrview.capability.contract import (
    CapabilityRequest,
    CapabilityContext,
    RiskClassification,
)
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.provider import (
    CalculatorCapability,
    TextTransformCapability,
    ClockCapability,
    MockSensorCapability,
    MockActuatorCapability,
)
from chakrview.capability.gate import CapabilityGate
from chakrview.capability.environment import get_desktop_environment, get_edge_environment


def benchmark_capability_subsystem() -> Dict[str, Any]:
    print("=" * 70)
    print("CHAKRVIEW STEP 17: CAPABILITY & DEVICE ABSTRACTION BENCHMARK")
    print("=" * 70)

    # 1. Registration Overhead
    reg = CapabilityRegistry()
    caps = [
        CalculatorCapability(),
        TextTransformCapability(),
        ClockCapability(),
        MockSensorCapability(),
        MockActuatorCapability(),
    ]

    t0 = time.perf_counter()
    n_reg_iters = 5000
    for i in range(n_reg_iters):
        c = caps[i % len(caps)]
        reg.register(c, overwrite=True)
    t_reg = (time.perf_counter() - t0) / n_reg_iters
    reg_ops_sec = int(1.0 / t_reg) if t_reg > 0 else 0
    print(f"[1] Registration Latency: {t_reg * 1e6:.2f} µs/op ({reg_ops_sec:,} ops/sec)")

    # 2. Lookup Latency
    t0 = time.perf_counter()
    n_lookups = 20000
    for i in range(n_lookups):
        _ = reg.get("calculator")
    t_lookup = (time.perf_counter() - t0) / n_lookups
    lookup_ops_sec = int(1.0 / t_lookup) if t_lookup > 0 else 0
    print(f"[2] Registry Lookup Latency: {t_lookup * 1e6:.2f} µs/lookup ({lookup_ops_sec:,} lookups/sec)")

    # 3. Parameter Validation Overhead
    calc = reg.get("calculator")
    test_params = {"expression": "(12 + 8) * 5 / 2"}
    t0 = time.perf_counter()
    n_val = 10000
    for _ in range(n_val):
        _ = calc.validate_arguments(test_params)
    t_val = (time.perf_counter() - t0) / n_val
    print(f"[3] Schema Validation Latency: {t_val * 1e6:.2f} µs/val")

    # 4. CapabilityGate Authorization Latency
    gate = CapabilityGate(reg)
    req_auth = CapabilityRequest(capability_id="calculator", parameters=test_params)
    ctx_auth = CapabilityContext(granted_permissions={"capability.compute.math"})
    desktop_env = get_desktop_environment()

    t0 = time.perf_counter()
    n_auth = 10000
    for _ in range(n_auth):
        _ = gate.authorize(req_auth, context=ctx_auth, active_policy=desktop_env)
    t_auth = (time.perf_counter() - t0) / n_auth
    auth_ops_sec = int(1.0 / t_auth) if t_auth > 0 else 0
    print(f"[4] Gate Authorization Latency: {t_auth * 1e6:.2f} µs/check ({auth_ops_sec:,} checks/sec)")

    # 5. Mock Capability Execution Latency
    bench_caps = {
        "compute_calculator": (
            CapabilityRequest(capability_id="calculator", parameters={"expression": "10 * (5 + 3) / 2"}),
            ctx_auth,
        ),
        "compute_text_transform": (
            CapabilityRequest(capability_id="text_transform", parameters={"text": "chakrview indigenous ai", "operation": "title_case"}),
            CapabilityContext(granted_permissions={"capability.compute.text"}),
        ),
        "utility_clock": (
            CapabilityRequest(capability_id="system_clock", parameters={}),
            CapabilityContext(granted_permissions={"capability.read.clock"}),
        ),
        "sensor_reading": (
            CapabilityRequest(capability_id="mock_environmental_sensor", parameters={"metric": "all"}),
            CapabilityContext(granted_permissions={"capability.sensor.read"}),
        ),
        "actuator_control": (
            CapabilityRequest(capability_id="mock_motor_actuator", parameters={"action": "set_speed", "target_value": 50.0}),
            CapabilityContext(granted_permissions={"capability.actuator.control"}),
        ),
    }

    exec_results = {}
    print("\n[5] Individual Capability Execution Benchmarks:")
    for name, (req, ctx) in bench_caps.items():
        cap = reg.get(req.capability_id)
        # Warmup
        _ = cap.execute(req, ctx)
        t0 = time.perf_counter()
        n_exec = 5000
        for _ in range(n_exec):
            res = cap.execute(req, ctx)
        t_exec = (time.perf_counter() - t0) / n_exec
        exec_results[name] = {
            "latency_us": round(t_exec * 1e6, 2),
            "latency_ms": round(t_exec * 1000.0, 4),
            "ops_per_sec": int(1.0 / t_exec) if t_exec > 0 else 0,
            "success": res.success,
        }
        print(f"  - {name}: {t_exec * 1e6:.2f} µs ({exec_results[name]['ops_per_sec']:,} ops/sec)")

    # 6. Complete Governed Pipeline (Request -> Gate -> Auth -> Validation -> Exec -> Sanitize)
    print("\n[6] Complete Governed Pipeline Latency:")
    governed_results = {}
    for name, (req, ctx) in bench_caps.items():
        t0 = time.perf_counter()
        n_gov = 5000
        for _ in range(n_gov):
            _ = gate.execute_governed(req, context=ctx, active_policy=desktop_env)
        t_gov = (time.perf_counter() - t0) / n_gov
        governed_results[name] = {
            "latency_us": round(t_gov * 1e6, 2),
            "latency_ms": round(t_gov * 1000.0, 4),
            "ops_per_sec": int(1.0 / t_gov) if t_gov > 0 else 0,
        }
        print(f"  - {name} [Governed]: {t_gov * 1e6:.2f} µs ({governed_results[name]['ops_per_sec']:,} ops/sec)")

    # 7. Scale Benchmark: Scaling Registry with N capabilities
    print("\n[7] Registry Scaling Benchmark:")
    scale_bench = {}
    for n_scale in [10, 100, 1000]:
        scaled_reg = CapabilityRegistry()
        for i in range(n_scale):
            scaled_reg.register(
                MockSensorCapability(
                    provider_id=f"provider_{i}",
                    temperature_c=20.0 + (i % 15),
                ),
                overwrite=True,
            )
            # Re-register with unique ID by cloning descriptor
            cap_inst = MockSensorCapability(provider_id=f"provider_{i}")
            cap_inst.descriptor.capability_id = f"sensor_{i}"
            scaled_reg.register(cap_inst)

        # Measure lookup in scaled registry
        t0 = time.perf_counter()
        n_ops = 10000
        for i in range(n_ops):
            _ = scaled_reg.get(f"sensor_{i % n_scale}")
        t_scale_lookup = (time.perf_counter() - t0) / n_ops

        # Measure list with filtering
        t0 = time.perf_counter()
        for _ in range(100):
            _ = scaled_reg.list_capabilities(risk_level=RiskClassification.READ_ONLY)
        t_scale_list = (time.perf_counter() - t0) / 100

        scale_bench[f"scale_{n_scale}_caps"] = {
            "count": scaled_reg.count(),
            "lookup_latency_us": round(t_scale_lookup * 1e6, 2),
            "filter_list_latency_us": round(t_scale_list * 1e6, 2),
            "lookup_qps": int(1.0 / t_scale_lookup) if t_scale_lookup > 0 else 0,
        }
        print(f"  - Scale N={n_scale}: Lookup = {t_scale_lookup * 1e6:.2f} µs, Filter = {t_scale_list * 1e6:.2f} µs")

    report = {
        "timestamp": time.time(),
        "platform": {
            "system": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "registration_benchmark": {
            "latency_us": round(t_reg * 1e6, 2),
            "ops_per_sec": reg_ops_sec,
        },
        "lookup_benchmark": {
            "latency_us": round(t_lookup * 1e6, 2),
            "lookups_per_sec": lookup_ops_sec,
        },
        "validation_benchmark": {
            "latency_us": round(t_val * 1e6, 2),
        },
        "authorization_benchmark": {
            "latency_us": round(t_auth * 1e6, 2),
            "checks_per_sec": auth_ops_sec,
        },
        "execution_benchmarks": exec_results,
        "governed_pipeline_benchmarks": governed_results,
        "scale_benchmarks": scale_bench,
    }

    out_path = os.path.join("docs", "STEP_17_BENCHMARK_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark results successfully written to {out_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    benchmark_capability_subsystem()
