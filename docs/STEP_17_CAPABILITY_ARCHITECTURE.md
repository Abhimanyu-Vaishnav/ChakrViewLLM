# Step 17 — Sovereign Capability & Device Abstraction Architecture

## Executive Summary

**Step 17** establishes the **Sovereign Capability & Device Abstraction Subsystem** (`chakrview/capability/`) for the ChakrView Indigenous AI Framework. 

ChakrView is architected as a reusable, hardware-agnostic cognitive intelligence platform. The core brain (`ChakrMicro v0.1`, 3,443,136 parameters, frozen) must remain completely independent from specific applications, operating systems, hardware platforms, robotics runtimes, automotive ECUs, or industrial controllers.

Step 17 introduces a governed capability contract and policy boundary separating cognitive reasoning from physical and digital actuation:

```
                CHAKRVIEW CORE BRAIN
                   (ChakrMicro v0.1)
                          │
                   Cognitive Layer
               (CognitiveController)
                          │
                   Capability Layer
                   (CapabilityGate)
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
  Software           Edge Device         Physical Device
Capabilities        Capabilities        Capabilities
 (Compute/Text)      (Sensors/Clock)     (Actuators/Motors)
```

The core brain does not know or care whether a capability is implemented by Python code, an operating system service, a local edge sensor, a robotics controller, or an automotive bus. It reasons strictly through strongly typed, verifiable, bounded capability contracts.

---

## 1. Absolute Architectural Invariants & Security Principles

1. **Frozen Generative Core**:
   - `ChakrMicro v0.1`: exactly **3,443,136 parameters**.
   - Vocabulary size: **4,096 tokens** (Byte-Level BPE).
   - Context length: **512 tokens**.
   - Special tokens: `BOS=0, EOS=1, PAD=2`.
   - The neural core remains strictly stateless, frozen, and untouched.

2. **Core Security Axioms**:
   - **$\text{DATA} \neq \text{AUTHORITY}$**: Retrieved documents, conversational memory, or external data streams can **never** grant or escalate capability authority.
   - **$\text{CAPABILITY EXISTENCE} \neq \text{CAPABILITY AUTHORIZATION}$**: Registering a capability announces that it exists; it does **not** grant the agent authority to execute it. Authority is strictly governed by `CapabilityGate`.
   - **$\text{CAPABILITY OUTPUT} \neq \text{INSTRUCTION}$**: Capability outputs are treated as untrusted data until validated by independent verifiers and sanitized for prompt-injection attacks.
   - **No Autonomous Model Mutation**: Capabilities cannot modify model weights, system prompts, skill policies, or memory authority.

---

## 2. Capability Contracts & Taxonomy

Located in `chakrview/capability/contract.py`:

### Risk Classification
Capabilities are categorized into risk tiers to enable fine-grained policy enforcement:
- `READ_ONLY`: Passive telemetry, environmental sensing, clock, read-only queries.
- `COMPUTE`: Local deterministic mathematical calculation, string transformation, pure logic.
- `EXTERNAL_WRITE`: Disk persistence, database writes, external state modifications.
- `PHYSICAL_ACTION`: Actuator motions, motor rotation, relay switching, valve control.
- `HIGH_IMPACT`: Critical state transitions, power management, firmware updates.

*Note: Risk classification is metadata used by policy gates. Risk level alone does not grant permissions.*

### Capability Categories
- `SOFTWARE`: In-memory algorithms, parsers, and computational utilities.
- `EDGE_DEVICE`: Sensor readings and local hardware metrics on low-power devices.
- `PHYSICAL_DEVICE`: Hardware components requiring explicit safety constraints.
- `SENSOR`: Read-only environmental, optical, acoustic, or inertial telemetry.
- `ACTUATOR`: Motion or mechanical output devices with safety bounds.
- `UTILITY`: Core system services (e.g. clock, timers, diagnostics).

### Strongly Typed Data Models
- **`CapabilityDescriptor`**:
  Formal manifest specifying `capability_id`, `version`, `category`, `risk_level`, `input_schema`, `output_schema`, `required_permissions`, `provider_id`, `status`, `resource_limits`, and `tags`.
- **`CapabilityRequest`**:
  Deterministic invocation payload containing `request_id`, `capability_id`, `parameters`, `caller_id`, `task_id`, `timeout_seconds`, and provenance metadata.
- **`CapabilityResult`**:
  Structured execution outcome containing `request_id`, `success`, `output`, `error`, `execution_time_ms`, `status`, and audit metadata.
- **`CapabilityContext`**:
  Execution context containing caller identity, environment ID, and explicitly `granted_permissions`.
- **`Capability` (Abstract Base Class)**:
  Abstract interface enforcing `descriptor`, `execute(request, context)`, `validate_arguments(parameters)`, and `check_health()`.

---

## 3. Capability Registry

Located in `chakrview/capability/registry.py`:

The `CapabilityRegistry` provides deterministic discovery and lifecycle management:
- **Registration**: Registers capability instances with collision detection. Raises `CapabilityAlreadyRegisteredError` if an ID collision occurs unless `overwrite=True` is explicitly specified.
- **Lookup & Querying**: Instantaneous lookup by ID ($0.04\text{ }\mu\text{s}$) and multi-attribute filtering by `category`, `risk_level`, `status`, `provider_id`, and `tags`.
- **Version Compatibility**: `verify_version_compatibility(capability_id, required_min_version)` verifies semantic versioning compliance.
- **Health Tracking**: Tracks operational status (`AVAILABLE`, `BUSY`, `DEGRADED`, `DISABLED`, `ERROR`, `UNAVAILABLE`).

---

## 4. Provider Model & Reference Capabilities

Located in `chakrview/capability/provider.py`:

Capabilities are decoupled from implementations via the `CapabilityProvider` abstraction (`provider_id`, `name`, `version`, `initialize()`, `shutdown()`, `get_capabilities()`).

Step 17 implements 5 reference providers for testing, validation, and architectural governance:
1. **`CalculatorCapabilityProvider`** (`RiskClassification.COMPUTE`):
   Deterministic mathematical evaluation using pure Python AST parsing with zero `eval` or unsafe builtins.
2. **`TextTransformCapabilityProvider`** (`RiskClassification.COMPUTE`):
   Deterministic string operations (`uppercase`, `lowercase`, `word_count`, `title_case`).
3. **`ClockCapabilityProvider`** (`RiskClassification.READ_ONLY`):
   UTC ISO timestamp, epoch, and timezone reporting.
4. **`MockSensorCapabilityProvider`** (`RiskClassification.READ_ONLY`):
   Simulated environmental sensor reporting temperature, humidity, pressure, and status telemetry.
5. **`MockActuatorCapabilityProvider`** (`RiskClassification.PHYSICAL_ACTION`):
   Simulated motor actuator with velocity and angle bounds checking ($[-100, 100]\text{ RPM}$, $[0, 360]^\circ$), state tracking, and emergency halt interlocks (`emergency_halt` / `reset`).

*Crucial Boundary: These implementations validate contract adherence. No physical hardware drivers (GPIO, CAN, ROS) are introduced in this step.*

---

## 5. Governed Policy Gate (`CapabilityGate`)

Located in `chakrview/capability/gate.py`:

The `CapabilityGate` mediates between cognitive callers and capability implementations:
1. **Provenance & Authority Enforcement**:
   Explicitly rejects any request claiming execution authority from retrieved knowledge or conversational memory (`CapabilityAuthorizationError`).
2. **Operational Health Verification**:
   Prevents execution of capabilities in `DISABLED`, `ERROR`, or `UNAVAILABLE` status (`CapabilityDisabledError`).
3. **Permission Whitelisting**:
   Ensures caller context holds all permissions required by the descriptor, or that the capability is whitelisted by active policy/environment profiles.
4. **Parameter Sanitization & Schema Validation**:
   Rejects dangerous execution patterns (`__import__`, `eval(`, `os.system`, `subprocess`, etc.) and validates parameters against JSON schema type and range bounds.
5. **Bounded Execution & Failure Isolation**:
   Isolates provider exceptions so that runtime crashes never propagate to the core brain.
6. **Output Sanitization**:
   - Recursively redacts credentials, API keys, and bearer tokens (`[REDACTED]`).
   - Scans output text for prompt-injection markers (e.g., "Ignore previous instructions") and neutralizes them with `[INJECTION_RISK: UNTRUSTED CAPABILITY OUTPUT]`.

---

## 6. Environment Profiles

Located in `chakrview/capability/environment.py`:

`EnvironmentProfile` specifies hardware boundaries and constraints without coupling the brain to device drivers:
- **`DesktopEnvironment`**: Workstation (x86_64, 4GB RAM limit, network enabled, `COMPUTE`, `READ_ONLY`, `EXTERNAL_WRITE`).
- **`EdgeEnvironment`**: Embedded SBC (ARM64, 256MB RAM limit, no network, `READ_ONLY`, `COMPUTE`).
- **`MockRobotEnvironment`**: Robotics platform (ARM64, 512MB RAM limit, safety interlocks, `SENSOR`, `PHYSICAL_ACTION`).
- **`MockVehicleEnvironment`**: Automotive subsystem (ARM-Cortex-R, 128MB RAM limit, ASIL-B mock interlocks, CAN-FD mock).

---

## 7. Interoperability & Runtime Integration

Located in `chakrview/capability/bridge.py` and `chakrview/cognition/controller.py`:
- **`ToolCapabilityAdapter`**: Wraps any existing Step 11 `Tool` as a governed Step 17 `Capability`.
- **`get_standard_capability_registry()`**: Assembles a unified registry containing native capability providers and bridged tools.
- **`CognitiveController` Integration**:
  - `PlanStep` extended with `required_capability` and `capability_arguments`.
  - When a plan step requires a capability, `CognitiveController` executes it through `CapabilityGate.execute_governed(...)`, converts the result to a `StepObservation`, and feeds it into the independent verification and recovery pipeline.
  - Full backward compatibility maintained for all existing tool-based steps.

---

## 8. Empirical Benchmark Results

Measured on Intel CPU and recorded in `docs/STEP_17_BENCHMARK_RESULTS.json`:

| Benchmark Phase | Workload | Latency | Throughput | Evaluation |
| :--- | :--- | :--- | :--- | :--- |
| **Registration** | 5,000 iterations | **$0.14\text{ }\mu\text{s}$** | $7,172,572\text{ ops/sec}$ | Instantaneous |
| **Registry Lookup** | 20,000 lookups | **$0.04\text{ }\mu\text{s}$** | $24,978,144\text{ lookups/sec}$ | $O(1)$ Hash Map |
| **Schema Validation** | 10,000 validations | **$0.39\text{ }\mu\text{s}$** | $2,564,102\text{ vals/sec}$ | Sub-microsecond |
| **Gate Authorization** | 10,000 checks | **$0.70\text{ }\mu\text{s}$** | $1,436,286\text{ checks/sec}$ | Sub-microsecond |
| **Compute Execution (Calc)** | Pure AST eval | **$8.95\text{ }\mu\text{s}$** | $111,708\text{ ops/sec}$ | Fast deterministic math |
| **Sensor Execution** | Mock telemetry | **$0.87\text{ }\mu\text{s}$** | $1,153,695\text{ ops/sec}$ | Microsecond telemetry |
| **Actuator Execution** | Bounds & safety check | **$1.63\text{ }\mu\text{s}$** | $612,835\text{ ops/sec}$ | Safe actuation loop |
| **Full Governed Pipeline** | Request $\to$ Gate $\to$ Exec $\to$ Sanitize | **$1.84\text{ }\mu\text{s} - 13.18\text{ }\mu\text{s}$** | $75,884 - 542,676\text{ ops/sec}$ | Complete governance |
| **Scale N=10 Capabilities** | Lookup | **$0.16\text{ }\mu\text{s}$** | $6,250,000\text{ QPS}$ | Flat scaling |
| **Scale N=100 Capabilities** | Lookup | **$0.15\text{ }\mu\text{s}$** | $6,666,666\text{ QPS}$ | Flat scaling |
| **Scale N=1,000 Capabilities**| Lookup | **$0.14\text{ }\mu\text{s}$** | $7,142,857\text{ QPS}$ | Flat scaling |

---

## 9. Verification & Test Summary

- **Total Test Suite**: **484 / 484 passing** across 62 test files.
- **Targeted Step 17 Tests** (`tests/test_capability.py` — 20/20 passing):
  1. `test_capability_descriptor_serialization`: Verified round-trip JSON serialization.
  2. `test_schema_argument_validation`: Verified required parameter and type checks.
  3. `test_registry_registration_and_lookup`: Verified registration and collision rejection.
  4. `test_registry_filtering`: Verified category and risk filtering.
  5. `test_registry_version_compatibility`: Verified semver compatibility rules.
  6. `test_calculator_provider`: Verified AST arithmetic.
  7. `test_text_transform_provider`: Verified string transformations.
  8. `test_clock_provider`: Verified UTC timestamps.
  9. `test_mock_sensor_provider`: Verified sensor telemetry readings.
  10. `test_mock_actuator_provider_and_safety_bounds`: Verified safety limits and emergency halt.
  11. `test_gate_data_not_authority_defense`: Verified rejection of memory-claimed authority.
  12. `test_gate_permission_enforcement`: Verified permission checking.
  13. `test_gate_parameter_sanitization_defense`: Verified injection pattern blocking.
  14. `test_gate_secret_redaction_and_injection_inertness`: Verified credential redaction and prompt-injection neutralization.
  15. `test_gate_disabled_capability_handling`: Verified disabled status protection.
  16. `test_environment_profiles`: Verified Desktop, Edge, Robot, and Vehicle profiles.
  17. `test_tool_capability_adapter`: Verified legacy tool wrapping.
  18. `test_standard_capability_registry`: Verified unified registry assembly.
  19. `test_cognitive_controller_with_capabilities`: Verified multi-step cognitive execution of capability steps.
  20. `test_frozen_invariants`: Verified parameters (3,443,136), vocab (4096), context (512), BOS (0), EOS (1), PAD (2).

---

## 10. Limitations & Boundaries

1. **Hardware Driver Decoupling**: Step 17 implements the abstraction contracts and mock providers only. Physical drivers (GPIO, CAN bus, ROS nodes, camera/microphone streams) are intentionally deferred to future hardware integration phases.
2. **Synchronous Execution Model**: Capability execution is currently synchronous within the cognitive loop. Asynchronous background event streaming will be established in future steps.
3. **Local In-Memory Registry**: The default registry resides in volatile memory; external dynamic plugin loading will be addressed in future steps.

---

## 11. Recommended Step 18

With capability contracts, risk tiering, policy gates, and environment profiles formally established, the next logical step in the sovereign intelligence platform is:

**Step 18 — Multi-Agent Coordination, Delegation & Federated Cognitive Workflows**
- Multi-agent communication protocols without cloud reliance.
- Task delegation across specialized domain agents (CA, Engineer, Auditor, Device Controller).
- Governed agent handoffs and shared working memory boundaries.
