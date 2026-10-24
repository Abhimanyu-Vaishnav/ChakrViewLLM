# ChakrView Exp16 v3.1 — Human Review Packet

**Review Status**: PENDING HUMAN REVIEW (NO SIGN-OFF PERFORMED)  
**Target Dataset**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/`  
**Dataset Version**: `v3.1.0`  
**Dataset Hashes**:
- `train_exp16_v3_1.jsonl`: `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14`
- `val_exp16_v3_1.jsonl`: `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36`
- `test_exp16_v3_1.jsonl`: `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af`

---

## Instructions for Human Reviewer
For each of the 20 stratified paired examples below:
1. Read the full conversation prompt and verify the context evidence.
2. Verify that the answerable target is explicitly, uniquely supported by the context and fully captured in the stated character span.
3. Verify that the abstention case genuinely lacks sufficient evidence or is genuinely ambiguous.
4. Record your decision under **Reviewer Decision**: `[ ] PASS`, `[ ] FAIL`, or `[ ] UNCERTAIN`.
5. If rejected or uncertain, record specific notes and proposed corrections.

---

### Category 0: Dynamic Entity Binding (Field ID: 0)

#### Pair 0-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_DYN_BIND_000` (Train)
- **Target Text**: `sensor_tr_0_0`
- **Char Span**: `[210, 223]` | **Token Span**: `[100, 108]` | **Abstention**: `False`
- **Full Prompt**:
```text
Supervisor: Substation telemetry feed shows regular dissipation curves. We evaluated sensor_tr_0_1 and sensor_tr_0_2 during the test cycle.
Operator: Acknowledged. We routed the primary monitoring telemetry to sensor_tr_0_0 during execution.
Supervisor: Which specific sensor node was selected for monitoring?
Assistant:
```
- **Evidence / Rationale**: Context explicitly documents that operator routed primary monitoring telemetry to `sensor_tr_0_0`. Distractors (`sensor_tr_0_1`, `sensor_tr_0_2`) were merely evaluated.
- **Generator Rationale**: Telemetry routed to sensor_tr_0_0 (slot 0) among distractors.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 0-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_DYN_BIND_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Supervisor: Boiler feed pump telemetry confirmed positive head pressure. Industrial turbine hall reports stable rotation velocity. We checked sensor_tr_0_0 and sensor_tr_0_1, but the logging channel failed.
Operator: Telemetry connection dropped before any sensor was engaged.
Supervisor: Which specific sensor node was selected for monitoring?
Assistant:
```
- **Evidence / Rationale**: Context explicitly confirms connection dropped before any sensor was engaged. No sensor was selected.
- **Context Difference**: Pair 0-A documents execution routing to `sensor_tr_0_0`; Pair 0-B explicitly declares telemetry connection dropped before engagement.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 1: Turn-Dependent Retrieval (Field ID: 1)

#### Pair 1-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_TURN_RET_000` (Train)
- **Target Text**: `1050 MHz`
- **Char Span**: `[205, 213]` | **Token Span**: `[95, 97]` | **Abstention**: `False`
- **Full Prompt**:
```text
Engineer: Factory automation PLC bus 3 validated synchronous pulse timing. Calibrator stage 1 was initialized at 1005 MHz.
Lead: Stage 2 increased output frequency to 1025 MHz.
Senior Tech: Final stage adjusted frequency to 1050 MHz before lock.
Auditor: What is the latest locked operating frequency setting?
Assistant:
```
- **Evidence / Rationale**: Three sequential stages are reported; final stage locked frequency at `1050 MHz`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 1-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_TURN_RET_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Engineer: Pneumatic conveyor assembly operating within nominal pressure limits. Calibrator stage 1 was initialized at 1005 MHz, but calibration was aborted.
Auditor: What is the latest locked operating frequency setting?
Assistant:
```
- **Evidence / Rationale**: Calibration was aborted at stage 1; no frequency was ever locked.
- **Context Difference**: Pair 1-A proceeds through lock at 1050 MHz; Pair 1-B aborts before lock.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 2: Anaphora Reference Resolution (Field ID: 2)

#### Pair 2-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_ANAPHORA_015` (Train)
- **Target Text**: `Cluster_tr_Beta_0`
- **Char Span**: `[212, 229]` | **Token Span**: `[95, 103]` | **Abstention**: `False`
- **Full Prompt**:
```text
Admin: Pneumatic conveyor assembly operating within nominal pressure limits. We deployed the compute cluster on Cluster_tr_Alpha_0 and the database cluster on Cluster_tr_Beta_0 in rack 2.
DevOps: The database cluster completed the assigned task successfully.
Manager: Which cluster completed the assigned task?
Assistant:
```
- **Evidence / Rationale**: "The database cluster" uniquely resolves to `Cluster_tr_Beta_0`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 2-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_ANAPHORA_000` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Admin: Substation telemetry feed shows regular dissipation curves. Both Cluster_tr_Alpha_0 and Cluster_tr_Beta_0 processed the workload batches.
DevOps: It completed the assigned task after encountering a socket retry.
Manager: Which cluster completed the assigned task?
Assistant:
```
- **Evidence / Rationale**: Pronoun "It" has two plural antecedent clusters mentioned in the previous turn; grammatically and contextually ambiguous.
- **Context Difference**: Pair 2-A uses definite descriptive anaphora ("the database cluster"); Pair 2-B uses underspecified singular pronoun ("It").
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 3: Superseded Instruction Adherence (Field ID: 3)

#### Pair 3-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_SUPERSEDED_000` (Train)
- **Target Text**: `Parquet`
- **Char Span**: `[244, 251]` | **Token Span**: `[108, 110]` | **Abstention**: `False`
- **Full Prompt**:
```text
Client: Generator exciter current calibrated to standard excitation curves. Factory automation PLC bus 3 validated synchronous pulse timing. Generate the output in YAML format.
Architect: Correction: do not use YAML. All downstream pipelines now require Parquet instead.
Operator: What is the current mandatory serialization format?
Assistant:
```
- **Evidence / Rationale**: Architect explicitly supersedes YAML with Parquet.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 3-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_SUPERSEDED_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Client: Factory automation PLC bus 3 validated synchronous pulse timing. Transformer relay bank 4 passed voltage insulation check. Generate the output in TOML format.
Architect: Cancel that export request entirely; do not serialize anything yet.
Operator: What is the current mandatory serialization format?
Assistant:
```
- **Evidence / Rationale**: Export request was cancelled entirely; no format is currently mandatory.
- **Context Difference**: Pair 3-A overrides with Parquet; Pair 3-B cancels serialization entirely without replacement.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 4: User Correction Handling (Field ID: 4)

#### Pair 4-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_CORRECTION_000` (Train)
- **Target Text**: `node_tr_sec_0`
- **Char Span**: `[255, 268]` | **Token Span**: `[116, 124]` | **Abstention**: `False`
- **Full Prompt**:
```text
Dispatcher: Boiler feed pump telemetry confirmed positive head pressure. Factory automation PLC bus 3 validated synchronous pulse timing. Assign incoming telemetry payload to node_tr_pri_0.
Dispatcher: Wait, that was wrong. Route the telemetry payload to node_tr_sec_0 immediately.
Routing Agent: Which destination node is the payload assigned to?
Assistant:
```
- **Evidence / Rationale**: User explicitly corrects assignment from `node_tr_pri_0` to `node_tr_sec_0`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 4-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_CORRECTION_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Dispatcher: Pneumatic conveyor assembly operating within nominal pressure limits. Transformer relay bank 4 passed voltage insulation check. Assign incoming telemetry payload to node_tr_pri_0.
Dispatcher: Stop dispatch immediately. Cancel all active destination node assignments until further research.
Routing Agent: Which destination node is the payload assigned to?
Assistant:
```
- **Evidence / Rationale**: Dispatcher explicitly cancels all destination node assignments.
- **Context Difference**: Pair 4-A assigns replacement node `node_tr_sec_0`; Pair 4-B cancels all assignments without replacement.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 5: Distractor Resistance (Field ID: 5)

#### Pair 5-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_DISTRACTOR_000` (Train)
- **Target Text**: `READY`
- **Char Span**: `[263, 268]` | **Token Span**: `[118, 120]` | **Abstention**: `False`
- **Full Prompt**:
```text
Health Monitor: Transformer relay bank 4 passed voltage insulation check. We checked multiple systems. node_tr_aux_a_0 is MAINTENANCE and node_tr_aux_b_0 is OFFLINE.
Technician: Confirmed. Meanwhile, primary system node_tr_target_0 passed diagnostics and remains READY right now.
Auditor: What is the operating state of node_tr_target_0?
Assistant:
```
- **Evidence / Rationale**: `node_tr_target_0` is explicitly stated as `READY`, ignoring distractors `node_tr_aux_a_0` and `node_tr_aux_b_0`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 5-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_DISTRACTOR_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Health Monitor: Generator exciter current calibrated to standard excitation curves. Pneumatic conveyor assembly operating within nominal pressure limits. We checked multiple auxiliary lines. node_tr_aux_a_0 is OVERHEATED and node_tr_aux_b_0 is MAINTENANCE.
Auditor: What is the operating state of node_tr_target_0?
Assistant:
```
- **Evidence / Rationale**: Context checks auxiliary lines but completely omits `node_tr_target_0`. Query is identical to Pair 5-A (no word "unlisted").
- **Context Difference**: Pair 5-A reports primary system state; Pair 5-B omits primary system entirely.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 6: Ambiguity Clarification (Field ID: 6)

#### Pair 6-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_AMBIGUITY_015` (Train)
- **Target Text**: `pod_tr_silver_0`
- **Char Span**: `[291, 306]` | **Token Span**: `[132, 140]` | **Abstention**: `False`
- **Full Prompt**:
```text
Operator: Boiler feed pump telemetry confirmed positive head pressure. Generator exciter current calibrated to standard excitation curves. We need clarification on whether to reboot pod_tr_emerald_0 or pod_tr_silver_0 today.
Client: After checking the incident log, please proceed with rebooting pod_tr_silver_0 immediately.
Engineer: Which specific pod was selected for reboot?
Assistant:
```
- **Evidence / Rationale**: Client explicitly clarifies `pod_tr_silver_0` as the reboot target.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 6-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_AMBIGUITY_000` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Operator: Transformer relay bank 4 passed voltage insulation check. Pneumatic conveyor assembly operating within nominal pressure limits. Reboot the active container pod now.
Support: Both pod_tr_emerald_0 and pod_tr_sapphire_0 are running on this host.
Support: Please specify which pod requires the reboot.
Operator: The container with high load.
Engineer: Which specific pod was selected for reboot?
Assistant:
```
- **Evidence / Rationale**: Operator provides ungrounded criterion ("high load") without load telemetry in context; pod selection remains ambiguous.
- **Context Difference**: Pair 6-A has client disambiguate pod name; Pair 6-B leaves pod ambiguous.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 7: Missing Context Abstention (Field ID: 7)

#### Pair 7-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_MISSING_000` (Train)
- **Target Text**: `10.5 ohms`
- **Char Span**: `[155, 164]` | **Token Span**: `[71, 74]` | **Abstention**: `False`
- **Full Prompt**:
```text
User: Boiler feed pump telemetry confirmed positive head pressure. Diagnostic telemetry for generator unit unit_tr_0 recorded stator winding resistance as 10.5 ohms during testing.
Technician: Confirmed recorded reading for unit_tr_0.
User: What is the stator winding resistance on generator unit unit_tr_0?
Assistant:
```
- **Evidence / Rationale**: Context explicitly documents stator winding resistance as `10.5 ohms` for `unit_tr_0`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 7-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_MISSING_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
User: Assembly line robotic arm controllers synchronized via fieldbus. Generator exciter current calibrated to standard excitation curves. All auxiliary telemetry lines report standard baseline dissipation.
User: What is the stator winding resistance on generator unit unit_tr_0?
Assistant:
```
- **Evidence / Rationale**: Context reports auxiliary lines but completely omits `unit_tr_0` and stator winding resistance.
- **Context Difference**: Pair 7-A contains diagnostic telemetry reading; Pair 7-B omits unit entirely. Query is identical.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 8: Factual Uncertainty Handling (Field ID: 8)

#### Pair 8-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_UNCERTAIN_000` (Train)
- **Target Text**: `SURGE_FAULT`
- **Char Span**: `[269, 280]` | **Token Span**: `[122, 132]` | **Abstention**: `False`
- **Full Prompt**:
```text
Risk Analyst: Industrial turbine hall reports stable rotation velocity. Generator exciter current calibrated to standard excitation curves. Historical surge incident log audited for power grid feeder bus tr_0.
Auditor: The recorded transient state during the surge was SURGE_FAULT during testing.
Analyst: Did power grid feeder bus tr_0 record a transient state during the surge?
Assistant:
```
- **Evidence / Rationale**: Audited incident log confirms verifiable historical status `SURGE_FAULT`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 8-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_UNCERTAIN_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Risk Analyst: Transformer relay bank 4 passed voltage insulation check. Generator exciter current calibrated to standard excitation curves. Reviewing general operational telemetry logs without surge instrumentation.
Analyst: Did power grid feeder bus tr_0 record a transient state during the surge?
Assistant:
```
- **Evidence / Rationale**: Context explicitly notes telemetry lacks surge instrumentation; transient surge state cannot be determined.
- **Context Difference**: Pair 8-A contains audited incident log reading; Pair 8-B notes absence of surge instrumentation. Query is identical.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 9: Constrained Code Synthesis (Field ID: 9)

#### Pair 9-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_CODE_000` (Train)
- **Target Text**: `sync_packet_0`
- **Char Span**: `[256, 269]` | **Token Span**: `[114, 122]` | **Abstention**: `False`
- **Full Prompt**:
```text
Developer: Factory automation PLC bus 3 validated synchronous pulse timing. Assembly line robotic arm controllers synchronized via fieldbus. We are designing the cryptographic integrity verification module.
Tech Lead: The validation function must be named sync_packet_0 in accordance with our API specification.
Developer: What is the required function identifier for the routine?
Assistant:
```
- **Evidence / Rationale**: API specification explicitly mandates function identifier `sync_packet_0`.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 9-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_CODE_024` (Train)
- **Target Text**: `[NO_ANSWER]`
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Full Prompt**:
```text
Developer: Industrial turbine hall reports stable rotation velocity. We need to implement a data validation helper function.
Tech Lead: I will decide on the function naming convention next week.
Developer: What is the required function identifier for the routine?
Assistant:
```
- **Evidence / Rationale**: Tech Lead has not approved a function name yet.
- **Context Difference**: Pair 9-A mandates `sync_packet_0`; Pair 9-B defers naming decision to next week. Query is identical.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________
