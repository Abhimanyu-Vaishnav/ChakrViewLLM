"""SFT V3 Grounded Dataset Builder for ChakrView.

Generates a fully grounded, reproducible, leakage-free dataset for SFT V3.
Features:
- Target: 200 train examples, 40 validation examples.
- 70% Grounded JSON extraction across 6 distinct capability categories.
- 30% Natural language replay continuation with 100% disjoint document sources.
- Strict template-family disjointness between train and validation splits.
- Disjoint entity names, metrics, and numerical values.
- Independent ground truth generation and verification.
- Completely deterministic with fixed seed.

SAFETY NOTICE:
- Generates files in experiments/sft_v3_dataset_builder/ only.
- Does NOT perform training or optimizer steps.
- Does NOT touch canonical checkpoints, tokenizers, or test.jsonl.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "sft_v3_dataset_builder"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

RANDOM_SEED = 42

# ==============================================================================
# 1. GROUNDED FACT BASES (Disjoint between Train and Validation)
# ==============================================================================

# Category 1: Server Status (status: str, code: int)
TRAIN_SERVER_FACTS = [
    ("alpha_node_1", "ready", 200, "Node alpha_node_1 initialized successfully with HTTP status ready (200)."),
    ("alpha_node_2", "active", 200, "Telemetry ping from alpha_node_2 confirms active operation returning 200."),
    ("alpha_node_3", "healthy", 200, "Health probe for alpha_node_3 reports healthy condition with return code 200."),
    ("alpha_node_4", "idle", 200, "Workload scheduler places alpha_node_4 in idle state with status code 200."),
    ("alpha_node_5", "running", 200, "Compute job running on alpha_node_5 with standard HTTP code 200."),
    ("alpha_node_6", "standby", 200, "Backup cluster member alpha_node_6 placed on standby with code 200."),
    ("alpha_node_7", "connected", 200, "Gateway link alpha_node_7 connected to edge fabric with HTTP code 200."),
    ("alpha_node_8", "online", 200, "Primary database alpha_node_8 is now online returning status 200."),
    ("alpha_node_9", "operational", 200, "Inference runner alpha_node_9 is fully operational with code 200."),
    ("alpha_node_10", "starting", 202, "Container container_10 starting execution asynchronously, returning code 202."),
    ("alpha_node_11", "accepted", 202, "Task dispatch alpha_node_11 accepted batch request with HTTP 202."),
    ("alpha_node_12", "processing", 202, "Worker thread alpha_node_12 is processing queue items with code 202."),
    ("alpha_node_13", "paused", 200, "Maintenance script marked alpha_node_13 paused with status 200."),
    ("alpha_node_14", "busy", 200, "Buffer capacity full, alpha_node_14 is busy with HTTP return code 200."),
    ("alpha_node_15", "initialized", 200, "Boot cycle complete, alpha_node_15 initialized with status 200."),
    ("alpha_node_16", "degraded", 206, "Storage volume degraded on alpha_node_16, returning partial code 206."),
    ("alpha_node_17", "unhealthy", 500, "Hardware diagnostics report alpha_node_17 unhealthy with internal error 500."),
    ("alpha_node_18", "error", 500, "Kernel panic triggered on alpha_node_18, fatal error with code 500."),
    ("alpha_node_19", "failed", 500, "Process memory alloc failed on alpha_node_19 returning HTTP code 500."),
    ("alpha_node_20", "timeout", 504, "Upstream connection to alpha_node_20 reached timeout with gateway code 504."),
    ("alpha_node_21", "unavailable", 503, "Cluster alpha_node_21 is temporarily unavailable with code 503."),
    ("alpha_node_22", "bad_request", 400, "Client sent malformed payload to alpha_node_22 yielding bad_request (400)."),
    ("alpha_node_23", "unauthorized", 401, "Token expired on alpha_node_23, returning unauthorized code 401."),
    ("alpha_node_24", "forbidden", 403, "Access policy blocked request to alpha_node_24 with forbidden code 403."),
    ("alpha_node_25", "not_found", 404, "Endpoint resource missing on alpha_node_25 returning not_found (404)."),
]

VAL_SERVER_FACTS = [
    ("val_host_01", "queued", 202, "Job queue manager reports val_host_01 is queued with acceptance code 202."),
    ("val_host_02", "draining", 200, "Traffic controller set val_host_02 to draining mode with HTTP code 200."),
    ("val_host_03", "internal_error", 500, "Uncaught exception on val_host_03 caused internal_error with code 500."),
    ("val_host_04", "service_down", 503, "Outage notice confirms val_host_04 has service_down status with code 503."),
    ("val_host_05", "closed", 410, "Registration endpoint on val_host_05 has been closed with code 410."),
]

# Category 2: Telemetry Metric (metric: str, value: float)
TRAIN_METRIC_FACTS = [
    ("cpu_utilization", 42.5, "Node performance monitor: cpu_utilization reading is 42.5 percent."),
    ("memory_consumed", 78.2, "Resource monitor reports memory_consumed at 78.2 gigabytes."),
    ("disk_throughput", 12.4, "Storage interface reports disk_throughput measured at 12.4 megabytes per second."),
    ("network_latency", 5.8, "Network diagnostic ping records network_latency at 5.8 milliseconds."),
    ("packet_drop_rate", 0.02, "Interface eth0 reports packet_drop_rate observed at 0.02 percent."),
    ("channel_bandwidth", 8.4, "Fiber link reports channel_bandwidth steady at 8.4 gigabits per second."),
    ("system_load_avg", 1.85, "Scheduler metrics: system_load_avg registered at 1.85 over fifteen minutes."),
    ("processor_temp", 54.2, "Thermal probe shows processor_temp at 54.2 degrees Celsius."),
    ("rail_voltage", 12.04, "Power distribution unit: rail_voltage measured at 12.04 volts DC."),
    ("cooling_fan_speed", 65.0, "Fan chassis controller records cooling_fan_speed at 65.0 percent PWM."),
    ("incoming_req_rate", 240.5, "Load balancer reports incoming_req_rate at 240.5 queries per second."),
    ("cache_miss_ratio", 4.12, "L2 cache monitor indicates cache_miss_ratio at 4.12 percent."),
    ("transaction_error", 0.05, "Payment gateway records transaction_error at 0.05 percent of total volume."),
    ("clock_jitter_ms", 1.25, "Time sync service reports clock_jitter_ms measured at 1.25 milliseconds."),
    ("round_trip_ping", 14.8, "WAN tunnel ping test records round_trip_ping at 14.8 milliseconds."),
    ("queue_depth_count", 18.0, "Message broker indicates queue_depth_count standing at 18.0 messages."),
    ("active_connections", 450.0, "Socket manager reports active_connections currently at 450.0 sockets."),
    ("stream_throughput", 95.4, "Video pipeline measures stream_throughput at 95.4 megabits per second."),
    ("accelerator_load", 88.5, "GPU runtime records accelerator_load at 88.5 percent utilization."),
    ("total_power_draw", 185.0, "Wattmeter reading indicates total_power_draw at 185.0 watts."),
    ("headroom_capacity", 32.5, "Capacity planning: headroom_capacity evaluated at 32.5 percent."),
    ("line_pressure_psi", 14.7, "Pneumatic sensor registers line_pressure_psi at 14.7 pounds per square inch."),
    ("clock_frequency", 3.6, "CPU core clock_frequency clocked at 3.6 gigahertz."),
    ("heap_allocated", 512.5, "Runtime heap analysis reveals heap_allocated at 512.5 megabytes."),
    ("thread_pool_active", 64.0, "Worker pool monitors thread_pool_active at 64.0 threads."),
]

VAL_METRIC_FACTS = [
    ("chassis_vibration", 0.12, "Accelerometer records chassis_vibration intensity at 0.12 G force."),
    ("ambient_room_temp", 24.5, "Facility environmental sensor shows ambient_room_temp at 24.5 Celsius."),
    ("frame_drop_ratio", 0.03, "Display pipeline telemetry measures frame_drop_ratio at 0.03 percent."),
    ("acoustic_noise_db", 42.0, "Microphone array registers acoustic_noise_db at 42.0 decibels."),
    ("wireless_rssi_dbm", -65.4, "Transceiver telemetry notes wireless_rssi_dbm signal at -65.4 decibel milliwatts."),
]

# Category 3: Feature Flag (feature: str, enabled: bool)
TRAIN_FLAG_FACTS = [
    ("dark_mode", True, "User preference manager: dark_mode is active in system settings."),
    ("beta_chat", True, "Experiment controller: feature beta_chat is currently enabled for users."),
    ("telemetry_stream", True, "Compliance policy: telemetry_stream has been activated by administrator."),
    ("auto_save", True, "Editor state: auto_save is enabled for real-time document sync."),
    ("gpu_acceleration", True, "Compute pipeline: gpu_acceleration is turned on in driver configuration."),
    ("caching_layer", True, "Data layer: caching_layer is active across redis nodes."),
    ("access_logging", True, "Security daemon: access_logging has been enabled for auditing."),
    ("data_compression", True, "Network layer: data_compression is enabled for payload transit."),
    ("audit_trail", True, "Governance module: audit_trail is active for compliance tracking."),
    ("metrics_export", True, "Monitoring agent: metrics_export is currently enabled."),
    ("strict_validation", True, "Schema engine: strict_validation is active for API requests."),
    ("query_tracing", True, "Profiler daemon: query_tracing is turned on for deep diagnostics."),
    ("sandbox_isolation", True, "Container security: sandbox_isolation is enabled for untrusted code."),
    ("tls_verification", True, "Transport security: tls_verification is active on all outgoing ports."),
    ("keep_alive_ping", True, "HTTP client: keep_alive_ping is enabled for socket preservation."),
    ("desktop_notify", False, "Notification service: desktop_notify is disabled by user override."),
    ("debug_symbols", False, "Release build options: debug_symbols is disabled in production."),
    ("mock_api_backend", False, "Integration test flag: mock_api_backend is turned off for live testing."),
    ("legacy_auth_v1", False, "Security audit: legacy_auth_v1 is disabled to block deprecated tokens."),
    ("verbose_output", False, "Command line runner: verbose_output is turned off by default."),
    ("cpu_profiler", False, "Diagnostics config: cpu_profiler is disabled during benchmark runs."),
    ("dry_run_mode", False, "Deployment pipeline: dry_run_mode is turned off for actual apply."),
    ("unsafe_evaluation", False, "Scripting engine: unsafe_evaluation is strictly disabled."),
    ("experimental_v3", False, "Feature toggle: experimental_v3 is disabled pending QA approval."),
    ("maintenance_mode", False, "Platform gateway: maintenance_mode is turned off, site is operational."),
]

VAL_FLAG_FACTS = [
    ("memory_snapshots", True, "Snapshot service: memory_snapshots is currently active in the manager."),
    ("vector_indexing", True, "Search subsystem: vector_indexing is enabled for semantic queries."),
    ("relay_fallback", False, "Routing matrix: relay_fallback has been turned off by network administrator."),
    ("strict_json_mode", True, "Parser configuration: strict_json_mode is active for inference pipelines."),
    ("guest_access", False, "Authentication policy: guest_access is disabled on private repositories."),
]

# Category 4: Inventory Item (name: str, count: int)
TRAIN_INVENTORY_FACTS = [
    ("server_rack", 12, "Warehouse logistics register: exactly 12 units of server_rack in stock."),
    ("power_supply", 45, "Supply inventory confirms 45 units of power_supply available."),
    ("memory_module", 128, "Stock ledger counts 128 units of memory_module ready for shipment."),
    ("solid_state_drive", 64, "Component depot records 64 units of solid_state_drive in storage."),
    ("ethernet_switch", 8, "Network distribution center has 8 units of ethernet_switch on shelf."),
    ("fiber_patch_cord", 250, "Cabling bin contains 250 units of fiber_patch_cord ready to deploy."),
    ("cooling_fan_pack", 90, "Thermal components inventory holds 90 units of cooling_fan_pack."),
    ("cpu_copper_heatsink", 32, "Assembly inventory notes 32 units of cpu_copper_heatsink available."),
    ("chassis_bracket", 180, "Hardware stock records 180 units of chassis_bracket in rack A."),
    ("optical_transceiver", 55, "Telecom supply lists 55 units of optical_transceiver available."),
    ("battery_ups_unit", 16, "Facility backup holds 16 units of battery_ups_unit in depot."),
    ("terminal_jumper", 220, "Electronics stock confirms 220 units of terminal_jumper on hand."),
    ("temperature_probe", 40, "Sensor supply bin contains 40 units of temperature_probe."),
    ("rack_mount_shelf", 24, "Enclosure storage confirms 24 units of rack_mount_shelf available."),
    ("surge_protector", 38, "Power inventory records 38 units of surge_protector ready for issue."),
    ("glass_fuse_cartridge", 150, "Electrical parts storage lists 150 units of glass_fuse_cartridge."),
    ("gigabit_nic_card", 72, "Peripheral depot reports 72 units of gigabit_nic_card on inventory."),
    ("serial_usb_adapter", 85, "Interface supply counts 85 units of serial_usb_adapter available."),
    ("thermal_grease_tube", 30, "Maintenance kit stock holds 30 units of thermal_grease_tube."),
    ("nylon_cable_straps", 500, "Packaging inventory records 500 units of nylon_cable_straps."),
    ("laptop_dock_station", 20, "IT hardware depot lists 20 units of laptop_dock_station available."),
    ("video_display_port", 44, "Accessory stock holds 44 units of video_display_port adapters."),
    ("rf_antenna_pole", 10, "Radio infrastructure inventory notes 10 units of rf_antenna_pole."),
    ("anti_static_strap", 100, "Cleanroom inventory counts 100 units of anti_static_strap on shelf."),
    ("status_led_module", 300, "Component store contains 300 units of status_led_module in stock."),
]

VAL_INVENTORY_FACTS = [
    ("radiator_core", 14, "Cooling inventory audit lists 14 units of radiator_core in inventory."),
    ("hepa_air_filter", 95, "Facility stock register counts 95 units of hepa_air_filter available."),
    ("drive_enclosure", 18, "Storage hardware supply records 18 units of drive_enclosure in aisle 4."),
    ("numeric_keypad", 35, "Input device register confirms 35 units of numeric_keypad ready for deployment."),
    ("ribbon_cable_harness", 110, "Wiring depot records 110 units of ribbon_cable_harness on hand."),
]

# Category 5: Anti-Code Attractor Guard (varied 2-key grounded facts with negative prompts)
TRAIN_ANTI_CODE_FACTS = [
    ("service_worker", "operational", 0, "status", "code", "Worker service_worker completed task normally with operational status and code 0."),
    ("script_runner", "terminated", 1, "status", "code", "Automation script_runner exited abruptly with terminated status and code 1."),
    ("sync_daemon", "finished", 0, "status", "code", "Data replication sync_daemon reports finished state and return code 0."),
    ("index_builder", "ready", 200, "status", "code", "Search index_builder initialized successfully with ready status and code 200."),
    ("event_bus", "active", 200, "status", "code", "Pub-sub event_bus is fully active with status code 200."),
    ("cache_purger", "idle", 200, "status", "code", "Maintenance cache_purger finished sweeps and is idle with code 200."),
    ("auth_verifier", "verified", 200, "status", "code", "Security auth_verifier passed all checks with verified status and code 200."),
    ("stream_encoder", "encoding", 202, "status", "code", "Video stream_encoder is actively encoding chunks with HTTP code 202."),
    ("db_migrator", "applied", 200, "status", "code", "Schema db_migrator applied migrations successfully with status applied (200)."),
    ("health_beacon", "alive", 200, "status", "code", "Watchdog health_beacon pulses alive status with code 200."),
    ("backup_agent", "completed", 200, "status", "code", "Nightly backup_agent run completed with return code 200."),
    ("proxy_router", "routing", 200, "status", "code", "Edge proxy_router is currently routing requests with HTTP code 200."),
    ("log_shipper", "streaming", 202, "status", "code", "Centralized log_shipper is streaming records with code 202."),
    ("alert_dispatcher", "armed", 200, "status", "code", "Threshold alert_dispatcher is armed and monitoring with code 200."),
    ("cert_renewer", "renewed", 200, "status", "code", "SSL cert_renewer verified root chain with renewed status and code 200."),
    ("job_reaper", "cleaned", 200, "status", "code", "Zombie job_reaper cleaned orphan processes with status cleaned and code 200."),
    ("dns_resolver", "bound", 200, "status", "code", "Internal dns_resolver successfully bound port 53 with code 200."),
    ("metric_collector", "polling", 200, "status", "code", "Prometheus metric_collector is polling endpoints with code 200."),
    ("package_syncer", "synced", 200, "status", "code", "Repository package_syncer completed sync with code 200."),
    ("firewall_agent", "enforcing", 200, "status", "code", "Packet firewall_agent is actively enforcing rules with status code 200."),
]

VAL_ANTI_CODE_FACTS = [
    ("kernel_module", "loaded", 0, "status", "code", "Driver kernel_module loaded into memory with status loaded and code 0."),
    ("telemetry_sender", "dispatched", 202, "status", "code", "Metrics telemetry_sender dispatched batch with status dispatched and code 202."),
    ("snapshot_engine", "saved", 200, "status", "code", "State snapshot_engine saved state with status saved and code 200."),
    ("quota_enforcer", "throttled", 429, "status", "code", "Rate quota_enforcer throttled excessive requests with status throttled and code 429."),
    ("audit_collector", "archived", 200, "status", "code", "Compliance audit_collector archived logs with status archived and code 200."),
]

# Category 6: Missing Information Abstention (schema requests key, but context lacks data)
TRAIN_ABSTENTION_FACTS = [
    ("node_charlie_1", "Cluster node_charlie_1 rebooted, but no telemetry status or error code was provided.", "status", "code"),
    ("node_charlie_2", "Network scan detected node_charlie_2 alive, however numerical latency was unmeasured.", "metric", "value"),
    ("node_charlie_3", "User profile node_charlie_3 updated preferences, but dark_mode flag was not specified.", "feature", "enabled"),
    ("node_charlie_4", "Warehouse pallet node_charlie_4 contains fiber cables, but total quantity is uncounted.", "name", "count"),
    ("node_charlie_5", "System event logged on node_charlie_5 without operational state or return code.", "status", "code"),
    ("node_charlie_6", "Sensor node_charlie_6 signaled an interrupt, but the physical value was omitted.", "metric", "value"),
    ("node_charlie_7", "Security manifest for node_charlie_7 was created without specifying auto_save state.", "feature", "enabled"),
    ("node_charlie_8", "Logistics intake registered node_charlie_8 parts, but unit count remains blank.", "name", "count"),
    ("node_charlie_9", "Service heartbeat from node_charlie_9 lacked HTTP status and code information.", "status", "code"),
    ("node_charlie_10", "Probe on node_charlie_10 triggered, but metric reading timed out.", "metric", "value"),
    ("node_charlie_11", "Feature register for node_charlie_11 omitted the compression flag setting.", "feature", "enabled"),
    ("node_charlie_12", "Hardware delivery node_charlie_12 received, but item quantity was missing.", "name", "count"),
    ("node_charlie_13", "Process monitor pinged node_charlie_13 without receiving status or code.", "status", "code"),
    ("node_charlie_14", "Voltage sensor on node_charlie_14 reported an error without value telemetry.", "metric", "value"),
    ("node_charlie_15", "Configuration file node_charlie_15 did not define the logging flag.", "feature", "enabled"),
    ("node_charlie_16", "Inventory audit note for node_charlie_16 lacks the final stock count.", "name", "count"),
    ("node_charlie_17", "Diagnostics run on node_charlie_17 concluded without providing a status.", "status", "code"),
    ("node_charlie_18", "Telemetry gateway for node_charlie_18 failed to capture packet_loss reading.", "metric", "value"),
    ("node_charlie_19", "System setting for node_charlie_19 lacks an enabled flag for sandbox mode.", "feature", "enabled"),
    ("node_charlie_20", "Receiving dock node_charlie_20 received brackets with count omitted.", "name", "count"),
]

VAL_ABSTENTION_FACTS = [
    ("val_abstain_01", "Dispatch record for val_abstain_01 is pending with status and return code absent.", "status", "code"),
    ("val_abstain_02", "Acoustic monitor val_abstain_02 detected sound without recording a decibel value.", "metric", "value"),
    ("val_abstain_03", "Application options for val_abstain_03 do not contain an enabled setting for notifications.", "feature", "enabled"),
    ("val_abstain_04", "Storage shelf val_abstain_04 holds heatsinks but quantity was not inventoried.", "name", "count"),
    ("val_abstain_05", "Health check for val_abstain_05 failed to transmit status or numeric return code.", "status", "code"),
]


# ==============================================================================
# 2. TEMPLATE PHRASINGS (Strict Template-Family Disjointness)
# ==============================================================================

TRAIN_TEMPLATES = {
    "server_status": [
        ("tmpl_trn_srv_1", "Context: {context}\nInstruction: Extract the server status and code as a JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_srv_2", "Context: {context}\nInstruction: Output a JSON object containing keys '{k1}' and '{k2}' indicating health status and code.\nResponse:"),
        ("tmpl_trn_srv_3", "Context: {context}\nInstruction: Parse the given telemetry and return a JSON object with fields '{k1}' and '{k2}'.\nResponse:"),
    ],
    "telemetry_metric": [
        ("tmpl_trn_met_1", "Context: {context}\nInstruction: Represent this telemetry measurement as a JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_met_2", "Context: {context}\nInstruction: Format the sensor reading as a JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_met_3", "Context: {context}\nInstruction: Extract the telemetry metric name and measured value as a JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
    ],
    "feature_flag": [
        ("tmpl_trn_flg_1", "Context: {context}\nInstruction: Return a JSON object with keys '{k1}' and '{k2}' reflecting this feature state.\nResponse:"),
        ("tmpl_trn_flg_2", "Context: {context}\nInstruction: Format the configuration state as a JSON object containing '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_flg_3", "Context: {context}\nInstruction: Output a JSON object specifying feature flag keys '{k1}' and '{k2}'.\nResponse:"),
    ],
    "inventory_item": [
        ("tmpl_trn_inv_1", "Context: {context}\nInstruction: Output a JSON object with keys '{k1}' and '{k2}' for this inventory item.\nResponse:"),
        ("tmpl_trn_inv_2", "Context: {context}\nInstruction: Generate a JSON object containing item keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_inv_3", "Context: {context}\nInstruction: Record the stock accounting as a JSON object with fields '{k1}' and '{k2}'.\nResponse:"),
    ],
    "anti_code": [
        ("tmpl_trn_anti_1", "Context: {context}\nInstruction: Do NOT write Python code, doctests, or import modules. Output ONLY a valid JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_trn_anti_2", "Context: {context}\nInstruction: Output strictly a JSON object with keys '{k1}' and '{k2}'. Do not generate script or comments.\nResponse:"),
        ("tmpl_trn_anti_3", "Context: {context}\nInstruction: Do NOT import doctest or write Python functions. Return ONLY a JSON object containing keys '{k1}' and '{k2}'.\nResponse:"),
    ],
    "abstention": [
        ("tmpl_trn_abs_1", "Context: {context}\nInstruction: Extract fields '{k1}' and '{k2}' as JSON. If the data is missing from context, return null values with error 'missing_information'.\nResponse:"),
        ("tmpl_trn_abs_2", "Context: {context}\nInstruction: Produce a JSON object for '{k1}' and '{k2}'. If the context lacks this information, output null values and error 'missing_information'.\nResponse:"),
        ("tmpl_trn_abs_3", "Context: {context}\nInstruction: Return a JSON object with keys '{k1}' and '{k2}'. Set values to null and set 'error' to 'missing_information' if unrecorded.\nResponse:"),
    ],
}

VAL_TEMPLATES = {
    "server_status": [
        ("tmpl_val_srv_1", "Context: {context}\nInstruction: From the report above, construct a JSON response with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_val_srv_2", "Context: {context}\nInstruction: Provide a JSON object with attributes '{k1}' and '{k2}' derived from the context.\nResponse:"),
    ],
    "telemetry_metric": [
        ("tmpl_val_met_1", "Context: {context}\nInstruction: From the text, capture the measurement as a JSON object with properties '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_val_met_2", "Context: {context}\nInstruction: Provide a JSON object with fields '{k1}' and '{k2}' from this reading.\nResponse:"),
    ],
    "feature_flag": [
        ("tmpl_val_flg_1", "Context: {context}\nInstruction: State the flag configuration as a JSON object specifying keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_val_flg_2", "Context: {context}\nInstruction: Provide a JSON response showing attributes '{k1}' and '{k2}'.\nResponse:"),
    ],
    "inventory_item": [
        ("tmpl_val_inv_1", "Context: {context}\nInstruction: Summarize the stock item as a JSON object with properties '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_val_inv_2", "Context: {context}\nInstruction: Provide a JSON object with attributes '{k1}' and '{k2}' based on the stock report.\nResponse:"),
    ],
    "anti_code": [
        ("tmpl_val_anti_1", "Context: {context}\nInstruction: Strictly avoid Python statements or doctest definitions. Return ONLY a JSON object with keys '{k1}' and '{k2}'.\nResponse:"),
        ("tmpl_val_anti_2", "Context: {context}\nInstruction: Produce purely a JSON object with keys '{k1}' and '{k2}'. Never output code blocks or doctest imports.\nResponse:"),
    ],
    "abstention": [
        ("tmpl_val_abs_1", "Context: {context}\nInstruction: Provide a JSON object for '{k1}' and '{k2}'. When information is missing, report error 'missing_information' and set fields to null.\nResponse:"),
        ("tmpl_val_abs_2", "Context: {context}\nInstruction: Construct a JSON response with keys '{k1}' and '{k2}'. If context omits data, abstain with error 'missing_information' and null fields.\nResponse:"),
    ],
}


# ==============================================================================
# 3. BUILDER LOGIC
# ==============================================================================

def create_structured_record(
    rid: str,
    split: str,
    category: str,
    template_tuple: Tuple[str, str],
    context: str,
    instruction_k1: str,
    instruction_k2: str,
    expected_dict: Dict[str, Any],
    field_types: Dict[str, str],
    required_keys: List[str],
    difficulty: str = "standard",
) -> Dict[str, Any]:
    tmpl_id, tmpl_fmt = template_tuple
    prompt = tmpl_fmt.format(context=context, k1=instruction_k1, k2=instruction_k2)
    # Target response format: compact, no space after comma or colon in keys, single space after colon
    # Standard json.dumps uses (', ', ': ') by default; compact json formatting:
    response = json.dumps(expected_dict)

    # Double verify parsing
    parsed = json.loads(response)
    assert parsed == expected_dict, f"Serialization mismatch: {parsed} vs {expected_dict}"

    return {
        "id": rid,
        "category": category,
        "template_family_id": tmpl_id,
        "context": context,
        "instruction": prompt.split("Instruction: ")[1].split("\nResponse:")[0].strip(),
        "prompt": prompt,
        "response": response,
        "expected_answer": expected_dict,
        "field_types": field_types,
        "required_keys": required_keys,
        "optional_keys": [],
        "difficulty": difficulty,
        "provenance": "SYNTHETIC_GROUNDED_CURRICULUM_V3",
        "data_role": split.upper(),
        "validation_rule": "exact_key_value_match",
    }


def build_structured_split(split: str) -> List[Dict[str, Any]]:
    records = []
    tmpls = TRAIN_TEMPLATES if split == "train" else VAL_TEMPLATES

    # 1. Server Status (25 train, 5 val)
    server_facts = TRAIN_SERVER_FACTS if split == "train" else VAL_SERVER_FACTS
    for idx, (node, status, code, ctx) in enumerate(server_facts, start=1):
        tmpl = tmpls["server_status"][idx % len(tmpls["server_status"])]
        expected = {"status": status, "code": code}
        ftypes = {"status": "string", "code": "integer"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_SRV_{idx:03d}",
            split=split,
            category="grounded_extraction_server_status",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1="status",
            instruction_k2="code",
            expected_dict=expected,
            field_types=ftypes,
            required_keys=["status", "code"],
        )
        records.append(rec)

    # 2. Telemetry Metric (25 train, 5 val)
    metric_facts = TRAIN_METRIC_FACTS if split == "train" else VAL_METRIC_FACTS
    for idx, (met, val, ctx) in enumerate(metric_facts, start=1):
        tmpl = tmpls["telemetry_metric"][idx % len(tmpls["telemetry_metric"])]
        expected = {"metric": met, "value": val}
        ftypes = {"metric": "string", "value": "number"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_MET_{idx:03d}",
            split=split,
            category="grounded_telemetry_metric",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1="metric",
            instruction_k2="value",
            expected_dict=expected,
            field_types=ftypes,
            required_keys=["metric", "value"],
        )
        records.append(rec)

    # 3. Feature Flag (25 train, 5 val)
    flag_facts = TRAIN_FLAG_FACTS if split == "train" else VAL_FLAG_FACTS
    for idx, (flag, enabled, ctx) in enumerate(flag_facts, start=1):
        tmpl = tmpls["feature_flag"][idx % len(tmpls["feature_flag"])]
        expected = {"feature": flag, "enabled": enabled}
        ftypes = {"feature": "string", "enabled": "boolean"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_FLG_{idx:03d}",
            split=split,
            category="grounded_feature_flag",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1="feature",
            instruction_k2="enabled",
            expected_dict=expected,
            field_types=ftypes,
            required_keys=["feature", "enabled"],
        )
        records.append(rec)

    # 4. Inventory Item (25 train, 5 val)
    inventory_facts = TRAIN_INVENTORY_FACTS if split == "train" else VAL_INVENTORY_FACTS
    for idx, (name, count, ctx) in enumerate(inventory_facts, start=1):
        tmpl = tmpls["inventory_item"][idx % len(tmpls["inventory_item"])]
        expected = {"name": name, "count": count}
        ftypes = {"name": "string", "count": "integer"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_INV_{idx:03d}",
            split=split,
            category="grounded_inventory_item",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1="name",
            instruction_k2="count",
            expected_dict=expected,
            field_types=ftypes,
            required_keys=["name", "count"],
        )
        records.append(rec)

    # 5. Anti-Code Attractor Guard (20 train, 5 val)
    anti_facts = TRAIN_ANTI_CODE_FACTS if split == "train" else VAL_ANTI_CODE_FACTS
    for idx, (item, val1, val2, k1, k2, ctx) in enumerate(anti_facts, start=1):
        tmpl = tmpls["anti_code"][idx % len(tmpls["anti_code"])]
        expected = {k1: val1, k2: val2}
        ftypes = {k1: "string", k2: "integer"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_ATC_{idx:03d}",
            split=split,
            category="anti_code_attractor_guard",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1=k1,
            instruction_k2=k2,
            expected_dict=expected,
            field_types=ftypes,
            required_keys=[k1, k2],
            difficulty="anti_attractor",
        )
        records.append(rec)

    # 6. Missing Information Abstention (20 train, 5 val)
    abs_facts = TRAIN_ABSTENTION_FACTS if split == "train" else VAL_ABSTENTION_FACTS
    for idx, (item, ctx, k1, k2) in enumerate(abs_facts, start=1):
        tmpl = tmpls["abstention"][idx % len(tmpls["abstention"])]
        expected = {"error": "missing_information", k1: None, k2: None}
        ftypes = {"error": "string", k1: "null", k2: "null"}
        rec = create_structured_record(
            rid=f"SFT3_{split.upper()[:3]}_ABS_{idx:03d}",
            split=split,
            category="missing_information_abstention",
            template_tuple=tmpl,
            context=ctx,
            instruction_k1=k1,
            instruction_k2=k2,
            expected_dict=expected,
            field_types=ftypes,
            required_keys=["error", k1, k2],
            difficulty="abstention",
        )
        records.append(rec)

    return records


def extract_isolated_replay(
    train_count: int = 60,
    val_count: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts strictly disjoint replay continuation examples from raw text corpora."""
    random.seed(RANDOM_SEED)

    # Completely disjoint corpus files for train vs val
    train_corpus_paths = [
        RAW_DATA_DIR / "english" / "english_computer_science.txt",
        RAW_DATA_DIR / "english" / "stage_b_systems.txt",
    ]
    val_corpus_paths = [
        RAW_DATA_DIR / "english" / "english_systems.txt",
    ]

    train_lines: List[Tuple[str, str]] = []
    for p in train_corpus_paths:
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if len(l.strip()) >= 50 and not l.strip().startswith("#") and not l.strip().startswith("assert")]
        for l in lines:
            train_lines.append((l, p.name))

    val_lines: List[Tuple[str, str]] = []
    for p in val_corpus_paths:
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if len(l.strip()) >= 50 and not l.strip().startswith("#") and not l.strip().startswith("assert")]
        for l in lines:
            val_lines.append((l, p.name))

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    train_selected = train_lines[:train_count]
    val_selected = val_lines[:val_count]

    assert len(train_selected) == train_count, f"Insufficient train replay lines: {len(train_selected)}"
    assert len(val_selected) == val_count, f"Insufficient val replay lines: {len(val_selected)}"

    def make_replay_records(items: List[Tuple[str, str]], split: str) -> List[Dict[str, Any]]:
        recs = []
        for idx, (line, prov) in enumerate(items, start=1):
            words = line.split()
            split_idx = max(3, len(words) // 2)
            prompt_str = " ".join(words[:split_idx]) + " "
            resp_str = " ".join(words[split_idx:])
            recs.append({
                "id": f"SFT3_{split.upper()[:3]}_REP_{idx:03d}",
                "category": "base_continuation_replay",
                "template_family_id": f"replay_continuation_{split}",
                "context": None,
                "instruction": None,
                "prompt": prompt_str,
                "response": resp_str,
                "expected_answer": resp_str,
                "field_types": None,
                "required_keys": [],
                "optional_keys": [],
                "difficulty": "replay",
                "provenance": f"PRETRAINING_CORPUS_{prov.upper().replace('.TXT', '')}",
                "data_role": split.upper(),
                "append_eos": False,  # Strict continuation policy
                "validation_rule": "replay_token_continuation",
            })
        return recs

    return make_replay_records(train_selected, "train"), make_replay_records(val_selected, "val")


def build_sft_v3_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Builds full SFT V3 train (200 records) and validation (40 records) datasets."""
    random.seed(RANDOM_SEED)

    train_structured = build_structured_split("train")  # 140 records
    val_structured = build_structured_split("val")      # 30 records

    train_replay, val_replay = extract_isolated_replay(train_count=60, val_count=10)

    train_all = train_structured + train_replay  # 140 + 60 = 200
    val_all = val_structured + val_replay        # 30 + 10 = 40

    random.shuffle(train_all)
    random.shuffle(val_all)

    assert len(train_all) == 200, f"Expected exactly 200 train records, got {len(train_all)}"
    assert len(val_all) == 40, f"Expected exactly 40 val records, got {len(val_all)}"

    return train_all, val_all


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_recs, val_recs = build_sft_v3_dataset()

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully generated {len(train_recs)} train records -> {train_path}")
    print(f"Successfully generated {len(val_recs)} val records -> {val_path}")


if __name__ == "__main__":
    main()
