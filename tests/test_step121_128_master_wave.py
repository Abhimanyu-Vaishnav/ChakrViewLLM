"""
Comprehensive test suite for Steps 121-128 Master Development Wave:
- test_step121_network_node_federation
- test_step122_secure_distributed_transport
- test_step123_distributed_cognitive_state_sync
- test_step124_federated_memory_plane
- test_step125_long_horizon_planning
- test_step126_neural_cognitive_boundary
- test_step127_federation_adaptation
- test_step128_canonical_baseline_bit_exact
"""

import time
from pathlib import Path
import pytest

from chakrview.cognition.multi_agent.contracts import WorkerRole, WorkerContract
from chakrview.cognition.multi_agent.transport import RequestEnvelope, PROTOCOL_VERSION_V1
from chakrview.cognition.multi_agent.network_federation import (
    TcpNetworkTransportChannel,
    TcpWorkerNodeServer,
)
from chakrview.cognition.multi_agent.secure_transport import (
    SecureEnvelope,
    DistributedSecurityValidator,
)
from chakrview.cognition.multi_agent.state_sync import (
    CognitiveStateCategory,
    VersionedCognitiveStateRecord,
    DistributedStateSynchronizer,
)
from chakrview.cognition.multi_agent.memory_plane import (
    ProvenanceMemoryRecord,
    FederatedMemoryPlane,
)
from chakrview.cognition.multi_agent.long_horizon import (
    PipelineStage,
    LongHorizonPlanner,
)
from chakrview.cognition.multi_agent.neural_boundary import (
    ModelTier,
    NeuralCognitiveBridge,
)
from chakrview.cognition.multi_agent.federation_adaptation import (
    FederationAdaptationManager,
)
from chakrview.cognition.multi_agent.federation import (
    FederatedScheduler,
    FederatedWorkerMetadata,
)
from chakrview.cognition.autonomous_benchmark import build_benchmark_tool_gate
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


@pytest.fixture(scope="module")
def net_wave_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("step121_128_env")
    from scripts.setup_step113_fixture import setup_step113_repository
    repo_dir = setup_step113_repository(tmp_dir)
    db_path = tmp_dir / "step128_brain.db"
    tool_gate, _ = build_benchmark_tool_gate(repo_dir)
    scheduler = FederatedScheduler(db_path, repo_dir, tool_gate)

    server = TcpWorkerNodeServer("tcp_node_test", host="127.0.0.1", port=0, workspace_root=repo_dir)
    port = server.start()
    time.sleep(0.1)

    yield {
        "root": tmp_dir,
        "repo": repo_dir,
        "db": db_path,
        "port": port,
        "scheduler": scheduler,
    }
    server.stop()


def test_step121_tcp_node_execution(net_wave_fixture):
    port = net_wave_fixture["port"]
    client = TcpNetworkTransportChannel("127.0.0.1", port)
    assert client.is_available() is True

    req = RequestEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id="proc_analyst",
        task_id="net_t1",
        package_id="pkg_net1",
        role=WorkerRole.PROJECT_ANALYST.value,
        context_hash="h1",
        context_token_count=50,
        allowed_tools=["read_file"],
        allowed_files=["app/models.py"],
        context_budget=512,
        package_payload={"task_id": "net_t1", "objective": "Read", "measured_context_tokens": 50},
    )
    resp = client.send_request(req, timeout_seconds=10.0)
    assert resp.result_status == "SUCCESS"


def test_step122_secure_transport_hmac_and_replay():
    validator = DistributedSecurityValidator()
    env = SecureEnvelope(
        inner_payload={"cmd": "run"},
        sender_node_id="master",
        recipient_node_id="worker1",
        nonce="nonce_abc",
    )
    env.sign()
    validator.validate_incoming_envelope(env)

    # Replay attempt fails
    with pytest.raises(ValueError, match="Replay detected"):
        validator.validate_incoming_envelope(env)


def test_step123_distributed_state_sync(net_wave_fixture):
    db_path = net_wave_fixture["db"]
    sync = DistributedStateSynchronizer(db_path)

    rec1 = VersionedCognitiveStateRecord("task_state_1", 1, "node_A", CognitiveStateCategory.TASK_STATE, {"status": "IN_PROGRESS"}, "hash_1")
    assert sync.publish_state_update(rec1) is True

    # Stale update fails
    rec_stale = VersionedCognitiveStateRecord("task_state_1", 1, "node_B", CognitiveStateCategory.TASK_STATE, {"status": "OLD"}, "hash_old")
    assert sync.publish_state_update(rec_stale) is False

    # Newer update succeeds
    rec2 = VersionedCognitiveStateRecord("task_state_1", 2, "node_B", CognitiveStateCategory.TASK_STATE, {"status": "DONE"}, "hash_2")
    assert sync.publish_state_update(rec2) is True


def test_step124_federated_memory_provenance(net_wave_fixture):
    db_path = net_wave_fixture["db"]
    mem = FederatedMemoryPlane(db_path)

    mem.store_fact(ProvenanceMemoryRecord("f1", "AccountService", "handles", "UserRegistration", "node_1", "task_1"))
    facts = mem.query_facts(subject="AccountService")
    assert len(facts) == 1
    assert facts[0].object_value == "UserRegistration"


def test_step125_long_horizon_checkpointing(net_wave_fixture):
    sched = net_wave_fixture["scheduler"]
    planner = LongHorizonPlanner(sched, "pipe_test")

    stage, _, _ = planner.get_pipeline_state()
    assert stage == PipelineStage.ANALYZE

    planner.checkpoint_stage(PipelineStage.TEST, 1, {"verified": True})
    new_stage, revs, _ = planner.get_pipeline_state()
    assert new_stage == PipelineStage.TEST
    assert revs == 1


def test_step126_neural_cognitive_boundary():
    bridge = NeuralCognitiveBridge(ModelTier.CANONICAL_BASELINE)
    ppl = bridge.score_sequence_perplexity([10, 20, 30, 40])
    assert ppl > 0.0
    assert bridge.verify_baseline_immutability() is True


def test_step127_federation_adaptation(net_wave_fixture):
    db_path = net_wave_fixture["db"]
    mgr = FederationAdaptationManager(db_path)

    mgr.record_outcome("bad_node", WorkerRole.IMPLEMENTER, False, "TIMEOUT")
    mgr.record_outcome("bad_node", WorkerRole.IMPLEMENTER, True)
    mgr.record_outcome("bad_node", WorkerRole.IMPLEMENTER, False, "PROCESS_CRASH")

    rel = mgr.get_worker_reliability("bad_node", WorkerRole.IMPLEMENTER)
    assert round(rel, 2) == 0.33  # 1 success out of 3


def test_step128_canonical_baseline_bit_exact():
    model = instantiate_frozen_baseline()
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH
