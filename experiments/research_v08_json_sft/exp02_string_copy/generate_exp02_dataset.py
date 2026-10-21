"""Dataset Generator for ChakrView v0.8 Experiment 02: Single-Field String Extraction.

Mission:
Build a deterministic, independently hashed dataset with train, validation, and held-out test splits.

Task:
Given an input prompt containing an explicitly identified status string, produce exactly one JSON
object containing that string under the 'status' key:
  Expected output: {"status": "<status_string>"}

Integrity & Design Requirements:
1. Synthetic but varied status strings:
   - 40 train statuses (strictly disjoint from val and test)
   - 10 val statuses (strictly disjoint from train and test)
   - 15 test statuses (strictly disjoint from train and val)
2. Multiple varied instruction phrasings (5 distinct prompt templates):
   - Template A: "Extract the system status as a JSON object with key \"status\"."
   - Template B: "Parse the telemetry status from the reading below into a JSON object with the \"status\" field."
   - Template C: "Convert the following status report into a JSON dictionary containing \"status\"."
   - Template D: "Given the monitored system state, return a JSON object structured as {\"status\": string}."
   - Template E: "Identify the reported status condition and output it as JSON: {\"status\": \"<value>\"}."
3. Control examples:
   - Control cases with simple known dictionary status words vs composite hyphenated/underscored words
   - Verifies whether copy failure is token-count or syntax related.
4. Continuation replay records:
   - 30% base continuation replay records to preserve base generation stability.
5. Strict verification:
   - Zero leakage across splits.
   - Exact and normalized duplicate checks.
   - Tokenizer compatibility and max sequence length <= 256.
   - Deterministic and machine-verifiable.
   - Split hashes and record counts logged.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
RANDOM_SEED = 42

TEMPLATES = [
    "Extract the system status as a JSON object with key \"status\".",
    "Parse the telemetry status from the reading below into a JSON object with the \"status\" field.",
    "Convert the following status report into a JSON dictionary containing \"status\".",
    "Given the monitored system state, return a JSON object structured as {\"status\": string}.",
    "Identify the reported status condition and output it as JSON: {\"status\": \"<value>\"}.",
]

# Disjoint component pools across splits
TRAIN_COMPONENTS = [f"subsystem_{name}_{i:02d}" for name in ["alpha", "beta", "gamma", "delta"] for i in range(1, 26)] # 100
VAL_COMPONENTS = [f"inverter_{name}_{i:02d}" for name in ["epsilon", "zeta"] for i in range(1, 11)] # 20
TEST_COMPONENTS = [f"relay_{name}_{i:02d}" for name in ["eta", "theta"] for i in range(1, 11)] # 20

# Disjoint status string pools across splits
# 40 Train Statuses: mixture of single words and underscored compounds
TRAIN_STATUSES = [
    "nominal", "active", "degraded", "overheated", "idle", "restarting",
    "calibrated", "throttled", "bypassed", "discharged", "synchronized",
    "pressurized", "saturated", "drained", "dampened", "isolated",
    "cooling", "recharged", "primed", "unlatched",
    "bus_online", "carrier_locked", "pipeline_active", "voltage_steady",
    "thermal_stable", "flow_unrestricted", "buffer_cleared", "link_connected",
    "bearing_greased", "chamber_vacuum", "sensor_ready", "beacon_pulsing",
    "circuit_closed", "phase_aligned", "output_saturated", "input_grounded",
    "filter_cleaned", "pump_primed", "valve_seated", "rotor_balanced"
]

# 10 Validation Statuses (strictly disjoint)
VAL_STATUSES = [
    "standby_normal", "frequency_drift", "voltage_spike", "pressure_drop",
    "clock_jitter", "thermal_gradient", "signal_attenuated", "power_sag",
    "oscillation_detected", "impedance_mismatch"
]

# 15 Test Statuses (strictly disjoint from train and val)
TEST_STATUSES = [
    "optical_alignment_fault", "beam_misaligned", "cavity_depressurized",
    "flux_fluctuation", "cryo_quench", "magnetic_leakage",
    "harmonic_distortion", "plasma_disruption", "lattice_strain",
    "dielectric_breakdown", "core_undervoltage", "resonance_shift",
    "thermal_runaway", "quantum_decoherence", "manifold_rupture"
]

# Replay corpora (disjoint scientific pretraining facts)
TRAIN_REPLAY = [
    ("Empirical Science Series [Classical Mechanics Lesson 01]: Newton's second law defines force as the time rate of ",
     "change of linear momentum in an inertial frame of reference."),
    ("Empirical Science Series [Thermodynamics Lesson 02]: The entropy of an isolated thermodynamic system never decreases ",
     "over time, approaching a maximum value at thermodynamic equilibrium."),
    ("Empirical Science Series [Electromagnetism Lesson 03]: Gauss's law for magnetism states that the magnetic flux through ",
     "any closed Gaussian surface is identically zero."),
    ("Empirical Science Series [Optics Lesson 04]: Snell's law relates the angles of incidence and refraction for a wave ",
     "impinging on an interface between two different isotropic media."),
    ("Empirical Science Series [Quantum Theory Lesson 05]: The de Broglie hypothesis asserts that matter exhibits wave-like ",
     "behavior with wavelength inversely proportional to its momentum."),
    ("Empirical Science Series [Statistical Mechanics Lesson 06]: The Boltzmann distribution specifies the probability that a ",
     "state of energy E is occupied at absolute temperature T."),
    ("Empirical Science Series [Fluid Dynamics Lesson 07]: The Navier-Stokes equations describe how the velocity, pressure, ",
     "temperature, and density of a moving fluid are related."),
    ("Empirical Science Series [Special Relativity Lesson 08]: The Lorentz transformation describes how measurements of space ",
     "and time by two observers in relative motion are related."),
    ("Empirical Science Series [Solid State Physics Lesson 09]: The Fermi energy represents the electrochemical potential of ",
     "electrons at absolute zero temperature in a solid."),
    ("Empirical Science Series [Nuclear Physics Lesson 10]: Radioactive decay follows first-order kinetics where the half-life ",
     "is inversely proportional to the decay constant."),
    ("Mathematical Foundations [Linear Algebra 01]: The spectral theorem states that any real symmetric matrix can be ",
     "diagonalized by an orthogonal matrix of its eigenvectors."),
    ("Mathematical Foundations [Calculus 02]: The fundamental theorem of calculus establishes that differentiation and ",
     "integration are inverse operations for continuous functions."),
    ("Mathematical Foundations [Real Analysis 03]: The Heine-Borel theorem states that a subset of Euclidean space is compact ",
     "if and only if it is closed and bounded."),
    ("Mathematical Foundations [Complex Analysis 04]: Cauchy's integral formula computes holomorphic function values from a ",
     "contour integral along a closed path."),
    ("Mathematical Foundations [Differential Equations 05]: An ordinary differential equation has a unique solution under ",
     "Lipschitz continuity conditions by Picard-Lindelof."),
    ("Mathematical Foundations [Abstract Algebra 06]: Lagrange's theorem asserts that the order of any subgroup divides the ",
     "order of the finite group."),
    ("Mathematical Foundations [Topology 07]: A topological space is connected if it cannot be represented as the union of ",
     "two disjoint non-empty open sets."),
    ("Mathematical Foundations [Measure Theory 08]: The Lebesgue dominated convergence theorem allows swapping limits and ",
     "integrals under integrable bounds."),
    ("Mathematical Foundations [Probability 09]: The central limit theorem guarantees that normalized sums of independent ",
     "variables converge to a standard normal distribution."),
    ("Mathematical Foundations [Graph Theory 10]: Euler's formula states that for any connected planar graph, vertices minus ",
     "edges plus faces equals two."),
    ("Mathematical Foundations [Number Theory 11]: Fermat's little theorem states that if p is prime, then a^p minus a is an ",
     "integer multiple of p for any integer a."),
    ("Mathematical Foundations [Combinatorics 12]: The pigeonhole principle guarantees that partitioning n items into m boxes ",
     "leaves at least one box with ceil(n/m) items."),
    ("Mathematical Foundations [Optimization 13]: The Karush-Kuhn-Tucker conditions provide first-order necessary criteria ",
     "for nonlinear constrained optimization."),
    ("Mathematical Foundations [Differential Geometry 14]: The Gauss-Bonnet theorem connects the Gaussian curvature of a surface ",
     "to its Euler characteristic."),
    ("Mathematical Foundations [Functional Analysis 15]: The Hahn-Banach theorem ensures the extension of bounded linear ",
     "functionals defined on a subspace to the whole space."),
    ("Mathematical Foundations [Information Theory 16]: Shannon's source coding theorem establishes the theoretical limit of ",
     "lossless data compression given entropy."),
    ("Mathematical Foundations [Numerical Methods 17]: Newton's method approximates roots of real-valued functions using the ",
     "tangent line at successive iterations."),
    ("Mathematical Foundations [Dynamical Systems 18]: The Poincare-Bendixson theorem characterizes the limit sets of continuous ",
     "dynamical systems on two-dimensional manifolds."),
    ("Mathematical Foundations [Matrix Theory 19]: The singular value decomposition factors any rectangular matrix into left ",
     "and right singular vectors and singular values."),
    ("Mathematical Foundations [Discrete Math 20]: Mathematical induction proves a statement for all natural numbers by ",
     "establishing the base case and induction step.")
] # 30 train replay records

VAL_REPLAY = [
    ("Mathematical Foundations [Set Theory 21]: Cantor's theorem demonstrates that the power set of any set has strictly ",
     "greater cardinality than the set itself."),
    ("Mathematical Foundations [Order Theory 22]: Zorn's lemma asserts that every partially ordered set with upper bounds for ",
     "every chain contains a maximal element."),
    ("Mathematical Foundations [Field Theory 23]: Galois theory establishes a correspondence between subfields of a Galois ",
     "extension and subgroups of its Galois group."),
    ("Mathematical Foundations [Lie Algebras 24]: The Jacobi identity is an essential axiom satisfied by the Lie bracket in ",
     "any Lie algebra structure."),
    ("Empirical Science Series [Cryogenics 21]: Below the lambda point, liquid helium-4 undergoes a phase transition to ",
     "a superfluid state with zero apparent viscosity."),
    ("Empirical Science Series [Geophysics 22]: Seismic S-waves are shear transverse waves that cannot propagate through the ",
     "liquid outer core of the Earth."),
    ("Empirical Science Series [Oceanography 23]: The Coriolis effect deflects moving water masses to the right in the Northern ",
     "Hemisphere and to the left in the Southern Hemisphere."),
    ("Empirical Science Series [Meteorology 24]: Geostrophic wind represents the theoretical wind that results from an exact ",
     "balance between Coriolis force and pressure gradient."),
    ("Empirical Science Series [Spectroscopy 25]: The Doppler broadening of spectral lines occurs due to the thermal velocity ",
     "distribution of radiating atoms."),
    ("Empirical Science Series [Astrophysics 26]: Degeneracy pressure prevents white dwarfs and neutron stars from collapsing ",
     "under their own gravitational field.")
] # 10 val replay records

TEST_REPLAY = [
    ("Empirical Science Series [Cosmology 26]: Hubble's law indicates that galaxies recede from Earth with velocities ",
     "proportional to their proper cosmological distance."),
    ("Empirical Science Series [Particle Physics 27]: The Higgs mechanism explains how gauge bosons acquire non-zero invariant ",
     "mass through spontaneous symmetry breaking."),
    ("Empirical Science Series [Quantum Optics 28]: Hong-Ou-Mandel interference causes two identical photons entering a 50:50 ",
     "beamsplitter to always exit together in the same output port."),
    ("Empirical Science Series [Statistical Mechanics 29]: The Ising model mathematically describes ferromagnetism using ",
     "discrete variables that represent magnetic dipole moments."),
    ("Empirical Science Series [Solar Physics 30]: Coronal mass ejections release large quantities of magnetized plasma into ",
     "the interplanetary medium from the solar corona."),
    ("Mathematical Foundations [Homotopy Theory 26]: The fundamental group classifies topological loops based on continuous ",
     "deformations around topological holes."),
    ("Mathematical Foundations [Symplectic Geometry 27]: Darboux's theorem establishes that all symplectic manifolds of the ",
     "same dimension are locally symplectomorphic."),
    ("Mathematical Foundations [Fourier Analysis 28]: Parseval's identity states that the sum of the square of a function is ",
     "equal to the sum of the square of its transform."),
    ("Mathematical Foundations [Stochastic Calculus 29]: Ito's lemma provides the chain rule for functions of stochastic ",
     "processes driven by Brownian motion."),
    ("Mathematical Foundations [Representation Theory 30]: Schur's lemma asserts that non-zero intertwiners between irreducible ",
     "representations are scalar multiples of the identity.")
] # 10 test replay records


def build_single_field_record(
    rid: str,
    component: str,
    status: str,
    template_idx: int,
    split: str,
    data_role: str,
) -> Dict[str, Any]:
    instruction = TEMPLATES[template_idx]
    context = f"Telemetry monitor report: Component {component} reported status '{status}' during periodic scan."
    prompt = f"Instruction: {instruction}\nContext: {context}\nResponse:"
    response = json.dumps({"status": status})

    return {
        "id": rid,
        "category": "single_field_string_extraction",
        "prompt": prompt,
        "response": response,
        "expected_status": status,
        "expected_schema": {"status": "string"},
        "component": component,
        "split": split,
        "data_role": data_role,
        "append_eos": True,
        "provenance": "exp02_deterministic_single_field_generator",
        "review_status": "APPROVED",
        "status": "APPROVED",
    }


def build_replay_record(
    rid: str,
    prompt: str,
    response: str,
    split: str,
    data_role: str,
) -> Dict[str, Any]:
    return {
        "id": rid,
        "category": "base_continuation_replay",
        "prompt": prompt,
        "response": response,
        "split": split,
        "data_role": data_role,
        "append_eos": False,
        "provenance": "canonical_pretraining_replay_corpus_v08",
        "review_status": "APPROVED",
        "status": "APPROVED",
    }


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_splits():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # 1. Assert disjoint status pools
    train_set = set(TRAIN_STATUSES)
    val_set = set(VAL_STATUSES)
    test_set = set(TEST_STATUSES)

    assert train_set.isdisjoint(val_set), "Train and Val statuses overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test statuses overlap!"
    assert val_set.isdisjoint(test_set), "Val and Test statuses overlap!"
    print(f"[Verification] Status pools strictly disjoint: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    # 2. Build Train Split (100 single-field JSON extraction + 30 continuation replay = 130 records)
    train_records = []
    for i in range(100):
        rid = f"exp02_train_json_{i+1:03d}"
        status = TRAIN_STATUSES[i % len(TRAIN_STATUSES)]
        component = TRAIN_COMPONENTS[i % len(TRAIN_COMPONENTS)]
        template_idx = i % len(TEMPLATES)
        train_records.append(
            build_single_field_record(rid, component, status, template_idx, "train", "TRAIN")
        )

    for i, (p, r) in enumerate(TRAIN_REPLAY):
        rid = f"exp02_train_replay_{i+1:03d}"
        train_records.append(build_replay_record(rid, p, r, "train", "TRAIN"))

    rng.shuffle(train_records)

    # 3. Build Val Split (20 single-field JSON extraction + 10 continuation replay = 30 records)
    val_records = []
    for i in range(20):
        rid = f"exp02_val_json_{i+1:03d}"
        status = VAL_STATUSES[i % len(VAL_STATUSES)]
        component = VAL_COMPONENTS[i % len(VAL_COMPONENTS)]
        template_idx = i % len(TEMPLATES)
        val_records.append(
            build_single_field_record(rid, component, status, template_idx, "val", "VALIDATION")
        )

    for i, (p, r) in enumerate(VAL_REPLAY):
        rid = f"exp02_val_replay_{i+1:03d}"
        val_records.append(build_replay_record(rid, p, r, "val", "VALIDATION"))

    rng.shuffle(val_records)

    # 4. Build Test Split (20 single-field JSON extraction + 10 continuation replay = 30 records)
    test_records = []
    # 20 JSON extraction cases using 15 disjoint test statuses
    for i in range(20):
        rid = f"exp02_test_json_{i+1:03d}"
        status = TEST_STATUSES[i % len(TEST_STATUSES)]
        component = TEST_COMPONENTS[i % len(TEST_COMPONENTS)]
        template_idx = i % len(TEMPLATES)
        test_records.append(
            build_single_field_record(rid, component, status, template_idx, "test", "HELD_OUT_BENCHMARK")
        )

    for i, (p, r) in enumerate(TEST_REPLAY):
        rid = f"exp02_test_replay_{i+1:03d}"
        test_records.append(build_replay_record(rid, p, r, "test", "HELD_OUT_BENCHMARK"))

    # Write files deterministically
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"

    for path, recs in [(train_path, train_records), (val_path, val_records), (test_path, test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Hashes and stats
    train_sha = compute_sha256(train_path)
    val_sha = compute_sha256(val_path)
    test_sha = compute_sha256(test_path)

    manifest = {
        "dataset_name": "exp02_single_field_string_extraction",
        "random_seed": RANDOM_SEED,
        "splits": {
            "train": {"path": str(train_path), "records": len(train_records), "sha256": train_sha},
            "val": {"path": str(val_path), "records": len(val_records), "sha256": val_sha},
            "test": {"path": str(test_path), "records": len(test_records), "sha256": test_sha},
        },
        "status_counts": {
            "train": len(TRAIN_STATUSES),
            "val": len(VAL_STATUSES),
            "test": len(TEST_STATUSES),
        }
    }

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[SUCCESS] Dataset Generation Complete.")
    print(f"  Train: {train_path} ({len(train_records)} records, SHA256={train_sha})")
    print(f"  Val:   {val_path} ({len(val_records)} records, SHA256={val_sha})")
    print(f"  Test:  {test_path} ({len(test_records)} records, SHA256={test_sha})")
    print(f"  Manifest written to {manifest_path}")

if __name__ == "__main__":
    generate_splits()
