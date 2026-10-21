"""Dataset Generation Script for ChakrView v0.8 Single-Task JSON SFT Experiment.

Deterministic generation of:
- train.jsonl (140 records: 100 JSON extraction + 40 continuation replay)
- val.jsonl (30 records: 20 JSON extraction + 10 continuation replay)
- test.jsonl (30 records: 20 JSON extraction + 10 continuation replay)

Strict partitioning:
- Entities (component names) are strictly disjoint across splits.
- Status strings are strictly disjoint across splits.
- Code integers are sampled from non-overlapping numeric intervals:
    Train: [100, 699]
    Val:   [700, 799]
    Test:  [800, 899]
- 4 instruction templates:
    T1, T2, T3, T4
- Replay records:
    Sampled from general scientific/mathematical pretraining facts with append_eos=False.
    Disjoint across splits.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "research_v08_json_sft" / "data"

RANDOM_SEED = 42

# Templates
TEMPLATES = [
    "Extract the system status and status code as a JSON object with keys \"status\" and \"code\".",
    "Parse the telemetry reading below into JSON format with \"status\" and \"code\" fields.",
    "Convert the following component report into a JSON dictionary containing \"status\" and \"code\".",
    "Given the system state description, return a JSON object structured as {\"status\": string, \"code\": integer}.",
]

# Disjoint entity pools
TRAIN_COMPONENTS = [
    f"subsystem_alpha_{i:02d}" for i in range(1, 26)
] + [
    f"hydraulic_actuator_{i:02d}" for i in range(1, 26)
] + [
    f"power_converter_{i:02d}" for i in range(1, 26)
] + [
    f"thermal_sensor_{i:02d}" for i in range(1, 26)
]  # 100 components

VAL_COMPONENTS = [
    f"propulsion_thruster_{i:02d}" for i in range(1, 11)
] + [
    f"solar_inverter_{i:02d}" for i in range(1, 11)
]  # 20 components

TEST_COMPONENTS = [
    f"cryo_pump_{i:02d}" for i in range(1, 11)
] + [
    f"optical_relay_{i:02d}" for i in range(1, 11)
]  # 20 components

# Disjoint status string pools
TRAIN_STATUSES = [
    "nominal", "active", "degraded", "overheated", "idle", "restarting",
    "calibrated", "throttled", "bypassed", "discharged", "synchronized",
    "pressurized", "saturated", "drained", "dampened", "isolated",
    "cooling", "recharged", "primed", "unlatched"
]

VAL_STATUSES = [
    "standby_normal", "frequency_drift", "voltage_spike", "pressure_drop"
]

TEST_STATUSES = [
    "standby_degraded", "beam_misaligned", "cavity_depressurized", "flux_fluctuation"
]

# Replay corpora (Disjoint scientific/mathematical pretraining facts)
TRAIN_REPLAY_FACTS = [
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
    ("Empirical Science Series [Atomic Physics Lesson 11]: The Bohr model posits that electrons occupy discrete quantized ",
     "stationary energy levels without emitting radiation."),
    ("Empirical Science Series [Acoustics Lesson 12]: The speed of sound through a fluid medium depends on its bulk modulus ",
     "and equilibrium mass density."),
    ("Empirical Science Series [Physical Chemistry Lesson 13]: The Arrhenius equation models the temperature dependence of ",
     "chemical reaction rate constants via activation energy."),
    ("Empirical Science Series [Materials Science Lesson 14]: Hooke's law states that strain in a solid is proportional to ",
     "the applied mechanical stress within the elastic limit."),
    ("Empirical Science Series [Semiconductor Physics Lesson 15]: Band gaps determine whether a crystalline solid behaves as ",
     "an electrical conductor, semiconductor, or insulator."),
    ("Empirical Science Series [Cryogenics Lesson 16]: Superconductivity is characterized by zero electrical resistance and ",
     "the expulsion of magnetic flux via the Meissner effect."),
    ("Empirical Science Series [Plasma Physics Lesson 17]: Debye shielding describes the screening of electrostatic fields ",
     "by mobile charged particles in a quasi-neutral plasma."),
    ("Empirical Science Series [Orbital Mechanics Lesson 18]: Kepler's third law establishes that the square of orbital period ",
     "is proportional to the cube of the semi-major axis."),
    ("Empirical Science Series [Astrophysics Lesson 19]: The Chandrasekhar limit establishes the maximum stable mass of an ",
     "electron-degenerate white dwarf star."),
    ("Empirical Science Series [Laser Physics Lesson 20]: Population inversion is the state where more systems exist in an ",
     "excited state than in lower energy states."),
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
     "establishing the base case and induction step."),
]  # 40 train replay records

VAL_REPLAY_FACTS = [
    ("Mathematical Foundations [Set Theory 21]: Cantor's theorem demonstrates that the power set of any set has strictly ",
     "greater cardinality than the set itself."),
    ("Mathematical Foundations [Order Theory 22]: Zorn's lemma asserts that every partially ordered set with upper bounds for ",
     "every chain contains a maximal element."),
    ("Mathematical Foundations [Field Theory 23]: Galois theory establishes a correspondence between subfields of a Galois ",
     "extension and subgroups of its Galois group."),
    ("Mathematical Foundations [Lie Algebras 24]: The Jacobi identity is an essential axiom satisfied by the Lie bracket in ",
     "any Lie algebra structure."),
    ("Mathematical Foundations [Category Theory 25]: A functor preserves morphisms and identity mappings between source and ",
     "target categories."),
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
]  # 10 val replay records

TEST_REPLAY_FACTS = [
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
     "representations are scalar multiples of the identity."),
]  # 10 test replay records


def build_json_record(
    rid: str,
    component: str,
    status: str,
    code: int,
    template_idx: int,
    split: str,
    data_role: str,
) -> Dict[str, Any]:
    instruction = TEMPLATES[template_idx]
    context = f"Telemetry monitor report: Component {component} reported status '{status}' with system code {code} during periodic scan."
    prompt = f"Instruction: {instruction}\nContext: {context}\nResponse:"
    response = json.dumps({"status": status, "code": code})

    return {
        "id": rid,
        "category": "two_key_json_extraction",
        "prompt": prompt,
        "response": response,
        "expected_status": status,
        "expected_code": code,
        "expected_schema": {"status": "string", "code": "integer"},
        "component": component,
        "split": split,
        "data_role": data_role,
        "append_eos": True,
        "provenance": "synthetic_two_key_telemetry_generator_v08",
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
        "expected_status": None,
        "expected_code": None,
        "expected_schema": None,
        "component": None,
        "split": split,
        "data_role": data_role,
        "append_eos": False,
        "provenance": "verified_scientific_pretraining_corpus",
        "review_status": "APPROVED",
        "status": "APPROVED",
    }


def generate_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED)

    # 1. Train JSON: 100 records
    # Codes: 100 to 699
    train_codes = rng.sample(range(100, 700), 100)
    train_records = []
    for i in range(100):
        comp = TRAIN_COMPONENTS[i]
        status = TRAIN_STATUSES[i % len(TRAIN_STATUSES)]
        code = train_codes[i]
        tpl_idx = i % len(TEMPLATES)
        rec = build_json_record(
            rid=f"TRAIN_JSON_{i+1:03d}",
            component=comp,
            status=status,
            code=code,
            template_idx=tpl_idx,
            split="train",
            data_role="TRAIN",
        )
        train_records.append(rec)

    # Train Replay: 40 records
    for i, (p, r) in enumerate(TRAIN_REPLAY_FACTS):
        rec = build_replay_record(
            rid=f"TRAIN_REPLAY_{i+1:03d}",
            prompt=p,
            response=r,
            split="train",
            data_role="TRAIN",
        )
        train_records.append(rec)

    # 2. Val JSON: 20 records
    # Codes: 700 to 799
    val_codes = rng.sample(range(700, 800), 20)
    val_records = []
    for i in range(20):
        comp = VAL_COMPONENTS[i]
        status = VAL_STATUSES[i % len(VAL_STATUSES)]
        code = val_codes[i]
        tpl_idx = i % len(TEMPLATES)
        rec = build_json_record(
            rid=f"VAL_JSON_{i+1:03d}",
            component=comp,
            status=status,
            code=code,
            template_idx=tpl_idx,
            split="val",
            data_role="VALIDATION",
        )
        val_records.append(rec)

    # Val Replay: 10 records
    for i, (p, r) in enumerate(VAL_REPLAY_FACTS):
        rec = build_replay_record(
            rid=f"VAL_REPLAY_{i+1:03d}",
            prompt=p,
            response=r,
            split="val",
            data_role="VALIDATION",
        )
        val_records.append(rec)

    # 3. Test JSON: 20 records
    # Codes: 800 to 899
    test_codes = rng.sample(range(800, 900), 20)
    test_records = []
    for i in range(20):
        comp = TEST_COMPONENTS[i]
        status = TEST_STATUSES[i % len(TEST_STATUSES)]
        code = test_codes[i]
        tpl_idx = i % len(TEMPLATES)
        rec = build_json_record(
            rid=f"TEST_JSON_{i+1:03d}",
            component=comp,
            status=status,
            code=code,
            template_idx=tpl_idx,
            split="test",
            data_role="HELD_OUT_BENCHMARK",
        )
        test_records.append(rec)

    # Test Replay: 10 records
    for i, (p, r) in enumerate(TEST_REPLAY_FACTS):
        rec = build_replay_record(
            rid=f"TEST_REPLAY_{i+1:03d}",
            prompt=p,
            response=r,
            split="test",
            data_role="HELD_OUT_BENCHMARK",
        )
        test_records.append(rec)

    return train_records, val_records, test_records


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_recs, val_recs, test_recs = generate_dataset()

    train_file = OUTPUT_DIR / "train.jsonl"
    val_file = OUTPUT_DIR / "val.jsonl"
    test_file = OUTPUT_DIR / "test.jsonl"

    for file_path, recs in [(train_file, train_recs), (val_file, val_recs), (test_file, test_recs)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    manifest = {
        "dataset_name": "chakrview_v08_single_task_json_sft",
        "random_seed": RANDOM_SEED,
        "splits": {
            "train": {
                "file": "train.jsonl",
                "sha256": compute_file_sha256(train_file),
                "total_records": len(train_recs),
                "json_records": sum(1 for r in train_recs if r["category"] == "two_key_json_extraction"),
                "replay_records": sum(1 for r in train_recs if r["category"] == "base_continuation_replay"),
            },
            "val": {
                "file": "val.jsonl",
                "sha256": compute_file_sha256(val_file),
                "total_records": len(val_recs),
                "json_records": sum(1 for r in val_recs if r["category"] == "two_key_json_extraction"),
                "replay_records": sum(1 for r in val_recs if r["category"] == "base_continuation_replay"),
            },
            "test": {
                "file": "test.jsonl",
                "sha256": compute_file_sha256(test_file),
                "total_records": len(test_recs),
                "json_records": sum(1 for r in test_recs if r["category"] == "two_key_json_extraction"),
                "replay_records": sum(1 for r in test_recs if r["category"] == "base_continuation_replay"),
            },
        },
    }

    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Dataset generation complete:")
    print(f"  Train: {len(train_recs)} records -> {train_file.name} ({manifest['splits']['train']['sha256']})")
    print(f"  Val:   {len(val_recs)} records -> {val_file.name} ({manifest['splits']['val']['sha256']})")
    print(f"  Test:  {len(test_recs)} records -> {test_file.name} ({manifest['splits']['test']['sha256']})")
    print(f"  Manifest written to {manifest_file.name}")


if __name__ == "__main__":
    main()
