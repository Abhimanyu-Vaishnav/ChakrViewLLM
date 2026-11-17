import sys
import torch
from pathlib import Path
REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner

for name, path in [
    ('UCIP-06 Medium', REPO_ROOT / 'releases' / 'ucip_06_candidate' / 'chakrview_ucip06_medium_model.pt'),
    ('UCIP-07 Medium', REPO_ROOT / 'releases' / 'ucip_07_candidates' / 'chakrview_ucip07_medium_model.pt'),
    ('UCIP-08 Medium', REPO_ROOT / 'releases' / 'ucip_08_candidates' / 'chakrview_ucip08_medium_model.pt'),
]:
    runner = DirectNeuralRunner(path)
    model = runner.model
    params = sum(p.numel() for p in model.parameters())
    print(f"=== {name} ===")
    print(f"  Architecture: {runner.arch_name}")
    print(f"  Total Params: {params:,}")
    print(f"  Config: {model.config}")
