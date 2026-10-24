import json, sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator
from experiments.overnight_runs.exp19_pretrain_readiness_20261024_125000.diversification_transform import Exp19ResponseDiversifier

gen = Exp18DatasetGenerator()
splits = gen.generate_all_splits()

diversifier = Exp19ResponseDiversifier()
trans_train, train_stats = diversifier.transform_split('train', splits['train'])
trans_val, val_stats = diversifier.transform_split('val', splits['val'])
trans_test, test_stats = diversifier.transform_split('test', splits['test'])

assert splits['val'] == trans_val
assert splits['test'] == trans_test

print('Validation and test splits verified 100% identical in memory!')

cat_max_pcts = {}
for cat in sorted(set(r['category'] for r in trans_train)):
    cat_recs = [r for r in trans_train if r['category'] == cat]
    first_two = Counter(' '.join(r['response'].split()[:2]) for r in cat_recs if len(r['response'].split()) >= 2)
    max_p, max_cnt = first_two.most_common(1)[0]
    pct = round(max_cnt / len(cat_recs) * 100, 2)
    cat_max_pcts[cat] = (max_p, max_cnt, pct)
    print(f'  {cat:25s}: max prefix "{max_p}" = {max_cnt}/150 ({pct}%)')
    assert pct <= 25.0, f'Category {cat} exceeded 25% threshold: {pct}%'

overall_first_two = Counter(' '.join(r['response'].split()[:2]) for r in trans_train if len(r['response'].split()) >= 2)
max_ov_p, max_ov_cnt = overall_first_two.most_common(1)[0]
ov_pct = round(max_ov_cnt / len(trans_train) * 100, 2)
print(f'Overall train max prefix: "{max_ov_p}" = {max_ov_cnt}/1200 ({ov_pct}%)')
assert ov_pct <= 25.0

print('ALL CATEGORIES STRICTLY <= 25.00%! Check PASSED!')
