import json

with open("experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/eval_outputs/cr16d_grand_evaluation_summary.json") as f:
    data = json.load(f)

for k, v in data.items():
    print("=" * 60)
    print(f"MODEL: {k}")
    print(f"  Copying: exact={v['copying']['exact']}/100, pres={v['copying']['presence']}/100, first={v['copying']['first_token']}/100")
    ret = v['retrieval']
    print(f"  Retrieval Total: {ret['total_exact']}/1500 ({ret['total_exact']/15:.2f}%)")
    for cat, score in ret['by_category'].items():
        print(f"    {cat}: {score}")
    comp = v['compositional']
    print(f"  Compositional Total: {comp['total_exact']}/1000 ({comp['total_exact']/10:.2f}%)")
    for grp, score in comp['by_group'].items():
        print(f"    {grp}: {score}")

