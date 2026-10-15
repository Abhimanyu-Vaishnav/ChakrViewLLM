from pathlib import Path
import json
import re
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant
from chakrview.runtime.retrieval import RetrievalQuery

corpus_dir = Path('experiments/batch1021_1060/data/sample_corpus')
bench_file = Path('experiments/batch1021_1060/data/held_out_50_benchmark.json')

with open(bench_file, 'r', encoding='utf-8') as f:
    benchmark = json.load(f)

a = LocalKnowledgeAssistant()
a.ingest_directory(corpus_dir)

pass_count = 0
failures = []
stops = {
    'what', 'when', 'where', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'from', 'with', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'have', 'has', 'had', 'does', 'did', 'doing', 'the', 'a', 'an', 'is', 'was',
    'are', 'were', 'in', 'on', 'at', 'by', 'for', 'and', 'or', 'to', 'of', 'how', 'many',
    'much', 'do', 'can', 'are', 'is', 'total', 'does'
}

for item in benchmark:
    a.reset_conversation()
    if item.get('context_query'):
        a.ask(item['context_query'])
    
    clean_q = item['query'].strip()
    q_lower = clean_q.lower()
    is_follow_up = False
    resolved = clean_q
    if a.conversation_turns and any(w in q_lower for w in ['that', 'it', 'the document', 'the second', 'what about', 'which', 'previous']):
        is_follow_up = True
        recent = [t for t in reversed(a.conversation_turns) if t.role.value == 'user']
        if recent:
            resolved = f"{recent[0].content} {clean_q}"
    
    gate_res = a.evidence_gate.process_query(resolved)
    if gate_res.decision.value == 'REFUTE':
        dec = 'REFUTE'
        ans = f"Contradiction detected: {gate_res.content}"
    else:
        ret = a.retriever.retrieve_candidates(RetrievalQuery(text=resolved, top_k=5, lexical_weight=0.8, semantic_weight=0.2))
        q_raw_tokens = [w.rstrip('s') for w in re.findall(r'\b[a-zA-Z0-9_\.-]{3,}\b|\b\d+(?:\.\d+)?\b', resolved.lower())]
        content_tokens = [w for w in q_raw_tokens if w not in stops]
        
        # Score candidates by exact content token overlap
        scored_cands = []
        for c in ret.candidates:
            c_tokens = set([w.rstrip('s') for w in re.findall(r'\b[a-zA-Z0-9_\.-]{3,}\b|\b\d+(?:\.\d+)?\b', c.text.lower())])
            overlap = set(content_tokens).intersection(c_tokens)
            score = len(overlap) * 3.0 + (c.score * 0.5)
            scored_cands.append((score, overlap, c))
        
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_overlap, best_cand = scored_cands[0] if scored_cands else (0, set(), None)
        
        if not best_cand or not best_overlap or len(best_overlap) < 1 or (len(content_tokens) >= 3 and len(best_overlap) < 2):
            dec = 'ABSTAIN'
            ans = 'I could not find sufficient verified evidence in the indexed local documents to answer this question.'
        else:
            dec = 'ANSWER'
            doc_id = best_cand.source_id
            start_c = best_cand.metadata.get('start_char', 0)
            end_c = best_cand.metadata.get('end_char', len(best_cand.text))
            ans = f"{best_cand.text.strip()} [Source: {doc_id} (char {start_c}-{end_c})]"
    
    exp_t = item['expected_type']
    exp_s = item.get('expected_substring')
    passed = (dec == exp_t) and (not exp_s or exp_s.lower() in ans.lower())
    if passed:
        pass_count += 1
    else:
        failures.append((item['id'], item['category'], exp_t, dec, item['query']))

print(f"Total passed: {pass_count}/50")
for f in failures:
    print(f"  Fail ID {f[0]} [{f[1]}] Exp:{f[2]} Got:{f[3]} | Q: {f[4]}")
