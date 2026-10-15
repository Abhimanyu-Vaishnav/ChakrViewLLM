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
    
    # Gate check
    gate_res = a.evidence_gate.process_query(resolved)
    if gate_res.decision.value == 'REFUTE':
        dec = 'REFUTE'
        ans = f"Contradiction detected: {gate_res.content}"
    else:
        ret = a.retriever.retrieve_candidates(RetrievalQuery(text=resolved, top_k=3, lexical_weight=0.8, semantic_weight=0.2))
        stops = {'what', 'when', 'where', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'from', 'with', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'have', 'has', 'had', 'does', 'did', 'doing', 'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'by', 'for', 'and', 'or', 'to', 'of', 'how', 'many', 'much', 'do', 'can', 'are', 'is'}
        content_tokens = {w.rstrip('s') for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', resolved.lower()) if w not in stops}
        
        cand = ret.candidates[0] if ret.candidates else None
        if not cand or cand.score < 0.25:
            dec = 'ABSTAIN'
            ans = 'I could not find sufficient verified evidence in the indexed local documents to answer this question.'
        else:
            cand_tokens = {w.rstrip('s') for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', cand.text.lower())}
            overlap = content_tokens.intersection(cand_tokens)
            if not overlap or len(overlap) < max(1, min(2, len(content_tokens))):
                dec = 'ABSTAIN'
                ans = 'I could not find sufficient verified evidence in the indexed local documents to answer this question.'
            else:
                dec = 'ANSWER'
                doc_id = cand.source_id
                start_c = cand.metadata.get('start_char', 0)
                end_c = cand.metadata.get('end_char', len(cand.text))
                ans = f"{cand.text.strip()} [Source: {doc_id} (char {start_c}-{end_c})]"
    
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
