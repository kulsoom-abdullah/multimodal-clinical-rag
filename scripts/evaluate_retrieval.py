#!/usr/bin/env python3
"""
Evaluation Script for Clinical Trial RAG.
Imports the EXACT logic from query_rag.py to ensure fair testing.
"""
import json
import sys
import time
from pathlib import Path

# Add current directory to path so we can import query_rag
sys.path.append(str(Path(__file__).parent.parent))

# Import from your FIXED script
from scripts.query_rag import load_resources, build_dynamic_retriever

def main():
    print("🧪 Starting Retrieval Evaluation...")
    
    eval_file = Path("data/eval.jsonl")
    if not eval_file.exists():
        print(f"❌ Error: Eval file not found at {eval_file}")
        sys.exit(1)
        
    with open(eval_file, "r") as f:
        test_cases = [json.loads(line) for line in f]

    # 1. Load Shared Resources
    try:
        # This matches the return signature of your fixed script
        vectorstore, docstore, all_text_docs = load_resources()
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        sys.exit(1)

    total_recall = 0
    total_latency = 0
    
    print(f"\n{'QUERY':<50} | {'TYPE':<15} | {'FOUND?':<8} | {'TIME':<8}")
    print(f"{'-'*90}")
    
    for case in test_cases:
        query = case["query"]
        expected = case["expected_trial_id"]
        q_type = case["type"]
        
        start_time = time.time()
        
        # 2. Build Retriever using the logic from query_rag.py
        retriever = build_dynamic_retriever(query, vectorstore, docstore, all_text_docs)
        
        # 3. Run
        docs = retriever.invoke(query)
        latency = time.time() - start_time
        
        # 4. Score (Check if expected ID is in top 3 docs)
        found_ids = [doc.metadata.get("trial_id") for doc in docs[:3]]
        is_found = expected in found_ids
        
        score = 1.0 if is_found else 0.0
        total_recall += score
        total_latency += latency
        
        icon = "✅" if is_found else "❌"
        print(f"{query[:47]+'...':<50} | {q_type:<15} | {icon} {score:<5.0f} | {latency:<8.2f}s")

    avg_recall = total_recall / len(test_cases)
    avg_latency = total_latency / len(test_cases)
    
    print(f"{'='*90}")
    print(f"📊 FINAL METRICS:")
    print(f"   • Recall@3: {avg_recall:.1%}  (Goal: >80%)")
    print(f"   • Latency:  {avg_latency:.2f}s")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()