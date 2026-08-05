#!/usr/bin/env python3
"""
Retrieval evaluation for the clinical trial RAG.

Imports the live routing/retrieval path from query_rag.py so the eval exercises
the same code the app does.

Scores every query under BOTH identity keys, always:

  trial_id        comparable to the committed baseline run
  protocol_number matches the corrected identity model, where a master protocol
                  and its separately-registered sub-studies share one number

Gold is a SET per query per key, because a single study can carry more than one
identifier -- this corpus mixes protocols with the publications reporting them.
Labels and the reasoning behind them are pre-registered in
data/eval_runs/GOLD_PREREGISTRATION.md; change them there, with a reason, not here.

Usage:
    python scripts/evaluate_retrieval.py [--json out.json] [--min-recall 1.0]

Exits non-zero if either key falls below --min-recall, so a regression fails
rather than printing a lower number.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.query_rag import load_resources, build_dynamic_retriever  # noqa: E402

# Pre-registered gold. See data/eval_runs/GOLD_PREREGISTRATION.md.
GOLD = {
    "What is the primary objective of study 3000-02-005?": {
        "trial_id": {"NCT05751629"},
        "protocol_number": {"3000-02-005"},
    },
    "What are the common side effects of PARP inhibitors?": {
        "trial_id": {"NCT05751629"},
        "protocol_number": {"3000-02-005"},
    },
    "What is the dosing schedule for Galunisertib?": {
        "trial_id": {"NCT02423343"},
        "protocol_number": {"H9H-MC-JBEF", "<UNKNOWN>"},
    },
    "Show me Phase 3 trials for NSCLC": {
        "trial_id": {"NCT02578680"},
        "protocol_number": {"189", "KEYNOTE-189"},
    },
}

KEYS = ("trial_id", "protocol_number")
TOP_K = 3


def random_floor(metadatas, gold_trials, k=TOP_K):
    """P(at least one of k uniformly random chunks belongs to the gold trial).

    Reported alongside recall so the headline number is read against chance on
    this corpus rather than against 0.
    """
    total = len(metadatas)
    floors = []
    for trial in gold_trials:
        count = sum(1 for m in metadatas if m.get("trial_id") == trial)
        q = 1.0
        for i in range(k):
            q *= (total - count - i) / (total - i)
        floors.append(1 - q)
    return sum(floors) / len(floors) if floors else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write a run artifact here")
    ap.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="exit non-zero if either key scores below this (e.g. 1.0)",
    )
    args = ap.parse_args()

    eval_file = Path("data/eval.jsonl")
    if not eval_file.exists():
        print(f"❌ Eval file not found at {eval_file}")
        return 1
    cases = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]

    missing = [c["query"] for c in cases if c["query"] not in GOLD]
    if missing:
        print(f"❌ No pre-registered gold for: {missing}")
        return 1

    # Resources load once, OUTSIDE the timed region. Previously the retriever was
    # constructed inside it, so each "latency" included a cross-encoder load and a
    # BM25 rebuild over every chunk -- neither retrieval time nor end-to-end time.
    print("📦 Loading retrieval resources...")
    vectorstore, docstore, all_text_docs = load_resources()

    total = vectorstore._collection.count()
    metadatas, offset = [], 0
    while offset < total:
        metadatas += vectorstore._collection.get(
            include=["metadatas"], limit=2000, offset=offset
        )["metadatas"]
        offset += 2000

    hits = {k: 0 for k in KEYS}
    rows = []
    print(f"\n{'QUERY':<44} | {'trial_id':<8} | {'protocol_number':<15} | TIME")
    print("-" * 88)

    for case in cases:
        query = case["query"]
        gold = GOLD[query]

        start = time.time()
        docs = build_dynamic_retriever(query, vectorstore, docstore, all_text_docs).invoke(query)
        latency = time.time() - start

        row = {"query": query, "type": case.get("type"), "latency_s": round(latency, 3)}
        for key in KEYS:
            got = {d.metadata.get(key) for d in docs[:TOP_K]}
            hit = bool(gold[key] & got)
            hits[key] += hit
            row[key] = {"hit": hit, "gold": sorted(gold[key]), "top3": sorted(str(g) for g in got)}
        row["n_retrieved"] = len(docs)
        rows.append(row)

        print(
            f"{query[:42]:<44} | {'HIT ' if row['trial_id']['hit'] else 'MISS':<8} | "
            f"{'HIT ' if row['protocol_number']['hit'] else 'MISS':<15} | {latency:.2f}s"
        )
        for key in KEYS:
            if not row[key]["hit"]:
                print(f"    {key}: expected {row[key]['gold']}, top{TOP_K} had {row[key]['top3']}")

    n = len(cases)
    recall = {k: hits[k] / n for k in KEYS}
    floor = random_floor(metadatas, {g for c in cases for g in GOLD[c["query"]]["trial_id"]})

    print("=" * 88)
    print(f"📊 Recall@{TOP_K}, n={n}")
    for key in KEYS:
        print(f"   • by {key:<16}: {hits[key]}/{n} = {recall[key]:.0%}")
    print(f"   • random floor (trial_id): {floor:.1%}  — chance on this corpus, not 0")
    print(f"   • mean latency (retrieval only, excludes generation): "
          f"{sum(r['latency_s'] for r in rows)/n:.2f}s")
    print("=" * 88)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "n": n, "top_k": TOP_K, "recall": recall,
            "random_floor_trial_id": floor,
            "index_chunks": total, "results": rows,
        }, indent=2))
        print(f"💾 Wrote {args.json}")

    if args.min_recall is not None:
        failed = [k for k in KEYS if recall[k] < args.min_recall]
        if failed:
            print(f"❌ Below --min-recall={args.min_recall}: {', '.join(failed)}")
            return 1
        print(f"✅ Both keys at or above --min-recall={args.min_recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
