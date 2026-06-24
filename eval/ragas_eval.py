"""
Day 8 evaluation script.
Runs 20 questions through the agent, evaluates
RAG questions with embedding similarity, logs to MLflow.

Run: python3 eval/ragas_eval.py
(All 3 MCP servers must be running)
"""
import sys, os, asyncio, json
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
from agent.graph import build_app
from agent.mlflow_config import setup_mlflow
from tools.mcp_tools import search_products
from eval.test_questions import TEST_DATASET, RAG_QUESTIONS

setup_mlflow()
mlflow.end_run()

# Load embedding model once
print("Loading embedding model for evaluation...")
EVAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Eval model ready\n")


# ── Step 1: Run all 20 through agent ──────────────
async def collect_results(app):
    """Run all 20 questions and collect responses."""
    print("Running 20 questions through agent...")
    results = []

    for i, (question, expected_worker, ground_truth) \
            in enumerate(TEST_DATASET):

        state = {
            "messages":      [HumanMessage(question)],
            "merchant_id":   "M001",
            "active_worker": "",
            "query_blocked": False,
            "error_state":   False
        }
        config = {
            "configurable": {"thread_id": f"eval-{i}"},
            "recursion_limit": 10
        }

        result = await app.ainvoke(state, config=config)

        if result.get("query_blocked"):
            actual_worker = "escalation"
        else:
            actual_worker = result.get(
                "active_worker", "escalation")

        answer = result["messages"][-1].content
        routed_correctly = (actual_worker == expected_worker)

        results.append({
            "question":         question,
            "expected_worker":  expected_worker,
            "actual_worker":    actual_worker,
            "answer":           answer,
            "ground_truth":     ground_truth,
            "routed_correctly": routed_correctly
        })

        status = "✅" if routed_correctly else "❌"
        print(f"  [{i+1:02d}/20] {status} "
              f"{expected_worker:15s} → "
              f"{actual_worker:15s} | "
              f"{question[:45]}...")

    return results


# ── Step 2: Embedding similarity evaluation ────────
def embedding_eval(results):
    """
    Measure answer quality using cosine similarity
    between agent answer and ground truth.
    Only applied to RAG worker results.
    """
    rag_results = [
        r for r in results
        if r["expected_worker"] == "rag_worker"
    ]

    print(f"\nEvaluating {len(rag_results)} "
          f"RAG answers with embedding similarity...")

    scores = []
    for r in rag_results:
        answer = r["answer"]
        gt     = r["ground_truth"]

        # Get retrieved context for this question
        context = search_products.invoke({
            "query":       r["question"],
            "merchant_id": "M001"
        })

        # Embed answer and ground truth
        vecs = EVAL_MODEL.encode(
            [answer, gt, context],
            normalize_embeddings=True
        )

        # Faithfulness: how similar is answer to context?
        faithfulness = float(np.dot(vecs[0], vecs[2]))

        # Answer relevancy: how similar is answer to ground truth?
        relevancy = float(np.dot(vecs[0], vecs[1]))

        scores.append({
            "question":    r["question"],
            "faithfulness": faithfulness,
            "relevancy":   relevancy
        })

        print(f"  {r['question'][:50][:50]}")
        print(f"    faithfulness={faithfulness:.3f} "
              f"relevancy={relevancy:.3f}")

    avg_faith  = np.mean([s["faithfulness"] for s in scores])
    avg_relev  = np.mean([s["relevancy"]    for s in scores])

    print(f"\nAverage faithfulness: {avg_faith:.3f}")
    print(f"Average relevancy:    {avg_relev:.3f}")

    return {
        "faithfulness":     float(avg_faith),
        "answer_relevancy": float(avg_relev),
        "per_question":     scores
    }


# ── Step 3: Log everything to MLflow ──────────────
def log_to_mlflow(results, eval_scores):
    """Log all metrics, per-worker accuracy, artifact."""

    correct  = sum(r["routed_correctly"] for r in results)
    accuracy = correct / len(results)
    errors   = len(results) - correct

    mlflow.end_run()
    with mlflow.start_run(run_name="day8_ragas_eval"):

        # Core evaluation scores
        mlflow.log_metric(
            "faithfulness",     eval_scores["faithfulness"])
        mlflow.log_metric(
            "answer_relevancy", eval_scores["answer_relevancy"])

        # Routing metrics
        mlflow.log_metric("routing_accuracy", accuracy)
        mlflow.log_metric("questions_correct", correct)
        mlflow.log_metric("questions_total",   len(results))
        mlflow.log_metric("routing_errors",    errors)

        # Per-worker accuracy
        for worker in ["rag_worker", "action_worker",
                       "data_worker", "escalation"]:
            w_res = [
                r for r in results
                if r["expected_worker"] == worker
            ]
            if w_res:
                w_acc = sum(
                    r["routed_correctly"]
                    for r in w_res
                ) / len(w_res)
                mlflow.log_metric(
                    f"{worker}_accuracy", w_acc)

        # Tags
        mlflow.set_tag("day",  "8")
        mlflow.set_tag("eval_method", "embedding_similarity")
        mlflow.set_tag(
            "faithfulness_target_met",
            str(eval_scores["faithfulness"] >= 0.75))

        # Save full results as artifact
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact("eval_results.json")
        os.remove("eval_results.json")

    print("✅ Logged to MLflow: day8_ragas_eval")
    return accuracy


# ── Step 4: Print summary ──────────────────────────
def print_summary(results, eval_scores, accuracy):
    faith = eval_scores["faithfulness"]
    relev = eval_scores["answer_relevancy"]

    print("\n" + "=" * 60)
    print("DAY 8 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nROUTING: {sum(r['routed_correctly'] for r in results)}"
          f"/20 = {accuracy*100:.0f}%")
    print(f"\nRAG QUALITY (embedding similarity):")
    print(f"  Faithfulness:     {faith:.3f}  "
          f"{'✅' if faith >= 0.75 else '❌'} (target >0.75)")
    print(f"  Answer relevancy: {relev:.3f}  "
          f"{'✅' if relev >= 0.70 else '❌'} (target >0.70)")

    failures = [r for r in results if not r["routed_correctly"]]
    if failures:
        print(f"\nFAILING QUERIES ({len(failures)}):")
        for r in failures:
            print(f"  ❌ Expected {r['expected_worker']:15s} "
                  f"| Got {r['actual_worker']:15s} "
                  f"| {r['question'][:50]}")
    else:
        print("\n✅ All 20 queries routed correctly!")

    print("\n" + "=" * 60)


# ── MAIN ───────────────────────────────────────────
async def main():
    print("Building agent...")
    app = await build_app()
    print("Agent ready!\n")

    results     = await collect_results(app)
    eval_scores = embedding_eval(results)
    accuracy    = log_to_mlflow(results, eval_scores)
    print_summary(results, eval_scores, accuracy)

asyncio.run(main())