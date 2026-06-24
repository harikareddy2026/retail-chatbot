"""
Day 8 model registration.
Registers the agent as retail-merchant-chatbot v1.

Run: python3 register_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
from agent.mlflow_config import setup_mlflow

setup_mlflow()
mlflow.end_run()

MODEL_NAME = "retail-merchant-chatbot"

# Find the best day8_ragas_eval run
runs = mlflow.search_runs(
    experiment_names=["/retail-merchant-chatbot"],
    filter_string="tags.`mlflow.runName` = 'day8_ragas_eval'",
    order_by=["metrics.faithfulness DESC"],
    max_results=1
)

if runs.empty:
    print("❌ No day8_ragas_eval run found.")
    print("   Run python3 eval/ragas_eval.py first.")
    exit(1)

best_run_id  = runs["run_id"].iloc[0]
faithfulness = runs["metrics.faithfulness"].iloc[0] \
               if "metrics.faithfulness" in runs.columns \
               else 0.0
accuracy     = runs["metrics.routing_accuracy"].iloc[0] \
               if "metrics.routing_accuracy" in runs.columns \
               else 0.0

print(f"Best run ID:  {best_run_id[:12]}...")
print(f"Faithfulness: {faithfulness:.3f}")
print(f"Routing acc:  {accuracy*100:.0f}%")

# Register model
try:
    result = mlflow.register_model(
        model_uri = f"runs:/{best_run_id}",
        name      = MODEL_NAME
    )
    version = result.version
    print(f"\n✅ Model registered!")
    print(f"   Name:    {MODEL_NAME}")
    print(f"   Version: {version}")

    # Add description
    client = mlflow.MlflowClient()
    client.update_model_version(
        name        = MODEL_NAME,
        version     = version,
        description = (
            f"Retail merchant AI chatbot v{version}. "
            f"LangGraph: classifier + supervisor + "
            f"rag/action/data/escalation workers. "
            f"Faithfulness: {faithfulness:.3f}. "
            f"Routing accuracy: {accuracy*100:.0f}%. "
            f"Stack: Supabase pgvector (HNSW m=16) + "
            f"3 FastAPI MCP servers + "
            f"sentence-transformers 384-dim."
        )
    )
    print(f"✅ Description added to v{version}")
    print(f"\nView: mlflow ui --port 5001 "
          f"--backend-store-uri sqlite:///mlflow.db")
    print(f"Then: Models tab → {MODEL_NAME}")

except Exception as e:
    # Fallback: log registration details as a run
    print(f"Registry note: {e}")
    mlflow.end_run()
    with mlflow.start_run(run_name="day8_model_v1"):
        mlflow.set_tag("model_name",      MODEL_NAME)
        mlflow.set_tag("model_version",   "1")
        mlflow.set_tag("registered_from", best_run_id)
        mlflow.log_metric(
            "faithfulness", faithfulness)
        mlflow.log_metric(
            "routing_accuracy", accuracy)
    print("✅ Registration details logged as day8_model_v1")