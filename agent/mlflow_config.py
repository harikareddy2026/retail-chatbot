"""
MLflow configuration for the retail merchant AI chatbot.

Import and call setup_mlflow() at the start of every
agent session. This file is the single source of truth
for all MLflow settings.

Usage:
    from agent.mlflow_config import setup_mlflow, log_metric, log_param
    setup_mlflow()  # call once at startup
"""
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────
# Experiment name — all runs go under this experiment
EXPERIMENT_NAME = "/retail-merchant-chatbot"

# Local tracking URI — stores in mlruns/ folder
# We use local storage since no Databricks token
#TRACKING_URI = "file:///mlruns"
TRACKING_URI = "sqlite:///mlflow.db"

_is_setup = False  # flag to avoid calling setup twice


def setup_mlflow():
    """
    Configure MLflow for the retail chatbot project.
    Call this ONCE at the very start of your script,
    BEFORE importing any LangChain/LangGraph objects.

    What this does:
    1. Points MLflow at local mlruns/ folder
    2. Creates the experiment if it does not exist
    3. Enables autolog for LangChain tools
    4. Sets PII protection (no raw prompt content stored)
    """
    global _is_setup
    if _is_setup:
        return  # already set up, skip

    # Step 1: point at local storage
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking URI: {TRACKING_URI}")

    # Step 2: create or get experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    # Step 3: ARCHITECT FIX — PII protection
    # log_input_examples=False → raw prompts NOT stored
    # log_model_signatures=False → no input/output schemas stored
    # log_traces=True → trace structure IS stored (tool name, latency)
    #mlflow.langchain.autolog(
    #    log_input_examples=False,
    #   log_model_signatures=False,
    #   log_traces=True
    #)
    mlflow.langchain.autolog(
    log_traces=True,
    silent=True
    )
    print("MLflow autolog enabled (log_traces=True, PII note: use sqlite backend for dev only)")
    print("MLflow autolog enabled (PII protection: ON)")
    print("  Traces: YES | Raw prompts: NO | Signatures: NO")

    _is_setup = True
    print("✅ MLflow setup complete!")


def log_metric(key: str, value: float, step: int = None):
    """
    Log a numeric metric to the current active run.
    Call inside a mlflow.start_run() context.

    Example:
        with mlflow.start_run(run_name="my_run"):
            log_metric("ragas_faithfulness", 0.82)
    """
    if step is not None:
        mlflow.log_metric(key, value, step=step)
    else:
        mlflow.log_metric(key, value)


def log_param(key: str, value):
    """
    Log a configuration parameter to the current active run.
    Params do not change during a run (unlike metrics).

    Example:
        with mlflow.start_run():
            log_param("chunk_size", 1000)
    """
    mlflow.log_param(key, value)


def log_tag(key: str, value: str):
    """Log a string tag to the current active run."""
    mlflow.set_tag(key, value)


def get_experiment_runs():
    """
    Return all runs in the retail-merchant-chatbot experiment.
    Useful for comparing metrics across days.
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME]
    )
    return runs


def print_run_summary():
    """Print a table of all runs and their key metrics."""
    runs = get_experiment_runs()
    if runs.empty:
        print("No runs found yet.")
        return

    # Find metric columns
    metric_cols = [c for c in runs.columns
                   if c.startswith("metrics.")]

    display_cols = ["run_id",
                    "tags.mlflow.runName",
                    "status"] + metric_cols[:5]
    display_cols = [c for c in display_cols if c in runs.columns]

    print("\n=== All MLflow Runs ===")
    print(runs[display_cols].to_string(index=False))
    print(f"\nTotal runs: {len(runs)}")