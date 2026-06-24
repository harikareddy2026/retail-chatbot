import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow

# Local MLflow — stores in mlruns/ folder in project root
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:///Users/manmad/Documents/Harika/DE_projects/AI_PROJECT/retail-chatbot/mlruns")
mlflow.set_experiment("retail-merchant-chatbot")

with mlflow.start_run(run_name="day4_tools"):
    mlflow.log_metric("tools_created",  4)
    mlflow.log_metric("tools_passing",  4)

    mlflow.log_param("tool_1", "search_products")
    mlflow.log_param("tool_2", "check_inventory")
    mlflow.log_param("tool_3", "get_order_status")
    mlflow.log_param("tool_4", "get_pricing")
    mlflow.log_param("error_boundary", "try_except_returns_dict")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

    mlflow.set_tag("day", "4")
    mlflow.set_tag("status", "all_tools_passing")

print("✅ Day 4 metrics logged to MLflow!")
print("Run: mlflow ui --port 5000 to view in browser")