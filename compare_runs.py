import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlflow
from agent.mlflow_config import setup_mlflow, print_run_summary
setup_mlflow()
print_run_summary()
