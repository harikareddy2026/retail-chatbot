import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
from langchain_core.messages import HumanMessage
from agent.graph import build_app
from agent.mlflow_config import setup_mlflow

setup_mlflow()

TEST_QUERIES = [
    ("What is the return policy?",         "rag_worker"),
    ("Tell me about the yoga mat product", "rag_worker"),
    ("What products do you sell?",         "rag_worker"),
    ("Is SKU-001 in stock?",               "action_worker"),
    ("Show me my recent orders",           "action_worker"),
    ("What is the price of P001?",         "action_worker"),
    ("What is my total revenue?",          "data_worker"),
    ("How many orders do I have?",         "data_worker"),
    ("What is the weather today?",         "escalation"),
    ("Write me a poem",                    "escalation"),
]

async def run_routing_test():
    print("Building agent...")
    app = await build_app()
    print("Ready!\n")

    correct = 0
    results = []

    print("=" * 60)
    print("ROUTING ACCURACY TEST — 10 queries")
    print("=" * 60)

    for i, (query, expected) in enumerate(TEST_QUERIES):
        state = {
            "messages":      [HumanMessage(query)],
            "merchant_id":   "M001",
            "active_worker": "",
            "query_blocked": False,
            "error_state":   False
        }
        config = {
            "configurable": {"thread_id": f"route-test-{i}"},
            "recursion_limit": 10
        }

        result = await app.ainvoke(state, config=config)

        if result.get("query_blocked"):
            actual = "escalation"
        else:
            actual = result.get("active_worker", "escalation")

        passed = (actual == expected)
        correct += int(passed)
        results.append((query, expected, actual, passed))

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n[{i+1:02d}/10] {status}")
        print(f"  Query:    {query}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        if not passed:
            resp = result['messages'][-1].content[:80]
            print(f"  Response: {resp}...")

    accuracy = correct / len(TEST_QUERIES)

    print("\n" + "=" * 60)
    print(f"ROUTING ACCURACY: {correct}/10 = {accuracy*100:.0f}%")
    if accuracy >= 0.9:
        print("✅ EXCELLENT — above 90% target")
    elif accuracy >= 0.7:
        print("⚠ ACCEPTABLE — update supervisor keywords")
    else:
        print("❌ NEEDS WORK — update RETAIL_KEYWORDS")
    print("=" * 60)

    # Log to MLflow
    mlflow.end_run()  # clear any stuck run from autolog
    with mlflow.start_run(run_name="day7_routing_test"):
        mlflow.log_metric("routing_accuracy", accuracy)
        mlflow.log_metric("queries_correct",  correct)
        mlflow.log_metric("queries_total",    len(TEST_QUERIES))
        mlflow.set_tag("day", "7")

        for worker in ["rag_worker","action_worker",
                       "data_worker","escalation"]:
            w_res = [(q,e,a,p) for q,e,a,p
                     in results if e == worker]
            if w_res:
                w_acc = sum(p for _,_,_,p
                            in w_res) / len(w_res)
                mlflow.log_metric(
                    f"{worker}_accuracy", w_acc)

    print("\n✅ Results logged to MLflow: day7_routing_test")
    return accuracy

asyncio.run(run_routing_test())