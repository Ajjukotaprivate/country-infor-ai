import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Fix path so we can import our src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agent.graph import agent

# Quiet down the logs for clean output
logging.basicConfig(level=logging.WARNING)

DATASET = Path(__file__).parent / "dataset.json"

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

async def run_q(question, thread_id):
    res = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return res.get("answer", "")

def check_answer(item, answer):
    ans = answer.lower()
    gt = item.get("ground_truth", "")
    extra = item.get("ground_truth_extra", "")

    if gt and gt.lower() not in ans:
        return False
    if extra and extra.lower() not in ans:
        return False
        
    return True

async def main():
    with open(DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nRunning evals...")
    print(f"Total questions: {len(data)}\n")
    print(f"{'ID':<6} {'Category':<20} Status  Question")
    print("-" * 70)

    passed = 0

    for item in data:
        qid = item["id"]
        q = item["question"]
        cat = item.get("category", "unknown")
        
        start = time.monotonic()
        try:
            ans = await run_q(q, f"eval-{qid}")
        except Exception as e:
            ans = f"Error: {e}"
            
        elapsed = time.monotonic() - start
        
        ok = check_answer(item, ans)
        if ok:
            passed += 1

        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        preview = q[:45] + "..." if len(q) > 45 else q
        
        print(f"{qid:<6} {cat:<20} {status:<14} {preview} ({elapsed:.1f}s)")
        
        if not ok:
            print(f"    Expected: {item.get('ground_truth', '')}")
            print(f"    Got:      {ans[:100]}...\n")

    total = len(data)
    rate = (passed / total) * 100 if total else 0
    
    print("-" * 70)
    color = GREEN if rate >= 80 else RED
    print(f"\nScore: {color}{passed}/{total} ({rate:.1f}%){RESET}")
    
    if rate >= 80:
        print(f"{GREEN}Evals passed!{RESET}\n")
    else:
        print(f"{RED}Evals failed (need 80%){RESET}\n")

if __name__ == "__main__":
    asyncio.run(main())
