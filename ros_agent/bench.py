"""Score agents on the fault scenarios.

A scenario is solved when the agent names the right root cause and the right
faulty entity. Each run's full conversation is saved, so solved runs can be
reused as training data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ros_agent.agent import run_agent
from ros_agent.graph import Scenario
from ros_agent.tools import SimBackend


def _norm(value: str) -> str:
    return "".join(str(value).lower().split()).rstrip("/")


def score(answer: dict | None, scenario: Scenario) -> dict:
    if not answer:
        return {"root_cause_ok": False, "entity_ok": False, "solved": False}
    root_ok = _norm(answer.get("root_cause", "")) == _norm(scenario.root_cause)
    if scenario.root_cause == "none":
        entity_ok = True
    else:
        got = _norm(answer.get("entity", ""))
        entity_ok = bool(got) and any(_norm(e) == got or _norm(e) in got for e in scenario.entities)
    return {"root_cause_ok": root_ok, "entity_ok": entity_ok, "solved": root_ok and entity_ok}


def run_bench(scenarios: list[Scenario], client, out: Path | None = None, max_tool_calls: int = 12) -> dict:
    rows = []
    for scenario in scenarios:
        started = time.monotonic()
        result = run_agent(scenario.question, SimBackend(scenario.graph), client, max_tool_calls)
        row = {
            "scenario": scenario.id,
            "difficulty": scenario.difficulty,
            "expected": {"root_cause": scenario.root_cause, "entities": scenario.entities},
            "answer": result.answer,
            "tool_calls": result.tool_calls,
            "seconds": round(time.monotonic() - started, 1),
            "error": result.error,
            **score(result.answer, scenario),
            "messages": result.messages,
        }
        rows.append(row)
        mark = "PASS" if row["solved"] else "fail"
        got = (result.answer or {}).get("root_cause", result.error or "no answer")
        print(f"{mark}  {scenario.id:<32} {row['tool_calls']:>2} calls {row['seconds']:>6}s  got: {got}", flush=True)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    solved = sum(r["solved"] for r in rows)
    summary = {
        "solved": solved,
        "total": len(rows),
        "root_cause_accuracy": round(sum(r["root_cause_ok"] for r in rows) / max(len(rows), 1), 3),
        "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / max(len(rows), 1), 1),
    }
    print(f"\nSolved {solved}/{len(rows)}  root cause accuracy {summary['root_cause_accuracy']:.0%}  "
          f"mean tool calls {summary['mean_tool_calls']}")
    return summary
