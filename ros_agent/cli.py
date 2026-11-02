"""Command line: `ros-agent ask`, `ros-agent bench`, `ros-agent scenarios`, `ros-agent tool`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ros_agent.agent import OllamaClient, OpenAIClient, run_agent
from ros_agent.bench import run_bench
from ros_agent.graph import SCENARIO_DIR, load_scenario, load_scenarios
from ros_agent.tools import LiveBackend, SimBackend

DEFAULT_MODEL = os.environ.get("ROS_AGENT_MODEL", "qwen3:4b")


def _client(args):
    if args.api == "openai":
        return OpenAIClient(args.model, args.base_url or "http://127.0.0.1:8080/v1")
    return OllamaClient(args.model, args.base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), think=args.think)


def _scenario(name: str):
    matches = sorted(SCENARIO_DIR.glob(f"{name}*.yaml"))
    if not matches:
        sys.exit(f"No scenario matching '{name}'. Run `ros-agent scenarios` to list them.")
    return load_scenario(matches[0])


def _add_model_args(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default {DEFAULT_MODEL}, or $ROS_AGENT_MODEL)")
    parser.add_argument("--api", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", help="Ollama host or OpenAI-compatible base URL")
    parser.add_argument("--think", action="store_true", help="Enable thinking for models that support it")
    parser.add_argument("--max-tool-calls", type=int, default=12)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ros-agent", description="Offline ROS 2 debugging copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask about a live robot or a simulated scenario")
    ask.add_argument("question", nargs="?", help="Defaults to the scenario's question")
    where = ask.add_mutually_exclusive_group(required=True)
    where.add_argument("--live", action="store_true", help="Inspect the running ROS 2 system")
    where.add_argument("--scenario", help="Scenario id prefix, e.g. 01")
    ask.add_argument("--show-trace", action="store_true", help="Print every tool call and result")
    _add_model_args(ask)

    bench = sub.add_parser("bench", help="Score a model on all scenarios")
    bench.add_argument("--only", help="Run scenarios whose id starts with this")
    bench.add_argument("--out", type=Path, help="Write runs as JSONL")
    _add_model_args(bench)

    sub.add_parser("scenarios", help="List benchmark scenarios")

    tool = sub.add_parser("tool", help="Run one tool by hand against a scenario")
    tool.add_argument("scenario")
    tool.add_argument("name")
    tool.add_argument("args", nargs="*", help="key=value arguments")

    args = parser.parse_args(argv)

    if args.command == "scenarios":
        for s in load_scenarios():
            print(f"{s.id:<32} {s.difficulty:<7} {s.root_cause:<20} {s.question}")
    elif args.command == "tool":
        backend = SimBackend(_scenario(args.scenario).graph)
        print(backend.run(args.name, dict(a.split("=", 1) for a in args.args)))
    elif args.command == "ask":
        if args.live:
            backend, question = LiveBackend(), args.question
            if not question:
                sys.exit("Give a question when using --live.")
        else:
            scenario = _scenario(args.scenario)
            backend, question = SimBackend(scenario.graph), args.question or scenario.question
        result = run_agent(question, backend, _client(args), args.max_tool_calls)
        if args.show_trace:
            for m in result.messages[2:]:
                if m["role"] == "tool":
                    print(f"--- {m['name']} ---\n{m['content']}\n")
                elif m.get("tool_calls"):
                    for c in m["tool_calls"]:
                        print(f">>> {c['name']}({json.dumps(c['arguments'])})")
        final = result.messages[-1].get("content", "") if result.messages else ""
        print(final or result.error or "(no answer)")
    elif args.command == "bench":
        scenarios = [s for s in load_scenarios() if not args.only or s.id.startswith(args.only)]
        run_bench(scenarios, _client(args), args.out, args.max_tool_calls)


if __name__ == "__main__":
    main()
