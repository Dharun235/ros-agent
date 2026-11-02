"""Tool-calling loop for a local model, with no dependencies beyond the stdlib.

Supports Ollama's `/api/chat` and any OpenAI-compatible `/chat/completions`
server (llama.cpp, LM Studio, vLLM). The agent investigates with read-only
tools and finishes with a JSON verdict that the benchmark can score.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from ros_agent.tools import TOOLS, Backend

ROOT_CAUSES = (
    "qos_mismatch",
    "topic_name_mismatch",
    "type_mismatch",
    "missing_publisher",
    "stalled_publisher",
    "missing_transform",
    "wrong_parameter",
    "sim_time_mismatch",
    "none",
)

SYSTEM_PROMPT = f"""You are an offline ROS 2 debugging assistant running next to the robot.
Your tools are read-only views of the live ROS 2 graph; you cannot change anything.

Investigate before answering:
1. Start broad with node_list and topic_list.
2. For the topics the problem involves, check topic_info (types and QoS of every
   publisher and subscription), topic_hz (is data flowing?) and topic_echo
   (which frame is it stamped in?).
3. Check node parameters with param_list and param_get, and transforms with
   tf_frames and tf_echo.

Common causes: QoS reliability mismatch, a publisher and subscriber on different
topic names or message types, a dead or silent publisher, a missing transform,
a wrong parameter, or use_sim_time true without a /clock publisher.

When you are confident, stop calling tools. Reply with two or three sentences of
explanation, then this JSON object on its own line:
{{"root_cause": "<one of: {', '.join(ROOT_CAUSES)}>", "entity": "<the topic, node, node:parameter, or parent->child frame pair at fault>", "fix": "<one sentence>"}}
Use root_cause "none" and an empty entity if nothing is wrong."""

FINAL_NUDGE = "Stop investigating and give your final answer now, ending with the JSON object."


@dataclass
class AgentResult:
    answer: dict | None
    messages: list[dict]
    tool_calls: int
    error: str | None = None


@dataclass
class ChatReply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": dict}]


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", think: bool = False, timeout: float = 300):
        self.model, self.host, self.think, self.timeout = model, host.rstrip("/"), think, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None) -> ChatReply:
        wire = []
        for m in messages:
            entry = {"role": m["role"], "content": m.get("content", "")}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
            if m["role"] == "tool":
                entry["tool_name"] = m["name"]
            wire.append(entry)
        payload = {"model": self.model, "messages": wire, "stream": False, "think": self.think, "options": {"temperature": 0.2}}
        if tools:
            payload["tools"] = tools
        message = _post(f"{self.host}/api/chat", payload, self.timeout)["message"]
        calls = [
            {"id": f"call_{i}", "name": c["function"]["name"], "arguments": _arguments(c["function"].get("arguments"))}
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        return ChatReply(message.get("content") or "", calls)


class OpenAIClient:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8080/v1", timeout: float = 300):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def chat(self, messages: list[dict], tools: list[dict] | None) -> ChatReply:
        wire = []
        for m in messages:
            entry = {"role": m["role"], "content": m.get("content", "")}
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in m["tool_calls"]
                ]
            if m["role"] == "tool":
                entry["tool_call_id"] = m["tool_call_id"]
            wire.append(entry)
        payload = {"model": self.model, "messages": wire, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
        message = _post(f"{self.base_url}/chat/completions", payload, self.timeout)["choices"][0]["message"]
        calls = [
            {"id": c.get("id") or f"call_{i}", "name": c["function"]["name"], "arguments": _arguments(c["function"].get("arguments"))}
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        return ChatReply(message.get("content") or "", calls)


def parse_answer(text: str) -> dict | None:
    """The last JSON object in `text` that has a root_cause key."""
    decoder = json.JSONDecoder()
    found = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "root_cause" in value:
            found = value
    return found


def run_agent(question: str, backend: Backend, client, max_tool_calls: int = 12) -> AgentResult:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    used = 0
    try:
        while True:
            budget_left = used < max_tool_calls
            reply = client.chat(messages, TOOLS if budget_left else None)
            messages.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls})
            if not reply.tool_calls:
                answer = parse_answer(reply.content)
                if answer is None and budget_left:
                    messages.append({"role": "user", "content": FINAL_NUDGE})
                    used = max_tool_calls
                    continue
                return AgentResult(answer, messages, used)
            for call in reply.tool_calls:
                used += 1
                output = backend.run(call["name"], call["arguments"])
                messages.append({"role": "tool", "name": call["name"], "tool_call_id": call["id"], "content": output})
            if used >= max_tool_calls:
                messages.append({"role": "user", "content": FINAL_NUDGE})
    except (OSError, KeyError, ValueError) as exc:
        return AgentResult(None, messages, used, error=f"{type(exc).__name__}: {exc}")
