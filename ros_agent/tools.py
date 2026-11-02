"""Read-only ROS 2 inspection tools with two backends.

Each tool corresponds to a `ros2` CLI command. `LiveBackend` runs the real
command; `SimBackend` answers from a simulated `Graph` and prints text in the
same shape as the CLI, so a model tuned on simulated scenarios reads live
output the same way. Nothing here can change the robot.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from typing import Protocol

from ros_agent.graph import Graph


def _tool(tool_name: str, description: str, /, **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string", "description": text} for key, text in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS: list[dict] = [
    _tool("node_list", "List running nodes (ros2 node list)."),
    _tool("node_info", "Publishers and subscriptions of one node (ros2 node info).", node="Node name, e.g. /lidar_driver"),
    _tool("topic_list", "List topics with their message types (ros2 topic list -t)."),
    _tool(
        "topic_info",
        "Publishers and subscriptions of a topic with message types and QoS (ros2 topic info -v).",
        topic="Topic name, e.g. /scan",
    ),
    _tool("topic_hz", "Measure the publish rate of a topic (ros2 topic hz).", topic="Topic name"),
    _tool("topic_echo", "Print the header of one message on a topic (ros2 topic echo --once).", topic="Topic name"),
    _tool("param_list", "List parameters of a node (ros2 param list).", node="Node name"),
    _tool("param_get", "Get one parameter value (ros2 param get).", node="Node name", name="Parameter name"),
    _tool("tf_frames", "List TF edges with their publishers and rates (tf2_monitor)."),
    _tool(
        "tf_echo",
        "Look up the transform between two frames (tf2_echo).",
        source_frame="Frame to transform from, e.g. laser",
        target_frame="Frame to transform to, e.g. map",
    ),
]

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)


class Backend(Protocol):
    def run(self, tool: str, args: dict) -> str: ...


class SimBackend:
    """Answers tool calls from a simulated graph."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def run(self, tool: str, args: dict) -> str:
        if tool not in TOOL_NAMES:
            return f"Unknown tool '{tool}'. Available: {', '.join(sorted(TOOL_NAMES))}"
        try:
            return getattr(self, tool)(**args)
        except TypeError as exc:
            return f"Bad arguments for {tool}: {exc}"

    def node_list(self) -> str:
        return "\n".join(sorted(self.graph.nodes))

    def node_info(self, node: str) -> str:
        if node not in self.graph.nodes:
            return f"Unable to find node '{node}'"
        lines = [node, "  Subscribers:"]
        lines += [f"    {ep.topic}: {ep.type}" for ep in self.graph.subscriptions if ep.node == node]
        lines.append("  Publishers:")
        lines += [f"    {ep.topic}: {ep.type}" for ep in self.graph.publishers if ep.node == node]
        if any(edge.publisher == node for edge in self.graph.tf):
            lines.append("    /tf: tf2_msgs/msg/TFMessage")
        return "\n".join(lines)

    def topic_list(self) -> str:
        return "\n".join(f"{topic} [{', '.join(types)}]" for topic, types in self.graph.topics().items())

    def topic_info(self, topic: str) -> str:
        topics = self.graph.topics()
        if topic not in topics:
            return f"Unknown topic '{topic}'"
        pubs = [ep for ep in self.graph.publishers if ep.topic == topic]
        subs = [ep for ep in self.graph.subscriptions if ep.topic == topic]
        lines = [f"Type: {'; '.join(topics[topic])}", "", f"Publisher count: {len(pubs)}", ""]
        for ep in pubs:
            lines += _endpoint_block(ep, "PUBLISHER")
        lines += [f"Subscription count: {len(subs)}", ""]
        for ep in subs:
            lines += _endpoint_block(ep, "SUBSCRIPTION")
        return "\n".join(lines).rstrip()

    def topic_hz(self, topic: str) -> str:
        pubs = [ep for ep in self.graph.publishers if ep.topic == topic]
        if topic in ("/tf", "/tf_static"):
            rates = [edge.rate for edge in self.graph.live_tf() if edge.static == (topic == "/tf_static")]
            if rates:
                return _hz_text(max(rates))
        if not pubs:
            return f"WARNING: topic [{topic}] does not appear to be published yet"
        total = sum(ep.rate for ep in pubs)
        if total <= 0:
            return "(no new messages received in 5 seconds)"
        return _hz_text(total)

    def topic_echo(self, topic: str) -> str:
        pubs = [ep for ep in self.graph.publishers if ep.topic == topic and ep.rate > 0]
        if not pubs:
            return f"WARNING: topic [{topic}] does not appear to be published yet"
        ep = pubs[0]
        params = self.graph.params(ep.node)
        frame = params.get("frame_id") or params.get("odom_frame_id") or ""
        return f"header:\n  stamp:\n    sec: 1730540000\n    nanosec: 0\n  frame_id: {frame}\n---\n(message type {ep.type}; body omitted)"

    def param_list(self, node: str) -> str:
        if node not in self.graph.nodes:
            return f"Node not found: {node}"
        names = sorted({*self.graph.params(node), "use_sim_time"})
        return f"{node}:\n" + "\n".join(f"  {name}" for name in names)

    def param_get(self, node: str, name: str) -> str:
        if node not in self.graph.nodes:
            return f"Node not found: {node}"
        params = {"use_sim_time": False, **self.graph.params(node)}
        if name not in params:
            return "Parameter not set."
        value = params[name]
        if isinstance(value, bool):
            return f"Boolean value is: {value}"
        if isinstance(value, int):
            return f"Integer value is: {value}"
        if isinstance(value, float):
            return f"Double value is: {value}"
        return f"String value is: {value}"

    def tf_frames(self) -> str:
        edges = self.graph.live_tf()
        if not edges:
            return "No TF data received."
        lines = []
        for edge in edges:
            rate = "static" if edge.static else f"{edge.rate:g} Hz"
            lines.append(f"{edge.parent} -> {edge.child}  (publisher: {edge.publisher}, rate: {rate})")
        return "\n".join(lines)

    def tf_echo(self, source_frame: str, target_frame: str) -> str:
        edges = [e for e in self.graph.live_tf() if e.static or e.rate > 0]
        frames = {f for e in edges for f in (e.parent, e.child)}
        prefix = f"Could not transform {source_frame} to {target_frame}:"
        for frame, role in ((source_frame, "source_frame"), (target_frame, "target_frame")):
            if frame not in frames:
                return f'{prefix} "{frame}" passed to lookupTransform argument {role} does not exist.'
        if _connected(edges, source_frame, target_frame):
            return (
                "At time 0.0\n- Translation: [0.000, 0.000, 0.000]\n"
                "- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]"
            )
        return (
            f"{prefix} Could not find a connection between '{target_frame}' and '{source_frame}' "
            "because they are not part of the same tree. Tf has two or more unconnected trees."
        )


def _endpoint_block(ep, kind: str) -> list[str]:
    q = ep.qos
    return [
        f"Node name: {ep.node.lstrip('/')}",
        "Node namespace: /",
        f"Topic type: {ep.type}",
        f"Endpoint type: {kind}",
        "QoS profile:",
        f"  Reliability: {q['reliability'].upper()}",
        f"  History (Depth): KEEP_LAST ({q['depth']})",
        f"  Durability: {q['durability'].upper()}",
        "  Lifespan: Infinite",
        "  Deadline: Infinite",
        "  Liveliness: AUTOMATIC",
        "  Liveliness lease duration: Infinite",
        "",
    ]


def _hz_text(rate: float) -> str:
    period = 1.0 / rate
    return f"average rate: {rate:.3f}\n\tmin: {period:.3f}s max: {period:.3f}s std dev: 0.00010s window: {int(rate * 5)}"


def _connected(edges, a: str, b: str) -> bool:
    links: dict[str, set[str]] = {}
    for e in edges:
        links.setdefault(e.parent, set()).add(e.child)
        links.setdefault(e.child, set()).add(e.parent)
    seen, queue = {a}, deque([a])
    while queue:
        frame = queue.popleft()
        if frame == b:
            return True
        for nxt in links.get(frame, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


# Live commands; long-running ones are bounded with `timeout`.
_LIVE_COMMANDS = {
    "node_list": lambda a: ["ros2", "node", "list"],
    "node_info": lambda a: ["ros2", "node", "info", a["node"]],
    "topic_list": lambda a: ["ros2", "topic", "list", "-t"],
    "topic_info": lambda a: ["ros2", "topic", "info", "-v", a["topic"]],
    "topic_hz": lambda a: ["timeout", "5", "ros2", "topic", "hz", a["topic"]],
    "topic_echo": lambda a: ["timeout", "5", "ros2", "topic", "echo", "--once", "--no-arr", a["topic"]],
    "param_list": lambda a: ["ros2", "param", "list", a["node"]],
    "param_get": lambda a: ["ros2", "param", "get", a["node"], a["name"]],
    "tf_frames": lambda a: ["timeout", "5", "ros2", "run", "tf2_ros", "tf2_monitor"],
    "tf_echo": lambda a: ["timeout", "3", "ros2", "run", "tf2_ros", "tf2_echo", a["target_frame"], a["source_frame"]],
}


class LiveBackend:
    """Runs the real `ros2` CLI. Source your ROS 2 setup before starting."""

    def __init__(self, max_chars: int = 4000):
        if shutil.which("ros2") is None:
            raise RuntimeError("ros2 not found on PATH; source /opt/ros/<distro>/setup.bash first")
        self.max_chars = max_chars

    def run(self, tool: str, args: dict) -> str:
        if tool not in _LIVE_COMMANDS:
            return f"Unknown tool '{tool}'"
        try:
            command = _LIVE_COMMANDS[tool](args)
        except KeyError as exc:
            return f"Missing argument {exc} for {tool}"
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        output = (result.stdout + result.stderr).strip() or "(no output)"
        return output[: self.max_chars]
