"""A simulated ROS 2 graph loaded from YAML.

A graph describes running nodes, what each publishes and subscribes to (with
message type, rate and QoS), node parameters, and TF edges. Scenarios start
from a base graph and apply fault operations, so a broken robot is a few lines
of YAML on top of a working one.

Fault operations address the graph by key path, for example
`[nodes, /lidar_driver, publishes, /scan, qos, reliability]`:

- `set`: set the value at the path, creating maps on the way.
- `delete`: remove the key at the path.
- `rename`: move the key at the path to the name in `value`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_QOS = {"reliability": "reliable", "durability": "volatile", "depth": 10}


@dataclass
class Endpoint:
    node: str
    topic: str
    type: str
    qos: dict
    rate: float = 0.0  # publishers only; 0 means it publishes nothing


@dataclass
class TfEdge:
    parent: str
    child: str
    publisher: str
    static: bool = False
    rate: float = 0.0


@dataclass
class Graph:
    nodes: dict[str, dict] = field(default_factory=dict)
    tf: list[TfEdge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        tf = []
        for key, edge in (data.get("tf") or {}).items():
            parent, child = (part.strip() for part in key.split("->"))
            tf.append(
                TfEdge(
                    parent,
                    child,
                    edge["publisher"],
                    bool(edge.get("static", False)),
                    float(edge.get("rate", 0)),
                )
            )
        return cls(nodes=data.get("nodes") or {}, tf=tf)

    def _endpoints(self, kind: str) -> list[Endpoint]:
        out = []
        for node, spec in self.nodes.items():
            for topic, ep in (spec.get(kind) or {}).items():
                ep = ep or {}
                out.append(
                    Endpoint(
                        node=node,
                        topic=topic,
                        type=ep.get("type", "std_msgs/msg/String"),
                        qos={**DEFAULT_QOS, **(ep.get("qos") or {})},
                        rate=float(ep.get("rate", 0)),
                    )
                )
        return out

    @property
    def publishers(self) -> list[Endpoint]:
        return self._endpoints("publishes")

    @property
    def subscriptions(self) -> list[Endpoint]:
        return self._endpoints("subscribes")

    def topics(self) -> dict[str, list[str]]:
        """Topic name to the sorted message types seen on it."""
        types: dict[str, set[str]] = {}
        for ep in self.publishers + self.subscriptions:
            types.setdefault(ep.topic, set()).add(ep.type)
        if self.tf:
            types.setdefault("/tf", set()).add("tf2_msgs/msg/TFMessage")
            if any(edge.static for edge in self.tf):
                types.setdefault("/tf_static", set()).add("tf2_msgs/msg/TFMessage")
        return {topic: sorted(ts) for topic, ts in sorted(types.items())}

    def params(self, node: str) -> dict:
        return dict((self.nodes.get(node) or {}).get("params") or {})

    def live_tf(self) -> list[TfEdge]:
        """TF edges whose publishing node is running."""
        return [edge for edge in self.tf if edge.publisher in self.nodes]


def apply_ops(data: dict, ops: list[dict]) -> dict:
    data = copy.deepcopy(data)
    for op in ops:
        *parents, last = op["path"]
        target = data
        for key in parents:
            target = target.setdefault(key, {}) if op["op"] == "set" else target[key]
        if op["op"] == "set":
            target[last] = op["value"]
        elif op["op"] == "delete":
            del target[last]
        elif op["op"] == "rename":
            target[op["value"]] = target.pop(last)
        else:
            raise ValueError(f"unknown fault op {op['op']!r}")
    return data


@dataclass
class Scenario:
    id: str
    question: str
    graph: Graph
    root_cause: str
    entities: list[str]  # any of these names the faulty thing
    explanation: str = ""
    difficulty: str = "easy"


SCENARIO_DIR = Path(__file__).parent / "scenarios"


def load_scenario(path: Path) -> Scenario:
    spec = yaml.safe_load(path.read_text())
    base = yaml.safe_load((SCENARIO_DIR / "bases" / f"{spec.get('base', 'diff_drive')}.yaml").read_text())
    data = apply_ops(base, spec.get("faults") or [])
    expected = spec["expected"]
    entity = expected.get("entity") or []
    return Scenario(
        id=spec.get("id", path.stem),
        question=spec["question"].strip(),
        graph=Graph.from_dict(data),
        root_cause=expected["root_cause"],
        entities=[entity] if isinstance(entity, str) else list(entity),
        explanation=(expected.get("explanation") or "").strip(),
        difficulty=spec.get("difficulty", "easy"),
    )


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(directory.glob("*.yaml"))]
