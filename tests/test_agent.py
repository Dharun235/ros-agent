from ros_agent.agent import ChatReply, parse_answer, run_agent
from ros_agent.bench import run_bench, score
from ros_agent.graph import SCENARIO_DIR, load_scenario


class ScriptedClient:
    """Replays fixed replies and records what tools were offered."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.offered_tools = []

    def chat(self, messages, tools):
        self.offered_tools.append(bool(tools))
        return self.replies.pop(0)


QOS_ANSWER = (
    'The subscriber is RELIABLE while the lidar publishes BEST_EFFORT.\n'
    '{"root_cause": "qos_mismatch", "entity": "/scan", "fix": "Use best_effort in slam_toolbox."}'
)


def _scenario(prefix):
    return load_scenario(next(SCENARIO_DIR.glob(f"{prefix}*.yaml")))


def test_parse_answer_takes_last_verdict():
    text = 'Maybe {"root_cause": "none"} ... no: {"root_cause": "qos_mismatch", "entity": "/scan"}'
    assert parse_answer(text) == {"root_cause": "qos_mismatch", "entity": "/scan"}
    assert parse_answer("no json here") is None


def test_agent_runs_tools_then_answers():
    from ros_agent.tools import SimBackend

    client = ScriptedClient(
        [
            ChatReply("", [{"id": "a", "name": "topic_info", "arguments": {"topic": "/scan"}}]),
            ChatReply(QOS_ANSWER),
        ]
    )
    result = run_agent("map never updates", SimBackend(_scenario("01").graph), client)
    assert result.tool_calls == 1
    assert "Reliability: RELIABLE" in result.messages[3]["content"]
    assert result.answer["root_cause"] == "qos_mismatch"


def test_agent_forces_answer_after_budget():
    from ros_agent.tools import SimBackend

    looping = [ChatReply("", [{"id": str(i), "name": "node_list", "arguments": {}}]) for i in range(2)]
    client = ScriptedClient([*looping, ChatReply(QOS_ANSWER)])
    result = run_agent("q", SimBackend(_scenario("01").graph), client, max_tool_calls=2)
    assert result.tool_calls == 2
    assert client.offered_tools == [True, True, False]
    assert result.answer is not None


def test_score_accepts_any_listed_entity_and_none_case():
    remap = _scenario("02")
    assert score({"root_cause": "topic_name_mismatch", "entity": "/diff_drive_controller/cmd_vel_unstamped"}, remap)["solved"]
    assert not score({"root_cause": "qos_mismatch", "entity": "/cmd_vel"}, remap)["solved"]
    assert score({"root_cause": "none", "entity": ""}, _scenario("10"))["solved"]
    assert not score(None, remap)["solved"]


def test_bench_writes_runs(tmp_path):
    out = tmp_path / "runs.jsonl"
    summary = run_bench([_scenario("01")], ScriptedClient([ChatReply(QOS_ANSWER)]), out)
    assert summary["solved"] == 1
    assert '"solved": true' in out.read_text()
