import pytest

from ros_agent.graph import SCENARIO_DIR, load_scenario, load_scenarios
from ros_agent.tools import TOOL_NAMES, SimBackend


def _backend(prefix):
    return SimBackend(load_scenario(next(SCENARIO_DIR.glob(f"{prefix}*.yaml"))).graph)


def test_all_scenarios_load_with_known_causes():
    from ros_agent.agent import ROOT_CAUSES

    scenarios = load_scenarios()
    assert len(scenarios) >= 10
    assert {s.root_cause for s in scenarios} <= set(ROOT_CAUSES)


def test_healthy_robot_is_consistent():
    sim = _backend("10")
    assert "Reliability: BEST_EFFORT" in sim.run("topic_info", {"topic": "/scan"})
    assert "RELIABLE" not in sim.run("topic_info", {"topic": "/scan"})
    assert sim.run("topic_hz", {"topic": "/scan"}).startswith("average rate: 10.000")
    assert sim.run("tf_echo", {"source_frame": "laser", "target_frame": "map"}).startswith("At time")


def test_qos_mismatch_visible_in_topic_info_not_hz():
    sim = _backend("01")
    info = sim.run("topic_info", {"topic": "/scan"})
    assert "Reliability: BEST_EFFORT" in info and "Reliability: RELIABLE" in info
    # ros2 topic hz still sees the data; only the QoS view reveals the fault.
    assert sim.run("topic_hz", {"topic": "/scan"}).startswith("average rate")


def test_remapped_topic_leaves_cmd_vel_without_subscribers():
    info = _backend("02").run("topic_info", {"topic": "/cmd_vel"})
    assert "Publisher count: 1" in info and "Subscription count: 0" in info


def test_dead_node_and_missing_tf():
    assert "does not appear to be published" in _backend("03").run("topic_hz", {"topic": "/scan"})
    assert "does not exist" in _backend("04").run("tf_echo", {"source_frame": "laser", "target_frame": "map"})


@pytest.mark.parametrize(
    ("prefix", "tool", "args", "expected"),
    [
        ("05", "param_get", {"node": "/diff_drive_controller", "name": "wheel_separation"}, "Double value is: 0.0"),
        ("06", "topic_info", {"topic": "/cmd_vel"}, "Type: geometry_msgs/msg/Twist; geometry_msgs/msg/TwistStamped"),
        ("07", "param_get", {"node": "/slam_toolbox", "name": "use_sim_time"}, "Boolean value is: True"),
        ("08", "topic_hz", {"topic": "/odom"}, "no new messages"),
        ("09", "topic_echo", {"topic": "/scan"}, "frame_id: laser_frame"),
    ],
)
def test_each_fault_is_observable(prefix, tool, args, expected):
    assert expected in _backend(prefix).run(tool, args)


def test_unknown_inputs_return_cli_style_errors():
    sim = _backend("10")
    assert sim.run("node_info", {"node": "/nope"}) == "Unable to find node '/nope'"
    assert sim.run("topic_info", {"topic": "/nope"}) == "Unknown topic '/nope'"
    assert sim.run("param_get", {"node": "/lidar_driver", "name": "nope"}) == "Parameter not set."
    assert sim.run("reboot", {}).startswith("Unknown tool")
    assert sim.run("topic_info", {}).startswith("Bad arguments")
    assert "node_list" in TOOL_NAMES
