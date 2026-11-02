# ros-agent

**An offline ROS 2 debugging copilot.** Ask why your robot is misbehaving; a
small local model inspects the ROS 2 graph with read-only tools and tells you
the root cause and the fix. No cloud, no API keys, nothing leaves the robot.

```text
$ ros-agent ask --live "SLAM never updates the map"
>>> topic_info({"topic": "/scan"})
>>> param_get({"node": "/slam_toolbox", "name": "use_sim_time"})
/lidar_driver publishes /scan BEST_EFFORT but /slam_toolbox subscribes RELIABLE,
so they never connect. Set the subscriber to best_effort (sensor data QoS).
{"root_cause": "qos_mismatch", "entity": "/scan", "fix": "Use best_effort QoS in slam_toolbox."}
```

- **Offline**: runs on a laptop or robot computer with [Ollama](https://ollama.com)
  or any OpenAI-compatible local server (llama.cpp, LM Studio, vLLM).
- **Safe**: tools only read the graph (`ros2 node/topic/param` and TF lookups).
- **Measured**: a built-in benchmark of broken-robot scenarios that runs
  anywhere, even without ROS installed.
- **Light**: one dependency (PyYAML).

## Install

```bash
pip install -e .
ollama pull qwen3:4b-instruct   # or any tool-calling model
```

## Use

On a machine with ROS 2 sourced:

```bash
source /opt/ros/jazzy/setup.bash
ros-agent ask --live "My robot ignores teleop commands" --show-trace
```

Without ROS, try a simulated broken robot:

```bash
ros-agent scenarios
ros-agent ask --scenario 01 --show-trace
ros-agent tool 01 topic_info topic=/scan      # run one tool by hand
```

Choose the model with `--model` or `ROS_AGENT_MODEL`. For an OpenAI-compatible
server use `--api openai --base-url http://127.0.0.1:8080/v1`.

## Benchmark

Each scenario takes a working robot (`ros_agent/scenarios/bases/diff_drive.yaml`:
diff-drive base, lidar, camera, SLAM) and injects one fault: QoS mismatch,
remapped topic, dead or silent publisher, wrong message type, missing
transform, wrong parameter, `use_sim_time` without `/clock`, or no fault at
all. The simulated tools print the same text as the real `ros2` CLI, so what a
model learns here carries over to live robots.

```bash
ros-agent bench --model qwen3:4b-instruct --out data/bench.jsonl
```

A run is solved when the model names the right root cause and the faulty
topic, node, parameter or frame. Every conversation is saved, so solved runs can
become training data for a small specialised model.

### Results

First baseline, M2 MacBook (16 GB), Ollama, 2 November 2026:

| Model | Solved | Right root cause | Tool calls per scenario | Time per scenario |
| --- | --- | --- | --- | --- |
| qwen3:4b-instruct (Q4_K_M) | 2/10 | 30% | 7.0 | 42 s |
| qwen3:0.6b | 1/10 | 10% | 0 | 3 s |

The 4B model inspects the graph sensibly but over-guesses `missing_publisher`
and `missing_transform`; it solved the dead lidar driver and the missing laser
transform. `qwen3:4b` (the thinking variant) keeps reasoning even with
thinking disabled, which made it too slow to finish on this machine. These
scores are the starting point for better prompts and a fine-tuned model.

## Add a scenario

A scenario is a few lines of YAML on top of a base robot:

```yaml
# ros_agent/scenarios/11_my_fault.yaml
question: The camera image never reaches my detector.
faults:
  - {op: set, path: [nodes, /camera_driver, publishes, /camera/image_raw, rate], value: 0}
expected:
  root_cause: stalled_publisher
  entity: /camera/image_raw
  explanation: The camera driver advertises the topic but publishes nothing.
```

Fault operations are `set`, `delete` and `rename` on a key path into the base
graph. Run `pytest` to check it loads, then `ros-agent ask --scenario 11`.
Scenarios from real debugging sessions are the most valuable contribution.

## Roadmap

- More bases (manipulator with MoveIt, Nav2 stack) and many more scenarios
- Validate scenarios against real ROS 2 in Docker and MuJoCo simulations
- Fine-tune and publish a small model (Gemma 4 E4B class) for on-robot use
- MCP server mode for Claude, Cursor and ROSA users

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The earlier Ollama + OpenSearch + Phoenix web app is on the `main` branch.
