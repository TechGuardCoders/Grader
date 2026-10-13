# Grader

**Trajectory grading engine — the trajectory is the unit of accountability.**

Final-answer evals hide the exact step where agents break. Grader scores every tool call in an agent trajectory against a deterministic DAG of expected steps, and fails the run on:

- **Order violations** — the wrong tool ran at a position
- **Parameter violations** — missing required params, failed validators, forbidden params
- **Hallucinations** — unknown tools, or params no spec declares
- **Safety violations** — a safety-critical step (e.g. `isolate-host`) ran without its required preceding confirmation

## The model

A **trajectory** is a recorded list of tool calls:

```json
{"tool": "search", "params": {"q": "kv cache"}}
{"tool": "fetch", "params": {"url": "https://example.com"}}
{"tool": "summarize", "params": {"text": "..."}}
```

A **TrajectorySpec** is the expected DAG: ordered `StepSpec`s, each declaring required params (with validators), optional params, forbidden params, and a `safety_critical` flag. Safety-critical steps must be preceded by a `confirm` step — positionally, not just anywhere in the run.

## Quick start

```python
from grader import StepSpec, TrajectorySpec, grade_trajectory

spec = TrajectorySpec(name="containment", steps=[
    StepSpec(tool="confirm", optional_params={"action"}),
    StepSpec(tool="isolate-host", required_params={"ip"}, safety_critical=True),
])

trajectory = [
    {"tool": "confirm", "params": {"action": "isolate 10.0.0.5"}},
    {"tool": "isolate-host", "params": {"ip": "10.0.0.5"}},
]

r = grade_trajectory(trajectory, spec)
r.verdict        # 'pass'
r.to_dict()      # full machine-readable result
```

CLI-friendly: `spec_from_dict()` and `trajectory_from_jsonl()` load both sides from plain data.

## Verdicts

| Verdict | Meaning |
|---|---|
| `pass` | every step in order, params valid, no hallucinations, no safety violations |
| `incomplete` | trajectory shorter than spec (agent stopped early) |
| `fail` | any order/param violation, hallucination, or safety violation |

## Part of the TechGuardCoders portfolio

Grader's trajectory format is the artifact type downstream projects consume: Shadow Route diffs trajectories against it, Debug Spray replays them, Piranha attacks them. It also grades swarm hunts step-level in [NK-NODE](https://github.com/TechGuardCoders/NK-NODE) — did `isolate-host` follow the safety-check order? Did any step hallucinate a parameter?

## Tests

```bash
python -m pytest tests/ -q   # 18 tests
```

Every failure class the grader claims to catch has a test proving it catches it — and the safety checks are tested positionally (a `confirm` *after* the dangerous step doesn't count).
