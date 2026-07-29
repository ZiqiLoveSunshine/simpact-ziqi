"""Replay a recorded plan through its task's simulator — no VLM, no replanning.

Loads a committed plan (e.g. a trial's ``runs/refined_plan.json``), constructs the
task's rollout via the registry (simpact/tasks.py), and re-runs it with rendering +
video recording on. Useful for regenerating the ``final_rollout`` artifacts (stills
+ mp4) of a bundled run deterministically, or for watching your own saved plans.

The MuJoCo push replay is exact (same scene + waypoints => same trajectory), which
``--check`` asserts by comparing final object positions against a previously
recorded rollout JSON. MPM/ARAP replays are deterministic in practice but are
regenerated artifacts, not byte-identical guarantees.

  MUJOCO_GL=egl uv run python scripts/replay_rollout.py --task push \\
      --scene examples/push_real2sim/0103_push_0 \\
      --plan  examples/push_real2sim/0103_push_0/runs/refined_plan.json \\
      --out_dir /tmp/push_replay \\
      --check examples/push_real2sim/0103_push_0/runs/final_rollout/rollout_00.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from simpact.actions import ProposalSet
from simpact.tasks import TASKS


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=sorted(TASKS))
    task = TASKS[pre.parse_known_args()[0].task]

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=task.example_trial)
    ap.add_argument("--plan", required=True, help="ProposalSet JSON (e.g. refined_plan.json)")
    ap.add_argument("--proposal", type=int, default=0, help="proposal index inside the set")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--instruction", default="(replay of a recorded plan)")
    ap.add_argument("--check", default=None,
                    help="recorded rollout JSON: assert the replay's final object "
                         "positions match (push replays are exact)")
    for flag, kw in task.cli:
        ap.add_argument(flag, **kw)
    a = ap.parse_args()

    plan = ProposalSet.from_json(a.plan).action_proposals[a.proposal]
    roll = task.make_rollout(Path(a.scene), a)
    path = Path(roll.run(plan, 0, a.out_dir))
    print(f"replayed {a.plan} -> {path}")
    d = json.loads(path.read_text())
    if d.get("video"):
        print(f"video: {path.parent / d['video']}")

    if a.check:
        ref = json.loads(Path(a.check).read_text())
        got = d["snapshots"][-1]["objects"]
        want = ref["snapshots"][-1]["objects"]
        for name, w in want.items():
            delta = float(np.linalg.norm(np.asarray(got[name]["position"])
                                         - np.asarray(w["position"])))
            status = "ok  " if delta < 1e-3 else "FAIL"
            print(f"  {status} {name}: final-position delta {delta * 100:.3f} cm")
            assert delta < 1e-3, f"replay diverged for {name}"
        print("replay matches the recorded rollout")


if __name__ == "__main__":
    main()
