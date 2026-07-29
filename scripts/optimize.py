"""Unified closed-loop optimizer: propose -> rollout -> verify <-> regress, for any task.

One driver for all four release tasks (the former per-task demo_optimize_loop_*
scripts). Task-specific behaviour — rollout class, prompt templates, allowed
primitives, measured gate, extra CLI — comes from the task registry
(simpact/tasks.py); the loop skeleton below is identical for every material:

  1. propose   VLMProposer        real photo + context + instruction -> N plans
  2. rollout   task rollout       each candidate -> rollout JSON + renders
  3. loop      OptimizationLoop   verify each rollout (TaskVerifier, plus the
                                  task's measured gate when it has one); if none
                                  solve the task, regress over all rollouts-so-far
                                  -> refined plan -> roll out -> verify -> repeat,
                                  until verified done or --max_iters.
  4. confirm   re-roll the chosen plan and verify it independently.

Needs GOOGLE_API_KEY; dough/sweep need CUDA + warp; push needs a GL context.

  python scripts/optimize.py --task rope \\
      --scene examples/rope_real2sim/1102_rope_11 --out_dir /tmp/rope_loop
  MUJOCO_GL=egl python scripts/optimize.py --task dough \\
      --scene examples/dough_real2sim/1104_sand_6 --out_dir /tmp/dough_loop
  MUJOCO_GL=egl python scripts/optimize.py --task sweep \\
      --scene examples/sweep_real2sim/0118_sweep_0 --out_dir /tmp/sweep_loop
  MUJOCO_GL=egl python scripts/optimize.py --task push \\
      --scene examples/push_real2sim/0103_push_0/build --out_dir /tmp/push_loop
"""
import argparse
import json
from pathlib import Path

from simpact.generator.context import build_context, resolve_initial_ee
from simpact.generator.optimize_loop import OptimizationLoop, attach_verdict
from simpact.generator.propose import VLMProposer
from simpact.generator.regress import RegressOptimizer
from simpact.generator.verify import TaskVerifier
from simpact.tasks import TASKS
from simpact.utils.layout import find_scene_file


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=sorted(TASKS))
    task = TASKS[pre.parse_known_args()[0].task]

    ap = argparse.ArgumentParser(
        description=__doc__, parents=[pre],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=task.example_trial,
                    help="trial dir (bundled capture/sim layout or a flat assets dir)")
    ap.add_argument("--instruction", default=task.default_instruction)
    ap.add_argument("--object", default=task.default_object,
                    help="object string for the VLM context (push: derived from the trial)")
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=3,
                    help="max regress<->verify iterations after the candidates (0 = no regress)")
    ap.add_argument("--out_dir", required=True)
    for flag, kw in task.cli:
        ap.add_argument(flag, **kw)
    a = ap.parse_args()

    scene = Path(a.scene)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rollouts_dir = out / "rollouts"

    # rollout first: push discovers its objects + EE pose during scene assembly
    roll = task.make_rollout(scene, a)
    ee = getattr(roll, "context_ee", None) or resolve_initial_ee(scene)[0]
    obj_str = a.object or ". ".join(getattr(roll, "object_names", [])) + "."
    context = build_context(obj_str, scene, task.context_template, ee, cam_id=a.cam)
    photo = (find_scene_file(scene, f"camera{a.cam}_rgb.png", required=False)
             or find_scene_file(scene, f"camera{a.cam}_rgb.npy"))

    # 1. PROPOSE (the context template carries the task's rules)
    print("\n== 1. propose (VLM) ==", flush=True)
    proposals = VLMProposer(prompt_template="primitive").propose(
        a.instruction, photo, context=context, allowed_types=task.allowed_prims)
    proposals.to_json(out / "propose.json")  # initial proposals — uniform eval artifact
    for i, p in enumerate(proposals.action_proposals):
        print(f"  proposal {i}: {'+'.join(x.TYPE for x in p.action_sequence)} — "
              f"{(p.description or '')[:60]}")

    def do_rollout(plan, index, out_dir=rollouts_dir):
        path = Path(roll.run(plan, index, out_dir))
        print(f"    rollout {index}: "
              f"{task.describe_rollout(json.loads(path.read_text()))}", flush=True)
        return str(path)

    # 2. ROLLOUT each candidate (a proposal the rollout can't execute is skipped)
    print(f"\n== 2. rollout candidates ({task.name}) ==", flush=True)
    candidates = []
    for i, p in enumerate(proposals.action_proposals):
        try:
            candidates.append((p, do_rollout(p, i)))
        except ValueError as e:
            print(f"    proposal {i} skipped: {e}", flush=True)
    if not candidates:
        raise SystemExit("no proposal could be rolled out")

    # 3. CLOSED LOOP: verify each rollout (VLM + measured gate when the task has
    #    one); regress + roll out + verify until done
    gate = task.make_gate(scene, a, roll) if task.make_gate else None
    verifier = TaskVerifier(prompt_template=task.verify_template, success_gate=gate)
    print(f"\n== 3. optimize loop (verify{'[+gate]' if gate else ''} <-> regress, "
          f"max_iters={a.max_iters}) ==", flush=True)

    def log_event(stage, att):
        v = att.verdict
        print(f"  [{att.kind} {att.index}] verify -> {'SUCCESS' if v.success else 'failure'} "
              f"(conf {v.confidence:.2f}): {v.reason}", flush=True)
        if not v.success and v.remaining:
            print(f"      still needed: {v.remaining}", flush=True)

    loop = OptimizationLoop(
        RegressOptimizer(prompt_template=task.regress_template,
                         parse_rollout=task.parse_rollout),
        verifier, rollout_fn=do_rollout, max_iters=a.max_iters,
        accumulate_dir=rollouts_dir, parse_rollout=task.parse_rollout)
    result = loop.run(candidates, a.instruction, context, on_event=log_event)
    result.best_proposalset.to_json(out / "refined_plan.json")

    print(f"\n== loop result: {'SOLVED' if result.success else 'best-effort'} "
          f"after {result.iterations} regress iteration(s) ==")
    for act in result.best_plan.action_sequence:
        vals = " ".join(f"{k}={round(v, 4)}" for k, v in act.attr_values().items())
        print(f"    {act.TYPE:<16} {vals}")

    # 4. FINAL CONFIRMATION: re-roll the chosen plan and verify it independently —
    #    the loop's in-flight verdict can be a false positive, so this is the
    #    authoritative "did the returned plan ACTUALLY solve the task".
    print("\n== 4. confirm chosen plan ==", flush=True)
    fpath = Path(do_rollout(result.best_plan, 0, out_dir=out / "final_rollout"))
    fv = verifier.verify(fpath, a.instruction, context,
                         action_trace=task.parse_rollout(fpath).get("text", ""))
    attach_verdict(fpath, fv)
    if gate is not None:
        print(f"  measured gate: {gate(fpath)[1]}")
    print(f"  final plan verified: {'SUCCESS' if fv.success else 'NOT SUCCESSFUL'} "
          f"(conf {fv.confidence:.2f}) — {fv.reason}")
    print("DEMO_OK" if fv.success else "DEMO_DONE (chosen plan not verified successful)")


if __name__ == "__main__":
    main()
