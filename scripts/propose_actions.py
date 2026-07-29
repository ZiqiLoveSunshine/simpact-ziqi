"""Driver: sample action proposals for a scene (random and/or LLM).

simpact equivalent of the original ``generator/propose.sh``: build scene context, then
produce candidate action-primitive sequences with the random sampler and/or the
vision-LLM proposer, and write them as a ``ProposalSet`` JSON.

Random backend needs no scene/LLM (context-free sampling). The LLM backend needs
a scene image + context (object poses + an EE pose) and ``GOOGLE_API_KEY``.

Examples:
  # random only (no GPU/LLM/robot)
  python scripts/propose_actions.py --backend random --task push \
      --n 20 --out /tmp/proposals.json

  # LLM (or both), offline context from a recorded trial + a saved EE pose
  python scripts/propose_actions.py --backend both \
      --data_dir /path/to/data/0103_push_0 --cam 1 \
      --objects "orange bottle. brown purple box." --task push \
      --instruction "Push the orange bottle to the right, avoiding the box." \
      --ee-pose-file /path/to/ee_pose.txt --out /tmp/proposals.json
"""
import argparse
from pathlib import Path

from simpact.actions import ProposalSet
from simpact.generator.sampling import DEFAULT_RANGES, RandomProposer

# task -> (context-template name, allowed primitive types) — from the task registry
# (simpact/tasks.py), the single source of truth shared with scripts/optimize.py.
from simpact.tasks import TASKS  # noqa: E402

TASK_PROFILES = {t.name: (t.context_template, sorted(t.allowed_prims)) for t in TASKS.values()}


def resolve_image(data_dir: Path, cam: int, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for ext in (".png", ".jpg", ".npy"):
        p = data_dir / f"camera{cam}_rgb{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no scene image found in {data_dir} (camera{cam}_rgb.*); pass --image")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["random", "llm", "both"], default="random")
    ap.add_argument("--task", default=None, help=f"one of {sorted(TASK_PROFILES)} (sets context template + allowed primitives)")
    ap.add_argument("--allowed", default=None, help="comma-separated primitive types (overrides task profile)")
    ap.add_argument("--out", required=True)
    # random sampler
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    # LLM proposer / context
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--objects", default=None, help='e.g. "orange bottle. brown purple box."')
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt-template", default="primitive")
    ap.add_argument("--ee-pose-file", default=None, help="4x4 matrix or 'x y z qx qy qz qw'")
    ap.add_argument("--host", default=None, help="Franka FCI IP for a live EE-pose read (needs franky)")
    a = ap.parse_args()

    allowed = (
        [s.strip() for s in a.allowed.split(",") if s.strip()] if a.allowed
        else (TASK_PROFILES[a.task][1] if a.task in TASK_PROFILES else None)
    )

    proposals = []

    if a.backend in ("random", "both"):
        types = allowed or list(DEFAULT_RANGES)
        ps = RandomProposer(seed=a.seed).sample(
            n=a.n, action_types=types, min_len=a.min_len, max_len=a.max_len
        )
        proposals.extend(ps.action_proposals)
        print(f"[random] sampled {len(ps.action_proposals)} proposals over {types}")

    if a.backend in ("llm", "both"):
        if not (a.data_dir and a.objects and a.instruction):
            ap.error("--backend llm/both needs --data_dir, --objects and --instruction")
        from simpact.generator.context import EEPose, build_context
        from simpact.generator.propose import VLMProposer

        data_dir = Path(a.data_dir)
        if a.ee_pose_file:
            ee = EEPose.from_file(a.ee_pose_file)
        elif a.host:
            ee = EEPose.from_robot(a.host)
        else:
            ap.error("--backend llm/both needs --ee-pose-file (offline) or --host (live robot)")
        ctx_template = TASK_PROFILES[a.task][0] if a.task in TASK_PROFILES else (a.task or "push")
        context = build_context(a.objects, data_dir, ctx_template, ee, cam_id=a.cam)
        image = resolve_image(data_dir, a.cam, a.image)
        ps = VLMProposer(prompt_template=a.prompt_template).propose(
            a.instruction, image, context=context,
            allowed_types=set(allowed) if allowed else None,
        )
        proposals.extend(ps.action_proposals)
        print(f"[llm] {len(ps.action_proposals)} proposals from {image.name}")

    out_set = ProposalSet(proposals)
    errs = out_set.validate(
        allowed_types=set(allowed) if allowed else None, ranges=DEFAULT_RANGES
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out_set.to_json(a.out)
    print(f"wrote {len(proposals)} proposals -> {a.out}")
    if errs:
        print(f"WARNING: {len(errs)} validation issue(s); first few:")
        for e in errs[:5]:
            print(f"  - {e}")
    else:
        print("all proposals valid")


if __name__ == "__main__":
    main()
