"""Closed-loop action optimization: regress <-> rollout <-> verify, with memory.

the original pipeline's ``regress`` is open-loop — analyze N candidate rollouts once, emit one plan,
stop. This wraps it into a closed loop:

  1. verify each candidate rollout; if one already accomplishes the task, return it.
  2. otherwise regress over ALL rollouts-so-far (each carrying its verifier verdict)
     -> one refined plan -> roll it out -> verify it.
  3. if it fails, its rollout (with verdict) is appended to the rollouts folder and
     becomes context for the next regress. Repeat until success or ``max_iters``.

The rollout step is **injected** as ``rollout_fn(plan, index) -> rollout_json``, so
this module depends only on ``generator`` (the demo wires it from
``executor.MuJoCoRollout``) and is fully stub-testable without a VLM or MuJoCo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from simpact.actions import ActionProposal, ProposalSet
from simpact.generator.regress import RegressOptimizer, parse_rigid_rollout
from simpact.generator.verify import TaskVerifier, Verdict

# Roll a plan out and return the saved rollout JSON path. index controls the
# rollout's filename so refined attempts accumulate in the rollouts folder.
RolloutFn = Callable[[ActionProposal, int], Union[str, Path]]


@dataclass
class Attempt:
    kind: str  # "candidate" | "refined"
    index: int
    plan: ActionProposal
    rollout_path: Path
    verdict: Verdict


@dataclass
class LoopResult:
    success: bool
    best_plan: ActionProposal
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def best_proposalset(self) -> ProposalSet:
        return ProposalSet([self.best_plan])

    @property
    def iterations(self) -> int:
        return sum(1 for a in self.attempts if a.kind == "refined")


def attach_verdict(rollout_path: Path, verdict: Verdict) -> None:
    """Persist the verdict inside the rollout JSON so a later regress reads it."""
    data = json.loads(rollout_path.read_text())
    data["verdict"] = verdict.to_dict()
    rollout_path.write_text(json.dumps(data, indent=2))


class OptimizationLoop:
    """Iterate regress -> rollout -> verify until the task is verified done."""

    def __init__(
        self,
        regressor: RegressOptimizer,
        verifier: TaskVerifier,
        rollout_fn: RolloutFn,
        *,
        max_iters: int = 5,
        accumulate_dir: Optional[Union[str, Path]] = None,
        parse_rollout=parse_rigid_rollout,
    ):
        self.regressor = regressor
        self.verifier = verifier
        self.rollout_fn = rollout_fn
        self.max_iters = max_iters
        self.accumulate_dir = Path(accumulate_dir) if accumulate_dir else None
        self.parse_rollout = parse_rollout

    def _verify(self, rollout_path: Path, instruction: str, context: str) -> Verdict:
        trace = ""
        try:
            trace = self.parse_rollout(rollout_path).get("text", "")
        except Exception:
            pass
        verdict = self.verifier.verify(rollout_path, instruction, context, action_trace=trace)
        attach_verdict(rollout_path, verdict)
        return verdict

    def run(
        self,
        candidates: Sequence[tuple],
        instruction: str,
        context: str = "",
        *,
        on_event: Optional[Callable[[str, Attempt], None]] = None,
    ) -> LoopResult:
        """Run the closed loop.

        Args:
            candidates: ``[(ActionProposal, rollout_json_path), ...]`` — the proposals
                already rolled out (their JSONs live in ``accumulate_dir``).
            instruction: the high-level task.
            context: real-world context string (frame, object layout).
            on_event: optional callback ``(stage, attempt)`` for progress logging.
        """
        candidates = [(p, Path(rp)) for p, rp in candidates]
        if self.accumulate_dir is None and candidates:
            self.accumulate_dir = candidates[0][1].parent
        attempts: list[Attempt] = []

        # 1. Verify candidates; return the first that already accomplishes the task.
        for i, (plan, rp) in enumerate(candidates):
            verdict = self._verify(rp, instruction, context)
            att = Attempt("candidate", i, plan, rp, verdict)
            attempts.append(att)
            if on_event:
                on_event("verify_candidate", att)
            if verdict.success:
                return LoopResult(True, plan, attempts)

        # 2. Closed loop: regress over the accumulating memory, roll out, verify.
        n = len(candidates)
        last_refined: Optional[ActionProposal] = None
        for it in range(self.max_iters):
            refined = self.regressor.optimize(self.accumulate_dir, instruction, context)
            plan = refined.action_proposals[0]
            last_refined = plan
            idx = n + it
            rp = Path(self.rollout_fn(plan, idx))
            verdict = self._verify(rp, instruction, context)
            att = Attempt("refined", idx, plan, rp, verdict)
            attempts.append(att)
            if on_event:
                on_event("verify_refined", att)
            if verdict.success:
                return LoopResult(True, plan, attempts)
            # else: rp (with verdict) now lives in accumulate_dir -> next regress sees it

        # 3. No verified success — return the last, most-informed refined plan
        #    (fall back to the last candidate if no refine ran).
        best = last_refined or (candidates[-1][0] if candidates else None)
        return LoopResult(False, best, attempts)
