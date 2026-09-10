---
name: pose-refiner
description: Refines per-frame object poses for ONE frame window against a frozen scene.py, verifying every sweep winner visually before accepting it. Spawn one per window from multiagent.windows plan; point it at its iterations/NNNNNN/windows/<wid>/brief.md.
---

Before doing any work, read and follow the window brief you were given (the
absolute path to `RUN_DIR/iterations/NNNNNN/windows/<wid>/brief.md` is in your prompt).
The brief is SELF-CONTAINED: it defines your entire task, every rule, and every
path — do not go hunting for other instruction documents. Your only deliverable
is the window's `poses.json`.

<!--
Claude Code mirror of .codex/agents/pose-refiner.toml (see
harness/multiagent/windows.md, "the pose-refiner subagent bootstrap"). The two
definitions must say the same thing: only the bootstrap is runtime-specific,
the brief is the single instruction document either refiner reads.

Runtime notes for the orchestrator running under Claude Code:
- Spawn the round as one batch of Agent calls with subagent_type "pose-refiner",
  each prompt just "Follow <absolute brief path>" plus the round goal.
- A finished Claude Code subagent releases its slot when the Agent call
  returns; there are no stale agents to close before the next round (the
  "Round hygiene" section of windows.md describes the Codex runtime's cap).
- Subagents inherit the orchestrator's model and provider.
-->
