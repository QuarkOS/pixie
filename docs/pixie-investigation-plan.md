# Pixie stack investigation workflow

The goal is higher useful training throughput, lower chat latency and better
training data on the existing local GPUs. The operator supplies hardware runs.
The code author uses source inspection and CPU fixtures, without touching live
jobs or discovering local assets.

## Completion gates

- One report covers training, inference and data gathering. Every performance
  number identifies supplied measurement, source-derived calculation or estimate.
- A ranked table includes expected benefit, effort, risk and acceptance tests.
- Each code change has its own branch/PR and a runnable test or benchmark.
- Frozen holdout files and existing runs are byte-for-byte untouched.
- No GPU performance or real-dataset quality claim relies on a CPU fixture.

## Work sequence

1. Read the current merged code and reconcile it with the operator's results.
2. Audit three independent seams in isolated worktrees. Training owns step
   phase profiling, inference owns exact-shape blank-feature reuse, and data
   gathering owns server restart reliability followed by slow-streak telemetry.
3. Add a read-only holdout-byte preflight on another branch. Keep strict raw
   hashes; do not canonicalize JSON or rewrite original holdouts.
4. Produce the report and shape-derived checkpoint/cost arithmetic. Inspect
   accepted and rejected changes before proposing another optimization.
5. Run CPU behavior tests, review each diff, open separate PRs and record the
   commands for hardware verification. Leave numerical and quality questions
   open until the operator supplies their results.

## Throughput checkpoint

- Blocking first step was fetching merged main and reading the current guards.
- Independent workstreams own disjoint worktrees. No agent shares a branch.
- Shared state is limited to integration/reporting by the parent. Checkpoints,
  GPU jobs and source data remain outside this work.
- The smallest safe units are one correction or experiment per PR. A server
  lifecycle fix precedes telemetry; profiling precedes speculative loader work.

## Review rules

The Explain the Number principle keeps short-segment overhead separate from
long-run throughput. Encode Lessons in Structure requires byte and parity
checks after the reported failures. Prove It Works requires executed tests and
inspection of the final PR diffs. Model the Domain keeps raw holdout identity,
full-batch CLIP cache keys and phase timing explicit.
