# Changelog

All notable changes to `dsh-compaction-opencode`.

The runtime is a single file, `lib/index.js`. Releases that only change tests or
packaging say so explicitly, because it means no reinstall is required for the
running behaviour to be current.

## 0.2.2

Test and packaging release. `lib/index.js` is byte-identical to 0.2.1
(`9E126DB5DBE1`, 58918 bytes), so the runtime behaviour is unchanged.

Added — `.scratch/verify-host.mjs` grew from 75 to 93 real-package checks. Three of
the new ones cover guards that a live session actually executed on 2026-10-13
while its threshold was temporarily lowered to force a trigger; before this
release none of them had a test:

- `COMPACTION_UNSHRINKABLE` — the summarization budget cannot absorb the newest
  exchange, so `flattenAndDropOldest` refuses to drop it. Asserts the warning and
  that the surface is left untouched.
- `COMPACTION_NOT_SMALLER` — the model replied with a summary no smaller than what
  it replaces. Asserts the warning and that the surface is left untouched.
- `still above the pressure threshold after N attempt(s)` — the retained tail alone
  exceeds the threshold. Asserts the engine runs exactly `compactionRetries + 1`
  attempts, then stops and warns instead of thrashing.
- The retained tail must begin at a user message (`_select`), which is the direct
  cause of a single oversized turn flooring the surface. Asserted so it is not
  later "fixed" into unfaithfulness.

Added — real-package end-to-end coverage for `compactNow`, the only path
`/compact` uses, pinning its two contracts: a session with an open turn is
refused (`ManualCompactionError`, code `busy`), and an idle session compacts even
when it sits below the pressure threshold.

Fixed — packaging. `dsh.plugin.json` was missing from `package.json#files`, so a
packed install silently omitted it. `LICENSE` and `CHANGELOG.md` are now listed
too, and `dsh.plugin.json` is exported by name.

Changed — `peerDependencies` on the `@deepseek-ai/*` host packages moved from exact
`0.2.0-rc.2` pins to `>=0.2.0-rc.2`, matching the convention already used by
`dsh-context`. `dsh.compatibility.dshReleases` now lists both `0.1.7-rc.2` and
`0.2.0-rc.2`.

## 0.2.1

Test release. `lib/index.js` unchanged.

Fixed — the smoke host's `BlockAssembler` stub accepted a fabricated block
protocol (`{type:"text"}`, `{type:"finish",finish}`) and silently discarded every
chunk it did not recognise. The real protocol is `block-start` / `text-delta` /
`reasoning-delta` / `tool-call-delta` / `block-end` / `usage` / `finish`(`reason`),
and the real `BlockAssembler.push` ends in `assertNever` — an unknown chunk must
throw. The suite therefore could not have detected a wrong stream protocol. The
stub now follows the real protocol and throws on unknown blocks, and the real
end-to-end compaction was frozen into `verify-host.mjs`.

## 0.2.0-rc.2

Initial port of OpenCode's `packages/core/src/session/compaction.ts` (963 lines) as
a DeepSeek Harness `CompactionEngine`:

- `calculateCeiling` with `RESERVE_MIN = 16000` and `UNKNOWN_WINDOW = 200000`, plus
  the Harness-only completion-reservation constraint
  `min(ceiling, window - reservedCompletionTokens - headroomTokens)`.
- `splitConversation` / `recentStart`: a verbatim recent tail whose start snaps to a
  user message, then to a tool-pairing-balanced boundary.
- The structured `SUMMARY_TEMPLATE`, `filled()` validation, and a single `NUDGE`
  retry for an unfilled reply.
- The `<conversation-checkpoint>` wire format and `compactCheckpointSource`.
- `SHRINK_STEPS = [0.7, 0.5, 0.35]` for progressive request shrinkage.
- `flattenAndDropOldest` with `[N older exchanges omitted]` markers.
- `TOOL_OUTPUT_MAX_CHARS = 1250`, and 1500 / 2000 pricing for image and PDF blocks.
- The Harness transaction: `compaction/start` lock → `compaction/summary` →
  `user/message` with `surfaceOp: { op: "replace" }` → `compaction/end`, with the
  `compaction/summary-error` waterfall as the recovery hook.
