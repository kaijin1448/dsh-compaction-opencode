/**
 * `dsh-compaction-opencode` — an OpenCode-flavoured compaction backend for the
 * DeepSeek Harness.
 *
 * It ports the context-management policy of OpenCode's
 * `packages/core/src/session/compaction.ts` onto DSH's durable session log:
 *
 *   - the structured checkpoint template (`## Objective` … `## Important Context`)
 *     with single-nudge validation and fail-closed rejection;
 *   - the `<conversation-checkpoint><summary>…</summary><recent-context>…</recent-context></conversation-checkpoint>`
 *     wire format for the replacement user message;
 *   - a keep-token verbatim tail selected at a user-message boundary;
 *   - the `[0.7, 0.5, 0.35]` shrink ladder after provider context-window rejections;
 *   - flatten-and-drop-oldest exchange reduction with a `[N older exchanges omitted]` note;
 *   - 1250-character tool-output truncation in every text rendering.
 *
 * OpenCode's Effect runtime, its drizzle/SQLite projection, its event bus, its
 * provider-native compaction windows and its model/cost catalog are NOT ported:
 * those are supplied by the host (`ctx.llm`, `ctx.tokenMeter`, `ctx.sessions`,
 * the session log, and the `compaction/*` durable markers).
 *
 * @module dsh-compaction-opencode
 */

import z from "@deepseek-ai/schemastery";
import {
	CompactionEngine,
	CompactionId,
	ManualCompactionError,
	compactCheckpointSource,
	isCompactCheckpointSource,
	toolPairingBalancedAfter,
	toolPairingBalancedBefore
} from "@deepseek-ai/dsh-compaction";
import {
	BlockAssembler,
	CONTEXT_WINDOW_EXCEEDED_CODE,
	LlmError,
	contentHasImage,
	createUserMessage,
	errorChain
} from "@deepseek-ai/dsh-llm";
import { assertNever, deepFreeze } from "@deepseek-ai/dsh-util-values";
import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { SessionSeq } from "@deepseek-ai/dsh-session";

//#region ported policy — OpenCode `packages/core/src/session/compaction.ts`

/** After each "too long" rejection, the next attempt aims at this share of the first rejected request. */
const SHRINK_STEPS = [0.7, 0.5, 0.35];
/** The least of the window kept free for the last reply before compaction and for the summary itself. */
const RESERVE_MIN = 16_000;
/** A common window size, assumed for the compaction request when the model's window is unknown. */
const UNKNOWN_WINDOW = 200_000;
/** Longest tool output kept in any text rendering of history. */
const TOOL_OUTPUT_MAX_CHARS = 1_250;
const IMAGE_TOKEN_ESTIMATE = 1_500;
const PDF_TOKEN_ESTIMATE = 2_000;
/** OpenCode's `Token.estimate`: a cheap character-count proxy, not a real tokenizer. */
const CHARS_PER_TOKEN = 4;

/** OpenCode `Token.estimate`. */
const estimateText = (text) => Math.round(text.length / CHARS_PER_TOKEN);

const SUMMARY_TEMPLATE = `You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Relevant Files
List the files and directories, other than the current working directory, that another agent would need to open to continue this work. Include at most 15, most important first. Do not list every file that was read or changed. Include paths outside the current working directory when relevant. If none, write "(none)".
- \`[file or directory path]\`: [brief reason it matters]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>`;

const SUMMARY_RULES = `Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Prefer short references over detailed restatement. It is fine to leave out information the next agent can recover from the code or the files listed above.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Preserve consequential workflow state, including whether changes are uncommitted, committed, pushed, under review, or merged.
- Do not mention the summary process or that context was compacted.`;

/** Summaries written with OpenCode's previous template carry this catch-all heading. */
const LEGACY_HEADING = "## Additional Context";

const NUDGE =
	"The previous response did not fill in the required summary template. Do not call tools. Return the summary as text using the exact section headings from the template.";

/** Section headings a compliant reply must contain at least one of. */
const HEADINGS = SUMMARY_TEMPLATE.split("\n").filter((line) => line.startsWith("##"));

/**
 * OpenCode's `buildPrompt`: the summarization directive, delivered as the final
 * user message after the replayed history so the auxiliary call stays a genuine
 * prefix of the conversation and reuses the provider's prefix cache.
 *
 * @param update - whether a previous checkpoint is present in the replayed history.
 * @param legacy - whether that checkpoint used the pre-#48058 template.
 * @returns the complete summary instruction.
 */
export function buildPrompt(update, legacy = false) {
	const shared = [
		"Summarize only what the user and the assistant said and did. Leave out instructions and setup the assistant was given rather than told by the user: repository conventions, instruction files such as AGENTS.md, and environment details like the session ID. The next agent receives current versions of all of these separately.",
		SUMMARY_TEMPLATE,
		SUMMARY_RULES,
		"Do not continue the task or call tools.",
		"Return only the structured summary in the requested format. Do not include a preamble, explanation, or other commentary."
	];
	if (!update) {
		return [
			"You MUST summarize the conversation above into a structured summary that will be given to another agent to resume the work.",
			...shared
		].join("\n\n");
	}
	return [
		"Update the existing checkpoint in the conversation above into one consolidated summary.",
		...(legacy
			? [
					"The existing checkpoint was written with an earlier format that recorded far more detail than this one asks for. Rewrite it at the level of detail described below rather than carrying its detail forward. Keep its requirements, decisions, and open questions; they came from earlier conversation with the user."
				]
			: []),
		"Newer history always takes precedence over the existing checkpoint. Preserve previous information unless newer history clearly contradicts, supersedes, resolves, or makes it stale. If something is no longer relevant to continuing the work, you may remove it.",
		"Incorporate newer requirements, decisions, progress, and context. Reconcile Work State and Next Move: move completed work out of Active, remove resolved blockers and answered questions, and preserve unresolved or pending work.",
		"Return only the updated Markdown sections. Do not reproduce the `<conversation-checkpoint>`, `<summary>`, or `<recent-context>` wrapper tags from the previous checkpoint.",
		...shared
	].join("\n\n");
}

/** Whether a reply filled at least one required section heading. */
export const filled = (text) => text.split("\n").some((line) => HEADINGS.includes(line.trim()));

//#endregion

//#region checkpoint wire format — OpenCode `runner/to-llm-message.ts`

const CHECKPOINT_OPEN = "<conversation-checkpoint>";
const CHECKPOINT_CLOSE = "</conversation-checkpoint>";
const SUMMARY_OPEN = "<summary>";
const SUMMARY_CLOSE = "</summary>";
const RECENT_OPEN = "<recent-context>";
const RECENT_CLOSE = "</recent-context>";

/**
 * Framing that makes the replacement user message established context rather
 * than a fresh instruction. Ported from `to-llm-message.ts`.
 */
const CHECKPOINT_PREAMBLE =
	"This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.";

/**
 * Render one checkpoint as OpenCode does: a single user message carrying the
 * summary and, when the tail was shadowed rather than left on the surface, the
 * verbatim recent context beside it.
 *
 * @param summaryText - the validated structured summary.
 * @param recentText - verbatim recent history, or `""` when the tail stays on the surface.
 * @returns content blocks for the replacement user message.
 */
export function frameCheckpoint(summaryText, recentText) {
	const parts = [CHECKPOINT_OPEN, `${SUMMARY_OPEN}\n${summaryText}\n${SUMMARY_CLOSE}`];
	if (recentText.length > 0) parts.push(`${RECENT_OPEN}\n${recentText}\n${RECENT_CLOSE}`);
	parts.push(CHECKPOINT_CLOSE);
	return [{ type: "text", text: `${CHECKPOINT_PREAMBLE}\n\n${parts.join("\n")}` }];
}

/** Extract the `<summary>` body of an existing checkpoint message, if any. */
export function checkpointSummaryText(message) {
	const text = message.content
		.filter((block) => block.type === "text")
		.map((block) => block.text)
		.join("");
	const start = text.indexOf(SUMMARY_OPEN);
	const end = text.lastIndexOf(SUMMARY_CLOSE);
	if (start === -1 || end === -1 || end <= start) return text;
	return text.slice(start + SUMMARY_OPEN.length, end).trim();
}

//#endregion

//#region rendering — OpenCode `messageToText` / `flattenMessage` / `truncateToolOutput`

/** OpenCode's `truncateToolOutput`, counting code points so a surrogate pair is never split. */
export function truncateToolOutput(value) {
	if (value.length <= TOOL_OUTPUT_MAX_CHARS) return value;
	let end = 0;
	let kept = 0;
	for (const char of value) {
		if (kept === TOOL_OUTPUT_MAX_CHARS) break;
		end += char.length;
		kept += 1;
	}
	if (end === value.length) return value;
	return `${value.slice(0, end)}\n[truncated]`;
}

/** Render one block of an attachment as OpenCode's `[Attached <mime>: <name>]` note. */
function attachmentNote(block) {
	if (block.type === "image") {
		const name = block.attachment.name;
		return name === undefined ? `[Attached ${block.attachment.mediaType}]` : `[Attached ${block.attachment.mediaType}: ${name}]`;
	}
	return `[Attached ${block.attachment.name}]`;
}

/**
 * One message as OpenCode's `messageToText` renders it for the verbatim recent
 * section. Empty for messages that section leaves out.
 *
 * DSH keeps a tool result as its own `tool`-role message, where OpenCode nests
 * it inside the assistant turn, so the result line is emitted by its own case.
 *
 * @param message - a derived DSH message.
 * @returns the text rendering, or `""` to leave the message out.
 */
export function messageToText(message) {
	switch (message.role) {
		case "system":
		case "developer":
			return "";
		case "user": {
			if (message.source !== undefined && isCompactCheckpointSource(message.source)) return "";
			const lines = [];
			for (const block of message.content) {
				if (block.type === "text") lines.push(`[User]: ${block.text}`);
				else if (block.type === "image" || block.type === "file") lines.push(`[User]: ${attachmentNote(block)}`);
			}
			return lines.join("\n");
		}
		case "assistant": {
			const lines = [];
			for (const block of message.content) {
				if (block.type === "text") lines.push(`[Assistant]: ${block.text}`);
				else if (block.type === "reasoning") {
					if (block.text) lines.push(`[Assistant reasoning]: ${block.text}`);
				} else if (block.type === "tool-call") lines.push(`[Assistant tool call]: ${block.name}(${block.arguments})`);
			}
			return lines.join("\n");
		}
		case "tool": {
			const parts = [];
			for (const block of message.content) {
				if (block.type === "text") parts.push(block.text);
				else if (block.type === "image" || block.type === "file") parts.push(attachmentNote(block));
			}
			const label = message.isError === true ? "Tool error" : "Tool result";
			return `[${label}]: ${truncateToolOutput(parts.join("\n"))}`;
		}
		/* v8 ignore next 2 -- closed-union exhaustiveness guard */
		default:
			return assertNever(message.role, "message role");
	}
}

/**
 * One message as a flattened transcript line, used only by the last-resort
 * reduction. Reasoning and instruction updates are dropped, and long tool
 * outputs are truncated, exactly as OpenCode's `flattenMessage` does.
 *
 * @param message - a request message.
 * @returns the transcript line(s), or `""` to leave the message out.
 */
export function flattenMessage(message) {
	if (message.role === "system" || message.role === "developer") return "";
	const who = message.role === "assistant" ? "Assistant" : "User";
	const lines = [];
	for (const block of message.content) {
		if (block.type === "text") {
			if (block.text) lines.push(`[${who}]: ${block.text}`);
		} else if (block.type === "image" || block.type === "file") {
			lines.push(`[${block.attachment.mediaType ?? block.attachment.name} omitted]`);
		} else if (block.type === "tool-call") {
			lines.push(`[Assistant tool call]: ${block.name}(${block.arguments})`);
		}
	}
	if (message.role === "tool" && message.content.some((block) => block.type === "text")) {
		const label = message.isError === true ? "Tool error" : "Tool result";
		return `[${label}]: ${truncateToolOutput(message.content.map((block) => (block.type === "text" ? block.text : attachmentNote(block))).join("\n"))}`;
	}
	return lines.join("\n");
}

//#endregion

//#region sizing — OpenCode's `estimateMessage` / `oldestToDrop` / `recentStart`

/** OpenCode's per-part token estimate, adapted to DSH's content blocks. */
function estimateBlock(block) {
	switch (block.type) {
		case "text":
		case "reasoning":
			return estimateText(block.text);
		case "image":
			return block.offloaded === true ? 0 : IMAGE_TOKEN_ESTIMATE;
		case "file":
			return block.attachment.name.toLowerCase().endsWith(".pdf") ? PDF_TOKEN_ESTIMATE : 0;
		case "tool-call":
			return estimateText(block.name + block.arguments);
		default:
			return 0;
	}
}

/** OpenCode's `estimateMessage`. */
export const estimateMessage = (message) => message.content.reduce((sum, block) => sum + estimateBlock(block), 0);

/** Estimate a prepared auxiliary-call prefix: tool schemas plus every message. */
export function estimateInput(input) {
	let total = 0;
	for (const tool of input.tools ?? []) total += estimateText(tool.name + tool.description + JSON.stringify(tool.parameters));
	for (const message of input.messages) total += estimateMessage(message);
	return total;
}

/**
 * OpenCode's `oldestToDrop`: the index of the earliest item the remaining
 * suffix of `budget` can still hold.
 */
export function oldestToDrop(items, size, budget) {
	let total = 0;
	let start = items.length;
	while (start > 0) {
		const next = total + size(items[start - 1]);
		if (next > budget) break;
		total = next;
		start -= 1;
	}
	return start;
}

//#endregion

//#region configuration

const POLICY_KEYS = new Set([
	"auto",
	"buffer",
	"keep",
	"headroomTokens",
	"maxTokens",
	"recentMode",
	"summarizationProvider",
	"summarizationModel",
	"compactionRetries",
	"maxOverflowRetries"
]);

const RECENT_MODES = new Set(["surface", "text"]);

function assertNonNegativeInteger(name, value) {
	if (typeof value !== "number" || !Number.isInteger(value) || value < 0) throw new Error(`${name} (${String(value)}) must be a non-negative integer`);
}

function assertPositiveInteger(name, value) {
	if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) throw new Error(`${name} (${String(value)}) must be a positive integer`);
}

/**
 * Validate and freeze the plugin configuration.
 *
 * `keep` is OpenCode's `Settings.keep` (tokens of recent conversation kept
 * verbatim); `buffer` is OpenCode's `Settings.buffer` (tokens kept free below
 * the window); `auto` is OpenCode's `Settings.auto`.
 *
 * @param config - untrusted plugin configuration after loader normalization.
 * @returns detached immutable defaults.
 */
export function resolveConfig(config = {}) {
	for (const key of Object.keys(config)) if (!POLICY_KEYS.has(key)) throw new Error(`OpencodeCompactionConfig: unknown key "${key}"`);
	const auto = config.auto ?? true;
	if (typeof auto !== "boolean") throw new Error("OpencodeCompactionConfig: auto must be a boolean");
	const buffer = config.buffer;
	if (buffer !== undefined) assertNonNegativeInteger("OpencodeCompactionConfig.buffer", buffer);
	const keep = config.keep ?? 15_000;
	assertNonNegativeInteger("OpencodeCompactionConfig.keep", keep);
	const headroomTokens = config.headroomTokens ?? 0;
	assertNonNegativeInteger("OpencodeCompactionConfig.headroomTokens", headroomTokens);
	const maxTokens = config.maxTokens ?? 32_000;
	assertPositiveInteger("OpencodeCompactionConfig.maxTokens", maxTokens);
	const recentMode = config.recentMode ?? "surface";
	if (typeof recentMode !== "string" || !RECENT_MODES.has(recentMode)) throw new Error(`OpencodeCompactionConfig: recentMode must be one of ${[...RECENT_MODES].join(", ")}`);
	const summarizationProvider = config.summarizationProvider ?? "";
	const summarizationModel = config.summarizationModel ?? "";
	if (typeof summarizationProvider !== "string" || typeof summarizationModel !== "string") throw new Error("OpencodeCompactionConfig: summarizationProvider and summarizationModel must be strings");
	if (summarizationProvider.length === 0 !== (summarizationModel.length === 0)) throw new Error("OpencodeCompactionConfig: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair");
	const compactionRetries = config.compactionRetries ?? 1;
	assertNonNegativeInteger("OpencodeCompactionConfig.compactionRetries", compactionRetries);
	const maxOverflowRetries = config.maxOverflowRetries ?? 1;
	assertNonNegativeInteger("OpencodeCompactionConfig.maxOverflowRetries", maxOverflowRetries);
	return deepFreeze({ auto, buffer, keep, headroomTokens, maxTokens, recentMode, summarizationProvider, summarizationModel, compactionRetries, maxOverflowRetries });
}

/**
 * OpenCode's `calculateCeiling`: the largest request the model takes while
 * leaving room for its reply — 10% of the window, or `RESERVE_MIN` when that is
 * more, unless an explicit `buffer` names the reserve.
 *
 * @param window - the routed model's context window in tokens.
 * @param buffer - explicit reserve, or `undefined` for OpenCode's ratio rule.
 * @returns the input ceiling, `Infinity` when the window is unknown.
 */
export function calculateCeiling(window, buffer) {
	if (!Number.isFinite(window) || window <= 0) return Number.POSITIVE_INFINITY;
	if (buffer !== undefined) return window - buffer;
	return window - Math.max(Math.floor(window * 0.1), window >= 2 * RESERVE_MIN ? RESERVE_MIN : 0);
}

/** Target-specific pressure configuration failure, reported once per routed target. */
class TargetPressureConfigError extends Error {
	constructor(targetKey, message) {
		super(message);
		this.targetKey = targetKey;
	}
}

/**
 * Scale the resolved policy into concrete token budgets for one routed model.
 *
 * DSH adaptation: OpenCode's ceiling ignores the reserved completion tokens
 * because its adapters treat the window as an input budget. DSH charges the
 * routed `maxTokens` to the same window, so the pressure budget additionally
 * subtracts the reservation and any configured headroom.
 *
 * @param policy - resolved plugin configuration.
 * @param contextWindow - the routed model's positive context window.
 * @param reservedCompletionTokens - output tokens one routed request reserves.
 * @returns frozen pressure and retention budgets.
 */
export function resolveSpec(policy, contextWindow, reservedCompletionTokens) {
	const targetKey = policy.targetKey;
	if (!Number.isInteger(contextWindow) || contextWindow <= 0) throw new TargetPressureConfigError(targetKey, `opencode-compaction: no context capacity for ${targetKey}; configure contextWindow on that adapter model`);
	if (!Number.isInteger(reservedCompletionTokens) || reservedCompletionTokens < 0) throw new TargetPressureConfigError(targetKey, `opencode-compaction: ${targetKey} reserves an invalid maxTokens (${reservedCompletionTokens})`);
	const messageBudget = contextWindow - reservedCompletionTokens;
	if (messageBudget <= 0) throw new TargetPressureConfigError(targetKey, `opencode-compaction: ${targetKey} reserves ${reservedCompletionTokens} completion tokens of its ${contextWindow}-token window, leaving no message budget; lower the model's maxTokens or raise its contextWindow`);
	const ceiling = calculateCeiling(contextWindow, policy.buffer);
	const thresholdTokens = Math.floor(Math.min(ceiling, messageBudget - policy.headroomTokens));
	if (thresholdTokens <= 0) throw new TargetPressureConfigError(targetKey, `opencode-compaction: ${targetKey} leaves no pressure budget after reserving ${reservedCompletionTokens} completion tokens and ${policy.headroomTokens} headroom tokens`);
	if (policy.keep >= thresholdTokens) throw new TargetPressureConfigError(targetKey, `OpencodeCompactionConfig: keep (${policy.keep}) must be less than the threshold tokens ${thresholdTokens} for ${targetKey}`);
	return deepFreeze({
		targetKey,
		contextWindow,
		thresholdTokens,
		retainTokens: policy.keep,
		maxTokens: policy.maxTokens
	});
}

//#endregion

//#region surface transaction — identical in shape to `@deepseek-ai/dsh-compaction-basic`

/**
 * Rejects a summary whose replacement boundaries are no longer the ones it was
 * built from, distinguished from summarizer and shrink failures so a manual
 * caller can report the two causes differently.
 */
class SurfaceChangedError extends Error {}

/** The `system/message` holding surface node 0, or `undefined` without one. */
function systemHead(session, headSeq) {
	if (headSeq === undefined) return undefined;
	const head = session.eventAt(headSeq);
	return head.type === "system/message" ? head : undefined;
}

/** Validate one requested surface-position span before asynchronous work begins. */
function validateSurfaceRegion(session, start, end) {
	const nodes = session.surface.nodes;
	const startIdx = nodes.indexOf(start);
	const endIdx = nodes.indexOf(end);
	if (startIdx === -1) throw new Error(`compactRegion: start seq ${start} not found in surface`);
	if (endIdx === -1) throw new Error(`compactRegion: end seq ${end} not found in surface`);
	if (startIdx > endIdx) throw new Error(`compactRegion: start seq ${start} (position ${startIdx}) is after end seq ${end} (position ${endIdx}) on the surface`);
	if (!toolPairingBalancedBefore(session, nodes[startIdx])) throw new Error(`compactRegion: start seq ${start} is not a balanced boundary (would split a step's tool-call/result pair)`);
	if (!toolPairingBalancedAfter(session, nodes[endIdx])) throw new Error(`compactRegion: end seq ${end} is not a balanced boundary (would split a step, or the step is still open)`);
	return { start, end, startIdx, endIdx, shadowedSeqs: nodes.slice(startIdx, endIdx + 1) };
}

/** Reject a durable unmatched compaction marker unless a later seed proves an earlier lifecycle. */
function assertCompactionInactive(unmatchedCompactionStart, latestEndSeedSeq, stage) {
	if (unmatchedCompactionStart === undefined || (latestEndSeedSeq !== undefined && latestEndSeedSeq > unmatchedCompactionStart.seq)) return;
	throw new ManualCompactionError("busy", `${stage}: compaction already in progress; the session compaction lock is already active`);
}

/** Recheck the durable compaction lock after an asynchronous policy decision. */
function assertNoActiveCompaction(session, stage) {
	const entryState = inspectCompactionEntryState(session);
	assertCompactionInactive(entryState.unmatchedCompactionStart, entryState.latestEndSeedSeq, stage);
}

/** Inspect open-turn, unmatched-compaction, and latest seed-boundary state independently. */
function inspectCompactionEntryState(session) {
	let openTurn = null;
	let openTurnStateKnown = false;
	let unmatchedCompactionStart;
	let compactionEntryStateKnown = false;
	let latestEndSeedSeq;
	for (let seq = session.seq - 1; seq >= 0; seq -= 1) {
		const event = session.eventAt(SessionSeq(seq));
		if (latestEndSeedSeq === undefined && event.type === "session/end-seed") latestEndSeedSeq = event.seq;
		if (!compactionEntryStateKnown) {
			if (event.type === "compaction/start") {
				unmatchedCompactionStart = event;
				compactionEntryStateKnown = true;
			} else if (event.type === "compaction/end") compactionEntryStateKnown = true;
		}
		if (!openTurnStateKnown) {
			if (event.type === "turn/start") {
				openTurn = event.data.turn;
				openTurnStateKnown = true;
			} else if (event.type === "turn/end") openTurnStateKnown = true;
		}
		if (openTurnStateKnown && compactionEntryStateKnown && latestEndSeedSeq !== undefined) break;
	}
	return { openTurn, unmatchedCompactionStart, latestEndSeedSeq };
}

/**
 * Reconstruct the cacheable prefix the region stands for: the system head, the
 * header's tool schemas, then the region's derived messages in surface order.
 * The summarizer appends only its instruction after this, so the auxiliary call
 * is a genuine prefix of the conversation and reuses the provider's KV cache.
 *
 * @param session - session supplying the surface head, request header, and projection.
 * @param seqs - the surface-node seqs, in order, to replay.
 * @returns the replayed conversation prefix to condense.
 */
function buildSummarizationInput(session, seqs) {
	const header = session.requestHeader();
	const head = systemHead(session, session.surface.nodes[0]);
	const system = head === undefined ? null : session.deriveEventMessage(head);
	const regionMessages = seqs.map((seq) => session.deriveEventMessage(session.eventAt(seq))).filter((message) => message !== null);
	return {
		...(header?.tools === undefined ? {} : { tools: header.tools }),
		messages: system === null ? regionMessages : [system, ...regionMessages]
	};
}

/**
 * OpenCode's `flattenAndDropOldest`: replace the region's messages with one text
 * transcript, then drop whole exchanges oldest-first until it fits. The leading
 * system message and an existing checkpoint are kept as the lead and never
 * dropped. Returns `undefined` when even the newest exchange does not fit.
 *
 * @param input - the prepared auxiliary-call prefix.
 * @param target - the token budget the request must fit.
 * @returns a replacement prefix, or `undefined` when nothing can be kept.
 */
export function flattenAndDropOldest(input, target) {
	let lead = [];
	let rest = input.messages;
	if (rest[0]?.role === "system") {
		lead = [rest[0]];
		rest = rest.slice(1);
	}
	if (rest[0]?.role === "user" && rest[0].source !== undefined && isCompactCheckpointSource(rest[0].source)) {
		lead = [...lead, rest[0]];
		rest = rest.slice(1);
	}

	// A user message and everything after it, up to the next one, is one exchange: kept or dropped whole.
	const exchanges = [];
	for (const message of rest) {
		const text = flattenMessage(message);
		if (text.length === 0) continue;
		if (message.role === "user" || exchanges.length === 0) exchanges.push(text);
		else exchanges[exchanges.length - 1] += `\n${text}`;
	}
	if (exchanges.length === 0) return undefined;

	const note = (omitted) => (omitted === 0 ? "" : `[${omitted} older ${omitted === 1 ? "exchange" : "exchanges"} omitted]\n\n`);
	let tools = 0;
	for (const tool of input.tools ?? []) tools += estimateText(tool.name + tool.description + JSON.stringify(tool.parameters));
	// Room for the longest possible note is set aside before choosing what to keep.
	const room = target - tools - lead.reduce((sum, message) => sum + estimateMessage(message), 0) - estimateText(note(exchanges.length));
	const kept = exchanges.slice(oldestToDrop(exchanges, (text) => estimateText(text) + 1, room));
	if (kept.length === 0) return undefined;
	return {
		...(input.tools === undefined ? {} : { tools: input.tools }),
		messages: [...lead, deepFreeze({ role: "user", content: [{ type: "text", text: note(exchanges.length - kept.length) + kept.join("\n\n") }] })]
	};
}

/** Reject a summary prepared against any earlier surface generation. */
function assertWholeSurfaceUnchanged(meter, session, prepared) {
	if (!isDeepStrictEqual(meter.measure(session).nodes, prepared.measurement.nodes)) throw new SurfaceChangedError("opencode-compaction: session surface changed during summarization");
}

/** Require only that the selected span remain the same present, contiguous, equally priced, balanced target. */
function assertSelectedSpanStable(meter, session, prepared) {
	let current;
	try {
		current = validateSurfaceRegion(session, prepared.start, prepared.end);
	} catch (error) {
		throw new SurfaceChangedError("opencode-compaction: the selected span is no longer a valid replacement target", { cause: error });
	}
	if (!isDeepStrictEqual([...current.shadowedSeqs], [...prepared.shadowedSeqs])) throw new SurfaceChangedError("opencode-compaction: the selected span changed during summarization");
	if (!isDeepStrictEqual(meter.measure(session).nodes.slice(current.startIdx, current.endIdx + 1), prepared.selectedNodes)) throw new SurfaceChangedError("opencode-compaction: the selected span was rewritten during summarization");
}

/** Classify one closed manual attempt without weakening cancellation precedence. */
function throwManualFailure(failure) {
	if (failure.stage === "commit") throw new ManualCompactionError("commit", "manual compaction did not commit cleanly", { cause: failure.error });
	if (failure.error instanceof SurfaceChangedError) throw new ManualCompactionError("changed", "the compacted history changed during manual compaction", { cause: failure.error });
	throw new ManualCompactionError("summary", "manual compaction could not produce a smaller summary", { cause: failure.error });
}

/** Append one completed summary record and replacement body without yielding. */
function commitCompactionBody(session, startEvent, summarized) {
	const { start, end, shadowedSeqs, shadowedTokenCount, summary, recentText, provider, model, maxTokens, usage, checkpointMessage } = summarized;
	const callRecord = summarized.llmStreamCall === true ? { rawOutput: summarized.rawOutput, llmStreamCall: true } : summarized.rawOutput === undefined ? {} : { rawOutput: summarized.rawOutput };
	const summaryEvent = session.append("compaction/summary", {
		compactionId: startEvent.data.compactionId,
		...(startEvent.data.sourceCommandId === undefined ? {} : { sourceCommandId: startEvent.data.sourceCommandId }),
		summary,
		...callRecord,
		// DSH-visible extension: the verbatim tail embedded beside the summary.
		recentText,
		shadowedRange: { start, end },
		shadowedSeqs: [...shadowedSeqs],
		shadowedTokenCount,
		provider,
		model,
		...(maxTokens === undefined ? {} : { maxTokens }),
		...(usage === undefined ? {} : { usage })
	});
	session.append("user/message", checkpointMessage, {
		surfaceOp: { op: "replace", startSeq: start, endSeq: end },
		sourceEventSeqs: [startEvent.seq, summaryEvent.seq, ...shadowedSeqs]
	});
	return {
		compactionId: startEvent.data.compactionId,
		...(startEvent.data.sourceCommandId === undefined ? {} : { sourceCommandId: startEvent.data.sourceCommandId }),
		startSeq: startEvent.seq,
		summarySeq: summaryEvent.seq,
		summary,
		shadowedRange: { start, end },
		shadowedSeqs: [...shadowedSeqs],
		shadowedTokenCount
	};
}

//#endregion

//#region the ported engine

/** Resolve the exact provider/model durably routed for the latest request. */
function routedTarget(session) {
	const config = session.requestHeader()?.config;
	if (config === undefined || config.provider.length === 0 || config.model.length === 0) return undefined;
	return { provider: config.provider, model: config.model };
}

/** Resolve the conversation target used to select an optional policy override. */
function conversationTarget(agent) {
	const routed = routedTarget(agent.session);
	if (routed !== undefined) return routed;
	if (!agent.options.provider || !agent.options.model) return undefined;
	return { provider: agent.options.provider, model: agent.options.model };
}

/** Output tokens the routed request reserves, which the provider charges to the same window as the prompt. */
function reservedCompletionTokens(agent, defaultMaxTokens) {
	return agent.session.requestHeader()?.config.maxTokens ?? defaultMaxTokens ?? 0;
}

/** A terminal summarization finish mapped to its fail-closed error. */
function finishError(finish) {
	switch (finish.kind) {
		case "error":
		case "aborted":
			return new LlmError(finish.failure.message, finish.failure.code, finish.failure);
		case "max-tokens": {
			const error = new Error("compaction summary truncated at the output token cap (incomplete checkpoint)");
			error.code = "MAX_TOKENS";
			return error;
		}
		default:
			return undefined;
	}
}

/** Whether a failure is the provider rejecting the request as too long for its window. */
export function isContextOverflow(error) {
	for (let current = error; current !== undefined && current !== null; current = current.cause) {
		if (typeof current.code === "string" && current.code === CONTEXT_WINDOW_EXCEEDED_CODE) return true;
		if (current.name === "LlmError" && current.code === CONTEXT_WINDOW_EXCEEDED_CODE) return true;
	}
	return false;
}

/** The message text of an existing checkpoint at the head of the surface, if any. */
function previousCheckpoint(session, firstIdx) {
	const seq = session.surface.nodes[firstIdx];
	if (seq === undefined) return undefined;
	const event = session.eventAt(seq);
	if (event === undefined || event.type !== "user/message") return undefined;
	const message = session.deriveEventMessage(event);
	if (message === null || !isCompactCheckpointSource(message.source)) return undefined;
	return { seq, message, summaryText: checkpointSummaryText(message) };
}

/**
 * The core transaction. Identical in structure to `compactSurfaceRegion`, with
 * OpenCode's `deliver` loop, template validation, and one-nudge retry inside the
 * summarization stage.
 */
async function compactSurfaceRegion(dependencies, session, selection, agent, options, signal) {
	if (options.owner === null) signal?.throwIfAborted();
	const validated = validateSurfaceRegion(session, selection.start, selection.end);
	const entryState = inspectCompactionEntryState(session);
	assertCompactionInactive(entryState.unmatchedCompactionStart, entryState.latestEndSeedSeq, "opencode-compaction");
	let owner;
	if (options.owner === null) {
		if (entryState.openTurn !== null) throw new ManualCompactionError("busy", "manual compaction: the session already has an open turn");
		owner = null;
	} else {
		if (entryState.openTurn === null) throw new Error("compactRegion: no open turn — automatic compaction events must be enclosed in a turn");
		owner = entryState.openTurn;
	}

	const compactionId = CompactionId(randomUUID());
	const lifecycle = { compactionId, ...(options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId }), turn: owner };
	const startEvent = session.append("compaction/start", lifecycle);

	let failure;
	let flushFailure;
	let result;
	let closed = false;
	let closing = false;
	let stage = "summary";
	try {
		const summarized = await summarizeCompaction(dependencies, session, validated, selection, agent, compactionId, options.sourceCommandId, signal);
		if (options.owner === null) signal?.throwIfAborted();
		if (options.stability === "whole-surface") assertWholeSurfaceUnchanged(dependencies.meter, session, summarized);
		else assertSelectedSpanStable(dependencies.meter, session, summarized);
		stage = "commit";
		const pending = commitCompactionBody(session, startEvent, summarized);
		closing = true;
		const endEvent = session.append("compaction/end", lifecycle);
		closed = true;
		result = { ...pending, endSeq: endEvent.seq };
	} catch (error) {
		failure = { error, stage: closing ? "commit" : stage };
		if (!closing) {
			closing = true;
			try {
				session.append("compaction/end", { ...lifecycle, error: errorChain(error) });
				closed = true;
			} catch (closeError) {
				failure = { error: closeError, stage: "commit" };
			}
		}
	}
	if (closed && options.flush !== undefined) {
		try {
			await options.flush();
		} catch (error) {
			flushFailure = error;
		}
	}
	if (options.owner === null) signal?.throwIfAborted();
	if (failure !== undefined) {
		if (options.owner === null) throwManualFailure(failure);
		throw failure.error;
	}
	if (flushFailure !== undefined) throw new ManualCompactionError("persistence", "manual compaction durability checkpoint failed", { cause: flushFailure });
	/* v8 ignore next -- every path without a result records and throws a failure above. */
	if (result === undefined) throw new Error("compaction committed without a result");
	return result;
}

/**
 * Summarize the selected span through the ported OpenCode policy.
 *
 * Stage 1 (`send`): replay the older part, append the summary instruction, and
 * retry once with a nudge if the reply skipped the template.
 *
 * Stage 2 (`deliver`): send as-is when the estimate fits; otherwise replace the
 * region with a text transcript and drop whole exchanges oldest-first. Each
 * provider context-window rejection re-aims the target at 70%, 50%, then 35% of
 * the first rejected request, after which the attempt fails.
 */
async function summarizeCompaction(dependencies, session, validated, selection, agent, compactionId, sourceCommandId, signal) {
	const firstIdx = systemHead(session, session.surface.nodes[0]) === undefined ? 0 : 1;
	const previous = previousCheckpoint(session, firstIdx);
	const prompt = buildPrompt(previous !== undefined, previous !== undefined && previous.summaryText.includes(LEGACY_HEADING));
	const olderSeqs = selection.olderSeqs;
	const input = buildSummarizationInput(session, olderSeqs);
	let budget = Math.max(1, selection.thresholdTokens - estimateText(prompt) - estimateText(NUDGE));

	const send = async (request, lastUserText) => {
		const assembler = new BlockAssembler();
		const options = {
			provider: dependencies.target.provider,
			model: dependencies.target.model,
			messages: [...request.messages, deepFreeze({ role: "user", content: [{ type: "text", text: lastUserText }] })],
			toolHistory: agent.session.toolHistory(),
			...(request.tools === undefined ? {} : { tools: [...request.tools] }),
			maxTokens: dependencies.maxTokens,
			sessionId: agent.session.id,
			purpose: "compaction",
			...(signal === undefined ? {} : { signal })
		};
		for await (const chunk of dependencies.llm.stream(options)) assembler.push(chunk);
		const error = finishError(assembler.finish);
		if (error !== undefined) throw error;
		const rawOutput = assembler.blocks();
		if (contentHasImage(rawOutput)) throw new LlmError("compaction summary cannot contain image output", "UNSUPPORTED_CONTENT");
		const text = rawOutput.filter((block) => block.type === "text").map((block) => block.text).join("");
		return { text, rawOutput, usage: assembler.usage };
	};

	let rejections = 0;
	let rejected;
	let recovered = false;
	let useTranscript = false;
	let request = input;
	for (;;) {
		signal?.throwIfAborted();
		if (!useTranscript && estimateInput(request) > budget) {
			const flattened = flattenAndDropOldest(request, budget);
			if (flattened === undefined) throw new LlmError("The compaction request cannot be reduced further without losing the latest exchange or checkpoint", "COMPACTION_UNSHRINKABLE");
			request = flattened;
			useTranscript = true;
		}
		let reply;
		try {
			reply = await send(request, prompt);
			if (!filled(reply.text)) reply = await send(request, NUDGE);
		} catch (error) {
			if (signal?.aborted === true) throw error;
			if (isContextOverflow(error)) {
				const step = SHRINK_STEPS[rejections];
				if (step !== undefined) {
					rejected ??= estimateInput(request);
					budget = Math.floor(rejected * step);
					rejections += 1;
					useTranscript = false;
					request = input;
					continue;
				}
			}
			// One `compaction/summary-error` recovery attempt: a listener only
			// returns true after recording durable progress, so retry the same
			// selection once from a fresh attempt rather than recursing.
			if (recovered || !await dependencies.recover(error, agent, validated.shadowedSeqs, signal)) throw error;
			recovered = true;
			rejections = 0;
			rejected = undefined;
			useTranscript = false;
			request = input;
		}
		if (!filled(reply.text)) {
			const message = reply.text.trim().length > 0 ? "Compaction summary did not match the required template" : "Compaction produced no summary";
			throw new LlmError(message, "COMPACTION_TEMPLATE");
		}
		const summary = [{ type: "text", text: reply.text }];
		const checkpointMessage = createUserMessage({ content: frameCheckpoint(reply.text, selection.recentText), source: compactCheckpointSource(compactionId, sourceCommandId) });
		const framedTokenCount = dependencies.meter.estimateMessage(checkpointMessage);
		if (framedTokenCount >= selection.shadowedRouteTokenCount) {
			throw new LlmError(`summary is not smaller than the shadowed content (${framedTokenCount} estimated framed tokens >= ${selection.shadowedRouteTokenCount})`, "COMPACTION_NOT_SMALLER");
		}
		return {
			start: validated.start,
			end: validated.end,
			shadowedSeqs: validated.shadowedSeqs,
			shadowedTokenCount: selection.shadowedTokenCount,
			selectedNodes: selection.selectedNodes,
			measurement: selection.measurement,
			summary,
			recentText: selection.recentText,
			llmStreamCall: true,
			rawOutput: reply.rawOutput,
			provider: dependencies.target.provider,
			model: dependencies.target.model,
			maxTokens: dependencies.maxTokens,
			...(reply.usage === undefined ? {} : { usage: reply.usage }),
			checkpointMessage
		};
	}
}

/**
 * Compaction backend implementing OpenCode's context-management policy on
 * DSH's durable session surface.
 *
 * The surface transaction (durable `compaction/start` lock, one `compaction/end`
 * attempt, cited source events, `toolPairingBalancedBefore`/`After` boundaries,
 * `compactCheckpointSource`) matches `@deepseek-ai/dsh-compaction-basic` exactly,
 * because that part is the host's contract rather than a policy choice. What is
 * ported from OpenCode is the trigger arithmetic, the retention split, the
 * summary template and its validation, the shrink ladder, the flatten fallback,
 * and the checkpoint wire format.
 */
export class OpencodeCompactionEngine extends CompactionEngine {
	static inject = ["llm", "tokenMeter", "sessions"];

	static Config = z.object({
		auto: z.boolean(),
		buffer: z.number().step(1).min(0),
		keep: z.number().step(1).min(0),
		headroomTokens: z.number().step(1).min(0),
		maxTokens: z.number().step(1).min(1),
		recentMode: z.string(),
		summarizationProvider: z.string(),
		summarizationModel: z.string(),
		compactionRetries: z.number().step(1).min(0),
		maxOverflowRetries: z.number().step(1).min(0)
	});

	/** Resolved and validated compaction configuration. */
	config;
	warnedPressureConfigTargets = new Set();
	overflowRetries = new WeakMap();
	overflowAgents = new WeakMap();

	constructor(ctx, config = {}) {
		super(ctx);
		this.config = resolveConfig(config);
		if (this.config.auto) this._registerAutomaticCompaction();
	}

	/** Register automatic between-step pressure and model-request overflow recovery. */
	_registerAutomaticCompaction() {
		const { ctx } = this;
		const logResult = (result, trigger) => {
			ctx.logger.info(`opencode-compaction (${trigger}): shadowed ${result.shadowedSeqs.length} surface nodes (seqs ${result.shadowedRange.start}-${result.shadowedRange.end}, ~${result.shadowedTokenCount} tokens)`);
		};
		ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
			if (!signal.aborted) {
				try {
					const result = await this.compactIfNeeded(agent, "pressure", signal);
					if (result !== null) logResult(result, "step pressure");
				} catch (error) {
					if (error instanceof TargetPressureConfigError) {
						if (this.warnedPressureConfigTargets.has(error.targetKey)) return next();
						this.warnedPressureConfigTargets.add(error.targetKey);
					}
					const message = error instanceof Error ? error.message : String(error);
					ctx.logger.warn(`step compaction failed: ${message}; continuing the turn`);
				}
			}
			return next();
		});
		ctx.on("agent/status", ({ agent, status }) => {
			if (status === "idle") this.overflowRetries.delete(agent);
		});
		ctx.on("session/event", (session, event) => {
			if (event.type !== "assistant/message") return;
			const agent = this.overflowAgents.get(session);
			if (agent !== undefined) this.overflowRetries.delete(agent);
		});
		ctx.on("agent/request-error", async ({ agent, failure, signal }, next) => {
			if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next();
			this.overflowAgents.set(agent.session, agent);
			const target = routedTarget(agent.session);
			if (target === undefined) return next();
			const retries = this.overflowRetries.get(agent) ?? 0;
			if (retries >= this.config.maxOverflowRetries) return next();
			const generation = agent.session.surface.replaceGeneration;
			let result;
			try {
				result = await this.compactIfNeeded(agent, "context-overflow", signal);
			} catch (recoveryError) {
				const message = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
				if (!signal.aborted && agent.session.surface.replaceGeneration > generation) {
					ctx.logger.warn(`context-overflow compaction failed after durable surface progress: ${message}; retrying from the replacement surface`);
					this.overflowRetries.set(agent, retries + 1);
					return { kind: "retry" };
				}
				ctx.logger.warn(`context-overflow compaction failed: ${message}; ${signal.aborted ? "cancellation prevents retry" : "preserving the original request error"}`);
				return next();
			}
			if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next();
			if (result !== null) logResult(result, "context overflow recovery");
			this.overflowRetries.set(agent, retries + 1);
			return { kind: "retry" };
		});
	}

	/**
	 * Resolve the auxiliary call target and the pressure budgets for the routed
	 * model. OpenCode picks a dedicated `"compaction"` agent here; DSH has no such
	 * roster, so an explicit configuration pair wins, then the latest durable
	 * routed request, then the agent's own options.
	 */
	async _prepare(agent, signal) {
		const latest = agent.session.requestHeader()?.config;
		const configured = this.config.summarizationProvider.length === 0 ? undefined : { provider: this.config.summarizationProvider, model: this.config.summarizationModel };
		const agentTarget = conversationTarget(agent);
		const target = configured ?? (latest === undefined ? undefined : { provider: latest.provider, model: latest.model }) ?? agentTarget;
		if (target === undefined) throw new Error("opencode-compaction: no provider/model available for summarization: set summarizationProvider and summarizationModel, route one request, or set both AgentOptions fields");
		const info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal);
		const targetKey = `${target.provider}/${target.model}`;
		const spec = resolveSpec({ ...this.config, targetKey }, info.context?.contextWindow, reservedCompletionTokens(agent, info.defaultMaxTokens));
		return { target, spec, targetKey };
	}

	/**
	 * Select the compactable span with OpenCode's retention rule: accumulate the
	 * newest surface nodes until they cover `keep` tokens, snap the retained
	 * tail's start back to a user message so an assistant's tool calls and results
	 * stay together, then enforce DSH's tool-pairing balance.
	 *
	 * @param session - session supplying the authoritative current surface.
	 * @param measurement - unified pressure and per-node pricing.
	 * @param retainTokens - the verbatim tail budget.
	 * @param thresholdTokens - the pressure threshold the summary must come in under.
	 * @returns the selected span and its split, or `null` when nothing can be compacted.
	 */
	_select(session, measurement, retainTokens, thresholdTokens) {
		const pricedNodes = measurement.nodes;
		if (pricedNodes.length === 0) return null;
		const surfaceNodes = session.surface.nodes;
		if (surfaceNodes.length !== pricedNodes.length || surfaceNodes.some((seq, index) => seq !== pricedNodes[index]?.seq)) throw new Error("opencode-compaction: token-meter surface does not match the current session surface");

		const firstIdx = systemHead(session, surfaceNodes[0]) === undefined ? 0 : 1;
		let accumulated = 0;
		let keepFromIdx = pricedNodes.length;
		for (let index = pricedNodes.length - 1; index >= 0; index -= 1) {
			accumulated += pricedNodes[index].tokens;
			keepFromIdx = index;
			if (accumulated >= retainTokens) break;
		}
		if (keepFromIdx <= firstIdx) keepFromIdx = pricedNodes.length;

		// OpenCode's `recentStart`: start the retained tail at a user message.
		for (let index = Math.min(keepFromIdx, pricedNodes.length - 1); index > firstIdx; index -= 1) {
			if (session.eventAt(surfaceNodes[index])?.type === "user/message") {
				keepFromIdx = index;
				break;
			}
		}
		while (keepFromIdx > firstIdx && !toolPairingBalancedBefore(session, surfaceNodes[keepFromIdx])) keepFromIdx -= 1;

		// Everything fits. OpenCode keeps only the latest exchange so an older part
		// is left to summarize; when there is none, nothing useful can be compacted.
		if (keepFromIdx <= firstIdx) {
			const latestUser = pricedNodes.findLastIndex((_node, index) => index > firstIdx && session.eventAt(surfaceNodes[index])?.type === "user/message");
			if (latestUser <= firstIdx) return null;
			keepFromIdx = latestUser;
		}
		if (keepFromIdx <= firstIdx) return null;

		const olderSeqs = surfaceNodes.slice(firstIdx, keepFromIdx);
		if (olderSeqs.length === 0) return null;

		const text = this.config.recentMode === "text";
		const end = text ? surfaceNodes[surfaceNodes.length - 1] : surfaceNodes[keepFromIdx - 1];
		const recentText = text
			? surfaceNodes
					.slice(keepFromIdx)
					.map((seq) => session.deriveEventMessage(session.eventAt(seq)))
					.filter((message) => message !== null)
					.map((message) => messageToText(message))
					.filter((rendered) => rendered.length > 0)
					.join("\n\n")
			: "";

		const selectedNodes = pricedNodes.slice(firstIdx, surfaceNodes.indexOf(end) + 1);
		return {
			start: surfaceNodes[firstIdx],
			end,
			olderSeqs,
			recentText,
			selectedNodes,
			thresholdTokens,
			measurement,
			shadowedTokenCount: selectedNodes.reduce((total, node) => total + node.heuristicTokens, 0),
			shadowedRouteTokenCount: selectedNodes.reduce((total, node) => total + node.tokens, 0)
		};
	}

	/** Bind the effective token meter, the LLM, the routed target, and the recovery waterfall. */
	_regionDependencies(prepared) {
		return {
			meter: this.ctx.tokenMeter,
			llm: this.ctx.llm,
			target: prepared.target,
			maxTokens: prepared.spec.maxTokens,
			recover: (error, agent, sourceEventSeqs, signal) => this.ctx.waterfall("compaction/summary-error", { session: agent.session, sourceEventSeqs, error, ...(signal === undefined ? {} : { signal }) }, () => false)
		};
	}

	/**
	 * Compact for replayed step-boundary pressure or one provider-confirmed
	 * context overflow. Pressure uses OpenCode's ceiling; overflow skips the
	 * threshold and retention entirely so it can force one useful balanced
	 * reduction.
	 */
	async compactIfNeeded(agent, trigger, signal) {
		if (trigger !== "pressure" && trigger !== "context-overflow") assertNever(trigger, "compaction trigger");
		const prepared = await this._prepare(agent, signal);
		const meter = this.ctx.tokenMeter;
		let measurement = meter.measure(agent.session);
		const prune = this.ctx.get("toolResultPruner");

		if (trigger === "context-overflow") {
			if (prune !== undefined) {
				prune.pruneSession(agent.session);
				measurement = meter.measure(agent.session);
			}
			assertNoActiveCompaction(agent.session, "automatic overflow compaction");
			const selection = this._select(agent.session, measurement, 0, prepared.spec.thresholdTokens);
			if (selection === null) return null;
			return this._commit(agent, prepared, selection, "current-turn", signal, undefined, undefined);
		}

		assertNoActiveCompaction(agent.session, "automatic pressure compaction");
		if (measurement.totalTokens < prepared.spec.thresholdTokens) return null;
		if (prune !== undefined) {
			prune.pruneSession(agent.session);
			measurement = meter.measure(agent.session);
		}
		if (measurement.totalTokens < prepared.spec.thresholdTokens) return null;

		let result = null;
		for (let attempt = 0; attempt <= this.config.compactionRetries; attempt += 1) {
			const selection = this._select(agent.session, measurement, prepared.spec.retainTokens, prepared.spec.thresholdTokens);
			if (selection === null) {
				if (result === null) return null;
				break;
			}
			result = await this._commit(agent, prepared, selection, "current-turn", signal, undefined, undefined);
			measurement = meter.measure(agent.session);
			if (measurement.totalTokens < prepared.spec.thresholdTokens) return result;
		}
		// OpenCode has no post-compaction re-measure; it accepts the result. Keep the
		// signal without failing a compaction that durably landed, because the caller
		// must still see the replacement surface.
		this.ctx.logger.warn(`opencode-compaction: still above the pressure threshold after ${this.config.compactionRetries + 1} attempt(s) (${measurement.totalTokens} tokens >= threshold ${prepared.spec.thresholdTokens}); the retained tail may already exceed the configured budget`);
		return result;
	}

	/** Run the durable transaction for one already-selected span. */
	_commit(agent, prepared, selection, owner, signal, sourceCommandId, flush) {
		return compactSurfaceRegion(this._regionDependencies(prepared), agent.session, selection, agent, { owner, stability: owner === null ? "selected-span" : "whole-surface", ...(sourceCommandId === undefined ? {} : { sourceCommandId }), ...(flush === undefined ? {} : { flush }) }, signal);
	}

	/**
	 * Compact one inclusive positional range supplied by a caller. OpenCode has no
	 * equivalent entry point — it computes its own split — so an explicitly
	 * supplied range uses the whole range as the older part and leaves no verbatim
	 * tail, matching the surface semantics of the DSH contract.
	 */
	async compactRegion(start, end, agent, signal) {
		const prepared = await this._prepare(agent, signal);
		const validated = validateSurfaceRegion(agent.session, start, end);
		const measurement = this.ctx.tokenMeter.measure(agent.session);
		const selectedNodes = measurement.nodes.slice(validated.startIdx, validated.endIdx + 1);
		const selection = {
			start,
			end,
			olderSeqs: validated.shadowedSeqs,
			recentText: "",
			selectedNodes,
			thresholdTokens: prepared.spec.thresholdTokens,
			measurement,
			shadowedTokenCount: selectedNodes.reduce((total, node) => total + node.heuristicTokens, 0),
			shadowedRouteTokenCount: selectedNodes.reduce((total, node) => total + node.tokens, 0)
		};
		return this._commit(agent, prepared, selection, "current-turn", signal, undefined, undefined);
	}

	/**
	 * Force one useful idle-session compaction below the pressure threshold, and
	 * resolve only after its standalone marker pair is durably checkpointed.
	 */
	compactNow(agent, signal, sourceCommandId) {
		signal.throwIfAborted();
		try {
			return agent.runMaintenance(async (agentSignal) => {
				const operationSignal = AbortSignal.any([agentSignal, signal]);
				try {
					operationSignal.throwIfAborted();
					const prepared = await this._prepare(agent, operationSignal);
					const selection = this._select(agent.session, this.ctx.tokenMeter.measure(agent.session), 0, prepared.spec.thresholdTokens);
					if (selection === null) return null;
					return await this._commit(agent, prepared, selection, null, operationSignal, sourceCommandId, async () => {
						await this.ctx.sessions.flush(agent.session);
					});
				} catch (error) {
					if (agentSignal.aborted && operationSignal.reason === agentSignal.reason) throw new ManualCompactionError("cancelled", "manual compaction was cancelled", { cause: error });
					operationSignal.throwIfAborted();
					throw error;
				}
			});
		} catch (error) {
			if (error instanceof ManualCompactionError) throw error;
			throw new ManualCompactionError("busy", "manual compaction requires an idle agent with no waking queued work", { cause: error });
		}
	}
}

/**
 * The complete port surface. OpenCode's policy is a set of constants and pure
 * functions, so exposing them is what makes the port auditable: a reader can
 * diff each exported name against `packages/core/src/session/compaction.ts`
 * (or `token.ts`) without instantiating the engine.
 *
 * The DSH transaction helpers are exported for the same reason: they encode the
 * contract mechanics (lock inspection, surface validation, tool-pairing edges)
 * that the engine shares with `dsh-compaction-basic`.
 */
export {
	// OpenCode budget policy (compaction.ts).
	SHRINK_STEPS,
	RESERVE_MIN,
	UNKNOWN_WINDOW,
	// OpenCode tool/media pricing (compaction.ts) and token estimation (token.ts).
	TOOL_OUTPUT_MAX_CHARS,
	IMAGE_TOKEN_ESTIMATE,
	PDF_TOKEN_ESTIMATE,
	CHARS_PER_TOKEN,
	estimateText,
	// OpenCode's summary contract (compaction.ts).
	SUMMARY_TEMPLATE,
	SUMMARY_RULES,
	LEGACY_HEADING,
	NUDGE,
	HEADINGS,
	// OpenCode's checkpoint wire format (runner/to-llm-message.ts).
	CHECKPOINT_OPEN,
	CHECKPOINT_CLOSE,
	SUMMARY_OPEN,
	SUMMARY_CLOSE,
	RECENT_OPEN,
	RECENT_CLOSE,
	CHECKPOINT_PREAMBLE,
	// DSH configuration.
	POLICY_KEYS,
	RECENT_MODES,
	TargetPressureConfigError,
	// DSH surface/transaction helpers.
	SurfaceChangedError,
	systemHead,
	validateSurfaceRegion,
	assertCompactionInactive,
	assertNoActiveCompaction,
	inspectCompactionEntryState,
	buildSummarizationInput,
	assertWholeSurfaceUnchanged,
	assertSelectedSpanStable,
	throwManualFailure,
	commitCompactionBody,
	routedTarget,
	conversationTarget,
	reservedCompletionTokens,
	finishError,
	previousCheckpoint,
	compactSurfaceRegion,
	summarizeCompaction,
	attachmentNote,
	estimateBlock
};

export default OpencodeCompactionEngine;

//#endregion
