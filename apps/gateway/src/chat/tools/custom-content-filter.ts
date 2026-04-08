import { z } from "zod";

import { isCancellationError, isTimeoutError } from "@/lib/timeout-config.js";

import { logger } from "@llmgateway/logger";
import { getGatewayPublicBaseUrl } from "@llmgateway/shared";

import {
	getCustomContentFilterConfig,
	type CustomContentFilterConfig,
} from "./check-content-filter.js";
import { extractErrorCause } from "./extract-error-cause.js";
import { buildOpenAIContentFilterTextInput } from "./openai-content-filter.js";

import type { ModerationApiPayload } from "@llmgateway/db";
import type { BaseMessage, MessageContent } from "@llmgateway/models";

interface GatewayContentFilterContext {
	requestId: string;
	organizationId: string;
	projectId: string;
	apiKeyId: string;
}

interface ErrorWithCode extends Error {
	code?: string;
	cause?: unknown;
}

export interface CustomContentFilterCheckResult {
	flagged: boolean;
	model: string;
	upstreamRequestId: string | null;
	results: ModerationApiPayload["results"];
	responses: ModerationApiPayload[];
}

const CUSTOM_CONTENT_FILTER_TIMEOUT_MS = 60_000;
const CUSTOM_CONTENT_FILTER_MAX_TOKENS = 300;
const CUSTOM_CONTENT_FILTER_CATEGORIES = [
	"sexual",
	"sexual/minors",
	"harassment",
	"harassment/threatening",
	"hate",
	"hate/threatening",
	"illicit",
	"illicit/violent",
	"self-harm",
	"self-harm/intent",
	"self-harm/instructions",
	"violence",
	"violence/graphic",
] as const;
const CUSTOM_CONTENT_FILTER_SYSTEM_PROMPT = `You are the safety moderation layer for an LLM gateway.

Classify the supplied conversation against these categories:
${CUSTOM_CONTENT_FILTER_CATEGORIES.map((category) => `- ${category}`).join("\n")}

Return only valid JSON with this exact shape:
{
  "flagged": boolean,
  "categories": {
    "sexual": boolean,
    "sexual/minors": boolean,
    "harassment": boolean,
    "harassment/threatening": boolean,
    "hate": boolean,
    "hate/threatening": boolean,
    "illicit": boolean,
    "illicit/violent": boolean,
    "self-harm": boolean,
    "self-harm/intent": boolean,
    "self-harm/instructions": boolean,
    "violence": boolean,
    "violence/graphic": boolean
  },
  "category_scores": {
    "<category>": number
  },
  "reason": string
}

Rules:
- Scores must be numbers from 0 to 1.
- Set flagged to true when the content is unsafe enough to block.
- If evidence is insufficient, set flagged to false and keep scores low.
- Do not wrap the JSON in markdown.`;

const customContentFilterVerdictSchema = z
	.object({
		flagged: z.boolean(),
		categories: z.record(z.boolean()).optional(),
		category_scores: z.record(z.number()).optional(),
		reason: z.string().optional(),
	})
	.passthrough();

const gatewayChatCompletionResponseSchema = z
	.object({
		id: z.string().optional(),
		model: z.string().optional(),
		choices: z
			.array(
				z.object({
					message: z
						.object({
							content: z
								.union([
									z.string(),
									z
										.array(
											z
												.object({
													text: z.string().optional(),
												})
												.passthrough(),
										)
										.optional(),
								])
								.nullable()
								.optional(),
						})
						.passthrough()
						.optional(),
				}),
			)
			.min(1),
	})
	.passthrough();

function getCustomContentFilterUrl(): string {
	return new URL(
		"/v1/chat/completions",
		`${getGatewayPublicBaseUrl()}/`,
	).toString();
}

function getImageReference(part: MessageContent): string | null {
	if (part.type === "image_url") {
		return `remote-image: ${part.image_url.url}`;
	}

	if (part.type === "image") {
		return `inline-image: media_type=${part.source.media_type}, bytes=${part.source.data.length}`;
	}

	return null;
}

function buildCustomContentFilterInput(messages: BaseMessage[]): string {
	const textSummary = buildOpenAIContentFilterTextInput(messages);
	const imageReferences: string[] = [];

	for (const message of messages) {
		if (!Array.isArray(message.content)) {
			continue;
		}

		for (const part of message.content) {
			const imageReference = getImageReference(part);
			if (imageReference) {
				imageReferences.push(imageReference);
			}
		}
	}

	const sections = [
		`Conversation:\n${textSummary.length > 0 ? textSummary : "[no text content]"}`,
	];

	if (imageReferences.length > 0) {
		sections.push(`Image references:\n${imageReferences.join("\n")}`);
	}

	return sections.join("\n\n");
}

function extractAssistantText(content: unknown): string {
	if (typeof content === "string") {
		return content;
	}

	if (!Array.isArray(content)) {
		return "";
	}

	return content
		.map((part) => {
			if (
				typeof part === "object" &&
				part !== null &&
				"text" in part &&
				typeof part.text === "string"
			) {
				return part.text;
			}

			return "";
		})
		.join("");
}

function extractJsonObject(text: string): string | null {
	const trimmed = text.trim();
	if (trimmed.length === 0) {
		return null;
	}

	const fencedMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
	const unwrapped = fencedMatch?.[1]?.trim() ?? trimmed;
	const firstBraceIndex = unwrapped.indexOf("{");
	const lastBraceIndex = unwrapped.lastIndexOf("}");

	if (
		firstBraceIndex === -1 ||
		lastBraceIndex === -1 ||
		lastBraceIndex <= firstBraceIndex
	) {
		return null;
	}

	return unwrapped.slice(firstBraceIndex, lastBraceIndex + 1);
}

function getErrorCode(error: unknown): string | undefined {
	if (!(error instanceof Error)) {
		return undefined;
	}

	const directCode =
		typeof (error as ErrorWithCode).code === "string"
			? (error as ErrorWithCode).code
			: undefined;
	if (directCode) {
		return directCode;
	}

	const visited = new Set<Error>([error]);
	let current = (error as ErrorWithCode).cause;
	for (let depth = 0; depth < 5; depth++) {
		if (!(current instanceof Error) || visited.has(current)) {
			return undefined;
		}
		visited.add(current);

		if (typeof (current as ErrorWithCode).code === "string") {
			return (current as ErrorWithCode).code;
		}

		current = (current as ErrorWithCode).cause;
	}

	return undefined;
}

function buildModerationErrorDetails(error: unknown): Record<string, string> {
	if (!(error instanceof Error)) {
		return {
			error: String(error),
			errorName:
				error === null
					? "NullThrownValue"
					: typeof error === "undefined"
						? "UndefinedThrownValue"
						: typeof error,
		};
	}

	const errorCause = extractErrorCause(error);
	const errorCode = getErrorCode(error);

	return {
		error: error.message,
		errorName: error.name,
		...(errorCause ? { errorCause } : {}),
		...(errorCode ? { errorCode } : {}),
	};
}

function getFlaggedCategories(payload: ModerationApiPayload): string[] {
	const result = payload.results?.[0];
	if (!result) {
		return [];
	}

	const categories = new Set<string>();

	for (const [category, isFlagged] of Object.entries(result.categories ?? {})) {
		if (isFlagged) {
			categories.add(category);
		}
	}

	for (const [category, score] of Object.entries(
		result.category_scores ?? {},
	)) {
		if (score > 0.5) {
			categories.add(category);
		}
	}

	return [...categories];
}

function createFailedCustomContentFilterResult(
	model: string,
	upstreamRequestId: string | null = null,
): CustomContentFilterCheckResult {
	return {
		flagged: false,
		model,
		upstreamRequestId,
		results: [],
		responses: [],
	};
}

function logModerationResult(
	context: GatewayContentFilterContext,
	payload: Record<string, unknown>,
) {
	logger.debug("gateway_content_filter", {
		provider: "llmgateway",
		mode: "custom",
		requestId: context.requestId,
		organizationId: context.organizationId,
		projectId: context.projectId,
		apiKeyId: context.apiKeyId,
		...payload,
	});
}

function logModerationError(
	context: GatewayContentFilterContext,
	payload: Record<string, unknown>,
	error?: unknown,
) {
	const logPayload = {
		provider: "llmgateway",
		mode: "custom",
		requestId: context.requestId,
		organizationId: context.organizationId,
		projectId: context.projectId,
		apiKeyId: context.apiKeyId,
		...payload,
		...buildModerationErrorDetails(error),
	};

	if (error instanceof Error) {
		logger.error("gateway_content_filter_error", logPayload, error);
		return;
	}

	logger.error("gateway_content_filter_error", logPayload);
}

function buildCustomModerationPayload(
	responseId: string | undefined,
	responseModel: string | undefined,
	verdict: z.infer<typeof customContentFilterVerdictSchema>,
): ModerationApiPayload {
	const categories = verdict.categories ?? {};
	const categoryScores = verdict.category_scores ?? {};
	const inferredFlagged =
		verdict.flagged ||
		Object.values(categories).some(Boolean) ||
		Object.values(categoryScores).some((score) => score > 0.5);

	return {
		...(responseId ? { id: responseId } : {}),
		...(responseModel ? { model: responseModel } : {}),
		results: [
			{
				flagged: inferredFlagged,
				categories,
				category_scores: categoryScores,
				...(verdict.reason ? { reason: verdict.reason } : {}),
			},
		],
	};
}

async function runCustomContentFilterRequest(
	messages: BaseMessage[],
	context: GatewayContentFilterContext,
	config: CustomContentFilterConfig,
	signal: AbortSignal,
): Promise<CustomContentFilterCheckResult> {
	const startTime = Date.now();
	let upstreamResponse: Response;
	let upstreamText: string;

	try {
		upstreamResponse = await fetch(getCustomContentFilterUrl(), {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${config.apiKey}`,
				"X-Client-Request-Id": context.requestId,
			},
			body: JSON.stringify({
				model: config.model,
				temperature: 0,
				max_tokens: CUSTOM_CONTENT_FILTER_MAX_TOKENS,
				messages: [
					{
						role: "system",
						content: CUSTOM_CONTENT_FILTER_SYSTEM_PROMPT,
					},
					{
						role: "user",
						content: buildCustomContentFilterInput(messages),
					},
				],
			}),
			signal,
		});
		upstreamText = await upstreamResponse.text();
	} catch (error) {
		if (signal.aborted || isCancellationError(error)) {
			throw error;
		}

		logModerationError(
			context,
			{
				durationMs: Date.now() - startTime,
				url: getCustomContentFilterUrl(),
				timeout: isTimeoutError(error),
			},
			error,
		);

		return createFailedCustomContentFilterResult(config.model);
	}

	let responseJson: unknown = null;
	if (upstreamText.length > 0) {
		try {
			responseJson = JSON.parse(upstreamText);
		} catch {
			responseJson = upstreamText;
		}
	}

	const upstreamRequestId = upstreamResponse.headers.get("x-request-id");
	if (!upstreamResponse.ok) {
		logModerationError(context, {
			durationMs: Date.now() - startTime,
			status: upstreamResponse.status,
			statusText: upstreamResponse.statusText,
			upstreamRequestId,
			response: responseJson,
		});

		return createFailedCustomContentFilterResult(
			config.model,
			upstreamRequestId,
		);
	}

	const parsedResponse =
		gatewayChatCompletionResponseSchema.safeParse(responseJson);
	if (!parsedResponse.success) {
		logModerationError(context, {
			durationMs: Date.now() - startTime,
			upstreamRequestId,
			response: responseJson,
		});

		return createFailedCustomContentFilterResult(
			config.model,
			upstreamRequestId,
		);
	}

	const assistantText = extractAssistantText(
		parsedResponse.data.choices[0]?.message?.content,
	);
	const verdictJson = extractJsonObject(assistantText);
	if (!verdictJson) {
		logModerationError(context, {
			durationMs: Date.now() - startTime,
			upstreamRequestId,
			response: responseJson,
			assistantText,
		});

		return createFailedCustomContentFilterResult(
			config.model,
			upstreamRequestId,
		);
	}

	let verdictJsonValue: unknown;
	try {
		verdictJsonValue = JSON.parse(verdictJson);
	} catch (error) {
		logModerationError(
			context,
			{
				durationMs: Date.now() - startTime,
				upstreamRequestId,
				response: responseJson,
				assistantText,
			},
			error,
		);

		return createFailedCustomContentFilterResult(
			config.model,
			upstreamRequestId,
		);
	}

	const parsedVerdict =
		customContentFilterVerdictSchema.safeParse(verdictJsonValue);
	if (!parsedVerdict.success) {
		logModerationError(context, {
			durationMs: Date.now() - startTime,
			upstreamRequestId,
			response: responseJson,
			assistantText,
		});

		return createFailedCustomContentFilterResult(
			config.model,
			upstreamRequestId,
		);
	}

	const moderationPayload = buildCustomModerationPayload(
		parsedResponse.data.id,
		parsedResponse.data.model ?? config.model,
		parsedVerdict.data,
	);

	logModerationResult(context, {
		durationMs: Date.now() - startTime,
		flagged: moderationPayload.results?.[0]?.flagged === true,
		model: moderationPayload.model ?? config.model,
		upstreamRequestId,
		flaggedCategories: getFlaggedCategories(moderationPayload),
		results: moderationPayload.results ?? [],
	});

	return {
		flagged: moderationPayload.results?.[0]?.flagged === true,
		model: moderationPayload.model ?? config.model,
		upstreamRequestId,
		results: moderationPayload.results ?? [],
		responses: [moderationPayload],
	};
}

export async function checkCustomContentFilter(
	messages: BaseMessage[],
	context: GatewayContentFilterContext,
	requestSignal?: AbortSignal,
): Promise<CustomContentFilterCheckResult> {
	const startTime = Date.now();
	let config: CustomContentFilterConfig;

	try {
		config = getCustomContentFilterConfig();
	} catch (error) {
		logModerationError(
			context,
			{
				durationMs: Date.now() - startTime,
				url: getCustomContentFilterUrl(),
				timeout: false,
			},
			error,
		);

		return createFailedCustomContentFilterResult("custom-content-filter");
	}

	const signal = requestSignal
		? AbortSignal.any([
				AbortSignal.timeout(CUSTOM_CONTENT_FILTER_TIMEOUT_MS),
				requestSignal,
			])
		: AbortSignal.timeout(CUSTOM_CONTENT_FILTER_TIMEOUT_MS);

	try {
		return await runCustomContentFilterRequest(
			messages,
			context,
			config,
			signal,
		);
	} catch (error) {
		if (requestSignal?.aborted || isCancellationError(error)) {
			throw error;
		}

		logModerationError(
			context,
			{
				durationMs: Date.now() - startTime,
				url: getCustomContentFilterUrl(),
				timeout: isTimeoutError(error),
			},
			error,
		);

		return createFailedCustomContentFilterResult(config.model);
	}
}
