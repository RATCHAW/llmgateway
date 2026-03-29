import { createMiddleware } from "hono/factory";
import { HTTPException } from "hono/http-exception";

import {
	buildBaseLogEntry,
	type ChatCompletionLogState,
} from "@/chat/tools/chat-log-context.js";
import { insertLog as _insertLog } from "@/lib/logs.js";

import { logger } from "@llmgateway/logger";

import type { ServerTypes } from "@/vars.js";
import type { LogInsertData } from "@llmgateway/db";

function getSynthesizedClientErrorLog(
	baseLogEntry: ReturnType<typeof buildBaseLogEntry>,
	status: number,
	error: unknown,
): LogInsertData | null {
	if (!baseLogEntry) {
		return null;
	}

	const responseText =
		error instanceof HTTPException
			? error.message
			: error instanceof Error
				? error.message
				: "Client error";

	return {
		...baseLogEntry,
		content: null,
		responseSize: responseText.length,
		finishReason: "client_error",
		unifiedFinishReason: "client_error",
		promptTokens: null,
		completionTokens: null,
		totalTokens: null,
		reasoningTokens: null,
		cachedTokens: null,
		hasError: true,
		streamed:
			typeof baseLogEntry.rawRequest === "object" &&
			baseLogEntry.rawRequest !== null &&
			"stream" in baseLogEntry.rawRequest
				? Boolean(baseLogEntry.rawRequest.stream)
				: false,
		canceled: false,
		errorDetails: {
			statusCode: status,
			statusText:
				error instanceof HTTPException
					? "Client Error"
					: error instanceof Error
						? error.name
						: "Client Error",
			responseText,
		},
		duration: 0,
		timeToFirstToken: null,
		timeToFirstReasoningToken: null,
		inputCost: null,
		outputCost: null,
		cachedInputCost: null,
		requestCost: null,
		webSearchCost: null,
		imageInputTokens: null,
		imageOutputTokens: null,
		imageInputCost: null,
		imageOutputCost: null,
		cost: null,
		estimatedCost: false,
		discount: null,
		pricingTier: null,
		dataStorageCost: "0",
		cached: false,
		toolResults: null,
	};
}

export const chatCompletionLogMiddleware = createMiddleware<ServerTypes>(
	async (c, next) => {
		const state: ChatCompletionLogState = {
			pendingLogs: [],
			clientErrorSynthesized: false,
		};
		c.set("chatCompletionLogState", state);

		try {
			await next();
		} catch (error) {
			state.caughtError = error;
			throw error;
		} finally {
			try {
				await state.streamCompletion;
			} catch (error) {
				logger.error(
					"Error waiting for chat stream completion before flushing logs",
					error instanceof Error ? error : new Error(String(error)),
				);
			}

			const status =
				state.caughtError instanceof HTTPException
					? state.caughtError.status
					: c.res.status;
			const hasQueuedClientError = state.pendingLogs.some(
				(log) =>
					log.finishReason === "client_error" ||
					log.unifiedFinishReason === "client_error",
			);

			if (status >= 400 && status < 500 && !hasQueuedClientError) {
				const synthesizedLog = getSynthesizedClientErrorLog(
					buildBaseLogEntry(c),
					status,
					state.caughtError,
				);
				if (synthesizedLog) {
					state.pendingLogs.push(synthesizedLog);
					state.clientErrorSynthesized = true;
				}
			}

			for (const logData of state.pendingLogs) {
				try {
					await _insertLog({
						...logData,
						internalContentFilter: state.internalContentFilter
							? true
							: logData.internalContentFilter,
					});
				} catch (error) {
					logger.error(
						"Failed to flush queued chat completion log",
						error instanceof Error ? error : new Error(String(error)),
					);
				}
			}
		}
	},
);
