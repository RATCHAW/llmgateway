import "dotenv/config";
import { beforeAll, beforeEach, describe, expect, test } from "vitest";

import {
	beforeAllHook,
	beforeEachHook,
	generateTestRequestId,
	getConcurrentTestOptions,
	getTestOptions,
	logMode,
} from "@/chat-helpers.e2e.js";

import { app } from "./app.js";

// OpenAI prompt caching kicks in at >= 1024 prompt tokens. ~6.5k chars of
// padding repeated 50x easily clears that bar regardless of tokenizer.
function buildLongSystemPrompt(): string {
	return (
		"You are a helpful AI assistant. " +
		"This is detailed background context that should be cached for optimal request performance and consistent latency across many calls. ".repeat(
			50,
		) +
		"Please answer succinctly."
	);
}

async function sendUntilCacheRead(
	send: () => Promise<{ status: number; json: any }>,
	maxAttempts = 4,
): Promise<{ status: number; json: any; attempts: number }> {
	let last: { status: number; json: any } = { status: 0, json: null };
	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		last = await send();
		if (last.status !== 200) {
			return { ...last, attempts: attempt };
		}
		const cached =
			last.json?.usage?.prompt_tokens_details?.cached_tokens ??
			last.json?.usage?.input_tokens_details?.cached_tokens ??
			0;
		if (cached > 0) {
			return { ...last, attempts: attempt };
		}
		if (attempt < maxAttempts) {
			await new Promise((r) => setTimeout(r, 500 * attempt));
		}
	}
	return { ...last, attempts: maxAttempts };
}

const hasOpenAIKey = !!process.env.LLM_OPENAI_API_KEY;

describe("e2e openai prompt cache", getConcurrentTestOptions(), () => {
	beforeAll(beforeAllHook);
	beforeEach(beforeEachHook);

	test("empty", () => {
		expect(true).toBe(true);
	});

	// /v1/chat/completions: priming + warm call should report cached_tokens
	// under prompt_tokens_details. Also confirms prompt_cache_key is accepted
	// (forwarded as a routing hint, not echoed in the chat response).
	(hasOpenAIKey ? test : test.skip)(
		"chat-completions surfaces cached_tokens for openai",
		getTestOptions(),
		async () => {
			const longText = buildLongSystemPrompt();
			const body = {
				model: "openai/gpt-4.1",
				prompt_cache_key: "e2e-openai-cache-cc",
				messages: [
					{ role: "system", content: longText },
					{ role: "user", content: "Just reply OK." },
				],
			};

			const send = async () => {
				const requestId = generateTestRequestId();
				const res = await app.request("/v1/chat/completions", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"x-request-id": requestId,
						Authorization: `Bearer real-token`,
					},
					body: JSON.stringify(body),
				});
				const json = await res.json();
				if (logMode) {
					console.log(
						"openai chat cache",
						requestId,
						"status",
						res.status,
						"usage",
						JSON.stringify(json.usage),
					);
				}
				return { status: res.status, json };
			};

			const first = await send();
			expect(first.status).toBe(200);
			expect(first.json.usage).toBeDefined();
			expect(typeof first.json.usage.prompt_tokens).toBe("number");

			const second = await sendUntilCacheRead(send);
			expect(second.status).toBe(200);
			expect(second.json.usage.prompt_tokens_details).toBeDefined();
			expect(
				second.json.usage.prompt_tokens_details.cached_tokens,
				`expected cached_tokens > 0 after ${second.attempts} attempts`,
			).toBeGreaterThan(0);
		},
	);

	// /v1/responses: same surface but on the Responses API. Verifies the
	// schema accepts prompt_cache_key/prompt_cache_retention and that
	// input_tokens_details.cached_tokens is surfaced.
	(hasOpenAIKey ? test : test.skip)(
		"responses-api surfaces cached_tokens for openai",
		getTestOptions(),
		async () => {
			const longText = buildLongSystemPrompt();
			const body = {
				model: "openai/gpt-4.1",
				prompt_cache_key: "e2e-openai-cache-resp",
				prompt_cache_retention: "in_memory" as const,
				input: [
					{ role: "system" as const, content: longText },
					{ role: "user" as const, content: "Just reply OK." },
				],
			};

			const send = async () => {
				const requestId = generateTestRequestId();
				const res = await app.request("/v1/responses", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"x-request-id": requestId,
						Authorization: `Bearer real-token`,
					},
					body: JSON.stringify(body),
				});
				const json = await res.json();
				if (logMode) {
					console.log(
						"openai responses cache",
						requestId,
						"status",
						res.status,
						"usage",
						JSON.stringify(json.usage),
					);
				}
				return { status: res.status, json };
			};

			const first = await send();
			expect(first.status).toBe(200);

			const second = await sendUntilCacheRead(send);
			expect(second.status).toBe(200);
			const cached =
				second.json.usage?.input_tokens_details?.cached_tokens ??
				second.json.usage?.prompt_tokens_details?.cached_tokens ??
				0;
			expect(
				cached,
				`expected cached_tokens > 0 after ${second.attempts} attempts`,
			).toBeGreaterThan(0);
		},
	);

	// Extended retention is only valid for the docs-listed models. Sending
	// "24h" on gpt-4o (which is in_memory-only) should NOT cause OpenAI to
	// 400 — the gateway strips it. Verifies the request still succeeds.
	(hasOpenAIKey ? test : test.skip)(
		"chat-completions strips prompt_cache_retention=24h on unsupported model",
		getTestOptions(),
		async () => {
			const body = {
				model: "openai/gpt-4o",
				prompt_cache_retention: "24h" as const,
				messages: [{ role: "user", content: "Reply OK." }],
			};

			const requestId = generateTestRequestId();
			const res = await app.request("/v1/chat/completions", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"x-request-id": requestId,
					Authorization: `Bearer real-token`,
				},
				body: JSON.stringify(body),
			});
			const json = await res.json();
			if (logMode) {
				console.log(
					"openai 24h-strip",
					requestId,
					"status",
					res.status,
					"body",
					JSON.stringify(json).slice(0, 200),
				);
			}
			expect(res.status).toBe(200);
		},
	);

	// 24h retention should round-trip on an eligible model (gpt-4.1).
	// Successful 200 confirms upstream acceptance and gateway forwarding.
	(hasOpenAIKey ? test : test.skip)(
		"chat-completions forwards prompt_cache_retention=24h on supported model",
		getTestOptions(),
		async () => {
			const longText = buildLongSystemPrompt();
			const body = {
				model: "openai/gpt-4.1",
				prompt_cache_retention: "24h" as const,
				prompt_cache_key: "e2e-openai-cache-24h",
				messages: [
					{ role: "system", content: longText },
					{ role: "user", content: "Just reply OK." },
				],
			};

			const requestId = generateTestRequestId();
			const res = await app.request("/v1/chat/completions", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"x-request-id": requestId,
					Authorization: `Bearer real-token`,
				},
				body: JSON.stringify(body),
			});
			const json = await res.json();
			if (logMode) {
				console.log(
					"openai 24h-forward",
					requestId,
					"status",
					res.status,
					"usage",
					JSON.stringify(json.usage),
				);
			}
			expect(res.status).toBe(200);
			expect(json.usage).toBeDefined();
		},
	);
});
