import { describe, expect, it } from "vitest";

import {
	expandAllProviderRegions,
	expandProviderRegions,
	stripRegionFromModelName,
} from "./region-helpers.js";

import type { ProviderModelMapping } from "./models.js";

describe("stripRegionFromModelName", () => {
	it("strips region suffix when it matches", () => {
		expect(
			stripRegionFromModelName("deepseek-v3.2:singapore", "singapore"),
		).toBe("deepseek-v3.2");
	});

	it("strips aws-bedrock region suffix", () => {
		expect(
			stripRegionFromModelName(
				"anthropic.claude-sonnet-4-5-20250929-v1:0:us-east-1",
				"us-east-1",
			),
		).toBe("anthropic.claude-sonnet-4-5-20250929-v1:0");
	});

	it("returns model name unchanged when no region provided", () => {
		expect(stripRegionFromModelName("deepseek-v3.2:singapore")).toBe(
			"deepseek-v3.2:singapore",
		);
	});

	it("returns model name unchanged when region does not match suffix", () => {
		expect(
			stripRegionFromModelName("deepseek-v3.2:singapore", "us-east-1"),
		).toBe("deepseek-v3.2:singapore");
	});

	it("returns model name unchanged when it has no colon", () => {
		expect(stripRegionFromModelName("gpt-4", "us-east-1")).toBe("gpt-4");
	});
});

describe("expandProviderRegions", () => {
	it("returns mapping as-is when no regions defined", () => {
		const mapping = {
			providerId: "openai",
			modelName: "gpt-4",
			inputPrice: 10 / 1e6,
			outputPrice: 30 / 1e6,
		} as unknown as ProviderModelMapping;

		const result = expandProviderRegions(mapping);
		expect(result).toHaveLength(1);
		expect(result[0].modelName).toBe("gpt-4");
	});

	it("expands regions into separate entries with :region suffix", () => {
		const mapping = {
			providerId: "aws-bedrock",
			modelName: "anthropic.claude-v1:0",
			inputPrice: 3.0 / 1e6,
			outputPrice: 15.0 / 1e6,
			regions: [{ id: "us-east-1" }, { id: "eu-west-1" }],
		} as unknown as ProviderModelMapping;

		const result = expandProviderRegions(mapping);
		// 1 synthetic root + 2 region entries
		expect(result).toHaveLength(3);
		expect(result[0].modelName).toBe("anthropic.claude-v1:0");
		expect(result[0].region).toBeUndefined();
		expect(result[1].modelName).toBe("anthropic.claude-v1:0:us-east-1");
		expect(result[1].region).toBe("us-east-1");
		expect(result[2].modelName).toBe("anthropic.claude-v1:0:eu-west-1");
		expect(result[2].region).toBe("eu-west-1");
	});

	it("applies region-specific pricing overrides", () => {
		const mapping = {
			providerId: "alibaba",
			modelName: "qwen-plus",
			inputPrice: 0.4 / 1e6,
			outputPrice: 1.6 / 1e6,
			regions: [
				{ id: "singapore" },
				{ id: "cn-beijing", inputPrice: 0.115 / 1e6, outputPrice: 0.46 / 1e6 },
			],
		} as unknown as ProviderModelMapping;

		const result = expandProviderRegions(mapping);
		expect(result).toHaveLength(3);

		// Singapore inherits parent pricing
		const singapore = result.find((r) => r.region === "singapore");
		expect(singapore?.inputPrice).toBe(0.4 / 1e6);

		// Beijing overrides pricing
		const beijing = result.find((r) => r.region === "cn-beijing");
		expect(beijing?.inputPrice).toBe(0.115 / 1e6);
		expect(beijing?.outputPrice).toBe(0.46 / 1e6);
	});

	it("handles empty regions array", () => {
		const mapping = {
			providerId: "aws-bedrock",
			modelName: "anthropic.claude-v1:0",
			inputPrice: 3.0 / 1e6,
			outputPrice: 15.0 / 1e6,
			regions: [],
		} as unknown as ProviderModelMapping;

		const result = expandProviderRegions(mapping);
		expect(result).toHaveLength(1);
		expect(result[0].modelName).toBe("anthropic.claude-v1:0");
	});
});

describe("expandAllProviderRegions", () => {
	it("expands all providers with regions in a model", () => {
		const providers = [
			{
				providerId: "openai",
				modelName: "gpt-4",
				inputPrice: 10 / 1e6,
				outputPrice: 30 / 1e6,
			},
			{
				providerId: "aws-bedrock",
				modelName: "anthropic.claude-v1:0",
				inputPrice: 3.0 / 1e6,
				outputPrice: 15.0 / 1e6,
				regions: [{ id: "us-east-1" }, { id: "us-west-2" }],
			},
		] as unknown as ProviderModelMapping[];

		const result = expandAllProviderRegions(providers);
		// 1 (openai, no regions) + 1 (root) + 2 (region entries) = 4
		expect(result).toHaveLength(4);
		expect(result.filter((r) => r.providerId === "openai")).toHaveLength(1);
		expect(result.filter((r) => r.providerId === "aws-bedrock")).toHaveLength(
			3,
		);
	});
});
