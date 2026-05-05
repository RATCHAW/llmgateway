import { describe, expect, it } from "vitest";

import { models, type ProviderModelMapping } from "@llmgateway/models";

describe("Anthropic model pricing", () => {
	const anthropicProviderEntries = models.flatMap((model) =>
		model.family === "anthropic"
			? model.providers
					.filter((provider) => provider.providerId === "anthropic")
					.map((provider) => ({
						modelId: model.id,
						provider: provider as ProviderModelMapping,
					}))
			: [],
	);

	it("has at least one anthropic provider mapping to validate", () => {
		expect(anthropicProviderEntries.length).toBeGreaterThan(0);
	});

	it.each(anthropicProviderEntries)(
		"$modelId defines cacheWriteInputPrice1h whenever cacheWriteInputPrice is set",
		({ provider }) => {
			if (provider.cacheWriteInputPrice === undefined) {
				return;
			}
			expect(
				provider.cacheWriteInputPrice1h,
				`${provider.modelName}: cacheWriteInputPrice is set but cacheWriteInputPrice1h is missing — 1h cache writes would silently bill at the 5m rate`,
			).toBeDefined();
		},
	);

	it.each(anthropicProviderEntries)(
		"$modelId defines cacheWriteInputPrice1h on every pricing tier that sets cacheWriteInputPrice",
		({ provider }) => {
			const tiers = provider.pricingTiers ?? [];
			for (const tier of tiers) {
				if (tier.cacheWriteInputPrice === undefined) {
					continue;
				}
				expect(
					tier.cacheWriteInputPrice1h,
					`${provider.modelName} tier "${tier.name}": cacheWriteInputPrice is set but cacheWriteInputPrice1h is missing`,
				).toBeDefined();
			}
		},
	);
});
