import { createHash } from "node:crypto";

import {
	getProviderEnvVar,
	providers,
	type Provider,
} from "@llmgateway/models";

import { parseCommaSeparatedEnv } from "./round-robin-env.js";

const tokenFingerprintCache = new Map<string, string>();

function computeApiKeyFingerprint(token: string): string {
	return createHash("sha256").update(token).digest("hex");
}

function warmProviderEnvFingerprints(): void {
	const envVarNames = new Set<string>();

	for (const provider of providers) {
		const envVarName = getProviderEnvVar(provider.id as Provider);
		if (envVarName) {
			envVarNames.add(envVarName);
		}
	}

	for (const [envVarName, envValue] of Object.entries(process.env)) {
		if (!envValue) {
			continue;
		}

		const matchesProviderEnv = [...envVarNames].some(
			(baseEnvVarName) =>
				envVarName === baseEnvVarName ||
				envVarName.startsWith(`${baseEnvVarName}__`),
		);

		if (!matchesProviderEnv) {
			continue;
		}

		for (const token of parseCommaSeparatedEnv(envValue)) {
			if (!tokenFingerprintCache.has(token)) {
				tokenFingerprintCache.set(token, computeApiKeyFingerprint(token));
			}
		}
	}
}

warmProviderEnvFingerprints();

export function getApiKeyFingerprint(token: string): string {
	const existingFingerprint = tokenFingerprintCache.get(token);
	if (existingFingerprint) {
		return existingFingerprint;
	}

	const fingerprint = computeApiKeyFingerprint(token);
	tokenFingerprintCache.set(token, fingerprint);
	return fingerprint;
}
