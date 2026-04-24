export class InvalidToolCallArgumentsError extends Error {
	public readonly toolCallId?: string;
	public readonly toolName?: string;
	public readonly cause?: unknown;

	constructor(options: {
		toolCallId?: string;
		toolName?: string;
		cause?: unknown;
	}) {
		const { toolCallId, toolName, cause } = options;
		const suffix = [
			toolName ? `tool "${toolName}"` : null,
			toolCallId ? `id "${toolCallId}"` : null,
		]
			.filter(Boolean)
			.join(" ");
		const detail = cause instanceof Error ? `: ${cause.message}` : "";
		super(
			`Invalid JSON in tool_call.function.arguments${
				suffix ? ` for ${suffix}` : ""
			}${detail}`,
		);
		this.name = "InvalidToolCallArgumentsError";
		this.toolCallId = toolCallId;
		this.toolName = toolName;
		this.cause = cause;
	}
}
