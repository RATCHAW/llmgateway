#!/bin/bash

set -o pipefail

# Gemini 3 Flash Benchmark: direct providers vs local LLM Gateway
#
# Preferred env vars (set whichever endpoints you want to test):
#   LLM_GOOGLE_VERTEX_API_KEY   - Google Vertex API key
#   LLM_GOOGLE_CLOUD_PROJECT    - GCP project ID for Vertex AI
#   LLM_GOOGLE_VERTEX_REGION    - Vertex AI region (default: global)
#   LLM_GOOGLE_AI_STUDIO_API_KEY - Google AI Studio API key
#   LLM_OPENROUTER_API_KEY      - OpenRouter API key
#   LLM_GATEWAY_API_KEY         - LLM Gateway API key (defaults to test-token)
#
# Optional env vars:
#   REQUESTS_PER_ENDPOINT       - Number of requests per endpoint (default: 10)
#   PARALLEL_REQUESTS           - Parallel requests per endpoint (default: 4)
#   MODEL_NAME                  - Model to benchmark (default: gemini-3-flash-preview)
#   BENCHMARK_REASONING_EFFORT  - Reasoning level to send to all endpoints (default: low, use "off" to disable)
#   BENCHMARK_PROMPT            - Prompt to send for each request
#   OUTPUT_FILE                 - Where to write the benchmark JSON
#   LLM_GATEWAY_BASE_URL        - Override LLM Gateway base URL (default: http://localhost:4001)
#   LLM_GATEWAY_LOCAL_API_KEY   - Override the local gateway token (default: test-token)
#   LLM_OPENROUTER_BASE_URL     - Override OpenRouter base URL

REQUESTS=${REQUESTS_PER_ENDPOINT:-10}
PARALLEL_REQUESTS=${PARALLEL_REQUESTS:-4}
MODEL_NAME="${MODEL_NAME:-gemini-3-flash-preview}"
BENCHMARK_REASONING_EFFORT="${BENCHMARK_REASONING_EFFORT:-low}"
PROMPT="${BENCHMARK_PROMPT:-Write a haiku about programming}"
OUTPUT_FILE="${OUTPUT_FILE:-benchmark_gemini3_flash.json}"
GATEWAY_BASE_URL="${LLM_GATEWAY_BASE_URL:-http://localhost:4001}"
OPENROUTER_BASE_URL="${LLM_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
RESULTS_DIR=$(mktemp -d)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Millisecond timestamp (works on macOS which lacks date +%s%N)
now_ms() {
	perl -MTime::HiRes -e 'printf("%.0f\n", Time::HiRes::time() * 1000)'
}

format_percent_diff() {
	local value=$1
	local best=$2

	if [[ "$value" -eq 0 || "$best" -eq 0 ]]; then
		printf '0.0%%'
		return
	fi

	awk -v value="$value" -v best="$best" 'BEGIN { printf "%.1f%%", ((value - best) / best) * 100 }'
}

first_csv_value() {
	printf '%s' "$1" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}'
}

get_google_thinking_budget() {
	case "$1" in
		minimal)
			printf '512'
			;;
		low)
			printf '2048'
			;;
		medium)
			printf '8192'
			;;
		high)
			printf '24576'
			;;
		xhigh)
			printf '65536'
			;;
		off | none | "")
			printf ''
			;;
		*)
			echo "Unsupported BENCHMARK_REASONING_EFFORT: $1" >&2
			return 1
			;;
	esac
}

build_payload() {
	local request_mode=$1
	local model=$2
	local google_thinking_budget=""

	google_thinking_budget=$(get_google_thinking_budget "$BENCHMARK_REASONING_EFFORT") || return 1

	case "$request_mode" in
		openai-chat)
			jq -cn \
				--arg model "$model" \
				--arg prompt "$PROMPT" \
				--arg reasoning_effort "$BENCHMARK_REASONING_EFFORT" \
				'{
					model: $model,
					messages: [
						{role: "user", content: $prompt}
					],
					stream: true,
					max_tokens: 100
				}
				+ (
					if ($reasoning_effort == "off" or $reasoning_effort == "none" or $reasoning_effort == "")
					then {}
					else {reasoning_effort: $reasoning_effort}
					end
				)'
			;;
		google-native)
			jq -cn \
				--arg prompt "$PROMPT" \
				--argjson thinking_budget "${google_thinking_budget:-null}" \
				'{
					contents: [
						{
							role: "user",
							parts: [{text: $prompt}]
						}
					],
					generationConfig: {
						maxOutputTokens: 100
					}
				}
				+ (
					if $thinking_budget == null
					then {}
					else {
						generationConfig: {
							maxOutputTokens: 100,
							thinkingConfig: {
								includeThoughts: true,
								thinkingBudget: $thinking_budget
							}
						}
					}
					end
				)'
			;;
		*)
			echo "Unknown request mode: $request_mode" >&2
			return 1
			;;
	esac
}

extract_error_message() {
	local file=$1

	if [[ ! -s "$file" ]]; then
		return
	fi

	tr '\r\n' '  ' < "$file" | sed 's/[[:space:]]\+/ /g' | cut -c 1-200
}

VERTEX_API_KEY=$(first_csv_value "${LLM_GOOGLE_VERTEX_API_KEY:-}")
VERTEX_PROJECT=$(first_csv_value "${LLM_GOOGLE_CLOUD_PROJECT:-}")
VERTEX_REGION=$(first_csv_value "${LLM_GOOGLE_VERTEX_REGION:-global}")
GOOGLE_AI_STUDIO_API_KEY=$(first_csv_value "${LLM_GOOGLE_AI_STUDIO_API_KEY:-}")
OPENROUTER_API_KEY=$(first_csv_value "${LLM_OPENROUTER_API_KEY:-}")

if [[ "$GATEWAY_BASE_URL" == http://localhost:* || "$GATEWAY_BASE_URL" == http://127.0.0.1:* ]]; then
	GATEWAY_API_KEY=$(first_csv_value "${LLM_GATEWAY_LOCAL_API_KEY:-test-token}")
else
	GATEWAY_API_KEY=$(first_csv_value "${LLM_GATEWAY_API_KEY:-}")
fi

# --- Endpoint definitions ---
# Each endpoint: label|request_mode|url|auth_type|auth_value|model_field|extra_header
ENDPOINTS=()

# 1. Google AI Studio (direct, native API key flow)
if [[ -n "$GOOGLE_AI_STUDIO_API_KEY" ]]; then
	GOOGLE_AI_STUDIO_URL="https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:streamGenerateContent?alt=sse&key=${GOOGLE_AI_STUDIO_API_KEY}"
	ENDPOINTS+=("google-ai-studio-direct|google-native|${GOOGLE_AI_STUDIO_URL}|none||${MODEL_NAME}|")
else
	echo -e "${YELLOW}Skipping Google AI Studio (set LLM_GOOGLE_AI_STUDIO_API_KEY to enable)${NC}"
fi

# 2. Google Vertex AI (direct, native API key flow)
if [[ -n "$VERTEX_API_KEY" && -n "$VERTEX_PROJECT" ]]; then
	VERTEX_URL="https://aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_REGION}/publishers/google/models/${MODEL_NAME}:streamGenerateContent?alt=sse&key=${VERTEX_API_KEY}"
	ENDPOINTS+=("google-vertex-direct|google-native|${VERTEX_URL}|none||${MODEL_NAME}|")
else
	echo -e "${YELLOW}Skipping Google Vertex (set LLM_GOOGLE_VERTEX_API_KEY + LLM_GOOGLE_CLOUD_PROJECT to enable)${NC}"
fi

# 3. LLM Gateway (provider-pinned, no fallback)
if [[ -n "$GATEWAY_API_KEY" ]]; then
	ENDPOINTS+=("llm-gw-ai-studio|openai-chat|${GATEWAY_BASE_URL%/}/v1/chat/completions|bearer|${GATEWAY_API_KEY}|google-ai-studio/${MODEL_NAME}|x-no-fallback: true")
	ENDPOINTS+=("llm-gw-vertex|openai-chat|${GATEWAY_BASE_URL%/}/v1/chat/completions|bearer|${GATEWAY_API_KEY}|google-vertex/${MODEL_NAME}|x-no-fallback: true")
else
	echo -e "${YELLOW}Skipping LLM Gateway (set LLM_GATEWAY_API_KEY to enable)${NC}"
fi

# 4. OpenRouter
if [[ -n "$OPENROUTER_API_KEY" ]]; then
	ENDPOINTS+=("openrouter|openai-chat|${OPENROUTER_BASE_URL%/}/chat/completions|bearer|${OPENROUTER_API_KEY}|google/${MODEL_NAME}|")
else
	echo -e "${YELLOW}Skipping OpenRouter (set LLM_OPENROUTER_API_KEY to enable)${NC}"
fi

if [[ ${#ENDPOINTS[@]} -eq 0 ]]; then
	echo -e "${RED}No endpoints configured. Set at least one API key env var.${NC}"
	exit 1
fi

echo ""
echo -e "${BOLD}Gemini 3 Flash Benchmark${NC}"
echo "Endpoints: ${#ENDPOINTS[@]}"
echo "Requests per endpoint: $REQUESTS"
echo "Parallel requests per endpoint: $PARALLEL_REQUESTS"
echo "Prompt: $PROMPT"
echo "Reasoning effort: $BENCHMARK_REASONING_EFFORT"
echo "Gateway base URL: $GATEWAY_BASE_URL"
echo ""

# Initialize results
results="[]"

# Benchmark a single streaming request
benchmark_request() {
	local url=$1
	local auth_type=$2
	local auth_value=$3
	local model=$4
	local label=$5
	local request_num=$6
	local request_mode=$7
	local extra_header=$8

	local response_file=$(mktemp)
	local timing_file=$(mktemp)
	local status_file=$(mktemp)
	local curl_error_file=$(mktemp)
	local payload
	payload=$(build_payload "$request_mode" "$model") || {
		rm -f "$response_file" "$timing_file" "$status_file" "$curl_error_file"
		return 1
	}

	local start_time=$(now_ms)
	local curl_args=(
		-sS
		-N
		-X POST
		"$url"
		-H "Content-Type: application/json"
		-d "$payload"
	)

	case "$auth_type" in
		bearer)
			curl_args+=(-H "Authorization: Bearer ${auth_value}")
			;;
		x-goog-api-key)
			curl_args+=(-H "x-goog-api-key: ${auth_value}")
			;;
		none)
			;;
		*)
			echo "Unknown auth type: $auth_type" >&2
			rm -f "$response_file" "$timing_file" "$status_file" "$curl_error_file"
			return 1
			;;
	esac

	if [[ -n "$extra_header" ]]; then
		curl_args+=(-H "$extra_header")
	fi

	# Stream response; write TTFT timestamp to timing_file on first data chunk
	curl "${curl_args[@]}" \
		--write-out $'\n__LLM_BENCH_HTTP_STATUS__:%{http_code}\n' \
		2>"$curl_error_file" | while IFS= read -r line; do
		if [[ $line == __LLM_BENCH_HTTP_STATUS__:* ]]; then
			printf '%s\n' "${line#__LLM_BENCH_HTTP_STATUS__:}" > "$status_file"
			continue
		fi

		# First SSE data line = time to first token
		if [[ ! -s "$timing_file" && $line == data:* ]]; then
			now_ms > "$timing_file"
		fi

		printf '%s\n' "$line" >> "$response_file"
	done
	local curl_exit=${PIPESTATUS[0]}

	local end_time=$(now_ms)

	# Read TTFT
	local ttft_ms="null"
	if [[ -s "$timing_file" ]]; then
		local ttft_time=$(cat "$timing_file")
		ttft_ms=$(( ttft_time - start_time ))
	fi

	local total_ms=$(( end_time - start_time ))
	local http_status="000"
	if [[ -s "$status_file" ]]; then
		http_status=$(cat "$status_file")
	fi

	# Check success
	local status="error"
	local error_msg=""
	if [[ $curl_exit -eq 0 && $http_status =~ ^2[0-9][0-9]$ ]] && grep -q '^data:' "$response_file"; then
		status="success"
	elif [[ $curl_exit -ne 0 ]]; then
		error_msg=$(extract_error_message "$curl_error_file")
	else
		error_msg=$(extract_error_message "$response_file")
		if [[ -n "$error_msg" ]]; then
			error_msg="HTTP ${http_status}: ${error_msg}"
		else
			error_msg="HTTP ${http_status}"
		fi
	fi

	rm -f "$response_file" "$timing_file" "$status_file" "$curl_error_file"

	jq -cn \
		--arg endpoint "$label" \
		--arg model "$model" \
		--argjson request "$request_num" \
		--argjson ttft_ms "$ttft_ms" \
		--argjson total_ms "$total_ms" \
		--arg status "$status" \
		--arg error "$error_msg" \
		'{
			endpoint: $endpoint,
			model: $model,
			request: $request,
			ttft_ms: $ttft_ms,
			total_ms: $total_ms,
			status: $status,
			error: $error
		}'
}

wait_for_parallel_slot() {
	local max_parallel=$1

	while true; do
		local running_jobs
		running_jobs=$(jobs -pr | wc -l | tr -d ' ')
		if [[ "$running_jobs" -lt "$max_parallel" ]]; then
			return
		fi
		sleep 0.1
	done
}

run_endpoint_benchmark() {
	local label=$1
	local request_mode=$2
	local url=$3
	local auth_type=$4
	local auth_value=$5
	local model=$6
	local extra_header=$7
	local endpoint_dir="${RESULTS_DIR}/${label}"

	mkdir -p "$endpoint_dir"

	echo -e "${CYAN}[$label]${NC} ${YELLOW}$model${NC}"

	for i in $(seq 1 "$REQUESTS"); do
		wait_for_parallel_slot "$PARALLEL_REQUESTS"
		(
			local result
			local ttft
			local total
			local status
			local error_msg

			result=$(benchmark_request "$url" "$auth_type" "$auth_value" "$model" "$label" "$i" "$request_mode" "$extra_header")
			printf '%s\n' "$result" > "${endpoint_dir}/${i}.json"

			ttft=$(printf '%s' "$result" | jq -r '.ttft_ms // "null"')
			total=$(printf '%s' "$result" | jq -r '.total_ms')
			status=$(printf '%s' "$result" | jq -r '.status')

			if [[ "$status" == "success" ]]; then
				echo -e "  Request $i/$REQUESTS... ${GREEN}OK${NC} TTFT: ${ttft}ms, Total: ${total}ms"
			else
				error_msg=$(printf '%s' "$result" | jq -r '.error' | cut -c 1-120)
				echo -e "  Request $i/$REQUESTS... ${RED}FAIL${NC} ${error_msg}"
			fi
		) &
	done

	wait

	local endpoint_results="[]"
	for result_file in "$endpoint_dir"/*.json; do
		if [[ -f "$result_file" ]]; then
			endpoint_results=$(printf '%s' "$endpoint_results" | jq -c --argjson item "$(cat "$result_file")" '. + [$item]')
		fi
	done

	results=$(printf '%s' "$results" | jq -c --argjson endpoint_results "$endpoint_results" '. + $endpoint_results')
	echo ""
}

# Run benchmarks
for endpoint_def in "${ENDPOINTS[@]}"; do
	IFS='|' read -r label request_mode url auth_type auth_value model extra_header <<< "$endpoint_def"
	run_endpoint_benchmark "$label" "$request_mode" "$url" "$auth_type" "$auth_value" "$model" "$extra_header"
done

# --- Statistics ---
echo -e "${BOLD}Statistics${NC}"
echo ""

stats='{}'

declare -a seen_labels=()
for endpoint_def in "${ENDPOINTS[@]}"; do
	IFS='|' read -r label _ <<< "$endpoint_def"
	seen_labels+=("$label")
done

for label in "${seen_labels[@]}"; do
	ttfts=$(printf '%s' "$results" | jq -r --arg label "$label" '.[] | select(.endpoint == $label and .status == "success" and .ttft_ms != null) | .ttft_ms')
	totals=$(printf '%s' "$results" | jq -r --arg label "$label" '.[] | select(.endpoint == $label and .status == "success") | .total_ms')
	success_count=$(printf '%s' "$results" | jq -r --arg label "$label" '[.[] | select(.endpoint == $label and .status == "success")] | length')

	if [[ -n "$ttfts" ]]; then
		avg_ttft=$(echo "$ttfts" | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
		min_ttft=$(echo "$ttfts" | sort -n | head -1)
		max_ttft=$(echo "$ttfts" | sort -n | tail -1)
	else
		avg_ttft=0; min_ttft=0; max_ttft=0
	fi

	if [[ -n "$totals" ]]; then
		avg_total=$(echo "$totals" | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
		min_total=$(echo "$totals" | sort -n | head -1)
		max_total=$(echo "$totals" | sort -n | tail -1)
	else
		avg_total=0; min_total=0; max_total=0
	fi

	stats=$(printf '%s' "$stats" | jq -c \
		--arg label "$label" \
		--argjson avg_ttft "$avg_ttft" \
		--argjson min_ttft "$min_ttft" \
		--argjson max_ttft "$max_ttft" \
		--argjson avg_total "$avg_total" \
		--argjson min_total "$min_total" \
		--argjson max_total "$max_total" \
		--argjson requests "$REQUESTS" \
		--argjson successful_requests "$success_count" \
		'. + {
			($label): {
				avg_ttft_ms: $avg_ttft,
				min_ttft_ms: $min_ttft,
				max_ttft_ms: $max_ttft,
				avg_total_ms: $avg_total,
				min_total_ms: $min_total,
				max_total_ms: $max_total,
				requests: $requests,
				successful_requests: $successful_requests
			}
		}')

	echo -e "${CYAN}$label${NC}"
	echo "  TTFT:  avg=${avg_ttft}ms  min=${min_ttft}ms  max=${max_ttft}ms"
	echo "  Total: avg=${avg_total}ms  min=${min_total}ms  max=${max_total}ms"
	echo "  Success: ${success_count}/${REQUESTS}"
	echo ""
done

print_ranked_table() {
	local title=$1
	local metric_key=$2
	local value_label=$3
	local best_value

	best_value=$(printf '%s' "$stats" | jq -r --arg metric "$metric_key" '
		[to_entries[] | select(.value.successful_requests > 0) | .value[$metric]]
		| if length > 0 then min else empty end
	')

	echo -e "${BOLD}${title}${NC}"
	printf "%-30s %10s %12s\n" "Endpoint" "$value_label" "% diff"
	echo "----------------------------------------------------------"

	if [[ -z "$best_value" ]]; then
		echo "No successful endpoints"
		echo ""
		return
	fi

	while IFS=$'\t' read -r label metric success_count; do
		if [[ "$success_count" -gt 0 ]]; then
			diff=$(format_percent_diff "$metric" "$best_value")
			printf "%-30s %10sms %12s\n" "$label" "$metric" "$diff"
		else
			printf "%-30s %10s %12s\n" "$label" "N/A" "N/A"
		fi
	done < <(
		printf '%s' "$stats" | jq -r --arg metric "$metric_key" '
			to_entries
			| sort_by(
				if .value.successful_requests > 0 then .value[$metric] else 999999999 end
			)
			| .[]
			| [.key, (.value[$metric] | tostring), (.value.successful_requests | tostring)]
			| @tsv
		'
	)

	echo "----------------------------------------------------------"
	echo ""
}

print_ranked_table "Comparison By Avg TTFT" "avg_ttft_ms" "TTFT"
print_ranked_table "Comparison By Avg Total" "avg_total_ms" "Total"

# --- JSON output ---
endpoint_labels=$(printf '%s\n' "${seen_labels[@]}" | jq -R . | jq -s .)

output=$(cat <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "config": {
    "model": "$MODEL_NAME",
    "endpoints": $endpoint_labels,
    "requests_per_endpoint": $REQUESTS
  },
  "statistics": $stats,
  "raw_results": $results
}
EOF
)

echo "$output" | jq '.' > "$OUTPUT_FILE"

echo -e "${GREEN}Benchmark complete!${NC}"
echo "Results saved to: $OUTPUT_FILE"

rm -rf "$RESULTS_DIR"
