#!/bin/bash

# Gemini 3 Flash Benchmark: Direct (Google AI Studio / Vertex) vs LLM Gateway vs OpenRouter
#
# Required env vars (set whichever endpoints you want to test):
#   GOOGLE_AI_KEY          - Google AI Studio API key
#   LLM_GATEWAY_API_KEY    - LLM Gateway API key
#   OPENROUTER_API_KEY     - OpenRouter API key
#
# Optional env vars:
#   GOOGLE_VERTEX_TOKEN    - Vertex AI access token (from `gcloud auth print-access-token`)
#   VERTEX_PROJECT_ID      - GCP project ID for Vertex AI
#   VERTEX_REGION          - Vertex AI region (default: us-central1)
#   REQUESTS_PER_ENDPOINT  - Number of requests per endpoint (default: 5)

REQUESTS=${REQUESTS_PER_ENDPOINT:-5}
MODEL_NAME="gemini-3-flash-preview"
OUTPUT_FILE="benchmark_gemini3_flash.json"

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

# --- Endpoint definitions ---
# Each endpoint: label|url|auth_header|model_field
ENDPOINTS=()

# 1. Google AI Studio (direct)
if [[ -n "$GOOGLE_AI_KEY" ]]; then
	ENDPOINTS+=("google-ai-studio-direct|https://generativelanguage.googleapis.com/v1beta/openai/chat/completions|Bearer ${GOOGLE_AI_KEY}|${MODEL_NAME}")
else
	echo -e "${YELLOW}Skipping Google AI Studio (set GOOGLE_AI_KEY to enable)${NC}"
fi

# 2. Google Vertex AI (direct)
if [[ -n "$GOOGLE_VERTEX_TOKEN" && -n "$VERTEX_PROJECT_ID" ]]; then
	VERTEX_REGION="${VERTEX_REGION:-us-central1}"
	VERTEX_URL="https://${VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_REGION}/endpoints/openapi/chat/completions"
	ENDPOINTS+=("google-vertex-direct|${VERTEX_URL}|Bearer ${GOOGLE_VERTEX_TOKEN}|${MODEL_NAME}")
else
	echo -e "${YELLOW}Skipping Google Vertex (set GOOGLE_VERTEX_TOKEN + VERTEX_PROJECT_ID to enable)${NC}"
fi

# 3. LLM Gateway (both provider routes)
if [[ -n "$LLM_GATEWAY_API_KEY" ]]; then
	ENDPOINTS+=("llm-gw-ai-studio|https://api.llmgateway.io/v1/chat/completions|Bearer ${LLM_GATEWAY_API_KEY}|google-ai-studio/${MODEL_NAME}")
	ENDPOINTS+=("llm-gw-vertex|https://api.llmgateway.io/v1/chat/completions|Bearer ${LLM_GATEWAY_API_KEY}|google-vertex/${MODEL_NAME}")
else
	echo -e "${YELLOW}Skipping LLM Gateway (set LLM_GATEWAY_API_KEY to enable)${NC}"
fi

# 4. OpenRouter
if [[ -n "$OPENROUTER_API_KEY" ]]; then
	ENDPOINTS+=("openrouter|https://openrouter.ai/api/v1/chat/completions|Bearer ${OPENROUTER_API_KEY}|google/${MODEL_NAME}")
else
	echo -e "${YELLOW}Skipping OpenRouter (set OPENROUTER_API_KEY to enable)${NC}"
fi

if [[ ${#ENDPOINTS[@]} -eq 0 ]]; then
	echo -e "${RED}No endpoints configured. Set at least one API key env var.${NC}"
	exit 1
fi

echo ""
echo -e "${BOLD}Gemini 3 Flash Benchmark${NC}"
echo "Endpoints: ${#ENDPOINTS[@]}"
echo "Requests per endpoint: $REQUESTS"
echo ""

# Initialize results
results="[]"

# Benchmark a single streaming request
benchmark_request() {
	local url=$1
	local auth=$2
	local model=$3
	local label=$4
	local request_num=$5

	local response_file=$(mktemp)
	local timing_file=$(mktemp)

	local payload=$(cat <<EOF
{
  "model": "$model",
  "messages": [
    {"role": "user", "content": "Write a haiku about programming"}
  ],
  "stream": true,
  "max_tokens": 100
}
EOF
)

	local start_time=$(now_ms)

	# Stream response; write TTFT timestamp to timing_file on first data chunk
	curl -s -N -X POST "$url" \
		-H "Authorization: $auth" \
		-H "Content-Type: application/json" \
		-d "$payload" 2>/dev/null | while IFS= read -r line; do

		# First SSE data line = time to first token
		if [[ ! -s "$timing_file" && $line == data:* ]]; then
			now_ms > "$timing_file"
		fi

		echo "$line" >> "$response_file"
	done

	local end_time=$(now_ms)

	# Read TTFT
	local ttft_ms="null"
	if [[ -s "$timing_file" ]]; then
		local ttft_time=$(cat "$timing_file")
		ttft_ms=$(( ttft_time - start_time ))
	fi

	local total_ms=$(( end_time - start_time ))

	# Check success
	local status="error"
	local error_msg=""
	if [[ -f "$response_file" ]] && grep -q "data:" "$response_file"; then
		status="success"
	elif [[ -f "$response_file" && -s "$response_file" ]]; then
		# Capture error response, escape for JSON, truncate
		error_msg=$(cat "$response_file" | tr '\n' ' ' | sed 's/"/\\"/g' | head -c 300)
	fi

	rm -f "$response_file" "$timing_file"

	if [[ "$status" == "error" && -n "$error_msg" ]]; then
		# Print full error to stderr for debugging
		echo "  [ERROR] $error_msg" >&2
	fi

	echo "{\"endpoint\":\"$label\",\"model\":\"$model\",\"request\":$request_num,\"ttft_ms\":$ttft_ms,\"total_ms\":$total_ms,\"status\":\"$status\"}"
}

# Run benchmarks
for endpoint_def in "${ENDPOINTS[@]}"; do
	IFS='|' read -r label url auth model <<< "$endpoint_def"

	echo -e "${CYAN}[$label]${NC} ${YELLOW}$model${NC}"

	for i in $(seq 1 $REQUESTS); do
		echo -n "  Request $i/$REQUESTS... "

		result=$(benchmark_request "$url" "$auth" "$model" "$label" "$i")

		if [[ "$results" == "[]" ]]; then
			results="[$result]"
		else
			results="${results%]}, $result]"
		fi

		ttft=$(echo "$result" | grep -o '"ttft_ms":[0-9]*' | cut -d':' -f2)
		total=$(echo "$result" | grep -o '"total_ms":[0-9]*' | cut -d':' -f2)
		status=$(echo "$result" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

		if [[ "$status" == "success" ]]; then
			echo -e "${GREEN}OK${NC} TTFT: ${ttft:-null}ms, Total: ${total}ms"
		else
			echo -e "${RED}FAIL${NC} (see error above)"
		fi
	done
	echo ""
done

# --- Statistics ---
echo -e "${BOLD}Statistics${NC}"
echo ""

stats="{"
first=true

declare -a seen_labels=()
for endpoint_def in "${ENDPOINTS[@]}"; do
	IFS='|' read -r label _ <<< "$endpoint_def"
	seen_labels+=("$label")
done

for label in "${seen_labels[@]}"; do
	# Use grep -F for fixed-string match (no regex issues with special chars)
	ttfts=$(echo "$results" | grep -o "{[^}]*}" | grep -F "\"endpoint\":\"$label\"" | grep -o '"ttft_ms":[0-9]*' | cut -d':' -f2)
	totals=$(echo "$results" | grep -o "{[^}]*}" | grep -F "\"endpoint\":\"$label\"" | grep -o '"total_ms":[0-9]*' | cut -d':' -f2)

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

	if [[ "$first" == false ]]; then stats="$stats,"; fi
	first=false

	stats="$stats\"$label\":{\"avg_ttft_ms\":$avg_ttft,\"min_ttft_ms\":$min_ttft,\"max_ttft_ms\":$max_ttft,\"avg_total_ms\":$avg_total,\"min_total_ms\":$min_total,\"max_total_ms\":$max_total,\"requests\":$REQUESTS}"

	echo -e "${CYAN}$label${NC}"
	echo "  TTFT:  avg=${avg_ttft}ms  min=${min_ttft}ms  max=${max_ttft}ms"
	echo "  Total: avg=${avg_total}ms  min=${min_total}ms  max=${max_total}ms"
	echo ""
done

stats="$stats}"

# --- Comparison table ---
echo -e "${BOLD}Comparison (avg TTFT / avg Total)${NC}"
printf "%-30s %10s %10s\n" "Endpoint" "TTFT" "Total"
echo "----------------------------------------------------"
for label in "${seen_labels[@]}"; do
	avg_ttft=$(echo "$results" | grep -o "{[^}]*}" | grep -F "\"endpoint\":\"$label\"" | grep -o '"ttft_ms":[0-9]*' | cut -d':' -f2 | awk '{sum+=$1; count++} END {if(count>0) printf "%dms", int(sum/count); else printf "N/A"}')
	avg_total=$(echo "$results" | grep -o "{[^}]*}" | grep -F "\"endpoint\":\"$label\"" | grep -o '"total_ms":[0-9]*' | cut -d':' -f2 | awk '{sum+=$1; count++} END {if(count>0) printf "%dms", int(sum/count); else printf "N/A"}')
	printf "%-30s %10s %10s\n" "$label" "$avg_ttft" "$avg_total"
done
echo "----------------------------------------------------"
echo ""

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
