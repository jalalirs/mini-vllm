#!/bin/bash
# Benchmark mini-vLLM vs standard vLLM
# Usage: ./benchmark.sh <endpoint> [num_requests] [concurrency]
# Example: ./benchmark.sh http://localhost:8000 100 10

set -e

ENDPOINT="${1:-http://localhost:8000}"
NUM_REQUESTS="${2:-100}"
CONCURRENCY="${3:-10}"

echo "=========================================="
echo "Mini-vLLM Benchmark"
echo "Endpoint: $ENDPOINT"
echo "Requests: $NUM_REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "=========================================="

# Check if endpoint is accessible
echo "Checking endpoint..."
if ! curl -s "$ENDPOINT/health" > /dev/null; then
    echo "ERROR: Endpoint not accessible"
    exit 1
fi

# Get model info
echo ""
echo "Model info:"
curl -s "$ENDPOINT/v1/models" | python3 -m json.tool

# Warm up
echo ""
echo "Warming up..."
for i in {1..3}; do
    curl -s -X POST "$ENDPOINT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-120b",
            "prompt": "Hello",
            "max_tokens": 10
        }' > /dev/null
done

# Run benchmark
echo ""
echo "Running benchmark..."

# Simple Python benchmark script
python3 << 'EOF'
import asyncio
import aiohttp
import time
import statistics
import sys

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
NUM_REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 10

async def send_request(session, prompt):
    start = time.perf_counter()
    async with session.post(
        f"{ENDPOINT}/v1/completions",
        json={
            "model": "gpt-oss-120b",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7
        }
    ) as response:
        result = await response.json()
        latency = time.perf_counter() - start
        tokens = len(result.get("choices", [{}])[0].get("text", "").split())
        return latency, tokens

async def benchmark():
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        prompts = [f"Write a short story about adventure {i}:" for i in range(NUM_REQUESTS)]
        
        start_time = time.perf_counter()
        tasks = [send_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # Process results
        latencies = []
        total_tokens = 0
        errors = 0
        
        for r in results:
            if isinstance(r, Exception):
                errors += 1
            else:
                latencies.append(r[0])
                total_tokens += r[1]
        
        # Print results
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Total requests:     {NUM_REQUESTS}")
        print(f"Successful:         {len(latencies)}")
        print(f"Errors:             {errors}")
        print(f"Total time:         {total_time:.2f}s")
        print(f"Throughput:         {len(latencies)/total_time:.2f} req/s")
        print(f"Total tokens:       {total_tokens}")
        print(f"Tokens/second:      {total_tokens/total_time:.2f}")
        print()
        
        if latencies:
            print(f"Latency (avg):      {statistics.mean(latencies)*1000:.2f}ms")
            print(f"Latency (p50):      {statistics.median(latencies)*1000:.2f}ms")
            print(f"Latency (p95):      {sorted(latencies)[int(len(latencies)*0.95)]*1000:.2f}ms")
            print(f"Latency (p99):      {sorted(latencies)[int(len(latencies)*0.99)]*1000:.2f}ms")
        print(f"{'='*50}")

asyncio.run(benchmark())
EOF

echo ""
echo "Benchmark complete!"

