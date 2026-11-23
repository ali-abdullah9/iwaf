"""
iWAF Latency Benchmark
Measures end-to-end inference latency for the ML service.

Run from the ml-service folder (needs the trained artifacts in artifacts/):
    python benchmark_latency.py

Outputs:
    - Per-request latency distribution
    - Throughput (requests/second)
    - Comparison: with vs without ML (estimated)
"""

import sys
import time
import statistics
import json
from pathlib import Path

sys.path.insert(0, '.')


def benchmark():
    import torch
    from model import IWAFClassifier
    
    print("=" * 60)
    print("iWAF Latency Benchmark")
    print("=" * 60)
    
    # Load model
    classifier = IWAFClassifier(artifacts_dir='artifacts')
    
    # Test cases representing realistic traffic mix
    test_cases = [
        # Typical benign requests (60%)
        ("GET", "/", "", "", "benign-home"),
        ("GET", "/products", "page=1&limit=20", "", "benign-listing"),
        ("GET", "/api/v1/users/42", "", "", "benign-api"),
        ("POST", "/login", "", '{"email":"user@test.com","password":"secret"}', "benign-login"),
        ("GET", "/profile", "", "", "benign-profile"),
        ("GET", "/search", "q=python+tutorial", "", "benign-search"),
        # Attacks (40%)
        ("GET", "/search", "q=' OR 1=1--", "", "attack-sqli"),
        ("GET", "/search", "q=<script>alert(1)</script>", "", "attack-xss"),
        ("GET", "/view", "file=../../../etc/passwd", "", "attack-lfi"),
        ("GET", "/.env", "", "", "attack-probe"),
    ]
    
    # Warm-up (first calls are slower due to lazy initialization)
    print("\nWarming up...")
    for _ in range(20):
        for method, path, qs, body, _ in test_cases:
            classifier.predict(method=method, path=path, query=qs, body=body)
    
    # Benchmark
    print("\nMeasuring latencies...")
    n_iterations = 200
    latencies = []
    by_category = {}
    
    for i in range(n_iterations):
        for method, path, qs, body, name in test_cases:
            start = time.perf_counter()
            result = classifier.predict(method=method, path=path, query=qs, body=body)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            
            cat = name.split('-')[0]  # benign or attack
            by_category.setdefault(cat, []).append(elapsed_ms)
    
    # Stats
    latencies.sort()
    total_requests = len(latencies)
    
    def percentile(lst, p):
        return lst[int(len(lst) * p / 100)]
    
    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    p99_9 = percentile(latencies, 99.9) if len(latencies) >= 1000 else p99
    avg = statistics.mean(latencies)
    stddev = statistics.stdev(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    # Throughput estimate (single-threaded)
    throughput_rps = 1000 / avg
    
    # Report
    print(f"\n{'='*60}")
    print(f"RESULTS  ({total_requests} requests)")
    print(f"{'='*60}")
    print(f"\nLatency distribution (ms):")
    print(f"  Min:    {min_lat:7.2f}")
    print(f"  p50:    {p50:7.2f}  (median)")
    print(f"  Avg:    {avg:7.2f}  (mean)")
    print(f"  p90:    {p90:7.2f}")
    print(f"  p95:    {p95:7.2f}")
    print(f"  p99:    {p99:7.2f}")
    if total_requests >= 1000:
        print(f"  p99.9:  {p99_9:7.2f}")
    print(f"  Max:    {max_lat:7.2f}")
    print(f"  StdDev: {stddev:7.2f}")
    
    print(f"\nThroughput:")
    print(f"  Single-threaded: {throughput_rps:7.1f} req/sec")
    print(f"  With 4 workers:  {throughput_rps * 4:7.1f} req/sec (estimate)")
    
    print(f"\nLatency by request type:")
    for cat in sorted(by_category):
        lats = by_category[cat]
        cat_avg = statistics.mean(lats)
        cat_p99 = sorted(lats)[int(len(lats) * 0.99)]
        print(f"  {cat:8s}: avg={cat_avg:6.2f}ms, p99={cat_p99:6.2f}ms (n={len(lats)})")
    
    # Device check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nHardware: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY METRICS")
    print(f"{'='*60}")
    print(f"\nML inference adds:")
    print(f"  Median overhead:  +{p50:.1f}ms per request")
    print(f"  p95 overhead:     +{p95:.1f}ms per request")
    print(f"  p99 overhead:     +{p99:.1f}ms per request")
    print(f"\nFor a typical webpage with 20 concurrent requests:")
    print(f"  Total ML overhead: ~{p95 * 20 / 4:.0f}ms (assuming 4-worker parallel)")
    print(f"\nPerceived user impact:")
    if p95 < 50:
        print(f"  ✅ Imperceptible (< 50ms is below human perception)")
    elif p95 < 100:
        print(f"  ✅ Minimal (< 100ms feels instant)")
    elif p95 < 200:
        print(f"  ⚠️  Noticeable but acceptable (< 200ms feels responsive)")
    else:
        print(f"  ❌ Sluggish (>200ms - consider GPU deployment)")
    
    # Save results to JSON for the report
    results = {
        'total_requests': total_requests,
        'device': device,
        'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'latency_ms': {
            'min': min_lat,
            'p50': p50,
            'avg': avg,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'p99_9': p99_9 if total_requests >= 1000 else None,
            'max': max_lat,
            'stddev': stddev,
        },
        'throughput_rps': {
            'single_threaded': throughput_rps,
            'four_workers_estimate': throughput_rps * 4,
        },
        'by_category': {
            cat: {
                'avg': statistics.mean(lats),
                'p99': sorted(lats)[int(len(lats) * 0.99)],
                'count': len(lats),
            }
            for cat, lats in by_category.items()
        },
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to benchmark_results.json")


if __name__ == "__main__":
    benchmark()
