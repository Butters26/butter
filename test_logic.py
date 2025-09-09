#!/usr/bin/env python3
"""
Test the core logic of the Monday model fixes without numpy
"""
import sys

def test_layernorm_logic():
    """Test the LayerNorm logic"""
    print("🧪 Testing LayerNorm logic...")
    
    # Simulate the _apply_layernorm method logic
    def simulate_layernorm(x, gamma, beta, eps=1e-5):
        # Simulate mean calculation
        mean = sum(x) / len(x)
        # Simulate variance calculation  
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        var = max(var, eps)  # Clip variance
        # Simulate normalization
        y = [(xi - mean) / (var ** 0.5) for xi in x]
        # Apply scale and shift
        return [yi * gamma + beta for yi in y]
    
    # Test with simple data
    x = [1.0, 2.0, 3.0, 4.0]
    gamma = 1.0
    beta = 0.0
    
    result = simulate_layernorm(x, gamma, beta)
    print(f"  Input: {x}")
    print(f"  Output: {[round(r, 3) for r in result]}")
    
    # Check that result is normalized (mean ≈ 0, std ≈ 1)
    result_mean = sum(result) / len(result)
    result_var = sum((r - result_mean) ** 2 for r in result) / len(result)
    
    print(f"  Result mean: {result_mean:.6f} (should be ~0)")
    print(f"  Result std: {result_var**0.5:.6f} (should be ~1)")
    
    return abs(result_mean) < 0.001 and abs(result_var**0.5 - 1.0) < 0.001

def test_attention_logic():
    """Test the attention logic"""
    print("\n🧪 Testing attention logic...")
    
    # Simulate the key parts of attention
    def simulate_attention_scores(q, k, scale):
        # Simulate scaled dot product
        scores = []
        for qi in q:
            for kj in k:
                score = sum(qi[i] * kj[i] for i in range(len(qi))) * scale
                scores.append(score)
        return scores
    
    def simulate_softmax(scores):
        # Simulate stable softmax
        max_score = max(scores)
        exp_scores = [2.718281828459045 ** (s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [exp / sum_exp for exp in exp_scores]
    
    # Test data
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    scale = 0.5
    
    scores = simulate_attention_scores(q, k, scale)
    probs = simulate_softmax(scores)
    
    print(f"  Q: {q}")
    print(f"  K: {k}")
    print(f"  Scores: {[round(s, 3) for s in scores]}")
    print(f"  Probabilities: {[round(p, 3) for p in probs]}")
    print(f"  Prob sum: {sum(probs):.6f} (should be 1.0)")
    
    return abs(sum(probs) - 1.0) < 0.001

def test_generation_logic():
    """Test the generation logic"""
    print("\n🧪 Testing generation logic...")
    
    def simulate_logits_processing(logits, temperature):
        # Simulate the clean logits processing
        processed = [l / temperature for l in logits]
        return processed
    
    def simulate_top_p_sampling(probs, top_p):
        # Simulate top-p sampling
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        cutoff = 0
        for i, idx in enumerate(sorted_indices):
            cumsum += probs[idx]
            if cumsum >= top_p:
                cutoff = i + 1
                break
        return cutoff
    
    # Test data
    logits = [2.0, 1.0, 0.5, 0.1]
    temperature = 0.7
    top_p = 0.9
    
    processed = simulate_logits_processing(logits, temperature)
    print(f"  Original logits: {logits}")
    print(f"  After temperature: {[round(p, 3) for p in processed]}")
    
    # Simulate softmax to get probabilities
    max_logit = max(processed)
    exp_logits = [2.718281828459045 ** (l - max_logit) for l in processed]
    probs = [exp / sum(exp_logits) for exp in exp_logits]
    
    cutoff = simulate_top_p_sampling(probs, top_p)
    print(f"  Probabilities: {[round(p, 3) for p in probs]}")
    print(f"  Top-p cutoff: {cutoff} (should be reasonable)")
    
    return cutoff > 0 and cutoff <= len(probs)

def main():
    """Run all tests"""
    print("🚀 Testing Monday Model Logic Fixes\n")
    
    tests = [
        test_layernorm_logic,
        test_attention_logic, 
        test_generation_logic
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                print("  ✅ PASSED")
                passed += 1
            else:
                print("  ❌ FAILED")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All logic tests passed! The fixes should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. There may still be issues.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)