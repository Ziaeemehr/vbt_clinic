"""Quick test to check determinism of different scatter-add approaches."""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')

import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import numpy as np
import scipy.sparse

# Set up a simple test case
np.random.seed(42)
num_nodes = 68
num_connections = 500
num_item = 1

# Create random sparse connectivity
rows = np.random.randint(0, num_nodes, num_connections)
cols = np.random.randint(0, num_nodes, num_connections)
data = np.random.randn(num_connections).astype(np.float32)

# Sort by rows for segment_sum
sort_idx = np.argsort(rows, kind='stable')
sorted_rows = rows[sort_idx]
sorted_cols = cols[sort_idx]
sorted_data = data[sort_idx]

# Random input
test_input = np.random.randn(num_connections, num_item).astype(np.float32)

print("Testing different approaches for determinism...")
print("=" * 60)

# Approach 1: Original scatter-add (non-deterministic on GPU)
@jax.jit
def approach1_scatter_add(data, rows):
    cx = jnp.zeros((num_nodes, num_item))
    cx = cx.at[rows].add(data)
    return cx

# Approach 2: segment_sum with sorted indices
@jax.jit
def approach2_segment_sum_sorted(data, rows):
    return segment_sum(data, rows, num_segments=num_nodes)

# Approach 3: Dense matrix multiplication
scatter_matrix_dense = np.zeros((num_nodes, num_connections), dtype=np.float32)
scatter_matrix_dense[sorted_rows, np.arange(num_connections)] = 1.0
scatter_matrix_dense_jnp = jnp.array(scatter_matrix_dense)

@jax.jit
def approach3_dense_matmul(data):
    return scatter_matrix_dense_jnp @ data

# Approach 4: Sparse matrix multiplication (BCOO)
scatter_scipy = scipy.sparse.csr_matrix(
    (np.ones(num_connections, dtype=np.float32), 
     (sorted_rows, np.arange(num_connections))),
    shape=(num_nodes, num_connections)
)
from jax.experimental import sparse as jsparse
scatter_bcoo = jsparse.BCOO.from_scipy_sparse(scatter_scipy)

@jax.jit
def approach4_sparse_matmul(data):
    return scatter_bcoo @ data


def test_determinism(approach_func, name, data, rows=None):
    """Run the approach multiple times and check if results are identical."""
    print(f"\n{name}")
    print("-" * 60)
    
    results = []
    for i in range(5):
        if rows is not None:
            result = approach_func(data, rows)
        else:
            result = approach_func(data)
        result.block_until_ready()  # Ensure computation completes
        results.append(np.array(result))
    
    # Check if all results are identical
    all_same = all(np.allclose(results[0], r, rtol=0, atol=0) for r in results[1:])
    
    if all_same:
        print(f"✅ DETERMINISTIC: All 5 runs produced identical results")
    else:
        max_diff = max(np.max(np.abs(results[0] - r)) for r in results[1:])
        print(f"❌ NON-DETERMINISTIC: Max difference = {max_diff:.2e}")
    
    return all_same


# Run tests
print("\nDevice:", jax.devices()[0])
print()

test1 = test_determinism(
    approach1_scatter_add, 
    "Approach 1: scatter-add (original)",
    jnp.array(sorted_data.reshape(-1, num_item)),
    jnp.array(sorted_rows)
)

test2 = test_determinism(
    approach2_segment_sum_sorted,
    "Approach 2: segment_sum with sorted indices", 
    jnp.array(sorted_data.reshape(-1, num_item)),
    jnp.array(sorted_rows)
)

test3 = test_determinism(
    approach3_dense_matmul,
    "Approach 3: Dense matrix multiplication",
    jnp.array(sorted_data.reshape(-1, num_item))
)

test4 = test_determinism(
    approach4_sparse_matmul,
    "Approach 4: Sparse (BCOO) matrix multiplication",
    jnp.array(sorted_data.reshape(-1, num_item))
)

print("\n" + "=" * 60)
print("SUMMARY:")
print("-" * 60)
print(f"Approach 1 (scatter-add):        {'✅' if test1 else '❌'}")
print(f"Approach 2 (segment_sum sorted): {'✅' if test2 else '❌'}")
print(f"Approach 3 (dense matmul):       {'✅' if test3 else '❌'}")
print(f"Approach 4 (sparse matmul):      {'✅' if test4 else '❌'}")
print("=" * 60)
