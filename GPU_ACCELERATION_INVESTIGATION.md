# GPU Acceleration Investigation Report

## Executive Summary
This report investigates the feasibility of leveraging JAX or Numba-CUDA to provide further acceleration for the Stochastic Petri Net (SPN) dataset generation pipeline.

**Conclusion:** While minor mathematical operations could technically be ported to GPUs, **using JAX or Numba-CUDA would likely *decrease* performance and is not recommended without a complete, massive rewrite of the codebase.** The current algorithms are inherently sequential, branch-heavy, and rely on dynamically growing data structures, which are anti-patterns for GPU architectures. Furthermore, the overhead of transferring many small matrices between the CPU and GPU would bottleneck the pipeline.

## Detailed Analysis by Component

### 1. Reachability Graph Generation (`ArrivableGraph.py`)
- **Functions:** `generate_reachability_graph`, `get_next_markings`, `is_valid_marking`
- **Current implementation:** Uses Breadth-First Search (BFS) with a queue that grows dynamically. It sequentially explores states, evaluating branchy logic for valid markings.
- **GPU Compatibility (Poor):** GPUs excel at processing large, predictable arrays in parallel. BFS and dynamic graph exploration are notoriously difficult to accelerate on GPUs due to warp divergence (different threads taking different execution paths) and the need for dynamic memory allocation for the queue and visited sets.

### 2. Random SPN Generation (`PetriGenerate.py` & `DataTransformation.py`)
- **Functions:** `_generate_random_petri_net`, `_generate_candidate_matrices_numba`
- **Current implementation:** Sequentially adds nodes and edges to build valid connected components one step at a time, checking constraints iteratively.
- **GPU Compatibility (Poor):** This logic is entirely sequential. A thread on a GPU cannot efficiently generate a single graph this way without idling thousands of other threads.

### 3. State Equations and Steady-State Solving (`SPN.py`)
- **Functions:** `_compute_state_equation_numba`, `compute_average_markings`, `is_connected`, `solve_for_steady_state`
- **Current implementation:**
  - Connectivity check uses BFS (sequential, branchy).
  - State equations iterate through edges to populate a matrix.
  - Solving relies on `scipy.sparse.linalg.spsolve`.
- **GPU Compatibility (Mixed but impractical):**
  - **Vectorization:** `compute_average_markings` and `_compute_state_equation_numba` could be vectorized using JAX (e.g., `jax.numpy.add.at` or `jax.ops.segment_sum`).
  - **The Batching Problem:** To see any benefit from a GPU, these vectorized operations must be run on massive matrices. Processing one small SPN matrix at a time on a GPU incurs massive launch and PCIe transfer overheads, making it slower than running on the CPU via Numba.
  - To make JAX viable, the entire generation pipeline would have to be refactored to generate and solve *batches* of thousands of SPNs simultaneously. Because each SPN produces a reachability graph of a different size, batching would require heavy zero-padding, wasting significant GPU memory and compute.
  - **Sparse Solvers:** While GPU sparse solvers exist (like in CuPy/cuSPARSE), `spsolve` is being used on relatively small, varying-sized matrices, which again suffers from transfer overheads.

## Summary of GPU Anti-Patterns in the Current Codebase
1. **Dynamic Memory Allocation:** Lists and queues growing over time during graph generation. GPUs require pre-allocated, fixed-size arrays.
2. **Branching & Divergence:** Lots of `if/else` checks for graph constraints.
3. **Small Task Size:** Generating one graph at a time does not provide enough workload to saturate thousands of GPU cores.

## Alternative Recommendations for Acceleration
Instead of JAX or Numba-CUDA, consider the following CPU-based optimizations:
1. **Multiprocessing / Joblib:** The generation of independent SPNs is an "embarrassingly parallel" problem. Use Python's `multiprocessing` pool or `joblib` to generate different SPNs on different CPU cores simultaneously.
2. **Optimizing Sparse Solving:** Investigate if `scipy.sparse.linalg.gmres` or `bicgstab` provides faster convergence than the exact `spsolve` for these specific matrices.
3. **C++ / Cython rewrite:** If Numba is hitting its limits with BFS queues, a native C++ implementation with pybind11 might provide better control over memory and cache locality for graph traversal.