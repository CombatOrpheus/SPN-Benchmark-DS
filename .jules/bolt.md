## 2024-04-09 - [HDF5 Packed Columnar Data Packing & Heavy Top-Level Imports]
**Learning:** `numpy.array()` inside of loops when constructing stacked arrays from lists is highly inefficient because it incurs a massive conversion overhead and allocates memory dynamically for every list item. `flatten()` is similarly slow because it forces a copy of the array. Also, importing modules like `sklearn.model_selection` or `matplotlib.pyplot` / `seaborn` at the top of a file causes significant slow-down during script loading, even when the underlying functions that need them are not being used immediately.
**Action:** Always pre-calculate bounds/properties over data arrays natively (like extracting shapes and flattened data together into list comprehension), and use `array.reshape(-1)` rather than `array.flatten()` when constructing the flattened columnar values for fast HDF5 writes. Delay heavy module imports by putting them inside the specific function calls that require them if the module is sparsely used or not critical to script startup logic.

## 2024-04-10 - [Graph Property Calculation Overhead in Reachability Generators]
**Learning:** Computing qualitative graph properties (like deadlock-freedom and reversibility) inside Python functions using high-level structures like `set()`, list comprehensions `[[] for _ in range(N)]`, and `collections.deque` causes severe performance bottlenecks when analyzed over thousands of small subgraphs.
**Action:** When extracting properties from reachability graph edges, use `numba.jit(nopython=True)` along with flat pre-allocated numpy arrays (`np.empty(..., dtype=np.int32)`) to build adjacency tracking and BFS queues manually instead of relying on slow Python objects.

## 2024-04-16 - [Replacing np.array with np.asarray for Performance]
**Learning:** `numpy.array()` creates a full copy of the array if the input is already a NumPy array, leading to significant overhead inside performance-critical paths (e.g., inside generator loops like SPN generation). `numpy.asarray()` acts as a pass-through if the input is already a NumPy array, avoiding the unnecessary copy while still ensuring the input is wrapped properly.
**Action:** Always prefer `np.asarray()` over `np.array()` in functions that convert sequence arguments to numpy arrays if the input might already be an array, particularly in tight loops or data generators.
## 2026-04-19 - [Lazy Array Duplication]
**Learning:** [When generating permutations or augmentations of large data structures (like Petri net candidate matrices), eager copying of the arrays before sampling triggers extreme GC and memory overhead. Creating thousands of array copies just to discard 90% of them with `np.random.choice` destroys performance.]
**Action:** [Instead of eagerly generating the complete matrices for all possible mutations, collect lightweight "operation" tuples (e.g., `("add_edge", row, col)`). Sample the operations down to the desired limit (`max_candidates`), and only apply the matrix mutations to the selected subset.]

## 2024-04-20 - [Avoiding np.array for Shape Extraction]
**Learning:** Using `np.array()` on large nested lists (like parsed JSON representations of matrices) solely to access `.shape` attributes triggers massive unnecessary memory allocation, deep copying, and garbage collection overhead, converting an $O(1)$ dimension check into a slow $O(N)$ operation per item.
**Action:** When extracting dimensions or counts from pure Python lists or parsed JSON structures, use the native `len()` function on the nested lists directly (e.g., `len(data)` for rows and `len(data[0])` for columns) instead of casting to NumPy arrays.

## 2024-04-25 - [Direct Sparse Matrix Construction in Numba]
**Learning:** Creating COO sparse matrices (`scipy.sparse.coo_array`) from Numba-generated arrays and immediately calling `.tocsc()` or `.tocsr()`, followed by Python-level array slicing (`state_matrix[1:, :]`), incurs substantial overhead due to matrix reallocations and index sorting in Python.
**Action:** When building sparse matrices for state equations (e.g., in `src/spn_datasets/generator/SPN.py`), avoid intermediate dense numpy arrays and Python-level format conversions. Instead, compute row offsets manually and generate `indices`, `indptr`, and `data` arrays using Numba, and construct a square `scipy.sparse.csr_array` directly to prevent memory bottlenecks and `.tocsc()` conversion overhead.

## 2024-04-25 - [Numba Type Mismatch Recompilation]
**Learning:** If a Numba JIT function (`@numba.jit(cache=True)`) is repeatedly called with numpy arrays of varying types (e.g. `np.int64` vs `np.int32` depending on the platform or generation method), Numba will spend significant time recompiling the function for the new signature during runtime.
**Action:** To prevent Numba recompilation overhead on every unique array type signature (especially during batch generation), ensure inputs like `vertices`, `edges`, and `arc_transitions` are explicitly and consistently typed (e.g., using `np.asarray(..., dtype=np.int32)` or `np.float64`) before being passed to `@numba.jit` functions.
