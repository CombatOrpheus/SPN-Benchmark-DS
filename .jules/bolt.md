## 2024-04-09 - [HDF5 Packed Columnar Data Packing & Heavy Top-Level Imports]
**Learning:** `numpy.array()` inside of loops when constructing stacked arrays from lists is highly inefficient because it incurs a massive conversion overhead and allocates memory dynamically for every list item. `flatten()` is similarly slow because it forces a copy of the array. Also, importing modules like `sklearn.model_selection` or `matplotlib.pyplot` / `seaborn` at the top of a file causes significant slow-down during script loading, even when the underlying functions that need them are not being used immediately.
**Action:** Always pre-calculate bounds/properties over data arrays natively (like extracting shapes and flattened data together into list comprehension), and use `array.reshape(-1)` rather than `array.flatten()` when constructing the flattened columnar values for fast HDF5 writes. Delay heavy module imports by putting them inside the specific function calls that require them if the module is sparsely used or not critical to script startup logic.

## 2024-04-10 - [Graph Property Calculation Overhead in Reachability Generators]
**Learning:** Computing qualitative graph properties (like deadlock-freedom and reversibility) inside Python functions using high-level structures like `set()`, list comprehensions `[[] for _ in range(N)]`, and `collections.deque` causes severe performance bottlenecks when analyzed over thousands of small subgraphs.
**Action:** When extracting properties from reachability graph edges, use `numba.jit(nopython=True)` along with flat pre-allocated numpy arrays (`np.empty(..., dtype=np.int32)`) to build adjacency tracking and BFS queues manually instead of relying on slow Python objects.

## 2024-04-16 - [Replacing np.array with np.asarray for Performance]
**Learning:** `numpy.array()` creates a full copy of the array if the input is already a NumPy array, leading to significant overhead inside performance-critical paths (e.g., inside generator loops like SPN generation). `numpy.asarray()` acts as a pass-through if the input is already a NumPy array, avoiding the unnecessary copy while still ensuring the input is wrapped properly.
**Action:** Always prefer `np.asarray()` over `np.array()` in functions that convert sequence arguments to numpy arrays if the input might already be an array, particularly in tight loops or data generators.

## 2024-05-18 - [JSON Array Shape Parsing Overhead]
**Learning:** Extracting dimensions from parsed JSON arrays or large nested Python lists by casting them to NumPy arrays (`np.array()`) solely to use `.shape` is highly inefficient. It causes severe O(N) memory allocation and copying overhead.
**Action:** Use native Python `len()` on the lists directly to get the dimensions without triggering any copying or intermediate array allocations.
