"""
Management command to load additional Python coding challenges (round 2).
Brings each category up to at least 10 problems.

Usage:
    python manage.py load_more_challenges            # load all (skips duplicates)
    python manage.py load_more_challenges --flush    # delete all problems first
"""

from django.core.management.base import BaseCommand
from grader.models import Problem


CHALLENGES = [

    # ──────────────────────────────────────────────────────────────────────────
    # PYTHON BASICS  (need 5 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Python Basics] String Processing & Regex",
        "description": """\
Implement the following string-processing functions (you may import `re`):

1. `is_valid_email(s)` — returns True if `s` matches a basic email pattern:
   one or more word chars / dots / hyphens, an @, a domain, a dot, and a 2-4 char TLD.

2. `camel_to_snake(s)` — converts CamelCase (or mixedCase) to snake_case.
   e.g. "CamelCaseString" → "camel_case_string", "parseHTTPResponse" → "parse_http_response"

3. `extract_numbers(s)` — returns a list of all integers and floats found in a string.
   e.g. "Price is $3.99 and tax is 0.5" → [3.99, 0.5]

4. `word_frequency(text)` — returns a dict of word → count, case-insensitive,
   ignoring punctuation, sorted by count descending then alphabetically.
""",
        "grading_criteria": """\
- `is_valid_email`: uses re.fullmatch or re.match; handles edge cases (no @, multiple @, empty string returns False)
- `camel_to_snake`: handles consecutive capitals (HTTP → http); uses re.sub with a pattern like [A-Z]+[a-z]* or similar
- `extract_numbers`: finds both integers and floats; uses re.findall; returns list of numeric types (not strings)
- `word_frequency`: case-insensitive; punctuation stripped (re.sub or str.strip); sorted by count desc then alpha
- All functions handle empty string input gracefully
- Uses `import re` — no third-party libraries beyond stdlib
""",
        "example_input": 'is_valid_email("user@example.com")\ncamel_to_snake("parseHTTPResponse")\nextract_numbers("Price $3.99, qty 2")\nword_frequency("the cat sat on the mat")',
        "example_output": "True\n'parse_http_response'\n[3.99, 2]\n{'the': 2, 'cat': 1, 'mat': 1, 'on': 1, 'sat': 1}",
    },
    {
        "title": "[Python Basics] Itertools & Functional Programming",
        "description": """\
Using `itertools` and `functools` from the standard library, implement:

1. `running_total(lst)` — returns a list of running (cumulative) totals using `itertools.accumulate`.

2. `group_consecutive(lst)` — groups consecutive equal elements.
   e.g. [1,1,2,3,3,3,1] → [[1,1],[2],[3,3,3],[1]]
   Use `itertools.groupby`.

3. `cartesian_pairs(a, b)` — returns a list of all (x, y) pairs from lists a and b
   where x + y is even. Use `itertools.product`.

4. `compose(*funcs)` — returns a function that applies each func right-to-left (function composition).
   Use `functools.reduce`.
   e.g. compose(str, lambda x: x+1, abs)(-3) → "3"  (abs(-3)=3, 3+1=4, str(4)="4")
""",
        "grading_criteria": """\
- `running_total`: uses itertools.accumulate; returns a list; handles empty list → []
- `group_consecutive`: uses itertools.groupby; groups are lists not tuples; handles single-element input
- `cartesian_pairs`: uses itertools.product; filters where (x+y) % 2 == 0; returns list of tuples
- `compose`: uses functools.reduce; applies right-to-left; works with any number of functions; single function → identity
- All functions import only from stdlib (itertools, functools)
- Edge cases: empty lists, single-element lists handled correctly
""",
        "example_input": "running_total([1,2,3,4])\ngroup_consecutive([1,1,2,3,3])\ncompose(str, lambda x: x+1, abs)(-3)",
        "example_output": "[1, 3, 6, 10]\n[[1, 1], [2], [3, 3]]\n'4'",
    },
    {
        "title": "[Python Basics] Context Managers & File I/O",
        "description": """\
1. Implement a context manager class `Timer` that:
   - Records wall-clock time on `__enter__` using `time.perf_counter()`
   - On `__exit__`, stores the elapsed seconds as `self.elapsed`
   - Suppresses no exceptions
   - Usage: `with Timer() as t: ...; print(t.elapsed)`

2. Implement `csv_to_dict_list(filepath)` — reads a CSV file and returns a list of dicts
   (one per row, keys from the header row). Use only stdlib (`csv` module).

3. Implement `write_json_report(data, filepath)` — writes `data` (any JSON-serialisable object)
   to a JSON file with indent=2 and returns the number of bytes written.

4. Implement a context manager function (using `@contextmanager`) called `temp_file(suffix=".txt")`
   that creates a temporary file, yields its path as a string, and deletes it on exit even if an exception occurs.
""",
        "grading_criteria": """\
- `Timer`: implements __enter__ and __exit__; uses time.perf_counter(); elapsed attribute set correctly; does not suppress exceptions (returns None or False from __exit__)
- `csv_to_dict_list`: uses csv.DictReader; returns list of regular dicts; handles empty file → []
- `write_json_report`: uses json.dump with indent=2; opens file in write mode; returns byte count (can use os.path.getsize or len of written string)
- `temp_file`: uses @contextmanager from contextlib; creates file with tempfile module; cleans up in finally block; yields a string path
- All functions handle OS errors gracefully (FileNotFoundError for read functions)
""",
        "example_input": "with Timer() as t:\n    sum(range(1_000_000))\nprint(t.elapsed > 0)",
        "example_output": "True",
    },
    {
        "title": "[Python Basics] Data Structures — Stack, Queue & Heap",
        "description": """\
Implement the following data structures and algorithms using Python's stdlib:

1. `bracket_valid(s)` — returns True if bracket pairs `()[]{}` in `s` are correctly nested.
   Implement using a stack (use a list). e.g. "([{}])" → True, "([)]" → False.

2. `bfs_shortest_path(graph, start, end)` — given an adjacency dict representing an
   unweighted undirected graph, returns the shortest path as a list of nodes using BFS.
   Returns None if no path exists. Use `collections.deque` as the queue.

3. `top_k_frequent(lst, k)` — returns the k most frequent elements in `lst`, sorted by
   frequency descending. Use `heapq.nlargest` and `collections.Counter`.

4. `lru_cache_demo()` — implement a function `fib(n)` decorated with `@functools.lru_cache(maxsize=128)`,
   return `fib(30)` and the cache info object.
""",
        "grading_criteria": """\
- `bracket_valid`: uses a list as stack; correctly maps closing to opening brackets; empty string → True; unmatched brackets → False
- `bfs_shortest_path`: uses collections.deque; returns list including start and end; returns None when no path; handles start==end → [start]
- `top_k_frequent`: uses collections.Counter and heapq.nlargest; returns exactly k elements; handles ties consistently
- `lru_cache_demo`: fib correctly computes Fibonacci; @functools.lru_cache applied; returns tuple of (fib(30), cache_info)
- fib(30) == 832040
- All imports from stdlib only
""",
        "example_input": 'bracket_valid("([{}])")\nbfs_shortest_path({0:[1,2],1:[0,3],2:[0],3:[1]}, 0, 3)\ntop_k_frequent([1,1,2,2,2,3], 2)',
        "example_output": "True\n[0, 1, 3]\n[2, 1]",
    },
    {
        "title": "[Python Basics] Concurrency — Threading & Multiprocessing",
        "description": """\
Implement the following concurrency patterns:

1. `parallel_map(func, items, workers=4)` — applies `func` to each item in `items` in parallel
   using `concurrent.futures.ThreadPoolExecutor`. Returns a list of results in the original order.

2. `rate_limited_fetch(urls, delay=0.1)` — given a list of URL strings, simulates fetching each
   (use `time.sleep(delay)` instead of a real HTTP call) and returns a list of
   `{"url": url, "status": "ok"}` dicts. Use threading so all fetches run concurrently.

3. `cpu_bound_pool(n)` — computes the sum of squares of all integers 1..n using
   `multiprocessing.Pool.map`, splitting the work into 4 chunks. Returns the integer result.
   (Expected: n*(n+1)*(2n+1)//6)

4. `shared_counter()` — using `threading.Thread` and `threading.Lock`, spawn 10 threads
   each incrementing a shared counter 1000 times. Return the final counter value (should be 10000).
""",
        "grading_criteria": """\
- `parallel_map`: uses ThreadPoolExecutor; preserves order of results; handles empty list → []; workers parameter used
- `rate_limited_fetch`: all fetches run concurrently (not sequentially); returns list of dicts with correct keys; no real HTTP calls
- `cpu_bound_pool`: uses multiprocessing.Pool.map or starmap; splits work into chunks; result equals n*(n+1)*(2n+1)//6
- `shared_counter`: uses threading.Thread and threading.Lock correctly; final count is exactly 10000 (lock prevents race condition)
- All functions import only from stdlib (concurrent.futures, threading, multiprocessing)
- No deadlocks or race conditions
""",
        "example_input": "parallel_map(lambda x: x**2, [1,2,3,4,5])\nshared_counter()",
        "example_output": "[1, 4, 9, 16, 25]\n10000",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # NUMPY  (need 8 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[NumPy] Advanced Indexing & Slicing",
        "description": """\
Using NumPy, implement:

1. `diagonal_sum(M)` — returns the sum of main diagonal + anti-diagonal of a square 2D array.
   Do not double-count the centre for odd-sized matrices.

2. `sliding_window_view(arr, window)` — returns a 2-D array where each row is a window
   of length `window` slided by 1. Use `np.lib.stride_tricks.sliding_window_view` or stride tricks.
   e.g. arr=[1,2,3,4,5], window=3 → [[1,2,3],[2,3,4],[3,4,5]]

3. `fancy_replace(arr, condition_fn, new_val)` — returns a copy of `arr` where all elements
   satisfying `condition_fn(arr)` (a boolean mask function) are replaced with `new_val`.

4. `block_mean(arr, block_size)` — given a 1-D array whose length is divisible by `block_size`,
   reshape into blocks and return the mean of each block.
   e.g. arr=[1,2,3,4,5,6], block_size=2 → [1.5, 3.5, 5.5]
""",
        "grading_criteria": """\
- `diagonal_sum`: uses np.trace for main diagonal; np.fliplr for anti-diagonal; correctly avoids double-counting centre for odd n
- `sliding_window_view`: output shape is (len(arr)-window+1, window); uses stride tricks or np.lib.stride_tricks.sliding_window_view; no Python loops
- `fancy_replace`: returns a copy (not in-place); uses boolean masking with np.where or direct indexing
- `block_mean`: uses reshape to (-1, block_size) then .mean(axis=1); no Python loops; handles length not divisible by block_size gracefully (raise ValueError)
- All functions vectorised — no Python loops over array elements
""",
        "example_input": "sliding_window_view(np.array([1,2,3,4,5]), 3)\nblock_mean(np.array([1,2,3,4,5,6]), 2)",
        "example_output": "[[1 2 3]\n [2 3 4]\n [3 4 5]]\n[1.5 3.5 5.5]",
    },
    {
        "title": "[NumPy] Random Sampling & Simulation",
        "description": """\
Using NumPy (np.random.default_rng(42) for reproducibility), implement:

1. `monte_carlo_pi(n)` — estimate π by sampling `n` random (x,y) points in [0,1)²
   and counting how many fall inside the unit circle. Return the estimate as a float.

2. `bootstrap_ci(data, stat_fn, n_boot=1000, ci=95)` — given a 1-D array `data` and
   a statistic function `stat_fn` (e.g. np.mean), compute a bootstrap confidence interval.
   Return `(lower, upper)` floats rounded to 4 dp.

3. `random_walk_2d(steps)` — simulate a 2-D random walk starting at (0,0).
   Each step moves ±1 in x or y (not diagonal) with equal probability.
   Return the final (x, y) position as a tuple of ints.

4. `simulate_dice(n_rolls, n_dice=2)` — simulate rolling `n_dice` fair 6-sided dice `n_rolls` times.
   Return a dict of sum → frequency (all possible sums, even if frequency is 0).
""",
        "grading_criteria": """\
- All randomness uses np.random.default_rng(42)
- `monte_carlo_pi`: vectorised (no Python loop); uses x**2+y**2 <= 1; result within 0.05 of π for n=100000
- `bootstrap_ci`: resamples with replacement; applies stat_fn to each resample; uses np.percentile for CI bounds; works with any stat_fn
- `random_walk_2d`: each step is one of {up,down,left,right}; returns tuple (int,int); reproducible with seed
- `simulate_dice`: sum ranges from n_dice to 6*n_dice; all keys present even with zero freq; correct dtype
- No Python loops in monte_carlo_pi or bootstrap_ci (vectorised resampling)
""",
        "example_input": "monte_carlo_pi(100_000)  # should be close to 3.14159\nbootstrap_ci(np.array([1,2,3,4,5,6,7,8,9,10]), np.mean)",
        "example_output": "≈ 3.1416\n(approx 3.5 ± margin)",
    },
    {
        "title": "[NumPy] Signal Processing & FFT",
        "description": """\
Using NumPy, implement signal processing functions:

1. `dominant_frequency(signal, sample_rate)` — given a 1-D signal array and its sample rate (Hz),
   return the dominant frequency in Hz (highest amplitude in the FFT, ignoring DC component).

2. `apply_low_pass(signal, sample_rate, cutoff_hz)` — apply a simple ideal low-pass filter:
   compute FFT, zero out components above cutoff_hz, return the real part of the inverse FFT.

3. `generate_composite_signal(freqs, amplitudes, duration, sample_rate)` — generate a signal
   that is the sum of sinusoids with given frequencies and amplitudes.
   e.g. freqs=[5,10], amplitudes=[1,0.5], duration=1.0, sample_rate=1000
   → a 1000-sample array.

4. `signal_stats(signal)` — returns a dict: rms (root mean square), peak_to_peak, crest_factor (peak/rms), zero_crossing_rate.
""",
        "grading_criteria": """\
- `dominant_frequency`: uses np.fft.rfft; ignores index 0 (DC); computes freq axis with np.fft.rfftfreq; returns float Hz
- `apply_low_pass`: uses rfft/irfft; zeroes frequencies above cutoff; returns real array of same length; no distortion of kept frequencies
- `generate_composite_signal`: creates time axis with np.linspace; sums sine waves correctly; output length == int(duration*sample_rate)
- `signal_stats`: rms = sqrt(mean(x^2)); peak_to_peak = max-min; crest_factor = max(abs)/rms; zero_crossing_rate = count sign changes / N
- All vectorised; correct handling of even/odd length FFTs
- dominant_frequency test: generate_composite_signal([10,50],[1,0.5],1,1000) → dominant freq should be 10 Hz
""",
        "example_input": "sig = generate_composite_signal([10, 50], [1.0, 0.5], 1.0, 1000)\ndominant_frequency(sig, 1000)",
        "example_output": "10.0",
    },
    {
        "title": "[NumPy] Image Processing with Arrays",
        "description": """\
Treat images as NumPy arrays (height × width × channels, uint8, values 0-255).
Implement:

1. `to_grayscale(img)` — converts an RGB image array (H,W,3) to grayscale (H,W)
   using the luminosity formula: 0.2126·R + 0.7152·G + 0.0722·B. Return uint8.

2. `flip_rotate(img, mode)` — returns a transformed copy of `img`:
   mode='h' → horizontal flip, mode='v' → vertical flip, mode='90' → 90° clockwise rotation.
   Use numpy slicing / np.rot90 only — no PIL.

3. `apply_kernel(img_gray, kernel)` — applies a 2-D convolution kernel to a grayscale image
   using only NumPy (manual sliding window, no scipy). Clip output to [0,255], return uint8.

4. `histogram_equalize(img_gray)` — applies histogram equalisation to a grayscale image.
   Return uint8 array of same shape.

Test with: `img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)` with seed 42.
""",
        "grading_criteria": """\
- `to_grayscale`: correct luminosity weights; output dtype uint8; shape (H,W); uses np.dot or weighted sum
- `flip_rotate`: h → img[:,::-1], v → img[::-1], 90cw → np.rot90(img, k=-1) or equivalent; returns copy
- `apply_kernel`: handles border (crop or pad — state which); sliding window without scipy; clips to [0,255]; correct convolution (not correlation) or cross-correlation (state clearly)
- `histogram_equalize`: computes CDF from histogram; maps pixel values correctly; output in [0,255]
- All functions return numpy arrays with correct dtype (uint8)
- No PIL, cv2, or scipy imports
""",
        "example_input": "np.random.seed(42)\nimg = np.random.randint(0, 256, (8,8,3), dtype=np.uint8)\nto_grayscale(img).shape",
        "example_output": "(8, 8)",
    },
    {
        "title": "[NumPy] Polynomial Fitting & Numerical Methods",
        "description": """\
Using NumPy, implement:

1. `poly_fit_and_eval(x, y, degree)` — fits a polynomial of given degree to (x,y) data
   using `np.polyfit`, returns a dict with `"coefficients"` (list, highest degree first)
   and `"r_squared"` (float rounded to 4 dp).

2. `numerical_derivative(f, x, h=1e-5)` — computes the derivative of callable `f` at each
   point in array `x` using the central difference formula: (f(x+h) - f(x-h)) / (2h).

3. `trapezoidal_integral(f, a, b, n=1000)` — numerically integrates `f` from `a` to `b`
   using the trapezoidal rule with `n` intervals.

4. `find_root_bisection(f, a, b, tol=1e-6)` — finds a root of `f` in [a,b] using bisection.
   Assumes f(a) and f(b) have opposite signs. Returns the root as a float.
""",
        "grading_criteria": """\
- `poly_fit_and_eval`: uses np.polyfit; R² computed as 1 - SS_res/SS_tot; coefficients as Python list
- `numerical_derivative`: vectorised central difference; works when x is an array; handles scalar x too
- `trapezoidal_integral`: uses np.linspace for x points; np.trapz or manual formula; integral of sin from 0 to π ≈ 2.0
- `find_root_bisection`: correct bisection logic; terminates when |b-a| < tol; raises ValueError if f(a)*f(b) > 0
- All functions use NumPy operations (no scipy)
- Accuracy: integral of x² from 0 to 1 → ~0.3333; root of x²-2 in [1,2] → ~1.4142
""",
        "example_input": "trapezoidal_integral(np.sin, 0, np.pi)\nfind_root_bisection(lambda x: x**2 - 2, 1, 2)",
        "example_output": "≈ 2.0\n≈ 1.4142",
    },
    {
        "title": "[NumPy] Structured Arrays & Record Arrays",
        "description": """\
Using NumPy structured arrays, implement:

1. `create_student_array(names, grades, ages)` — creates a structured NumPy array with dtype:
   `[('name', 'U20'), ('grade', 'f4'), ('age', 'i4')]`

2. `top_students(arr, n)` — given a structured student array, returns the top-n students
   by grade (descending) as a structured array.

3. `grade_statistics(arr)` — returns a dict:
   - `"mean"`, `"std"`, `"min"`, `"max"` of the grade field (all rounded to 2 dp)
   - `"pass_rate"`: fraction of students with grade >= 60 (rounded to 4 dp)

4. `merge_student_arrays(arr1, arr2)` — concatenates two structured arrays of the same dtype,
   removes duplicate names (keep the one with the higher grade), returns sorted by name.
""",
        "grading_criteria": """\
- `create_student_array`: correct dtype specification; returns np.ndarray with structured dtype; handles mismatched list lengths with ValueError
- `top_students`: uses np.argsort or np.sort with order= parameter; returns structured array of length n; handles n > len(arr)
- `grade_statistics`: correct stats on arr['grade']; pass_rate uses boolean mask; all values rounded correctly
- `merge_student_arrays`: uses np.concatenate; deduplication keeps higher grade; sorted by name with np.sort(order='name')
- All operations use NumPy structured array syntax (arr['field']) — no conversion to dicts or plain arrays for operations
""",
        "example_input": "arr = create_student_array(['Alice','Bob','Carol'],[85.5,72.0,91.0],[20,22,21])\ngrade_statistics(arr)",
        "example_output": "{'mean': 82.83, 'std': 7.84, 'min': 72.0, 'max': 91.0, 'pass_rate': 1.0}",
    },
    {
        "title": "[NumPy] Sorting, Searching & Set Operations",
        "description": """\
Using NumPy, implement:

1. `argsort_2d(M, axis)` — returns the indices that would sort a 2-D array along `axis`
   (0=by column, 1=by row). Return the argsorted index array.

2. `binary_search_insert(sorted_arr, val)` — finds the index where `val` should be inserted
   into `sorted_arr` to keep it sorted, using `np.searchsorted`. Return the index.

3. `set_ops(a, b)` — given two 1-D arrays, returns a dict with:
   - `"union"`, `"intersection"`, `"difference"` (a - b), `"symmetric_difference"`
   All as sorted numpy arrays. Use np.union1d etc.

4. `top_n_per_row(M, n)` — given a 2-D matrix, return a same-shape boolean mask
   where True marks the top-n values per row. Handle ties by including all tied values.
""",
        "grading_criteria": """\
- `argsort_2d`: uses np.argsort with axis argument; returns index array of same shape as M
- `binary_search_insert`: uses np.searchsorted; returns correct insertion index for duplicates (right or left — state which)
- `set_ops`: uses np.union1d, np.intersect1d, np.setdiff1d; all outputs sorted; keys exactly as specified
- `top_n_per_row`: boolean mask; handles n > ncols (all True); uses np.partition or np.argsort for efficiency
- No Python loops for set ops or argsort_2d
""",
        "example_input": "set_ops(np.array([1,2,3,4]), np.array([3,4,5,6]))\ntop_n_per_row(np.array([[3,1,4,1,5],[9,2,6,5,3]]), 2)",
        "example_output": "{'union': [1,2,3,4,5,6], 'intersection': [3,4], 'difference': [1,2], 'symmetric_difference': [1,2,5,6]}\n[[True,False,False,False,True],[True,False,True,False,False]]",
    },
    {
        "title": "[NumPy] Performance & Memory — Vectorisation Challenges",
        "description": """\
Each function below has a slow Python-loop version. Rewrite it using only vectorised NumPy:

1. `pairwise_distances(X)` — given an (n, d) array, compute the n×n matrix of Euclidean
   distances between all pairs of rows. No loops, no scipy.

2. `encode_onehot(labels, n_classes)` — given a 1-D integer array of class indices,
   return a 2-D one-hot matrix of shape (len(labels), n_classes).

3. `batch_dot(A, B)` — given A of shape (batch, m, k) and B of shape (batch, k, n),
   return the batched matrix product of shape (batch, m, n) using np.einsum.

4. `running_max(arr)` — returns an array where each element is the maximum of all
   elements up to and including that position. Vectorised using np.maximum.accumulate.
""",
        "grading_criteria": """\
- `pairwise_distances`: uses broadcasting ((X[None]-X[:,None])**2).sum(-1)**0.5 or np.linalg.norm; O(n²d) but no Python loops; result is symmetric with zeros on diagonal
- `encode_onehot`: uses np.zeros + fancy indexing OR np.eye[labels]; shape (N, n_classes); correct for all labels in [0, n_classes)
- `batch_dot`: uses np.einsum('bik,bkj->bij', A, B) or np.matmul; correct shape; equivalent to a loop over batch
- `running_max`: uses np.maximum.accumulate; single line; handles all dtypes
- No Python for/while loops in any function
- Correct results verified against naive implementations
""",
        "example_input": "pairwise_distances(np.array([[0,0],[3,4]])).round(2)\nencode_onehot(np.array([0,2,1]), 3)\nrunning_max(np.array([3,1,4,1,5,9,2,6]))",
        "example_output": "[[0. 5.]\n [5. 0.]]\n[[1,0,0],[0,0,1],[0,1,0]]\n[3 3 4 4 5 9 9 9]",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PANDAS  (need 7 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Pandas] Merging, Joining & Reshaping",
        "description": """\
Given three DataFrames:
```python
employees = pd.DataFrame({'emp_id':[1,2,3,4], 'name':['Alice','Bob','Carol','Dave'], 'dept_id':[10,20,10,30]})
departments = pd.DataFrame({'dept_id':[10,20,40], 'dept_name':['Engineering','Marketing','HR']})
salaries = pd.DataFrame({'emp_id':[1,2,3], 'salary':[95000,55000,105000]})
```

Implement `reshape_analysis(employees, departments, salaries)` that returns a dict:
- `"full_join"`: inner join of all three tables on appropriate keys; columns: name, dept_name, salary
- `"dept_headcount"`: Series indexed by dept_name with employee counts (including depts with 0 employees from departments table)
- `"wide_salary_table"`: pivot table — index=dept_name, columns=name, values=salary (NaN where no data)
- `"unpivoted"`: melt the wide_salary_table back to long form, dropping NaN rows
""",
        "grading_criteria": """\
- `full_join`: correct three-way join; inner join drops employees without salary (Dave excluded); columns exactly name, dept_name, salary
- `dept_headcount`: left join departments → employees; includes departments with no employees (HR: 0); uses groupby or value_counts correctly
- `wide_salary_table`: pd.pivot_table with correct index/columns/values; aggfunc='first' or 'sum' (state which); NaN for missing combinations
- `unpivoted`: pd.melt on the wide table; NaN rows dropped; result is long-form with 3 rows (one per employee with salary)
- No hardcoded values — works on the given DataFrames
- Correct handling of merge keys (dept_id, emp_id)
""",
        "example_input": "r = reshape_analysis(employees, departments, salaries)\nprint(r['full_join'][['name','dept_name','salary']].to_dict('records'))",
        "example_output": "[{'name':'Alice','dept_name':'Engineering','salary':95000}, {'name':'Bob','dept_name':'Marketing','salary':55000}, {'name':'Carol','dept_name':'Engineering','salary':105000}]",
    },
    {
        "title": "[Pandas] String Operations & Text Analysis",
        "description": """\
Given a DataFrame of product reviews:
```python
reviews = pd.DataFrame({
    'review_id': range(1, 7),
    'text': [
        'Great product! Highly recommend.',
        'Terrible quality, very disappointed.',
        'Works as expected. Good value.',
        'AMAZING!!! Best purchase ever!!!',
        'Not worth the price. Poor build.',
        'Excellent quality and fast shipping.'
    ],
    'rating': [5, 1, 3, 5, 2, 4]
})
```

Implement `text_analysis(df)` that returns a dict:
- `"word_counts"`: Series indexed by review_id with word count per review
- `"avg_word_length"`: Series indexed by review_id with average word length (rounded to 2 dp)
- `"sentiment_label"`: Series indexed by review_id: 'positive' (rating>=4), 'neutral' (3), 'negative' (<=2)
- `"top_words"`: list of the 5 most common words across all reviews (lowercase, no punctuation, no stopwords: remove ['the','a','and','is','in','it','as','not'])
- `"has_exclamation"`: boolean Series — True if review contains '!'
""",
        "grading_criteria": """\
- `word_counts`: uses str.split(); counted per review; correct values for all 6 reviews
- `avg_word_length`: strips punctuation from words before measuring; uses str operations or apply; rounded to 2 dp
- `sentiment_label`: uses pd.cut or np.select or map; correct labels for all rating values
- `top_words`: lowercased, punctuation stripped, stopwords removed, counted across all reviews; returns list of 5 word strings
- `has_exclamation`: uses str.contains('!'); returns boolean Series
- All operations use pandas string methods (str accessor) — no Python loops over rows
""",
        "example_input": "text_analysis(reviews)['sentiment_label'].tolist()",
        "example_output": "['positive', 'negative', 'neutral', 'positive', 'negative', 'positive']",
    },
    {
        "title": "[Pandas] MultiIndex & Advanced GroupBy",
        "description": """\
Build the following MultiIndex DataFrame and analyse it:

```python
import numpy as np
np.random.seed(0)
idx = pd.MultiIndex.from_product(
    [['2023','2024'], ['Q1','Q2','Q3','Q4'], ['North','South','East','West']],
    names=['year','quarter','region']
)
df = pd.DataFrame({'sales': np.random.randint(100, 500, len(idx)),
                   'units': np.random.randint(10, 100, len(idx))}, index=idx)
```

Implement `multiindex_analysis(df)` that returns a dict:
- `"annual_totals"`: DataFrame — total sales and units per year (index=year)
- `"best_quarter_per_region"`: Series — for each region, the quarter with highest average sales across both years
- `"yoy_growth"`: DataFrame — year-over-year % growth in sales per (quarter, region), rounded to 2 dp
- `"top_performer"`: the (year, quarter, region) tuple with the single highest sales value
""",
        "grading_criteria": """\
- `annual_totals`: uses .groupby(level='year').sum(); shape (2, 2)
- `best_quarter_per_region`: groups by region and quarter; averages across years; returns quarter name per region
- `yoy_growth`: uses .unstack(level='year') or xs(); computes (2024-2023)/2023*100; rounded to 2 dp
- `top_performer`: uses .idxmax() on sales column; returns a tuple of 3 strings
- All operations use MultiIndex methods (xs, groupby with level=, unstack) — no reset_index then regroupby
""",
        "example_input": "r = multiindex_analysis(df)\nprint(r['annual_totals'].index.tolist())\nprint(type(r['top_performer']))",
        "example_output": "['2023', '2024']\n<class 'tuple'>",
    },
    {
        "title": "[Pandas] Apply, Transform & Window Functions",
        "description": """\
Using the following DataFrame:
```python
np.random.seed(7)
df = pd.DataFrame({
    'store': np.repeat(['A','B','C'], 12),
    'month': list(range(1,13)) * 3,
    'revenue': np.random.randint(10000, 50000, 36),
    'costs':   np.random.randint(5000,  30000, 36),
})
```

Implement `store_analysis(df)` that returns a dict:
- `"profit_margin"`: new column added to df: (revenue - costs) / revenue * 100, rounded to 2 dp
- `"store_rankings"`: for each month, rank the stores by revenue (rank 1 = highest); return as a DataFrame with columns [store, month, revenue, rank]
- `"rolling_3m_avg"`: for each store, 3-month rolling average of revenue (min_periods=1)
- `"pct_of_store_total"`: each row's revenue as a % of that store's total revenue (rounded to 2 dp)
""",
        "grading_criteria": """\
- `profit_margin`: correct formula; rounded to 2 dp; added to df or returned as Series
- `store_rankings`: uses groupby('month')['revenue'].rank(ascending=False); rank 1 is highest; correct shape
- `rolling_3m_avg`: uses groupby('store')['revenue'].transform(lambda x: x.rolling(3,min_periods=1).mean()); returns Series aligned with df
- `pct_of_store_total`: uses groupby('store')['revenue'].transform('sum') as denominator; rounded to 2 dp
- All operations use pandas transform/apply — no manual for loops
""",
        "example_input": "r = store_analysis(df)\nprint(r['pct_of_store_total'].sum().round(2))  # should sum to 300 (100% per store * 3 stores)",
        "example_output": "300.0",
    },
    {
        "title": "[Pandas] Categorical Data & Memory Optimisation",
        "description": """\
Implement `optimise_dataframe(df)` that takes any DataFrame and:

1. Converts object columns with fewer than 20 unique values to `category` dtype
2. Downcasts integer columns to the smallest integer type that fits (use `pd.to_numeric` with downcast)
3. Downcasts float columns to float32 where precision loss is acceptable (< 0.01% relative error for all values)
4. Returns a dict:
   - `"optimised_df"`: the memory-optimised DataFrame
   - `"memory_reduction_pct"`: % reduction in memory usage (rounded to 1 dp)
   - `"dtype_changes"`: dict of column_name → (old_dtype_str, new_dtype_str) for changed columns only

Also implement `categorical_stats(df, cat_col, num_col)` that returns:
- `"value_counts"`: dict of category → count, sorted descending
- `"num_stats_per_cat"`: DataFrame with mean, std, min, max of `num_col` grouped by `cat_col`
""",
        "grading_criteria": """\
- `optimise_dataframe`: correctly identifies object columns with < 20 unique values → category
- Integer downcast: tries int8, int16, int32 in order; uses pd.to_numeric(downcast='integer')
- Float downcast: converts to float32 and checks relative error < 0.01%; uses np.allclose or similar
- `memory_reduction_pct`: uses df.memory_usage(deep=True).sum(); correct percentage formula
- `dtype_changes`: only includes columns that actually changed; correct old/new dtype strings
- `categorical_stats`: correct groupby aggregation; value_counts sorted descending; returns dict and DataFrame
""",
        "example_input": "df = pd.DataFrame({'cat': ['a','b','a','c']*100, 'val': np.random.rand(400)})\nr = optimise_dataframe(df)\nprint(r['dtype_changes']['cat'])",
        "example_output": "('object', 'category')",
    },
    {
        "title": "[Pandas] Reading, Writing & Data Interchange",
        "description": """\
Implement a data pipeline function `etl_pipeline(source_csv_path, output_path)` that:

1. **Extract**: reads a CSV with `pd.read_csv` (handle `sep=','`, `encoding='utf-8'`, parse date columns automatically with `parse_dates=True`, `infer_datetime_format=True`)
2. **Transform**:
   - Drop columns where > 50% of values are NaN
   - Drop rows where ALL non-date columns are NaN
   - Strip whitespace from all string columns
   - Add a column `"row_hash"` = MD5 hash of each row's string representation (use `hashlib`)
3. **Load**: save to `output_path` as a Parquet file (use `df.to_parquet`; requires `pyarrow` or `fastparquet`)
4. **Return** a dict: `{"rows_in": int, "rows_out": int, "cols_dropped": list, "output_path": str}`

Also implement `df_to_json_records(df)` — converts df to a list of JSON-serialisable dicts,
converting Timestamps to ISO strings and NaN to None.
""",
        "grading_criteria": """\
- CSV reading: correct params; parse_dates used; encoding specified
- Column dropping: >50% NaN threshold correct; returns list of dropped column names
- Row dropping: dropna(how='all') on non-date columns only
- Whitespace stripping: uses str.strip() via applymap/apply on object columns only
- `row_hash`: uses hashlib.md5; encodes row string representation; correct hex digest
- Parquet output: df.to_parquet called with output_path; returns correct output_path string
- `df_to_json_records`: Timestamps → .isoformat(); NaN → None; returns list of dicts
- rows_in and rows_out are integers reflecting actual row counts
""",
        "example_input": "# Creates temp CSV and runs pipeline\netl_pipeline('sample.csv', 'output.parquet')",
        "example_output": "{'rows_in': N, 'rows_out': M, 'cols_dropped': [...], 'output_path': 'output.parquet'}",
    },
    {
        "title": "[Pandas] Time Series Resampling & Forecasting Prep",
        "description": """\
Using pandas time series functionality, implement:

1. `resample_ohlc(df, freq)` — given a DataFrame with columns `['timestamp', 'price', 'volume']`
   (timestamp is datetime), resample by `freq` (e.g. 'H', 'D') and return OHLC + total volume.

2. `lag_features(df, col, lags)` — given a DataFrame with a numeric column `col`,
   add lag columns named `f"{col}_lag_{n}"` for each n in `lags`. Return the augmented DataFrame.

3. `detect_seasonality(series, period)` — given a pd.Series with DatetimeIndex and a `period` (int),
   decompose into trend + seasonal + residual using a simple moving average approach.
   Return a DataFrame with columns `['original', 'trend', 'seasonal', 'residual']`.

4. `fill_missing_timestamps(df, freq, fill_method='ffill')` — given a time series DataFrame,
   reindex to a complete DatetimeIndex at `freq`, then forward-fill or back-fill as specified.
""",
        "grading_criteria": """\
- `resample_ohlc`: sets timestamp as index; uses .resample(freq).agg({'price':['first','max','min','last'],'volume':'sum'}); flattens MultiIndex columns
- `lag_features`: uses df[col].shift(n) for each lag; no data leakage (shift > 0); NaN at start accepted
- `detect_seasonality`: trend via rolling mean of `period` window; seasonal = original - trend; residual = original - trend - seasonal; handles NaN edges
- `fill_missing_timestamps`: uses pd.date_range for complete index; .reindex then .ffill() or .bfill(); correct freq string
- All operations use pandas time series methods (resample, shift, rolling, reindex)
""",
        "example_input": "s = pd.Series([1,4,2,5,3,6,2,5], index=pd.date_range('2024-01-01', periods=8, freq='D'))\nresult = detect_seasonality(s, period=4)\nprint(result.columns.tolist())",
        "example_output": "['original', 'trend', 'seasonal', 'residual']",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MATPLOTLIB  (need 8 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Matplotlib] 3D Plotting",
        "description": """\
Using `matplotlib` with `mpl_toolkits.mplot3d`, implement `create_3d_figure()` that saves a
2×2 grid (using a 3D projection for each) to `output_3d.png`:

1. **Top-left**: 3D surface plot of z = sin(√(x²+y²)) for x,y ∈ [-5,5] (50×50 grid), cmap='plasma'
2. **Top-right**: 3D scatter plot of 200 random points (seed=0) inside a sphere of radius 1,
   coloured by z-value with cmap='coolwarm'
3. **Bottom-left**: 3D wireframe of a saddle surface z = x² - y² for x,y ∈ [-2,2]
4. **Bottom-right**: 3D parametric line — a helix: x=cos(t), y=sin(t), z=t/10 for t ∈ [0, 6π]

- Figure size: 14×12 inches; each subplot titled; tight layout; DPI=100; no plt.show()
""",
        "grading_criteria": """\
- All four subplots use Axes3D (projection='3d') created with fig.add_subplot or plt.subplots with subplot_kw
- Top-left: np.meshgrid used; np.sqrt(x**2+y**2) inside sin; plot_surface with cmap='plasma'
- Top-right: random points inside unit sphere (rejection sampling or spherical coords); scatter3D with c=z values and cmap='coolwarm'
- Bottom-left: plot_wireframe on saddle z=x**2-y**2; np.meshgrid used
- Bottom-right: parametric t array with np.linspace; ax.plot(cos,sin,t/10) as a line
- All subplots titled; figure 14×12; tight_layout; saved to output_3d.png at dpi=100
- No plt.show()
""",
        "example_input": "create_3d_figure()  # produces output_3d.png",
        "example_output": "output_3d.png saved",
    },
    {
        "title": "[Matplotlib] Animations",
        "description": """\
Using `matplotlib.animation`, implement `create_animation()` that creates and saves an MP4
(or GIF if ffmpeg unavailable — try MP4 first with `writer='ffmpeg'`, fall back to GIF with `PillowWriter`)
to `wave_animation.mp4` (or `.gif`):

- Animate a sine wave y = sin(2πx - 2πt/20) for x ∈ [0,2] over 40 frames
- The wave should appear to travel left to right
- Include a title showing the current frame number
- Figure size: 8×4; line colour steelblue; y-axis fixed at [-1.5, 1.5]
- Return the filename that was actually saved (`.mp4` or `.gif`)

Also implement `plot_phase_portrait(ax, system_fn, xlim, ylim, n=20)` that draws a phase portrait
(vector field) of a 2D dynamical system using `ax.quiver`. `system_fn(x,y)` returns `(dx,dy)`.
""",
        "grading_criteria": """\
- Uses FuncAnimation from matplotlib.animation
- Sine wave formula correct: y = sin(2*pi*x - 2*pi*frame/20)
- 40 frames; animation saves successfully (returns correct filename)
- Title updates each frame with frame number
- Figure size 8×4; y-axis limits fixed; line colour steelblue
- Graceful fallback: tries ffmpeg writer, catches error, falls back to PillowWriter
- `plot_phase_portrait`: uses np.meshgrid for x,y grid; calls system_fn vectorised or in loop; ax.quiver with normalised arrows
- No plt.show() during save
""",
        "example_input": "fname = create_animation()\nprint(fname.endswith(('.mp4','.gif')))",
        "example_output": "True",
    },
    {
        "title": "[Matplotlib] Interactive Features & Widgets",
        "description": """\
Using `matplotlib.widgets`, implement `interactive_plot()` that creates a figure with:

1. A line plot of y = A·sin(2πf·x + φ) for x ∈ [0, 2]
2. Three sliders (use `matplotlib.widgets.Slider`):
   - Amplitude A ∈ [0.1, 3.0], initial=1.0
   - Frequency f ∈ [0.5, 5.0], initial=1.0
   - Phase φ ∈ [0, 2π], initial=0
3. A "Reset" button that restores all sliders to initial values

The plot must update in real-time as sliders move (connect update callbacks).
Save a static snapshot (sliders at initial values) to `interactive_snapshot.png`.

Also implement `multi_cursor_plot(data_list)` — plots multiple time series on the same axes
with a vertical cursor line that tracks mouse x-position (use `mpl_connect`). Save to `multicursor.png`.
""",
        "grading_criteria": """\
- `interactive_plot`: correct sine formula; 3 Sliders created with correct ranges and labels
- Slider callbacks connected correctly (slider.on_changed); line data updated with set_ydata; fig.canvas.draw_idle() called
- Reset button created with correct initial values; callback restores all three sliders
- Static snapshot saved to interactive_snapshot.png showing default A=1,f=1,φ=0 sine wave
- `multi_cursor_plot`: multiple lines plotted; mpl_connect('motion_notify_event') used; vertical line updates on mouse move; saved to multicursor.png
- No plt.show() in save paths
""",
        "example_input": "interactive_plot()  # produces interactive_snapshot.png",
        "example_output": "interactive_snapshot.png saved",
    },
    {
        "title": "[Matplotlib] Polar, Radar & Speciality Charts",
        "description": """\
Implement `specialty_charts()` that creates and saves a 2×2 figure to `specialty.png`:

1. **Top-left** (polar): Rose chart — plot 8 petals using r = cos(4θ) for θ ∈ [0, 2π].
   Use `subplot(projection='polar')`.

2. **Top-right** (radar/spider): Given skills dict
   `{'Python':9,'SQL':7,'ML':8,'Stats':6,'Viz':8,'Cloud':5}`,
   draw a radar chart (fill the polygon, add labels at each spoke).

3. **Bottom-left**: Broken axis plot — data has an outlier at y=1000 while others are 0-10.
   Show lower range [0,12] and upper range [990,1010] with a break indicator.

4. **Bottom-right**: Streamplot — plot the vector field `(u,v) = (y, -x)` (circular flow)
   for x,y ∈ [-2,2] using `ax.streamplot`.

Figure 14×12, tight_layout, DPI=100, no plt.show().
""",
        "grading_criteria": """\
- Top-left: correct projection='polar'; r=cos(4θ) formula; filled or line rose chart
- Top-right: radar chart manually constructed (angles evenly spaced, close the loop); polygon filled; labels at each vertex
- Bottom-left: two subplots with shared y-axis range trick or GridSpec; diagonal break lines drawn; correct ranges [0,12] and [990,1010]
- Bottom-right: np.meshgrid for x,y; u=y, v=-x correctly; ax.streamplot called with density parameter
- All four subplots titled; figure 14×12; saved to specialty.png; no plt.show()
""",
        "example_input": "specialty_charts()  # produces specialty.png",
        "example_output": "specialty.png saved",
    },
    {
        "title": "[Matplotlib] Log Scales, Twin Axes & Colour Maps",
        "description": """\
Implement `advanced_axes()` that creates and saves a 1×3 figure to `advanced_axes.png`:

1. **Left**: Twin-axis plot — x ∈ [0, 10]:
   - Left y-axis (blue): y = e^x
   - Right y-axis (red): y = x²
   Both plotted on the same x-axis using `ax.twinx()`. Label both y-axes and add a combined legend.

2. **Centre**: Log-log plot of y = x^2.5 for x ∈ [0.1, 1000].
   Mark power-law slope with a reference line. Use `ax.set_xscale('log')` and `ax.set_yscale('log')`.

3. **Right**: Custom discrete colormap — create a 10×10 heatmap where cell (i,j) = (i*j) % 7,
   using a custom ListedColormap of 7 distinct colours. Add a colorbar with integer tick labels 0-6.

Figure: 15×5 inches; each subplot titled; tight_layout; DPI=120; no plt.show().
""",
        "grading_criteria": """\
- Left: ax.twinx() used; both lines plotted on correct axes; y-axis labels coloured to match lines; combined legend (lines from both axes merged)
- Centre: log-log scale set correctly; power-law y=x^2.5 formula correct; reference line drawn with correct slope annotation
- Right: np.arange(10) meshgrid or loop for (i*j)%7; ListedColormap with exactly 7 colours; colorbar with ticks 0-6
- All subplots titled; figure 15×5; saved to advanced_axes.png; tight_layout; no plt.show()
""",
        "example_input": "advanced_axes()  # produces advanced_axes.png",
        "example_output": "advanced_axes.png saved",
    },
    {
        "title": "[Matplotlib] Error Bars, Fill Between & Statistical Plots",
        "description": """\
Using `np.random.seed(42)`, implement `statistical_plots()` that creates a 2×2 figure saved to `stats_plots.png`:

1. **Top-left**: Generate 5 groups, 30 samples each from N(μ_i, 1) for μ ∈ [1,5].
   Plot group means with 95% confidence interval error bars (use scipy.stats.sem or manual SEM×1.96).

2. **Top-right**: Time series with uncertainty band — plot a noisy signal (sin + noise) with
   `fill_between` showing ±1 std band computed from a 10-sample rolling window.

3. **Bottom-left**: Violin plot of the 5 groups from (1) using only matplotlib (ax.violinplot),
   with box-plot overlay (ax.boxplot with whiskers) on the same axes.

4. **Bottom-right**: Q-Q plot — compare 200 samples from a t-distribution (df=5) against
   a standard normal. Plot theoretical quantiles vs sample quantiles; add 45° reference line.

Figure 12×10, tight_layout, DPI=120, no plt.show().
""",
        "grading_criteria": """\
- Top-left: 5 groups generated with correct μ values; error bars use SEM×1.96 (or scipy.stats.sem); errorbar plot not bar chart
- Top-right: rolling std computed (window=10, min_periods=1); fill_between(x, mean-std, mean+std) with alpha
- Bottom-left: ax.violinplot with correct data; ax.boxplot overlaid; both visible on same axes
- Bottom-right: sorts sample quantiles; uses scipy.stats.norm.ppf or manual theoretical quantiles; 45° line plotted
- All subplots titled; figure 12×10; saved to stats_plots.png; no plt.show()
- np.random.seed(42) used at start
""",
        "example_input": "statistical_plots()  # produces stats_plots.png",
        "example_output": "stats_plots.png saved",
    },
    {
        "title": "[Matplotlib] Geographic & Network Visualisation",
        "description": """\
Implement two visualisation functions:

1. `network_graph(adj_matrix, labels)` — given a numpy adjacency matrix and node labels,
   draw a network graph using only matplotlib (no networkx):
   - Layout: circular layout (compute x,y positions from angles)
   - Draw edges as lines with width proportional to weight
   - Draw nodes as circles; label them; colour by degree (number of connections)
   - Save to `network.png`; figure 8×8

2. `bubble_chart(x, y, sizes, labels, colors)` — creates a bubble chart where:
   - Bubble area ∝ `sizes` values (scale to reasonable pixel sizes: 100-2000)
   - Each bubble labelled with `labels[i]` offset above the centre
   - Coloured by `colors` (a numeric array → viridis colormap)
   - Colorbar added; axes labelled "X Axis" and "Y Axis"; title "Bubble Chart"
   - Save to `bubble.png`; figure 10×7
""",
        "grading_criteria": """\
- `network_graph`: circular positions computed with cos/sin; edges drawn as plt.Line2D or ax.plot; edge widths scale with weight; nodes drawn as plt.Circle or scatter; colour by degree using a colormap; labels placed at node positions; saved to network.png
- `bubble_chart`: sizes scaled to [100,2000] range proportionally; ax.scatter with s=sizes and c=colors; colorbar added; labels placed above each bubble with ax.annotate or ax.text; correct axis labels and title; saved to bubble.png
- Both functions work with arbitrary valid inputs
- No networkx, basemap, or geopandas imports
""",
        "example_input": "adj = np.array([[0,1,1],[1,0,2],[1,2,0]])\nnetwork_graph(adj, ['A','B','C'])",
        "example_output": "network.png saved",
    },
    {
        "title": "[Matplotlib] Publication-Ready Figures & LaTeX",
        "description": """\
Implement `publication_figure()` that produces a high-quality single-column figure suitable for
a scientific paper, saved to `pub_figure.pdf` at 300 DPI:

- Use `plt.rcParams` to set: font.family='serif', font.size=11, axes.linewidth=1.2,
  mathtext.fontset='stix'
- Plot three curves: y = sin(x), cos(x), sin(x)cos(x) for x ∈ [0, 2π]
- Use LaTeX-style labels: r'$\\sin(x)$', r'$\\cos(x)$', r'$\\sin(x)\\cos(x)$'
- Add x-axis ticks at [0, π/2, π, 3π/2, 2π] with LaTeX labels: ['0','π/2','π','3π/2','2π']
- Shade the region where sin(x) > cos(x) with light yellow fill
- Add a legend inside the plot (loc='upper right'); grid with alpha=0.3
- Figure size: 5×4 inches (single-column width); tight layout with pad=0.5

Also implement `reset_rcparams()` that restores matplotlib defaults after the publication figure.
""",
        "grading_criteria": """\
- rcParams set correctly for all four specified keys
- Three curves plotted with correct formulas; LaTeX labels correctly formatted
- x-axis ticks at exactly [0, pi/2, pi, 3*pi/2, 2*pi]; labels use LaTeX pi notation
- Shaded region: fill_between where sin(x)>cos(x); correct color; correct condition
- Legend, grid, axis labels all present
- Figure 5×4; tight_layout(pad=0.5); saved to pub_figure.pdf at dpi=300
- `reset_rcparams` calls plt.rcdefaults() or matplotlib.rcdefaults()
- No plt.show()
""",
        "example_input": "publication_figure()  # produces pub_figure.pdf",
        "example_output": "pub_figure.pdf saved",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SEABORN  (need 8 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Seaborn] FacetGrid & Custom Mappings",
        "description": """\
Using the `diamonds` dataset (`sns.load_dataset('diamonds')`), implement `diamond_facets()` that
creates and saves a FacetGrid figure to `diamond_facets.png`:

1. Use `sns.FacetGrid` with `col='cut'` (5 levels) and `row='color'` limited to colors ['D','G','J']
2. Map a scatter plot of `carat` vs `price` onto each facet
3. Add a regression line to each facet using `grid.map_dataframe(sns.regplot, ...)`
4. Set axis labels and a overall title
5. Figure height=3 per facet; aspect=0.8; save at DPI=100

Also implement `custom_palette_plot()` — creates a bar plot of average diamond price by `cut`
using a custom 5-colour palette of your choice (define with `sns.set_palette`). Save to `custom_palette.png`.
""",
        "grading_criteria": """\
- `diamond_facets`: sns.FacetGrid with correct col='cut' and row= limited to ['D','G','J']; scatter mapped correctly
- regplot mapped to each facet (map or map_dataframe); regression line visible
- Axis labels set on FacetGrid (set_axis_labels); super title added (fig.suptitle or plt.subplots_adjust)
- height=3, aspect=0.8 set on FacetGrid; saved to diamond_facets.png at dpi=100
- `custom_palette_plot`: sns.barplot with x='cut', y='price'; custom palette with 5 colours; ci or errorbar shown; saved to custom_palette.png
- No plt.show()
""",
        "example_input": "diamond_facets()  # produces diamond_facets.png",
        "example_output": "diamond_facets.png saved",
    },
    {
        "title": "[Seaborn] Regression & Residual Plots",
        "description": """\
Using the `mpg` dataset (`sns.load_dataset('mpg')`), implement `regression_analysis()` that
creates and saves a 2×2 figure to `regression.png`:

1. **Top-left**: `sns.regplot` of `weight` vs `mpg` with 95% CI band; add correlation coefficient in title
2. **Top-right**: `sns.residplot` of the same regression; add a horizontal zero-line
3. **Bottom-left**: `sns.lmplot` of `horsepower` vs `mpg`, faceted by `origin` in a single row
4. **Bottom-right**: Polynomial regression (order=2) of `model_year` vs `mpg` using `sns.regplot(order=2)`

Drop rows with NaN before plotting.
Also return a dict `"correlations"` — Pearson correlation of mpg with each numeric column (rounded to 3 dp).
""",
        "grading_criteria": """\
- NaN rows dropped; mpg dataset loaded correctly
- Top-left: regplot with correct x/y; CI band visible; correlation computed and shown in title
- Top-right: residplot correct; zero horizontal line (ax.axhline(0)) added
- Bottom-left: lmplot with hue='origin' or col='origin'; displayed in single row; correct axes
- Bottom-right: order=2 polynomial regression; regplot call correct
- `correlations` dict: correct Pearson r values for all numeric columns; rounded to 3 dp
- All plots saved to regression.png; no plt.show()
""",
        "example_input": "r = regression_analysis()\nprint('mpg' in r['correlations'])",
        "example_output": "True (mpg correlates with itself = 1.0)",
    },
    {
        "title": "[Seaborn] Cluster Maps & Hierarchical Heatmaps",
        "description": """\
Using the `flights` dataset (`sns.load_dataset('flights')`):

1. Implement `flight_heatmap()`:
   - Pivot the data to a month × year matrix of passenger counts
   - Create a `sns.heatmap` with annotations, cmap='YlOrRd', linewidths=0.5
   - Save to `flight_heatmap.png`

2. Implement `flight_clustermap()`:
   - On the same pivot table, create a `sns.clustermap` with:
     - `method='ward'`, `metric='euclidean'`
     - `standard_scale=1` (scale columns)
     - `figsize=(10,8)`, cmap='vlag'
   - Save to `flight_clustermap.png`

3. Implement `correlation_clustermap(df)`:
   - Given any numeric DataFrame, computes the correlation matrix
   - Plots a clustermap of correlations with a diverging colormap, center=0
   - Saves to `correlation_clustermap.png`; returns the linkage order of rows as a list of column names
""",
        "grading_criteria": """\
- `flight_heatmap`: correct pivot with months as index; sns.heatmap with annot=True, fmt='d'; correct cmap and linewidths; saved to flight_heatmap.png
- `flight_clustermap`: sns.clustermap with method='ward', metric='euclidean', standard_scale=1; correct figsize and cmap; saved to flight_clustermap.png
- `correlation_clustermap`: .corr() computed; sns.clustermap with diverging cmap and center=0; returns row_order as list of column names from clustermap.dendrogram_row.reordered_ind; saved to correlation_clustermap.png
- All functions: no plt.show(); correct file names
""",
        "example_input": "flight_clustermap()  # produces flight_clustermap.png",
        "example_output": "flight_clustermap.png saved",
    },
    {
        "title": "[Seaborn] Distribution Plots & Comparison",
        "description": """\
Using the `titanic` dataset (`sns.load_dataset('titanic')`), implement `titanic_distributions()` that
creates a 2×3 figure saved to `titanic_dist.png`:

1. KDE plot of `age` coloured by `survived` (0 vs 1), with shaded fill
2. ECDF plot of `fare` for each passenger class (`pclass`)
3. Histogram + KDE of `age` split by `sex` on the same axes (two overlapping histograms)
4. Ridgeline-style plot: for each `pclass`, offset KDE of `fare` vertically (3 KDE curves stacked)
5. Box plot + strip plot overlay of `age` by `pclass` and `survived`
6. Violin plot of `fare` by `embarkation port` (`embarked`), with inner='quartile'

Drop NaN from relevant columns before each plot.
Figure 18×12, tight_layout, DPI=100, no plt.show().
""",
        "grading_criteria": """\
- All 6 subplots present and correctly positioned in 2×3 grid
- Subplot 1: sns.kdeplot with hue='survived', fill=True; correct x='age'
- Subplot 2: sns.ecdfplot with hue='pclass'; correct x='fare'
- Subplot 3: sns.histplot with hue='sex', kde=True; both sexes visible
- Subplot 4: manual KDE offsets using vertical shift (y + offset per class); 3 visible curves
- Subplot 5: sns.boxplot + sns.stripplot overlaid on same ax; hue used for survived
- Subplot 6: sns.violinplot with x='embarked', y='fare', inner='quartile'
- NaN handling per subplot; figure 18×12; saved to titanic_dist.png
""",
        "example_input": "titanic_distributions()  # produces titanic_dist.png",
        "example_output": "titanic_dist.png saved",
    },
    {
        "title": "[Seaborn] Joint Plots & Marginal Distributions",
        "description": """\
Using seaborn and the `penguins` dataset, implement `joint_analysis()` that creates and saves:

1. `joint_scatter.png`: A `sns.JointGrid` of `bill_length_mm` vs `flipper_length_mm`,
   coloured by `species`. Center: scatter; marginals: KDE per species.

2. `joint_hex.png`: A `sns.jointplot` with `kind='hex'` of `body_mass_g` vs `flipper_length_mm`.
   Include marginal histograms.

3. `joint_kde.png`: A bivariate KDE (`kind='kde'`) jointplot of `bill_length_mm` vs `bill_depth_mm`,
   coloured by `species` with `fill=True`.

Also implement `pairwise_correlation_matrix(df)` — returns a DataFrame of Spearman rank
correlations (not Pearson) for all numeric columns, rounded to 3 dp.

Drop NaN rows. No plt.show(). Each figure saved separately.
""",
        "grading_criteria": """\
- All three joint plots saved to correct filenames
- `joint_scatter.png`: JointGrid created manually; plot_joint(sns.scatterplot, hue='species'); plot_marginals(sns.kdeplot, hue='species')
- `joint_hex.png`: sns.jointplot with kind='hex'; correct x,y columns; marginal histograms present
- `joint_kde.png`: sns.jointplot with kind='kde'; hue='species'; fill=True applied
- `pairwise_correlation_matrix`: uses .corr(method='spearman'); returns DataFrame; rounded to 3 dp; square matrix
- NaN rows dropped; no plt.show()
""",
        "example_input": "pairwise_correlation_matrix(sns.load_dataset('penguins').dropna())",
        "example_output": "DataFrame of shape (4,4) with Spearman correlations on numeric columns",
    },
    {
        "title": "[Seaborn] Categorical Deep Dive",
        "description": """\
Using the `exercise` dataset (`sns.load_dataset('exercise')`), implement `exercise_analysis()` that
creates and saves a 2×2 figure to `exercise.png`:

1. **Top-left**: `sns.pointplot` of `pulse` vs `time`, hued by `kind` (type of exercise), with error bars
2. **Top-right**: `sns.barplot` of `pulse` by `diet` and `kind` (grouped bars); add value labels above each bar
3. **Bottom-left**: `sns.swarmplot` of `pulse` by `time`, hued by `kind`; add median lines manually
4. **Bottom-right**: `sns.catplot`-style: for each `diet`, show a box plot of `pulse` by `kind` side by side

Also implement `count_heatmap(df, col1, col2)` — creates a heatmap of value counts for
combinations of `col1` and `col2`, with annotations, saved to `count_heatmap.png`.
""",
        "grading_criteria": """\
- Dataset loaded with sns.load_dataset('exercise')
- Top-left: pointplot with correct x,y,hue; error bars visible (ci or errorbar parameter)
- Top-right: barplot with both diet and kind grouping (hue=); value labels added with ax.bar_label or ax.text
- Bottom-left: swarmplot with correct params; median lines added per group using ax.hlines or ax.plot
- Bottom-right: box plot faceted by diet and kind; side-by-side comparison visible
- `count_heatmap`: pd.crosstab used; sns.heatmap with annot=True; saved to count_heatmap.png
- Figure 14×12; tight_layout; no plt.show()
""",
        "example_input": "exercise_analysis()  # produces exercise.png",
        "example_output": "exercise.png saved",
    },
    {
        "title": "[Seaborn] Theming, Styling & Custom Aesthetics",
        "description": """\
Implement `styled_dashboard()` that demonstrates seaborn's theming system.
Creates a 2×2 figure saved to `styled.png` where each subplot uses a different seaborn style:

1. **Top-left**: `style='darkgrid'` — line plot of sin/cos with palette='deep'
2. **Top-right**: `style='whitegrid'` — bar chart with palette='pastel'
3. **Bottom-left**: `style='ticks'` — scatter plot with palette='bright'
4. **Bottom-right**: `style='white'` — KDE plot with palette='muted'

Each subplot should use `with sns.axes_style(style):` context manager.

Also implement `custom_theme(base_style='darkgrid', font_scale=1.2, palette='husl', n_colors=8)` that:
- Sets the seaborn theme globally
- Returns a list of `n_colors` hex colour strings from the palette
- Creates a swatch plot of the colours saved to `swatch.png`
""",
        "grading_criteria": """\
- `styled_dashboard`: uses `with sns.axes_style(style):` for each subplot; correct style name per panel; different seaborn palettes used; all 4 subplots titled
- Each subplot contains appropriate chart type (line, bar, scatter, kde) with sample data
- Figure 14×12; tight_layout; saved to styled.png; no plt.show()
- `custom_theme`: sns.set_theme or sns.set called with correct params; sns.color_palette(palette, n_colors) returns colour list; colours converted to hex
- Swatch plot: horizontal bars or circles showing each colour; saved to swatch.png; no plt.show()
""",
        "example_input": "colours = custom_theme('darkgrid', 1.2, 'husl', 8)\nprint(len(colours), colours[0].startswith('#'))",
        "example_output": "8 True",
    },
    {
        "title": "[Seaborn] Advanced: Multi-Dataset Comparison Dashboard",
        "description": """\
Build a comprehensive data comparison dashboard using three seaborn built-in datasets.
Implement `comparison_dashboard()` that creates a 3×3 figure saved to `dashboard.png`:

Row 1 (Tips dataset): violin | strip+box overlay | heatmap of correlations
Row 2 (Titanic dataset): survival rate bar by pclass+sex | age KDE by survived | fare by class ecdf
Row 3 (Iris/Penguins dataset): species scatter matrix diagonal | pair scatter (2 features) | 3-species KDE comparison

Requirements:
- Each cell has a descriptive title and labelled axes
- Use appropriate seaborn plot types for each cell
- Consistent colour palette across rows (one palette per dataset row)
- Figure 18×16 inches; DPI=100; tight_layout with hspace=0.4, wspace=0.3
- No plt.show(); saved to dashboard.png

Return a dict `"plot_types"` mapping subplot position (e.g. "(0,0)") → seaborn function name used.
""",
        "grading_criteria": """\
- 9 subplots in 3×3 grid; all present and non-empty
- Row 1: tips data; violin, combined strip/box, correlation heatmap
- Row 2: titanic data; bar (survival rate), KDE (age by survived), ECDF (fare by class)
- Row 3: iris or penguins; scatter, pair scatter, KDE comparison
- Each subplot has title and axis labels
- Consistent palette per row (sns.color_palette used per row)
- Figure 18×16; tight_layout with correct hspace/wspace; saved to dashboard.png; no plt.show()
- `plot_types` dict returned with 9 entries; keys are string position labels
""",
        "example_input": "r = comparison_dashboard()\nprint(len(r['plot_types']))",
        "example_output": "9",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SCIKIT-LEARN  (need 7 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Scikit-learn] Support Vector Machines",
        "description": """\
Using scikit-learn, implement `svm_analysis()` on the breast cancer dataset (`load_breast_cancer()`):

1. Split 80/20 (random_state=42, stratified)
2. Scale with `StandardScaler`
3. Train three SVMs: `SVC(kernel='linear')`, `SVC(kernel='rbf')`, `SVC(kernel='poly', degree=3)`
4. For each, compute: accuracy, precision, recall, F1 (all weighted, rounded to 4 dp)
5. Perform a `GridSearchCV` (cv=5) on `SVC(kernel='rbf')` over:
   `{'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}`
6. Return a dict:
   - `"results"`: dict of kernel_name → {accuracy, precision, recall, f1}
   - `"best_params"`: best params from GridSearchCV
   - `"best_cv_score"`: best CV score rounded to 4 dp
   - `"confusion_matrix"`: confusion matrix for best RBF model as a nested list
""",
        "grading_criteria": """\
- Correct dataset loading, splitting, scaling
- All three SVMs trained and evaluated; metrics use average='weighted'
- GridSearchCV on RBF kernel with specified param_grid, cv=5, scoring='accuracy'
- best_params returned as a dict from grid_search.best_params_
- best_cv_score ≥ 0.95 (breast cancer is well-separable)
- confusion_matrix: 2×2 matrix as nested list; computed on test set with best params
- No data leakage
""",
        "example_input": "r = svm_analysis()\nprint(r['best_params'])\nprint(r['best_cv_score'])",
        "example_output": "{'C': 1 or 10, 'gamma': 'scale'} (or similar)\n≥ 0.95",
    },
    {
        "title": "[Scikit-learn] Natural Language Processing Pipeline",
        "description": """\
Using scikit-learn's text feature extraction, implement `text_classifier()` on the 20 Newsgroups dataset:

1. Load only 4 categories: `['sci.med','sci.space','rec.sport.hockey','rec.sport.baseball']`
   using `fetch_20newsgroups(subset='train', categories=..., remove=('headers','footers','quotes'))`
2. Build a Pipeline:
   - `TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')`
   - `MultinomialNB(alpha=0.1)`
3. Train and evaluate on the corresponding test set
4. Return a dict:
   - `"accuracy"`: float rounded to 4 dp
   - `"top_features_per_class"`: dict of category → list of top-5 feature names (by TF-IDF weight)
   - `"classification_report"`: string from classification_report
   - `"most_confused_pair"`: tuple of (true_label, predicted_label) that appears most in the confusion matrix off-diagonal
""",
        "grading_criteria": """\
- Correct 4 categories loaded; headers/footers/quotes removed
- Pipeline with TfidfVectorizer (correct params) and MultinomialNB
- Accuracy ≥ 0.85 on test set (these categories are fairly distinct)
- top_features_per_class: uses clf.feature_log_prob_ and vectorizer.get_feature_names_out(); correct top-5 per class
- most_confused_pair: identifies the off-diagonal cell with highest count; returns (true, pred) as strings
- No manual text preprocessing beyond TfidfVectorizer's built-in
""",
        "example_input": "r = text_classifier()\nprint(r['accuracy'])\nprint(len(r['top_features_per_class']))",
        "example_output": "≥ 0.85\n4",
    },
    {
        "title": "[Scikit-learn] Feature Engineering & Selection",
        "description": """\
Using the California Housing dataset, implement `feature_engineering()`:

1. Load the dataset; create a DataFrame with feature names
2. Engineer new features:
   - `rooms_per_household` = AveRooms / AveOccup
   - `bedrooms_ratio` = AveBedrms / AveRooms
   - `population_density` = Population / AveOccup
3. Select features using three methods and compare:
   - `SelectKBest(f_regression, k=5)`
   - `RFE(LinearRegression(), n_features_to_select=5)`
   - `feature_importances_` from `GradientBoostingRegressor(n_estimators=50, random_state=42)`
4. Return a dict:
   - `"selected_kbest"`: list of 5 selected feature names
   - `"selected_rfe"`: list of 5 selected feature names
   - `"top5_gbm"`: list of 5 most important features from GBM (descending importance)
   - `"agreement"`: set of features selected by all 3 methods
""",
        "grading_criteria": """\
- Dataset loaded correctly; 3 new features engineered and added to DataFrame
- SelectKBest: fitted on training data; correct 5 feature names from get_support()
- RFE: fitted on training data; 5 features from get_support()
- GBM: fitted on training data; top 5 by feature_importances_ descending
- agreement: Python set intersection of all three; may be empty — that's ok
- All feature selection fitted on train set only (80/20 split, random_state=42)
- Engineered features included in the feature set for selection
""",
        "example_input": "r = feature_engineering()\nprint(len(r['selected_kbest']))\nprint(type(r['agreement']))",
        "example_output": "5\n<class 'set'>",
    },
    {
        "title": "[Scikit-learn] Ensemble Methods & Stacking",
        "description": """\
Using the Wine dataset (`load_wine()`), implement `ensemble_comparison()`:

1. Split 80/20, stratified, random_state=42
2. Train and evaluate (accuracy, rounded to 4 dp) on test set:
   - `RandomForestClassifier(n_estimators=100, random_state=42)`
   - `GradientBoostingClassifier(n_estimators=100, random_state=42)`
   - `AdaBoostClassifier(n_estimators=100, random_state=42)`
   - `VotingClassifier` (hard voting) combining all three above
   - `StackingClassifier` with the three above as estimators and `LogisticRegression` as final estimator
3. Return a dict:
   - `"accuracies"`: dict model_name → accuracy
   - `"best_model"`: name of model with highest accuracy
   - `"feature_importances"`: from RandomForest — dict feature_name → importance (top 5, sorted desc)
   - `"stacking_cv_score"`: 5-fold CV accuracy of the stacking classifier on the full dataset
""",
        "grading_criteria": """\
- All 5 models trained and evaluated; accuracy ≥ 0.90 for RF and GB (Wine is easy)
- VotingClassifier uses voting='hard' with all 3 estimators
- StackingClassifier uses all 3 as estimators with LogisticRegression() final
- best_model is the model name string with highest test accuracy
- feature_importances: correct feature names from load_wine().feature_names; top 5; sorted desc
- stacking_cv_score: cross_val_score on StackingClassifier, cv=5, scoring='accuracy'; mean score rounded to 4 dp
""",
        "example_input": "r = ensemble_comparison()\nprint(list(r['accuracies'].keys()))\nprint(r['best_model'])",
        "example_output": "['RandomForest', 'GradientBoosting', 'AdaBoost', 'Voting', 'Stacking']\n(one of the above)",
    },
    {
        "title": "[Scikit-learn] Cross-Validation Strategies & Learning Curves",
        "description": """\
Using the Digits dataset (`load_digits()`), implement `validation_analysis()`:

1. Compare three CV strategies on a `RandomForestClassifier(n_estimators=50, random_state=42)`:
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - `LeaveOneOut()` — only on the first 100 samples (it's slow)
2. Generate a learning curve: train sizes from 10% to 100% in 5 steps; compute mean train/val scores
3. Generate a validation curve: vary `n_estimators` in [10, 50, 100, 200]; compute mean train/val scores
4. Return a dict:
   - `"cv_scores"`: dict of strategy_name → list of fold scores (rounded to 4 dp)
   - `"learning_curve"`: dict with `"train_sizes"`, `"train_scores"`, `"val_scores"` (all lists, rounded to 4 dp)
   - `"validation_curve"`: dict with `"param_values"`, `"train_scores"`, `"val_scores"` (all lists, rounded to 4 dp)
   - `"best_n_estimators"`: n_estimators value with highest mean val score in validation curve
""",
        "grading_criteria": """\
- All three CV strategies used correctly with cross_val_score
- KFold and StratifiedKFold on full dataset; LOO on first 100 samples
- learning_curve uses sklearn.model_selection.learning_curve; train_sizes computed correctly
- validation_curve uses sklearn.model_selection.validation_curve with param_name='n_estimators'
- All scores are means over folds; rounded to 4 dp; represented as lists
- best_n_estimators: n_estimators from validation_curve param_values with max mean val score
- StratifiedKFold scores should be ≥ 0.90 for Digits with RF(50)
""",
        "example_input": "r = validation_analysis()\nprint(list(r['cv_scores'].keys()))\nprint(len(r['learning_curve']['train_sizes']))",
        "example_output": "['KFold', 'StratifiedKFold', 'LeaveOneOut']\n5",
    },
    {
        "title": "[Scikit-learn] Anomaly Detection & Outlier Analysis",
        "description": """\
Implement `anomaly_detection()` using three methods on a synthetic dataset:

1. Generate data with `np.random.seed(42)`:
   - 500 normal points from N(0, 1) in 2D
   - 20 outlier points from Uniform(-8, 8) in 2D
   - Combine into X (520×2) with true labels: 1=normal, -1=outlier

2. Fit and predict with:
   - `IsolationForest(contamination=20/520, random_state=42)`
   - `LocalOutlierFactor(n_neighbors=20, contamination=20/520)` (novelty=False)
   - `OneClassSVM(nu=20/520, kernel='rbf', gamma='auto')`

3. Return a dict:
   - `"results"`: dict model_name → `{"precision": float, "recall": float, "f1": float}` on outlier class (-1), rounded to 4 dp
   - `"best_model"`: model with highest F1 on the outlier class
   - `"isolation_forest_scores"`: anomaly scores for all 520 points (list of floats rounded to 4 dp)
   - `"plot_saved"`: True after saving scatter plot of all three detection results to `anomaly.png`
""",
        "grading_criteria": """\
- Data generated correctly with seed 42; correct shapes (500+20=520 samples, 2 features)
- All three models instantiated with correct contamination and params
- Metrics computed with precision_score, recall_score, f1_score with pos_label=-1
- best_model: model name with highest F1 for outlier class
- isolation_forest_scores: .score_samples() or .decision_function(); list of 520 floats
- anomaly.png: 3-panel scatter plot; normal vs detected-outlier coloured differently per model; saved correctly
- No plt.show()
""",
        "example_input": "r = anomaly_detection()\nprint(r['best_model'])\nprint(len(r['isolation_forest_scores']))",
        "example_output": "(one of the three model names)\n520",
    },
    {
        "title": "[Scikit-learn] Neural Networks with MLPClassifier",
        "description": """\
Using scikit-learn's `MLPClassifier` on the MNIST dataset (load via `fetch_openml('mnist_784', version=1)`
or use `load_digits()` as a lighter substitute — use `load_digits()` to avoid long download):

Implement `mlp_analysis()`:

1. Load `load_digits()`; split 80/20, stratified, random_state=42; scale with StandardScaler
2. Train three MLP architectures:
   - Small: `MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)`
   - Medium: `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)`
   - Large: `MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=200, random_state=42)`
3. For each: test accuracy, loss curve (list of loss values per iteration)
4. Return a dict:
   - `"accuracies"`: dict of arch_name → accuracy (rounded to 4 dp)
   - `"loss_curves"`: dict of arch_name → list of loss values (every 10th value, rounded to 4 dp)
   - `"best_architecture"`: name of highest accuracy model
   - `"convergence_iterations"`: dict of arch_name → number of iterations until convergence (n_iter_)
""",
        "grading_criteria": """\
- load_digits() used; correct split and scaling
- All three MLPClassifiers trained; accuracy ≥ 0.95 for at least the medium model
- loss_curves: clf.loss_curve_ sampled every 10th element (indices 0,10,20,...); list of floats
- best_architecture: name string with highest test accuracy
- convergence_iterations: clf.n_iter_ for each model; returns int per model
- No TensorFlow or PyTorch — sklearn MLP only
""",
        "example_input": "r = mlp_analysis()\nprint(list(r['accuracies'].keys()))\nprint(r['best_architecture'])",
        "example_output": "['Small', 'Medium', 'Large']\n'Medium' or 'Large'",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # TENSORFLOW  (need 7 more → total 10)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[TensorFlow] Convolutional Neural Network — CIFAR-10",
        "description": """\
Build and train a CNN on CIFAR-10 using tf.keras. Implement `train_cnn()`:

1. Load `tf.keras.datasets.cifar10`; normalise to [0,1]; one-hot encode labels
2. Build a model:
   - Conv2D(32, (3,3), activation='relu', padding='same') + BatchNormalization + MaxPooling2D(2,2)
   - Conv2D(64, (3,3), activation='relu', padding='same') + BatchNormalization + MaxPooling2D(2,2)
   - Conv2D(128,(3,3), activation='relu', padding='same') + BatchNormalization
   - GlobalAveragePooling2D
   - Dense(256, activation='relu') + Dropout(0.4)
   - Dense(10, activation='softmax')
3. Compile: Adam(lr=0.001), categorical_crossentropy, accuracy
4. Train: 10 epochs, batch_size=64, validation_split=0.1
5. Return: `{"test_accuracy": float (≥0.70), "test_loss": float, "params": int (total trainable params), "history": dict of final epoch metrics}`
""",
        "grading_criteria": """\
- CIFAR-10 loaded; normalised to [0,1]; labels one-hot encoded with to_categorical
- Exact architecture: Conv→BN→Pool (×2), Conv→BN, GlobalAveragePooling, Dense+Dropout, Dense softmax
- Compiled with Adam(0.001), categorical_crossentropy, accuracy metric
- 10 epochs, batch_size=64, validation_split=0.1
- test_accuracy ≥ 0.70 (achievable in 10 epochs with this architecture)
- params: model.count_params() returned as int
- history: dict with val_accuracy, val_loss, accuracy, loss for final epoch
""",
        "example_input": "r = train_cnn()\nprint(r['test_accuracy'] >= 0.70)\nprint('params' in r)",
        "example_output": "True\nTrue",
    },
    {
        "title": "[TensorFlow] Recurrent Neural Networks — Text Sequence",
        "description": """\
Build an LSTM-based sentiment classifier using tf.keras. Implement `train_lstm()`:

1. Load `tf.keras.datasets.imdb` with `num_words=10000`
2. Pad sequences to `maxlen=200` with `tf.keras.preprocessing.sequence.pad_sequences`
3. Build a Sequential model:
   - Embedding(10000, 128, input_length=200)
   - LSTM(64, return_sequences=True)
   - LSTM(32)
   - Dense(1, activation='sigmoid')
4. Compile: Adam(lr=0.001), binary_crossentropy, accuracy
5. Train: 5 epochs, batch_size=256, validation_split=0.1
6. Return: `{"test_accuracy": float (≥0.85), "test_loss": float, "model_params": int, "epoch_accuracies": list of 5 training accuracies}`
""",
        "grading_criteria": """\
- IMDB dataset loaded with num_words=10000; sequences padded to maxlen=200
- Exact architecture: Embedding(10000,128,200), LSTM(64, return_sequences=True), LSTM(32), Dense(1,sigmoid)
- Compiled with Adam(0.001), binary_crossentropy, accuracy
- 5 epochs, batch_size=256, validation_split=0.1
- test_accuracy ≥ 0.85 (standard LSTM on IMDB achieves 85-90%)
- epoch_accuracies: list of 5 floats from history.history['accuracy']
- model_params: model.count_params() as int
""",
        "example_input": "r = train_lstm()\nprint(r['test_accuracy'] >= 0.85)\nprint(len(r['epoch_accuracies']))",
        "example_output": "True\n5",
    },
    {
        "title": "[TensorFlow] Transfer Learning with MobileNetV2",
        "description": """\
Implement `transfer_learning_demo()` using MobileNetV2 as a feature extractor on CIFAR-10:

1. Load CIFAR-10; resize images from 32×32 to 96×96 using `tf.image.resize`; normalise to [0,1]
   (use only 5000 train samples and 1000 test samples to keep it fast)
2. Load `tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(96,96,3))`
3. Freeze all MobileNetV2 layers
4. Add a head: GlobalAveragePooling2D + Dense(128, relu) + Dropout(0.3) + Dense(10, softmax)
5. Compile: Adam(lr=0.001), sparse_categorical_crossentropy, accuracy
6. Train for 5 epochs, batch_size=32
7. Return: `{"test_accuracy": float, "trainable_params": int, "frozen_params": int, "note": "trainable_params should be << frozen_params"}`
""",
        "grading_criteria": """\
- CIFAR-10 loaded; resized to 96×96; subsets of 5000 train and 1000 test used
- MobileNetV2 loaded with include_top=False, weights='imagenet', correct input shape
- All MobileNetV2 layers frozen (trainable=False on base model)
- Head architecture correct: GAP + Dense(128,relu) + Dropout(0.3) + Dense(10,softmax)
- trainable_params ≈ head only (128*1280+128 + 10*128+10 ≈ 165,000); frozen_params ≈ 2.2M
- test_accuracy > 0.5 (expected given only 5 epochs and limited data)
- note string present in return dict
""",
        "example_input": "r = transfer_learning_demo()\nprint(r['trainable_params'] < r['frozen_params'])",
        "example_output": "True",
    },
    {
        "title": "[TensorFlow] Custom Layers & Model Subclassing",
        "description": """\
Implement custom TensorFlow components:

1. `class ResidualBlock(tf.keras.layers.Layer)` — a residual block with:
   - Two Dense layers (units=64, activation='relu') with BatchNormalization after each
   - A skip connection that adds the input to the output (add a projection Dense if shapes differ)
   - Implement `build()` and `call()` methods

2. `class SimpleTransformerBlock(tf.keras.layers.Layer)` — a simplified transformer block:
   - MultiHeadAttention(num_heads=2, key_dim=32)
   - Add & Norm (LayerNormalization)
   - FFN: Dense(64,relu) + Dense(units)
   - Add & Norm
   - Implement `call(x, training=False)`

3. `build_residual_model(input_dim, n_blocks=3)` — builds a model using `n_blocks` ResidualBlocks
   followed by a Dense(1) output; compile with Adam; return the model

4. Verify the model trains on synthetic data (X=random (100,32), y=random binary labels) for 2 epochs
   without error. Return `{"trained": True, "output_shape": (None,1)}`
""",
        "grading_criteria": """\
- `ResidualBlock`: correct build() initialises Dense and BN layers; call() implements skip connection; handles shape mismatch with projection
- `SimpleTransformerBlock`: MultiHeadAttention used correctly; two Add+Norm sublayers; FFN with correct units
- `build_residual_model`: uses n_blocks ResidualBlocks in sequence; Dense(1) output; compiled with Adam
- Model trains for 2 epochs without error on synthetic data
- Both classes extend tf.keras.layers.Layer; `__init__` calls super().__init__()
- Return dict contains trained=True and correct output_shape
""",
        "example_input": "r = build_residual_model(32, n_blocks=3)\nprint(r['trained'])",
        "example_output": "True",
    },
    {
        "title": "[TensorFlow] Data Pipelines with tf.data",
        "description": """\
Implement efficient data pipelines using `tf.data.Dataset`. Implement `build_pipeline()`:

1. Create a `tf.data.Dataset` from numpy arrays X (1000×10 random floats, seed=42) and y (1000 random binary labels)
2. Build a full preprocessing pipeline:
   - `.shuffle(buffer_size=1000, seed=42)`
   - `.batch(32)`
   - `.map()` a normalisation function: standardise each feature to zero mean/unit std (compute stats on the full dataset first)
   - `.prefetch(tf.data.AUTOTUNE)`
3. Build a second pipeline that reads from a list of filenames — implement `create_tfrecord_pipeline(filenames)` that creates a dataset reading TFRecord files (assume each has features 'x' float32 vector and 'y' int64 scalar)
4. Return a dict:
   - `"batches_per_epoch"`: number of batches in the dataset (1000//32 = 31 with drop_remainder=False)
   - `"first_batch_shape"`: shape of X in the first batch as a tuple
   - `"pipeline_steps"`: list of transformation names applied (e.g. ['shuffle','batch','map','prefetch'])
   - `"sample_output"`: first 3 y values from the first batch as a Python list of ints
""",
        "grading_criteria": """\
- tf.data.Dataset.from_tensor_slices used; correct shapes
- .shuffle with buffer_size=1000 and seed=42; .batch(32); .prefetch(tf.data.AUTOTUNE) — all applied in correct order
- Normalisation map function computes mean/std on the full dataset BEFORE creating the pipeline (no leakage); applied in .map()
- `create_tfrecord_pipeline`: tf.data.TFRecordDataset used; feature_description dict defined; tf.io.parse_single_example used in .map()
- batches_per_epoch = 32 (1000/32 = 31.25 → 32 batches with ceil, or 31 with floor — accept either, state which)
- first_batch_shape is (32, 10) or (batch_size, 10)
- pipeline_steps lists all 4 transformations
""",
        "example_input": "r = build_pipeline()\nprint(r['first_batch_shape'])\nprint(r['pipeline_steps'])",
        "example_output": "(32, 10)\n['shuffle', 'batch', 'map', 'prefetch']",
    },
    {
        "title": "[TensorFlow] Model Saving, Loading & TF-Serving Prep",
        "description": """\
Implement `save_load_workflow()` that demonstrates the full model lifecycle:

1. Build a simple model: Dense(64,relu) → Dense(32,relu) → Dense(1) on input_shape=(10,)
2. Compile with Adam, MSE loss; train for 3 epochs on synthetic data (X: 200×10, y: 200×1, seed=42)
3. Save the model in **three formats**:
   - `model.save('saved_model/')` — TensorFlow SavedModel format
   - `model.save('model.keras')` — Keras format
   - Save weights only: `model.save_weights('weights/model_weights')`
4. Reload from each format and verify predictions match original (within 1e-5)
5. Convert to TFLite: `tf.lite.TFLiteConverter.from_saved_model('saved_model/')` → save to `model.tflite`
6. Return: `{"formats_saved": list, "predictions_match": bool, "tflite_size_kb": float, "tflite_output_sample": float}`
""",
        "grading_criteria": """\
- All three save formats used correctly; no errors
- SavedModel format: tf.keras.models.load_model('saved_model/') works
- Keras format: tf.keras.models.load_model('model.keras') works
- Weights-only: model rebuilt before loading weights (same architecture); load_weights called correctly
- Predictions compared with np.allclose(tol=1e-5); predictions_match=True
- TFLite conversion: converter created from saved_model; .convert() called; saved to model.tflite
- tflite_size_kb: os.path.getsize('model.tflite') / 1024; float rounded to 2 dp
- formats_saved: list of 3 strings ['SavedModel', 'Keras', 'Weights']
""",
        "example_input": "r = save_load_workflow()\nprint(r['predictions_match'])\nprint(r['formats_saved'])",
        "example_output": "True\n['SavedModel', 'Keras', 'Weights']",
    },
    {
        "title": "[TensorFlow] Generative Model — Variational Autoencoder",
        "description": """\
Build a simple Variational Autoencoder (VAE) on MNIST. Implement `train_vae()`:

1. Load MNIST; normalise to [0,1]; flatten to (784,); use 10000 samples for speed
2. Build Encoder: Dense(256,relu) → Dense(128,relu) → two parallel Dense(32) heads: z_mean and z_log_var
3. Implement the reparametrisation trick: z = z_mean + exp(0.5*z_log_var) * epsilon (epsilon ~ N(0,I))
4. Build Decoder: Dense(128,relu) → Dense(256,relu) → Dense(784, sigmoid)
5. Custom loss = reconstruction loss (binary crossentropy) + KL divergence:
   KL = -0.5 * sum(1 + z_log_var - z_mean² - exp(z_log_var))
6. Train for 5 epochs, batch_size=128
7. Return: `{"final_loss": float, "reconstruction_loss": float, "kl_loss": float, "generated_sample_shape": (784,)}`
""",
        "grading_criteria": """\
- MNIST normalised to [0,1]; flattened; 10000 samples used
- Encoder produces z_mean and z_log_var with correct shapes (batch, 32)
- Reparametrisation trick implemented correctly with tf.random.normal
- Decoder reconstructs to (784,) with sigmoid activation
- Custom training step or model with custom train_step; both loss components present
- KL divergence formula correct: -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
- Training completes 5 epochs; final losses are finite (not NaN)
- generated_sample_shape is (784,) — from passing a random z through the decoder
""",
        "example_input": "r = train_vae()\nprint(r['generated_sample_shape'])\nprint(r['final_loss'] < 200)  # reasonable loss for VAE on MNIST",
        "example_output": "(784,)\nTrue",
    },
]


class Command(BaseCommand):
    help = "Load additional Python coding challenges (round 2 — brings each category to 10+)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--flush",
            action="store_true",
            help="Delete ALL existing problems before loading.",
        )

    def handle(self, *args, **options):
        if options["flush"]:
            count, _ = Problem.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Deleted {count} existing problem(s)."))

        created = 0
        skipped = 0
        for ch in CHALLENGES:
            obj, was_created = Problem.objects.get_or_create(
                title=ch["title"],
                defaults={
                    "description": ch["description"].strip(),
                    "grading_criteria": ch["grading_criteria"].strip(),
                    "example_input": ch.get("example_input", "").strip(),
                    "example_output": ch.get("example_output", "").strip(),
                },
            )
            if was_created:
                created += 1
                self.stdout.write(f"  ✓ Created: {obj.title}")
            else:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"  – Skipped (exists): {obj.title}"))

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone! {created} problem(s) created, {skipped} skipped."
            )
        )
