"""
Management command to load a curriculum of Python coding challenges.

Usage:
    python manage.py load_challenges            # load all (skips duplicates)
    python manage.py load_challenges --flush    # delete existing problems first
"""

from django.core.management.base import BaseCommand
from grader.models import Problem


CHALLENGES = [

    # ──────────────────────────────────────────────────────────────────────────
    # PYTHON BASICS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Python Basics] FizzBuzz",
        "description": """\
Write a function called `fizzbuzz(n)` that returns a list of strings for numbers 1 through n:
- "Fizz" for multiples of 3
- "Buzz" for multiples of 5
- "FizzBuzz" for multiples of both 3 and 5
- The number as a string otherwise

Example:
    fizzbuzz(15) → ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
""",
        "grading_criteria": """\
- The function is named exactly `fizzbuzz` and accepts one integer argument n
- Returns a list (not prints) of exactly n elements
- Multiples of 15 are correctly labelled "FizzBuzz" (checked before Fizz/Buzz)
- Multiples of 3 (only) are "Fizz", multiples of 5 (only) are "Buzz"
- All other numbers are returned as strings (e.g. "1", not 1)
- Handles edge cases: n=0 returns empty list, n=1 returns ["1"]
- Code is clean, readable, and uses a loop or list comprehension
""",
        "example_input": "fizzbuzz(15)",
        "example_output": '["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]',
    },
    {
        "title": "[Python Basics] Fibonacci Generator",
        "description": """\
Write a **generator function** called `fibonacci()` that yields Fibonacci numbers indefinitely,
starting from 0, 1, 1, 2, 3, 5, 8, …

Also write a helper `first_n_fibs(n)` that returns the first n Fibonacci numbers as a list
by consuming the generator.

Example:
    first_n_fibs(8) → [0, 1, 1, 2, 3, 5, 8, 13]
""",
        "grading_criteria": """\
- `fibonacci()` is implemented as a generator (uses `yield`, not `return`)
- The sequence starts correctly: 0, 1, 1, 2, 3, ...
- The generator runs indefinitely without crashing
- `first_n_fibs(n)` correctly returns a list of the first n values
- `first_n_fibs(0)` returns []
- `first_n_fibs(1)` returns [0]
- No pre-computed hardcoded lists — the sequence must be calculated
- Code is concise and Pythonic
""",
        "example_input": "first_n_fibs(8)",
        "example_output": "[0, 1, 1, 2, 3, 5, 8, 13]",
    },
    {
        "title": "[Python Basics] List & Dict Comprehensions",
        "description": """\
Implement the following three functions using comprehensions (no explicit for-loops):

1. `squares_of_evens(lst)` — returns a list of squares of even numbers in `lst`
2. `word_lengths(sentence)` — returns a dict mapping each unique word to its length (lowercase, strip punctuation using str.strip('.,!?'))
3. `flatten(matrix)` — returns a flat list from a 2-D list (list of lists)

Example:
    squares_of_evens([1,2,3,4,5,6]) → [4, 16, 36]
    word_lengths("Hello world hello") → {"hello": 5, "world": 5}
    flatten([[1,2],[3,4],[5]]) → [1, 2, 3, 4, 5]
""",
        "grading_criteria": """\
- All three functions use comprehensions (list or dict) — no explicit `for` loop bodies with append
- `squares_of_evens`: correctly filters evens and squares them; empty list → []
- `word_lengths`: correctly lowercases, strips punctuation, deduplicates (last occurrence wins or any consistent rule)
- `flatten`: handles jagged rows and empty sublists; returns a flat list
- No imports required
- Functions are named exactly as specified
- Code is clean and idiomatic Python
""",
        "example_input": "squares_of_evens([1,2,3,4,5,6])\nword_lengths('Hello world hello')\nflatten([[1,2],[3,4],[5]])",
        "example_output": "[4, 16, 36]\n{'hello': 5, 'world': 5}\n[1, 2, 3, 4, 5]",
    },
    {
        "title": "[Python Basics] Decorators & Error Handling",
        "description": """\
1. Write a decorator `retry(times)` that retries a function up to `times` attempts if it raises an exception.
   On final failure it should re-raise the last exception.

2. Write a function `safe_divide(a, b)` that returns a / b, or raises a `ValueError` with the message
   "Cannot divide by zero" when b == 0.

3. Apply the `@retry(3)` decorator to a function `flaky_fetch(data, fail_count)` that raises
   `ConnectionError("Network error")` for the first `fail_count` calls, then returns `data`.

Example:
    safe_divide(10, 2)          → 5.0
    safe_divide(10, 0)          → raises ValueError
    flaky_fetch("hello", 2)     → "hello"  (fails twice, succeeds on 3rd attempt)
    flaky_fetch("hello", 3)     → raises ConnectionError (all 3 attempts fail)
""",
        "grading_criteria": """\
- `retry(times)` is a parametrised decorator (decorator factory)
- It retries exactly `times` times before re-raising the last exception
- `safe_divide` raises `ValueError` with correct message on zero denominator
- `flaky_fetch` uses a mutable counter (e.g. a list or nonlocal) to track call count
- `@retry(3)` is correctly applied to `flaky_fetch`
- All four example cases behave correctly
- Code uses try/except appropriately; no bare `except:` clauses
""",
        "example_input": "flaky_fetch('hello', 2)",
        "example_output": "'hello'",
    },
    {
        "title": "[Python Basics] OOP — Bank Account",
        "description": """\
Implement a `BankAccount` class with:
- `__init__(self, owner, balance=0.0)` — sets owner and initial balance
- `deposit(amount)` — adds amount; raises `ValueError` if amount <= 0
- `withdraw(amount)` — subtracts amount; raises `ValueError` if amount <= 0 or exceeds balance
- `transfer(amount, other_account)` — withdraws from self and deposits into `other_account`
- `__repr__` — returns e.g. `"BankAccount(owner='Alice', balance=150.00)"`
- `__add__` — merges two accounts into a new one owned by "owner1 & owner2" with combined balance

Example:
    a = BankAccount("Alice", 100)
    b = BankAccount("Bob", 50)
    a.transfer(30, b)
    print(a)   # BankAccount(owner='Alice', balance=70.00)
    print(b)   # BankAccount(owner='Bob', balance=80.00)
    c = a + b  # BankAccount(owner='Alice & Bob', balance=150.00)
""",
        "grading_criteria": """\
- All methods implemented with correct signatures
- `deposit` and `withdraw` raise `ValueError` with sensible messages for invalid inputs
- `withdraw` raises `ValueError` when amount > balance (overdraft not allowed)
- `transfer` is atomic — uses existing deposit/withdraw methods
- `__repr__` formats balance to 2 decimal places
- `__add__` returns a new BankAccount (not modifying either original) with combined balance
- Class follows OOP principles — no global state
- All example operations produce correct results
""",
        "example_input": 'a = BankAccount("Alice", 100)\nb = BankAccount("Bob", 50)\na.transfer(30, b)\nprint(a)\nprint(a + b)',
        "example_output": "BankAccount(owner='Alice', balance=70.00)\nBankAccount(owner='Alice & Bob', balance=150.00)",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # NUMPY
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[NumPy] Array Operations & Broadcasting",
        "description": """\
Using NumPy, implement:

1. `normalize(arr)` — normalises a 1-D array to the range [0, 1].
   Formula: (x - min) / (max - min). Return zeros if max == min.

2. `moving_average(arr, window)` — returns a 1-D array of the moving average
   with the given window size (valid convolution, no padding).

3. `outer_diff(a, b)` — given two 1-D arrays, returns a 2-D matrix where
   element [i, j] = a[i] - b[j]. Must use broadcasting (no loops).

Example:
    normalize(np.array([2, 4, 6, 8])) → [0. , 0.25, 0.75, 1. ]  (wait — check: (2-2)/(8-2)=0, (4-2)/6≈0.333)
    moving_average(np.array([1,2,3,4,5]), 3) → [2., 3., 4.]
    outer_diff(np.array([1,2,3]), np.array([10,20])) → [[-9,-19],[-8,-18],[-7,-17]]
""",
        "grading_criteria": """\
- Uses `import numpy as np` — no Python loops for array computations
- `normalize`: correctly applies min-max formula; handles equal min/max (returns zeros)
- `moving_average`: uses np.convolve or stride tricks; output length is len(arr)-window+1
- `outer_diff`: uses broadcasting (reshaping with np.newaxis or reshape) — no loops
- All three functions return np.ndarrays
- Handles 1-D input arrays of varying lengths
- Code is concise and vectorised
""",
        "example_input": "normalize(np.array([2,4,6,8]))\nmoving_average(np.array([1,2,3,4,5]), 3)\nouter_diff(np.array([1,2,3]), np.array([10,20]))",
        "example_output": "[0.     0.333  0.667  1.   ]\n[2. 3. 4.]\n[[ -9 -19]\n [ -8 -18]\n [ -7 -17]]",
    },
    {
        "title": "[NumPy] Linear Algebra & Statistics",
        "description": """\
Using NumPy, implement:

1. `covariance_matrix(X)` — given a 2-D array X of shape (n_samples, n_features),
   return the covariance matrix of shape (n_features, n_features).
   Centre the data first (subtract column means). Do NOT use np.cov directly.

2. `solve_linear_system(A, b)` — solve Ax = b and return x.
   Return None if A is singular (catch np.linalg.LinAlgError).

3. `top_k_eigenvalues(M, k)` — return the top-k eigenvalues (largest magnitude, real part)
   of a symmetric matrix M, sorted descending.

Example:
    X = np.array([[1,2],[3,4],[5,6]])
    covariance_matrix(X)  → [[4., 4.], [4., 4.]]
    solve_linear_system(np.array([[2,1],[1,3]]), np.array([5,10])) → [1. 3.]
""",
        "grading_criteria": """\
- Uses numpy linalg functions appropriately (np.linalg.solve, np.linalg.eig or eigh)
- `covariance_matrix`: manually centres data; formula is X_c.T @ X_c / (n-1); shape correct
- `solve_linear_system`: returns None gracefully on singular matrix; correct for well-conditioned systems
- `top_k_eigenvalues`: returns real parts, sorted descending, correct length k
- No Python loops for matrix math
- Correct handling of edge cases (singular matrix, k > n_features)
""",
        "example_input": "covariance_matrix(np.array([[1,2],[3,4],[5,6]]))\nsolve_linear_system(np.array([[2,1],[1,3]]), np.array([5,10]))",
        "example_output": "[[4. 4.]\n [4. 4.]]\n[1. 3.]",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PANDAS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Pandas] DataFrame Wrangling",
        "description": """\
Given the following CSV-style data (load it with `pd.read_csv` or build it with `pd.DataFrame`):

```
name,age,department,salary
Alice,30,Engineering,95000
Bob,25,Marketing,55000
Carol,35,Engineering,105000
Dave,28,Marketing,60000
Eve,32,HR,70000
Frank,29,Engineering,88000
```

Implement a function `analyse(df)` that returns a dict with:
- `"highest_paid"`: name of the highest-paid employee
- `"avg_salary_by_dept"`: a dict mapping department → mean salary (rounded to 2 dp)
- `"engineering_seniors"`: list of names in Engineering with age >= 30, sorted alphabetically
- `"salary_zscore"`: a new column added to df in-place (or returned) — z-score of salary column
""",
        "grading_criteria": """\
- Uses pandas operations — no manual Python loops over rows
- `highest_paid`: correct name extracted (Alice → Carol? check data: Carol=105000) → "Carol"
- `avg_salary_by_dept`: dict with correct rounded means for all 3 departments
- `engineering_seniors`: correct filtering with & for multiple conditions; sorted alphabetically
- `salary_zscore`: correct z-score formula ((x - mean) / std); column added correctly
- Function returns a dict with all four keys
- Code uses idiomatic pandas (.groupby, .loc, boolean indexing, etc.)
""",
        "example_input": "analyse(df)['highest_paid']\nanalyse(df)['engineering_seniors']",
        "example_output": "'Carol'\n['Alice', 'Carol']",
    },
    {
        "title": "[Pandas] Time Series & GroupBy",
        "description": """\
Build and analyse a time-series sales DataFrame. Implement `sales_analysis()` that:

1. Creates a DataFrame with columns: `date` (daily from 2024-01-01 to 2024-12-31),
   `product` (cycling through ["Widget","Gadget","Doohickey"]),
   `sales` (integers: use `np.random.seed(42)` then `np.random.randint(10, 200, 365)`)

2. Returns a dict:
   - `"monthly_totals"`: Series of total sales per month (index = month number 1-12)
   - `"best_month"`: month number with highest total sales
   - `"product_avg"`: dict of product → mean sales across the year (rounded to 2 dp)
   - `"rolling_7day"`: Series of 7-day rolling average of daily sales (min_periods=1)

The function must be deterministic (seed is fixed).
""",
        "grading_criteria": """\
- `pd.date_range` used to create the date range correctly (365 days in 2024 — it's a leap year, so 366 — accept either approach with comment)
- `np.random.seed(42)` called before generating random numbers
- Products cycle correctly using a list repeated/tiled to match length
- `monthly_totals`: uses dt.month groupby, sums sales; correct index
- `best_month`: integer month number (1-12)
- `product_avg`: correct mean per product, rounded to 2 dp, returned as dict
- `rolling_7day`: uses .rolling(7, min_periods=1).mean()
- No hardcoded values — everything computed from the DataFrame
""",
        "example_input": "r = sales_analysis()\nprint(r['best_month'])\nprint(list(r['product_avg'].keys()))",
        "example_output": "(some integer 1-12)\n['Widget', 'Gadget', 'Doohickey']",
    },
    {
        "title": "[Pandas] Data Cleaning Pipeline",
        "description": """\
Implement `clean_dataframe(df)` that cleans a messy DataFrame with these rules:

1. **Drop duplicates** — remove exact duplicate rows
2. **Fix column names** — strip whitespace and convert to snake_case (lowercase, spaces→underscores)
3. **Handle missing values**:
   - Numeric columns: fill NaN with the column median
   - String/object columns: fill NaN with the string "unknown"
4. **Type coercion** — convert any column named ending in `_date` to datetime (errors="coerce")
5. **Outlier capping** — for numeric columns, cap values at [1st percentile, 99th percentile]
6. Return the cleaned DataFrame (do not modify in place)

The function should work on any DataFrame, not just a specific one.
""",
        "grading_criteria": """\
- Function returns a copy (not modifying the original df)
- Duplicate rows removed correctly
- Column name cleaning handles multiple spaces, mixed case, punctuation (at minimum spaces → underscores, lowercase)
- Numeric NaN filled with median (not mean); object NaN filled with "unknown"
- `_date` column detection uses endswith, not hardcoding; pd.to_datetime applied correctly
- Outlier capping uses np.percentile or .quantile with q=0.01 and q=0.99; clip applied correctly
- Works correctly on a fresh DataFrame with mixed types
- Code uses pandas idioms (select_dtypes, apply/map where appropriate)
""",
        "example_input": "df = pd.DataFrame({'Name ': ['Alice', 'Bob', None, 'Alice'], 'Score': [95, None, 70, 95], 'Join Date': ['2023-01-01', '2023-02-15', 'bad', None]})\nclean_dataframe(df)",
        "example_output": "  name  score  join_date\n0  Alice   95.0  2023-01-01\n1  Bob     82.5  2023-02-15\n2  unknown 70.0  NaT",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MATPLOTLIB
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Matplotlib] Multi-Panel Figure",
        "description": """\
Using `matplotlib` and `numpy`, create a function `create_figure()` that produces and **saves**
a 2×2 grid of subplots to a file called `output.png`. The four panels should be:

1. **Top-left**: Line plot of sin(x) and cos(x) for x in [0, 4π], with a legend and grid
2. **Top-right**: Scatter plot of 200 random points (np.random.seed(0)), coloured by a third
   random variable using the "viridis" colormap; include a colorbar
3. **Bottom-left**: Bar chart of the 26 letter frequencies in the string
   `"the quick brown fox jumps over the lazy dog"` (count each letter a-z)
4. **Bottom-right**: Histogram of 1000 samples from a normal distribution (μ=0, σ=1)
   with 30 bins, a KDE-style smooth curve overlaid (use `np.histogram` + plot), and labelled axes

Requirements:
- Figure size: 14×10 inches
- Each subplot must have a title
- Tight layout applied
- File saved as `output.png` at 150 dpi
""",
        "grading_criteria": """\
- Figure is 14×10 inches; 2×2 subplots created with plt.subplots
- Top-left: sin and cos plotted, legend present, grid enabled
- Top-right: scatter with c= argument and cmap='viridis', plt.colorbar called
- Bottom-left: letter frequencies computed from the given string; bar chart of 26 letters
- Bottom-right: histogram with 30 bins; smooth curve overlaid (computed, not seaborn); axes labelled
- All four subplots have titles
- plt.tight_layout() called
- Saved to 'output.png' at dpi=150
- np.random.seed(0) used for reproducibility
- No plt.show() call (non-interactive save only)
""",
        "example_input": "create_figure()  # should produce output.png",
        "example_output": "output.png saved to disk",
    },
    {
        "title": "[Matplotlib] Custom Styling & Annotations",
        "description": """\
Implement `annotated_plot()` that creates and saves a single publication-quality figure to `annotated.png`:

- Plot the function f(x) = x² · e^(-x) for x in [0, 8]
- Find and annotate the global maximum with a red dot and an arrow pointing to it,
  labelled "Maximum" with the (x, y) values shown to 2 dp
- Shade the area under the curve between x=1 and x=4 with a semi-transparent fill
- Add a horizontal dashed line at y = max/2 labelled "Half Maximum"
- Style requirements:
  - Use the `"seaborn-v0_8-whitegrid"` style
  - Line colour: steelblue, linewidth 2.5
  - Figure size: 10×6; DPI: 120
  - x-label: "x", y-label: "f(x)", title: "f(x) = x²·e^(-x)"
""",
        "grading_criteria": """\
- Correct function: f(x) = x**2 * np.exp(-x)
- Global maximum found numerically (np.argmax); red dot plotted at correct location
- Arrow annotation using ax.annotate with arrowprops; label shows correct (x, y) to 2 dp
- Shaded region using ax.fill_between for x in [1,4]
- Horizontal dashed line at correct y value with label
- Correct matplotlib style applied (seaborn-v0_8-whitegrid)
- Line properties (color, linewidth) set correctly
- Figure size 10×6, DPI 120
- All labels and title present
- Saved to annotated.png; no plt.show()
""",
        "example_input": "annotated_plot()  # produces annotated.png",
        "example_output": "annotated.png saved to disk",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SEABORN
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Seaborn] Statistical Visualisations",
        "description": """\
Using seaborn and the built-in `tips` dataset (`sns.load_dataset('tips')`),
implement `tips_dashboard()` that creates and saves a 2×2 dashboard to `tips_dashboard.png`:

1. **Top-left**: Violin plot of `total_bill` by `day`, split by `sex`, palette="muted"
2. **Top-right**: Scatter plot (`scatterplot`) of `total_bill` vs `tip`, hued by `smoker`,
   with a regression line per group (use `lmplot` or overlay manually)
3. **Bottom-left**: Correlation heatmap of all numeric columns — annotated with values,
   colormap "coolwarm", center=0
4. **Bottom-right**: KDE plot of `tip` distributions for each `day` overlaid on one axes

- Figure 16×12 inches, tight layout, saved at 150 dpi
- Each subplot titled appropriately
""",
        "grading_criteria": """\
- `sns.load_dataset('tips')` used — no manual data loading
- Top-left: violin plot with split=True or hue='sex'; correct x,y axes
- Top-right: scatter coloured by smoker; regression lines present (sns.lmplot or np.polyfit overlay)
- Bottom-left: .corr() computed; sns.heatmap with annot=True, cmap='coolwarm', center=0
- Bottom-right: kde plot per day overlaid (sns.kdeplot in a loop or hue=)
- All subplots have titles
- Figure 16×12, tight_layout, saved to tips_dashboard.png at 150 dpi
- No plt.show()
""",
        "example_input": "tips_dashboard()  # produces tips_dashboard.png",
        "example_output": "tips_dashboard.png saved to disk",
    },
    {
        "title": "[Seaborn] Pair Plots & Categorical Analysis",
        "description": """\
Using seaborn and the `penguins` dataset (`sns.load_dataset('penguins')`):

1. Implement `penguin_pairplot()` — creates a pair plot of the four numeric columns
   (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`),
   coloured by `species`. Diagonal: KDE plots. Save to `penguins_pair.png`.

2. Implement `penguin_catplot()` — creates a `catplot` (type="box") of `body_mass_g`
   by `species`, faceted by `sex` in a single row. Save to `penguins_cat.png`.

3. Implement `correlation_by_species(df)` — returns a dict mapping each species name
   to its Pearson correlation between `flipper_length_mm` and `body_mass_g` (rounded to 3 dp).

Drop rows with any NaN before plotting.
""",
        "grading_criteria": """\
- `sns.load_dataset('penguins')` used, NaN rows dropped before all operations
- `penguin_pairplot`: sns.pairplot with vars= set to the 4 numeric columns, hue='species', diag_kind='kde'; saved to penguins_pair.png
- `penguin_catplot`: sns.catplot with kind='box', x='species', y='body_mass_g', col='sex'; saved to penguins_cat.png
- `correlation_by_species`: groups by species, computes .corr() correctly, rounds to 3 dp, returns dict with 3 keys
- Both plot functions save files and don't call plt.show()
- No hardcoded species names in correlation function
""",
        "example_input": "correlation_by_species(sns.load_dataset('penguins').dropna())",
        "example_output": "{'Adelie': 0.468, 'Chinstrap': 0.642, 'Gentoo': 0.703}  # approximate values",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SCIKIT-LEARN
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Scikit-learn] Classification Pipeline",
        "description": """\
Using scikit-learn, build a classification pipeline on the Iris dataset:

Implement `iris_pipeline()` that:
1. Loads `sklearn.datasets.load_iris()`
2. Splits data 80/20 train/test (`random_state=42`, stratified)
3. Builds a `Pipeline` with:
   - `StandardScaler`
   - `RandomForestClassifier(n_estimators=100, random_state=42)`
4. Trains the pipeline on the training set
5. Returns a dict with:
   - `"accuracy"`: test accuracy (float, rounded to 4 dp)
   - `"classification_report"`: the string from `classification_report(y_test, y_pred)`
   - `"feature_importances"`: dict mapping feature name → importance (rounded to 4 dp), sorted descending by importance
   - `"cv_scores"`: list of 5-fold CV accuracy scores (rounded to 4 dp) on the full dataset
""",
        "grading_criteria": """\
- Correct use of sklearn Pipeline, StandardScaler, RandomForestClassifier
- train_test_split with stratify=y, test_size=0.2, random_state=42
- Accuracy ≥ 0.95 (Iris is an easy dataset — poor accuracy indicates a bug)
- classification_report present and non-empty
- feature_importances: correct mapping using iris.feature_names; sorted descending
- cv_scores: list of 5 floats from cross_val_score with cv=5; all between 0 and 1
- All numeric values rounded to 4 dp
- No data leakage (scaler fitted only on training data via pipeline)
""",
        "example_input": "r = iris_pipeline()\nprint(r['accuracy'])\nprint(list(r['feature_importances'].keys())[0])  # top feature",
        "example_output": "0.9667 (or similar)\n'petal length (cm)' or 'petal width (cm)'",
    },
    {
        "title": "[Scikit-learn] Regression & Model Selection",
        "description": """\
Using scikit-learn's California Housing dataset, implement `housing_model_comparison()` that:

1. Loads `sklearn.datasets.fetch_california_housing()`
2. Splits 80/20 (random_state=42)
3. Scales features with `StandardScaler`
4. Trains and evaluates three models:
   - `LinearRegression`
   - `Ridge(alpha=1.0)`
   - `GradientBoostingRegressor(n_estimators=100, random_state=42)`
5. Returns a dict:
   - `"results"`: dict of model_name → `{"rmse": float, "r2": float}` (both rounded to 4 dp)
   - `"best_model"`: name of the model with lowest RMSE
   - `"feature_coefficients"`: for LinearRegression only — dict of feature_name → coefficient (rounded to 4 dp)
""",
        "grading_criteria": """\
- Correct dataset loading and splitting
- StandardScaler fitted on train only (no leakage)
- All three models trained and evaluated on the test set
- RMSE computed as sqrt(mean_squared_error); R² from r2_score
- `results` dict has all three model names with correct keys
- `best_model` is the model name string with lowest RMSE (GradientBoosting usually wins)
- `feature_coefficients` maps housing feature names to LinearRegression coefficients correctly
- All floats rounded to 4 dp
- No leakage of test data into fitting
""",
        "example_input": "r = housing_model_comparison()\nprint(r['best_model'])\nprint(list(r['results'].keys()))",
        "example_output": "'GradientBoostingRegressor'\n['LinearRegression', 'Ridge', 'GradientBoostingRegressor']",
    },
    {
        "title": "[Scikit-learn] Unsupervised Learning & Dimensionality Reduction",
        "description": """\
Using scikit-learn, implement `cluster_and_reduce()` on the Digits dataset (`load_digits()`):

1. Scale the data with `StandardScaler`
2. Reduce to 2D with `PCA(n_components=2, random_state=42)`
3. Cluster the 2-D data with `KMeans(n_clusters=10, random_state=42, n_init=10)`
4. Evaluate clustering quality on the **original labels** using `adjusted_rand_score`
5. Save a scatter plot of the 2-D PCA projection coloured by predicted cluster to `digits_clusters.png`
6. Return a dict:
   - `"adjusted_rand_score"`: float rounded to 4 dp
   - `"pca_explained_variance"`: list of 2 floats (explained variance ratio per component, rounded to 4 dp)
   - `"cluster_sizes"`: dict of cluster_label (int) → count, sorted by label
""",
        "grading_criteria": """\
- Correct pipeline: scale → PCA(2) → KMeans(10)
- StandardScaler, PCA, KMeans all used from sklearn with specified params
- adjusted_rand_score computed against the true digit labels (0-9)
- ARI score should be > 0.5 (reasonable clustering of digits in PCA space)
- Scatter plot saved to digits_clusters.png; coloured by cluster (not true label)
- pca_explained_variance: list of exactly 2 floats from pca.explained_variance_ratio_
- cluster_sizes: dict with 10 keys (0-9), all sizes sum to 1797 (total samples)
- No plt.show()
""",
        "example_input": "r = cluster_and_reduce()\nprint(r['adjusted_rand_score'])\nprint(sum(r['cluster_sizes'].values()))",
        "example_output": "(float > 0.5)\n1797",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # TENSORFLOW / KERAS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[TensorFlow] Tensor Operations & Autodiff",
        "description": """\
Using TensorFlow, implement the following without converting to NumPy mid-computation:

1. `matrix_ops(A, B)` — given two tf.Tensors of shape (3,3), returns a dict:
   - `"product"`: matrix product A @ B
   - `"elementwise"`: element-wise A * B
   - `"trace_sum"`: scalar = trace(A) + trace(B)
   - `"frobenius_norm"`: Frobenius norm of (A - B)

2. `gradient_of(f, x_val)` — uses `tf.GradientTape` to compute the derivative of
   f (a Python callable returning a tf.Tensor) at x_val (a float). Return the gradient as a Python float.

3. `softmax_crossentropy(logits, labels)` — computes sparse softmax cross-entropy loss
   manually using tf operations (no tf.nn.sparse_softmax_cross_entropy_with_logits).
   Return a scalar tf.Tensor.

Example:
    gradient_of(lambda x: x**3 + 2*x, 3.0) → 29.0  (d/dx: 3x²+2 at x=3)
""",
        "grading_criteria": """\
- Uses tensorflow (import tensorflow as tf) throughout — no numpy for computations
- `matrix_ops`: all four outputs computed with correct tf operations (tf.linalg.matmul, tf.linalg.trace, tf.norm)
- `gradient_of`: uses tf.GradientTape correctly; x declared as tf.Variable; returns Python float
- gradient of x³+2x at x=3 correctly returns ~29.0
- `softmax_crossentropy`: manually applies log-softmax → gather → negative mean; no high-level loss functions
- All return types are tf.Tensors (except gradient_of which returns float)
- No tf.compat.v1 or session-based code
""",
        "example_input": "gradient_of(lambda x: x**3 + 2*x, 3.0)",
        "example_output": "29.0",
    },
    {
        "title": "[TensorFlow] Neural Network — MNIST Classifier",
        "description": """\
Build and train a neural network on MNIST using tf.keras. Implement `train_mnist()` that:

1. Loads `tf.keras.datasets.mnist`, normalises pixel values to [0,1]
2. Builds a Sequential model:
   - Flatten layer (28×28 → 784)
   - Dense(256, activation='relu')
   - Dropout(0.3)
   - Dense(128, activation='relu')
   - Dropout(0.2)
   - Dense(10, activation='softmax')
3. Compiles with Adam(lr=0.001), sparse_categorical_crossentropy loss, accuracy metric
4. Trains for 5 epochs, batch_size=128, validation_split=0.1
5. Evaluates on test set
6. Returns a dict:
   - `"test_accuracy"`: float rounded to 4 dp (should be ≥ 0.97)
   - `"test_loss"`: float rounded to 4 dp
   - `"model_summary"`: string from model.summary() captured via io.StringIO
   - `"history_keys"`: list of keys from the training history
""",
        "grading_criteria": """\
- Data normalised to [0,1] (divide by 255.0)
- Exact architecture: Flatten, Dense(256,relu), Dropout(0.3), Dense(128,relu), Dropout(0.2), Dense(10,softmax)
- Compiled with Adam, sparse_categorical_crossentropy, metrics=['accuracy']
- Trained for 5 epochs, batch_size=128, validation_split=0.1
- test_accuracy ≥ 0.97 (anything less indicates a training or architecture bug)
- model_summary captured correctly (not printed to stdout) using io.StringIO trick
- history_keys includes at least ['loss', 'accuracy', 'val_loss', 'val_accuracy']
- random_state / determinism not required (GPU variability acceptable)
""",
        "example_input": "r = train_mnist()\nprint(r['test_accuracy'])\nprint(r['history_keys'])",
        "example_output": "≥ 0.97\n['loss', 'accuracy', 'val_loss', 'val_accuracy']",
    },
    {
        "title": "[TensorFlow] Custom Training Loop & Callbacks",
        "description": """\
Implement a custom training loop for a regression problem using TensorFlow.

1. Generate synthetic data:
   - X = tf.linspace(-3.0, 3.0, 500), reshaped to (500,1)
   - y = 2*X**2 - 3*X + 1 + noise (tf.random.normal, stddev=0.5, seed=42)

2. Build a model using `tf.keras.Sequential`:
   - Dense(64, activation='relu', input_shape=(1,))
   - Dense(64, activation='relu')
   - Dense(1)

3. Write a custom training loop (do NOT use model.fit):
   - Use `tf.GradientTape` inside a loop
   - MSE loss computed manually: tf.reduce_mean((y_pred - y)**2)
   - Adam optimiser, learning_rate=0.01
   - Train for 200 epochs; store loss each epoch

4. Return a dict:
   - `"final_loss"`: loss at epoch 200, rounded to 4 dp (should be < 0.5)
   - `"loss_decreased"`: bool — True if final loss < loss at epoch 1
   - `"predictions_sample"`: model predictions for X[-5:] as a Python list of floats (rounded to 2 dp)
""",
        "grading_criteria": """\
- Synthetic data generated with tf.linspace and tf.random.normal with seed=42
- Correct model architecture (Dense 64→64→1)
- Custom training loop uses tf.GradientTape; tape.gradient and optimizer.apply_gradients called correctly
- MSE loss computed manually (not tf.keras.losses.MSE)
- 200 epochs, lr=0.01, Adam
- final_loss < 0.5 (model converges on this smooth function)
- loss_decreased is True
- predictions_sample is a Python list of 5 floats, rounded to 2 dp
- No model.fit() called
""",
        "example_input": "r = custom_training_loop()\nprint(r['final_loss'])\nprint(r['loss_decreased'])",
        "example_output": "(float < 0.5)\nTrue",
    },
]


class Command(BaseCommand):
    help = "Load Python coding challenges into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--flush",
            action="store_true",
            help="Delete all existing problems before loading.",
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
