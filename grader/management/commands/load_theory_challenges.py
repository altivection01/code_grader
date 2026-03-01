"""
Management command to load ML/AI theory questions (5 per topic × 5 topics = 25 questions).
Descriptions and grading criteria support Markdown + LaTeX.

Usage:
    python manage.py load_theory_challenges
    python manage.py load_theory_challenges --flush
"""
from django.core.management.base import BaseCommand
from grader.models import Problem

CHALLENGES = [

    # ──────────────────────────────────────────────────────────────────────────
    # REGRESSION (5)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Regression] Ordinary Least Squares & The Normal Equation",
        "subject_area": "regression",
        "description": r"""Derive the Ordinary Least Squares (OLS) estimator for linear regression from first principles.

Given the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, where:

- $\mathbf{y} \in \mathbb{R}^n$ is the response vector
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector to estimate
- $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$

Your answer must cover:

1. **The OLS objective** — write the loss function being minimised.
2. **Derivation of the normal equation** — take the gradient and set to zero.
3. **Closed-form solution** — state $\hat{\boldsymbol{\beta}}$ and conditions for it to exist.
4. **Geometric interpretation** — explain the hat matrix $\mathbf{H}$ and orthogonal projection.
5. **Gauss-Markov theorem** — state what it guarantees about OLS estimators.""",
        "grading_criteria": r"""- OLS objective stated correctly: $\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$
- Gradient derived: $\nabla_{\boldsymbol{\beta}} \text{RSS} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$
- Normal equation: $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$
- Solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, noting invertibility requires full column rank
- Hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ correctly described as orthogonal projection onto column space of $\mathbf{X}$
- Gauss-Markov correctly stated: OLS is BLUE (Best Linear Unbiased Estimator) under the GM assumptions
- Mathematical notation correct and consistent throughout""",
    },
    {
        "title": "[Regression] Bias-Variance Decomposition",
        "subject_area": "regression",
        "description": r"""Explain and derive the bias-variance tradeoff in supervised learning.

For a regression model $\hat{f}$ trained to approximate $f(x)$ with $y = f(x) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,\sigma^2)$:

1. **Derive** the expected prediction error decomposition into three components:
$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

2. **Explain intuitively** what bias and variance each represent about a model.

3. **Describe** how model complexity affects each term and leads to the U-shaped test error curve.

4. **Explain** how L1 and L2 regularisation manage the tradeoff by adding a penalty $\lambda \|\boldsymbol{\beta}\|_q^q$ to the OLS objective.""",
        "grading_criteria": r"""- Bias defined correctly: $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$
- Variance defined correctly: $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$
- Derivation shows correct algebraic expansion of MSE using $\text{add-and-subtract } \mathbb{E}[\hat{f}(x)]$
- $\sigma^2$ correctly identified as irreducible error
- Intuitive explanation: bias = systematic error, variance = sensitivity to training data fluctuations
- U-shaped curve explained: underfitting (high bias) → sweet spot → overfitting (high variance)
- L2 (Ridge) shrinks all coefficients; L1 (Lasso) produces sparsity by zeroing some coefficients
- Connection between $\lambda$ and bias-variance tradeoff clearly stated""",
    },
    {
        "title": "[Regression] Ridge (L2) vs Lasso (L1) Regularisation",
        "subject_area": "regression",
        "description": r"""Compare Ridge and Lasso regression through their mathematical formulations and geometric interpretations.

**Ridge** minimises:
$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2$$

**Lasso** minimises:
$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$$

Your answer should address:

1. **Closed-form solution for Ridge** — derive $\hat{\boldsymbol{\beta}}_\text{ridge}$.
2. **Why Lasso produces sparse solutions** — use the geometric (constrained optimisation) view.
3. **Effect of $\lambda$** on bias, variance, and coefficient magnitudes in each case.
4. **Elastic Net** — write its objective and explain when it is preferred.
5. **Practical guidance** — when would you choose Ridge over Lasso and vice versa?""",
        "grading_criteria": r"""- Ridge closed form: $\hat{\boldsymbol{\beta}}_\text{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$; notes that $\lambda\mathbf{I}$ ensures invertibility
- Geometric explanation: L1 constraint region (diamond) has corners on axes → solutions tend to lie at corners → sparsity; L2 (sphere) has no corners → continuous shrinkage
- $\lambda \to 0$: recovers OLS; $\lambda \to \infty$: all coefficients shrink to zero
- Elastic Net: $\lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2$, combining group selection and shrinkage
- Practical guidance: Ridge preferred when all features relevant; Lasso for feature selection; Elastic Net for correlated predictors""",
    },
    {
        "title": "[Regression] Logistic Regression: MLE and the Decision Boundary",
        "subject_area": "regression",
        "description": r"""Derive the logistic regression model and its training objective from maximum likelihood estimation.

1. **Model** — define the logistic (sigmoid) function $\sigma(z) = \frac{1}{1+e^{-z}}$ and write the model $P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$.

2. **Log-likelihood** — for a dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{0,1\}$, write and simplify the log-likelihood $\ell(\mathbf{w})$.

3. **Binary cross-entropy loss** — show the connection between minimising cross-entropy and maximising the log-likelihood.

4. **Decision boundary** — derive the equation of the decision boundary in feature space.

5. **Gradient** — compute $\nabla_{\mathbf{w}} \mathcal{L}$ and describe how gradient descent updates the weights.

6. **Multi-class extension** — briefly describe how softmax regression generalises logistic regression.""",
        "grading_criteria": r"""- Sigmoid function stated correctly; properties noted: $\sigma(z) \in (0,1)$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- Log-likelihood: $\ell = \sum_i [y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i))]$
- Cross-entropy minimisation correctly shown to equal negative log-likelihood maximisation
- Decision boundary: $\mathbf{w}^T\mathbf{x} + b = 0$ (a hyperplane); correctly derived from $P(y=1|x) = 0.5$
- Gradient: $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$ — clean, vectorised form
- Softmax: $P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_j e^{\mathbf{w}_j^T\mathbf{x}}}$ with categorical cross-entropy loss""",
    },
    {
        "title": "[Regression] Multicollinearity: Diagnosis and Remedies",
        "subject_area": "regression",
        "description": r"""Explain the problem of multicollinearity in multiple linear regression and how to address it.

1. **Definition** — what is multicollinearity and why does it arise?

2. **Mathematical consequence** — what happens to $(\mathbf{X}^T\mathbf{X})^{-1}$ when predictors are highly correlated? Use the condition number to quantify the problem.

3. **Variance Inflation Factor (VIF)** — define $\text{VIF}_j = \frac{1}{1 - R^2_j}$ where $R^2_j$ is from regressing $X_j$ on all other predictors. Explain what values indicate a problem.

4. **Effect on inference** — how does multicollinearity affect standard errors, t-statistics, and p-values?

5. **Remedies** — discuss at least four approaches: variable selection, ridge regression, PCA regression, and data collection.""",
        "grading_criteria": r"""- Multicollinearity defined: approximate linear dependence among predictors; note perfect collinearity makes $\mathbf{X}^T\mathbf{X}$ singular
- Condition number $\kappa = \lambda_\text{max}/\lambda_\text{min}$; large $\kappa$ means near-singularity, numerically unstable inversion
- VIF formula stated and interpreted correctly: VIF > 5 or 10 signals problematic collinearity
- SE inflation: $\text{Var}(\hat{\beta}_j) = \sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}$; near-singularity inflates variances → wide CIs, low t-stats, high p-values even when true $\beta_j \neq 0$
- Ridge adds $\lambda\mathbf{I}$ making the matrix better conditioned; PCA removes correlated directions; variable selection removes redundant predictors
- At least 4 remedies correctly described""",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # TREES & FORESTS (5)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Trees & Forests] Entropy, Gini Impurity & Information Gain",
        "subject_area": "trees_forests",
        "description": r"""Explain the mathematical criteria used to split nodes in decision trees.

1. **Entropy** — define the Shannon entropy of a discrete distribution:
$$H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$
What are its minimum and maximum values and when are they achieved?

2. **Information Gain** — define $\text{IG}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$ and explain what it measures.

3. **Gini Impurity** — define $G(S) = 1 - \sum_k p_k^2$ and compare it mathematically to entropy.

4. **CART vs ID3/C4.5** — explain which impurity measure each algorithm uses and what structural differences result.

5. **Regression trees** — what criterion replaces entropy/Gini when the target is continuous?""",
        "grading_criteria": r"""- Entropy formula stated correctly; $H = 0$ for pure node, $H = \log_2 K$ for uniform distribution over $K$ classes
- Information Gain: correct weighted sum of child entropies subtracted from parent; clear explanation as reduction in uncertainty
- Gini formula correct; Gini is computationally faster and numerically very similar to entropy; $G \in [0, 1-1/K]$
- CART uses Gini (classification) or MSE (regression); ID3 uses Information Gain (entropy); C4.5 uses gain ratio to correct for high-arity attributes
- Regression trees: minimise within-leaf variance (MSE), i.e. $\sum_v \sum_{x \in S_v}(y_i - \bar{y}_v)^2$""",
    },
    {
        "title": "[Trees & Forests] Overfitting in Decision Trees & Pruning",
        "subject_area": "trees_forests",
        "description": r"""Discuss why decision trees overfit and how pruning addresses this.

1. **Why trees overfit** — explain the relationship between tree depth and model variance. What does a fully grown tree do to training data?

2. **Pre-pruning (early stopping)** — describe at least three hyperparameters that prevent over-growth during training (e.g., `max_depth`, `min_samples_split`, `min_impurity_decrease`).

3. **Cost-complexity (post) pruning** — CART uses the objective:
$$R_\alpha(T) = R(T) + \alpha |T|$$
where $R(T)$ is the tree's misclassification rate and $|T|$ is the number of leaves. Explain how $\alpha$ is chosen and what effect increasing it has on the tree.

4. **Cross-validation role** — how is cross-validation used to select the optimal $\alpha$?

5. **Bias-variance view** — characterise a fully grown tree and a single-split tree in terms of bias and variance.""",
        "grading_criteria": r"""- Full tree memorises training data → near-zero training error, high test error (high variance)
- Pre-pruning: at least 3 of max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease, max_leaf_nodes described correctly
- Cost-complexity: $\alpha$ acts as regularisation; $\alpha = 0$ → full tree; increasing $\alpha$ prunes more aggressively; produces a nested sequence of subtrees
- Cross-validation: fit trees for a range of $\alpha$ values, evaluate on validation fold, select $\alpha$ with best CV error
- Full tree: zero bias, high variance; single-split stump: high bias, low variance — clear tradeoff stated""",
    },
    {
        "title": "[Trees & Forests] Random Forests: Bagging & the Variance Reduction Proof",
        "subject_area": "trees_forests",
        "description": r"""Explain how Random Forests reduce variance through bagging and feature randomisation.

1. **Bootstrap aggregation (bagging)** — if we have $B$ independent predictors each with variance $\sigma^2$, the average has variance $\sigma^2/B$. Why can't we simply average trees trained on the same data?

2. **Correlation effect** — if the predictors have pairwise correlation $\rho$, show that:
$$\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^B \hat{f}_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

3. **Feature subsampling** — how does using $m < p$ features at each split reduce $\rho$? What is the typical recommendation for $m$?

4. **Out-of-bag (OOB) error** — explain what OOB samples are and how they can be used as a free cross-validation estimate.

5. **Feature importance** — how is permutation importance computed, and what does it measure?""",
        "grading_criteria": r"""- Bagging rationale: bootstrap samples are correlated → trees are correlated → variance doesn't reduce to $\sigma^2/B$
- Correlation formula derived or stated correctly: $\rho\sigma^2 + (1-\rho)\sigma^2/B$; as $B \to \infty$ variance approaches $\rho\sigma^2$, not 0
- Feature subsampling reduces inter-tree correlation $\rho$ by preventing all trees from making the same best split; typical $m = \sqrt{p}$ for classification, $m = p/3$ for regression
- OOB: each bootstrap sample omits ~37% of data; OOB predictions average across trees that didn't include that sample; provides unbiased generalisation estimate without a separate test set
- Permutation importance: permute values of feature $j$, measure increase in OOB error; features that matter more cause larger degradation""",
    },
    {
        "title": "[Trees & Forests] Gradient Boosting: Mathematical Foundations",
        "subject_area": "trees_forests",
        "description": r"""Derive the gradient boosting algorithm from the perspective of functional gradient descent.

1. **Boosting principle** — explain how boosting builds an additive model $F_M(x) = \sum_{m=1}^M \gamma_m h_m(x)$ and contrast it with bagging.

2. **Gradient descent in function space** — the loss is $\mathcal{L}(F) = \sum_i L(y_i, F(x_i))$. Show that the negative gradient $-\frac{\partial L}{\partial F(x_i)}$ for squared-error loss $L = \frac{1}{2}(y-F)^2$ gives the residuals $r_i = y_i - F(x_i)$.

3. **Generic gradient boosting algorithm** — write the pseudocode/steps: initialise $F_0$, compute pseudo-residuals, fit a base learner $h_m$ to them, line-search for $\gamma_m$, update $F_m$.

4. **Regularisation** — explain shrinkage (learning rate $\eta$), tree depth, and subsampling (stochastic gradient boosting).

5. **XGBoost improvement** — how does XGBoost differ from vanilla gradient boosting in terms of the objective it minimises (second-order Taylor expansion)?""",
        "grading_criteria": r"""- Additive model structure and sequential fitting correctly explained; contrast with bagging (parallel, independent)
- Gradient derivation for squared loss: $-\partial L/\partial F = y_i - F(x_i)$; pseudo-residuals are the negative gradient
- Algorithm steps correct: initialise with constant, iterate (compute residuals, fit tree, find $\gamma$ by line search or leaf mean, add $\eta \cdot \gamma h_m$)
- Shrinkage $\eta \in (0,1]$ trades fewer trees for more regularisation; depth controls tree complexity; subsampling introduces randomness like RF
- XGBoost: second-order Taylor expansion of loss $L(y, F+h) \approx L + g_i h + \frac{1}{2}h_i^2$ where $g_i, h_i$ are first/second derivatives; enables analytical leaf weight computation and L1/L2 regularisation on leaf weights""",
    },
    {
        "title": "[Trees & Forests] Hyperparameter Tuning for Tree-Based Models",
        "subject_area": "trees_forests",
        "description": r"""Describe the key hyperparameters for decision trees, random forests, and gradient boosting, explaining the effect of each and how to tune them systematically.

1. **Decision tree hyperparameters** — explain `max_depth`, `min_samples_split`, `min_samples_leaf`, and how they each control the bias-variance tradeoff.

2. **Random Forest hyperparameters** — explain `n_estimators`, `max_features`, `bootstrap`, and `max_samples`. How does each affect variance reduction and computational cost?

3. **Gradient Boosting hyperparameters** — for XGBoost/LightGBM, explain `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha` (L1), and `reg_lambda` (L2).

4. **Interaction effects** — explain why `n_estimators` and `learning_rate` should be tuned jointly in gradient boosting (the shrinkage-iteration tradeoff).

5. **Search strategies** — compare grid search, random search, and Bayesian optimisation (e.g. using expected improvement) for hyperparameter optimisation. When would you use each?""",
        "grading_criteria": r"""- Decision tree params: max_depth limits growth → reduces variance; min_samples_split/leaf prevents splits on tiny nodes → reduces variance; all correctly linked to bias-variance
- RF params: n_estimators → more trees → lower variance until diminishing returns; max_features → lower → less correlation between trees; bootstrap and max_samples affect diversity
- GB params: all 7 listed with correct directional effects (e.g. lower learning_rate + more estimators = better but slower; subsample/colsample add stochasticity → regularisation)
- Shrinkage-iteration tradeoff: smaller $\eta$ requires more trees but often achieves lower generalisation error; should grid search jointly using early stopping
- Grid: exhaustive, good for small spaces; random: samples randomly, scales better, often finds good solutions faster; Bayesian: builds surrogate model of objective, targets promising regions — best for expensive evaluations""",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION & CLUSTERING (5)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Classification] K-Means: Algorithm, Convergence & Limitations",
        "subject_area": "classification",
        "description": r"""Analyse the K-Means clustering algorithm in depth.

1. **Lloyd's algorithm** — describe the E-step (assignment) and M-step (update) of K-Means. Write the objective being minimised:
$$\min_{\{C_k\}, \{\boldsymbol{\mu}_k\}} \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$

2. **Convergence proof sketch** — show that each step of Lloyd's algorithm is non-increasing in the objective and explain why convergence is guaranteed.

3. **K-Means++ initialisation** — describe the initialisation procedure and state its approximation guarantee relative to the optimal solution.

4. **Choosing $K$** — describe the elbow method and the silhouette score $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, defining $a(i)$ and $b(i)$.

5. **Limitations** — identify at least three failure modes of K-Means (e.g., sensitivity to initialisation, assumption of spherical clusters, inability to find non-convex clusters).""",
        "grading_criteria": r"""- Objective (WCSS) correctly stated; E-step assigns each point to nearest centroid; M-step updates centroid to cluster mean
- Convergence: both steps can only decrease or maintain WCSS → monotone decrease in objective → finite number of partitions → must converge (though possibly to local minimum)
- K-Means++: first centroid random; subsequent centroids chosen with probability $\propto d(\mathbf{x})^2$ (distance to nearest chosen centroid); provides $O(\log K)$ approximation guarantee
- Silhouette: $a(i)$ = mean intra-cluster distance; $b(i)$ = mean nearest-cluster distance; $s(i) \in [-1,1]$; closer to 1 is better
- At least 3 limitations: local minima, spherical cluster assumption, $K$ must be specified, sensitive to outliers, equal cluster sizes assumed, doesn't handle non-convex shapes""",
    },
    {
        "title": "[Classification] Gaussian Mixture Models & the EM Algorithm",
        "subject_area": "classification",
        "description": r"""Derive the EM algorithm for fitting a Gaussian Mixture Model (GMM).

A GMM models the data distribution as:
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k$ are mixing coefficients ($\sum_k \pi_k = 1$, $\pi_k \geq 0$).

1. **Latent variable formulation** — introduce the latent assignment $z_{ik}$ and write the complete-data log-likelihood.

2. **E-step** — derive the responsibility $r_{ik} = P(z_{ik}=1 | \mathbf{x}_i, \theta)$ using Bayes' theorem.

3. **M-step** — write the closed-form updates for $\hat{\pi}_k$, $\hat{\boldsymbol{\mu}}_k$, and $\hat{\boldsymbol{\Sigma}}_k$.

4. **Comparison to K-Means** — explain why GMM is a "soft" version of K-Means and what the hard-assignment limit corresponds to.

5. **Model selection** — how are BIC and AIC used to choose $K$?""",
        "grading_criteria": r"""- GMM model stated correctly; latent variable $z_{ik} \in \{0,1\}$ with $\sum_k z_{ik}=1$ introduced
- E-step: $r_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}$ derived via Bayes
- M-step: $\hat{\pi}_k = N_k/N$, $\hat{\boldsymbol{\mu}}_k = \frac{1}{N_k}\sum_i r_{ik}\mathbf{x}_i$, $\hat{\boldsymbol{\Sigma}}_k = \frac{1}{N_k}\sum_i r_{ik}(\mathbf{x}_i-\hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i-\hat{\boldsymbol{\mu}}_k)^T$ — all correct
- K-Means comparison: K-Means is limit where covariances are isotropic ($\boldsymbol{\Sigma}_k = \sigma^2\mathbf{I}$) and $\sigma^2 \to 0$ → hard assignments
- BIC: $-2\ell + p\ln N$; AIC: $-2\ell + 2p$; choose $K$ minimising criterion; BIC penalises complexity more heavily""",
    },
    {
        "title": "[Classification] SVM: Maximum Margin Classifier & the Kernel Trick",
        "subject_area": "classification",
        "description": r"""Derive the Support Vector Machine from the maximum-margin principle and explain the kernel trick.

1. **Hard-margin SVM** — for linearly separable data, the optimisation problem is:
$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \;\; \forall i$$
Explain why maximising the margin leads to this formulation (derive the margin width $= 2/\|\mathbf{w}\|$).

2. **Dual formulation** — use Lagrangian duality to write the dual problem. Show that the solution depends only on dot products $\mathbf{x}_i^T\mathbf{x}_j$.

3. **Soft-margin SVM** — introduce slack variables $\xi_i$ and the $C$ parameter. Explain the bias-variance tradeoff that $C$ controls.

4. **Kernel trick** — explain why replacing $\mathbf{x}_i^T\mathbf{x}_j$ with a kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$ allows nonlinear boundaries. Give two examples of kernels.

5. **Support vectors** — which training points are support vectors? What happens to non-support-vector points if they are removed from the training set?""",
        "grading_criteria": r"""- Margin width derivation correct: distance from hyperplane to closest point is $1/\|\mathbf{w}\|$ on each side → total $2/\|\mathbf{w}\|$; minimising $\|\mathbf{w}\|^2$ maximises margin
- Dual: Lagrangian $L = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_i \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1]$; dual depends on $\sum_i\sum_j \alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$
- Soft-margin: $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i\xi_i$; large $C$ → small margin, low bias, high variance; small $C$ → wide margin, high bias
- Kernel trick: compute $K(\mathbf{x},\mathbf{z})$ without explicit $\phi$; examples: RBF $K=\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$, polynomial $K=(\mathbf{x}^T\mathbf{z}+c)^d$
- Support vectors: points on or inside the margin ($\alpha_i > 0$); removing non-SVs doesn't change the decision boundary""",
    },
    {
        "title": "[Classification] Evaluation Metrics: ROC-AUC, Precision-Recall & F1",
        "subject_area": "classification",
        "description": r"""Explain and compare the key classification evaluation metrics and when to use each.

1. **Confusion matrix** — define TP, TN, FP, FN and derive the formulas for accuracy, precision, recall (sensitivity), and specificity.

2. **F1 score** — define $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ and its generalisation $F_\beta$. When does $F_\beta$ weight recall more than precision?

3. **ROC curve** — explain what is plotted (TPR vs FPR at varying thresholds) and what AUC represents probabilistically (the probability that a randomly chosen positive is ranked above a randomly chosen negative).

4. **Precision-Recall curve** — explain when this is preferred over ROC-AUC and why ROC-AUC can be misleadingly optimistic for imbalanced datasets.

5. **Multi-class metrics** — explain macro-averaging, micro-averaging, and weighted-averaging of F1.""",
        "grading_criteria": r"""- Confusion matrix: TP/TN/FP/FN clearly defined; all four derived metrics with correct formulas
- $F_\beta = (1+\beta^2) \cdot \frac{\text{P} \cdot \text{R}}{\beta^2\text{P} + \text{R}}$; $\beta > 1$ weights recall more; $\beta = 2$ (F2) for cases where missing positives is more costly
- ROC: TPR = TP/(TP+FN) vs FPR = FP/(FP+TN); AUC probabilistic interpretation correct; AUC = 0.5 for random, 1.0 for perfect
- PR curve preferred for imbalanced data: ROC can look good because TN is large inflating FPR denominator; PR focuses on positive class performance
- Macro: unweighted mean of per-class F1 (treats all classes equally); micro: aggregate TP/FP/FN then compute (influenced by class frequency); weighted: weight by support""",
    },
    {
        "title": "[Classification] Anomaly Detection: Statistical Foundations",
        "subject_area": "classification",
        "description": r"""Compare statistical and algorithmic approaches to anomaly (outlier) detection.

1. **Statistical approach** — for univariate data assumed $\sim \mathcal{N}(\mu, \sigma^2)$, the z-score method flags points where $|z_i| = |x_i - \hat{\mu}|/\hat{\sigma} > \tau$. Discuss its limitations and the Mahalanobis distance extension to multivariate data:
$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

2. **Isolation Forest** — explain the algorithm: how are random isolation trees built, and why do anomalies have shorter average path lengths? State the anomaly score formula.

3. **Local Outlier Factor (LOF)** — define the reachability distance, local reachability density, and LOF score. Why is LOF able to detect anomalies in varying-density regions where Isolation Forest struggles?

4. **One-Class SVM** — describe the objective and what it learns.

5. **Evaluation challenge** — why is evaluating anomaly detectors difficult in practice, and how is this typically handled?""",
        "grading_criteria": r"""- Z-score: simple, assumes normality; Mahalanobis: $D_M$ accounts for correlations and scales; follows $\chi^2_p$ distribution under normality; flags points with $D_M^2 > \chi^2_{p,\alpha}$
- Isolation Forest: builds trees by randomly selecting a feature and split value; anomalies are isolated in fewer splits → shorter average path length; score $s = 2^{-E[h(x)]/c(n)}$ where $c(n)$ is average path length in BST
- LOF: reachability distance smooths $k$-NN distances; LRD = inverse of average reachability distance; LOF = ratio of average LRD of neighbours to own LRD; LOF >> 1 indicates anomaly; adapts to local density variations
- One-Class SVM: finds hyperplane in feature space maximising distance from origin; maps normal data inside hypersphere; $\nu$ controls fraction of outliers
- Evaluation: anomalies are rare → imbalanced; often no labels; use precision@k, average precision, or synthetic contamination experiments""",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # NEURAL NETWORKS (5)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Neural Networks] Backpropagation: Chain Rule & Gradient Computation",
        "subject_area": "neural_networks",
        "description": r"""Derive the backpropagation algorithm for a fully-connected neural network.

Consider a network with $L$ layers, weights $\mathbf{W}^{(l)}$, biases $\mathbf{b}^{(l)}$, activation $\sigma$, and loss $\mathcal{L}$.

1. **Forward pass** — write the equations for pre-activation $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and activation $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$.

2. **Output layer gradient** — for cross-entropy loss with softmax output, derive $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} = \hat{\mathbf{y}} - \mathbf{y}$.

3. **Backpropagation recurrence** — derive the formula for $\delta^{(l)} = (\mathbf{W}^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$.

4. **Weight gradients** — show that $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}$.

5. **Computational complexity** — compare the number of operations in a forward pass vs. a full backward pass.""",
        "grading_criteria": r"""- Forward pass equations correct; notation consistent
- Output delta derivation: for softmax+cross-entropy, the simplification $\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$ derived or correctly stated with reference to the combined gradient
- Backprop recurrence: correct Hadamard product $\odot$ with $\sigma'(\mathbf{z}^{(l)})$; correct transposed weight matrix; reflects chain rule through both linear transform and activation
- Weight and bias gradient expressions correct and dimensionally consistent
- Complexity: forward pass $O(Np)$ where $N$ = params; backward pass also $O(Np)$ — approximately $2\times$ forward pass; this is the key advantage over finite differences which would be $O(Np \cdot N)$""",
    },
    {
        "title": "[Neural Networks] Activation Functions & the Vanishing Gradient Problem",
        "subject_area": "neural_networks",
        "description": r"""Analyse activation functions and their impact on gradient flow during training.

1. **Sigmoid and tanh** — write their formulas and derivatives. Show mathematically why saturating activations cause the vanishing gradient problem in deep networks.

2. **ReLU** — define $\text{ReLU}(z) = \max(0, z)$ and state its derivative. Explain the "dying ReLU" problem and the conditions under which a neuron can die permanently.

3. **Leaky ReLU and ELU** — write the formulas for both. How do they address the dying ReLU problem?

4. **GELU** — write the approximate formula $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ and explain why it is preferred in modern transformers.

5. **Exploding gradients** — describe the problem and two solutions: gradient clipping and careful weight initialisation (e.g., Xavier/He initialisation — state the variance formulas).""",
        "grading_criteria": r"""- Sigmoid: $\sigma(z) = 1/(1+e^{-z})$, $\sigma'(z) = \sigma(1-\sigma) \in (0, 0.25]$; tanh: $\tanh'(z) = 1 - \tanh^2(z) \in (0,1]$; both saturate → gradients $\to 0$ → product over $L$ layers $\to 0$ exponentially
- ReLU derivative: 1 for $z > 0$, 0 for $z \leq 0$; dying ReLU: if bias becomes very negative, neuron always outputs 0 → gradient always 0 → weight never updates
- Leaky ReLU: $\max(\alpha z, z)$ with $\alpha$ small; ELU: $\alpha(e^z - 1)$ for $z < 0$; both allow small negative gradient
- GELU: smooth, stochastic interpretation (multiplies input by probability of being kept); preferred for transformers due to smoother loss landscape
- Gradient clipping: scale gradient if $\|\mathbf{g}\| > \tau$; Xavier init: $\text{Var}(W) = 2/(n_{in}+n_{out})$; He init: $\text{Var}(W) = 2/n_{in}$ (for ReLU)""",
    },
    {
        "title": "[Neural Networks] Batch Normalisation: Theory & Training Dynamics",
        "subject_area": "neural_networks",
        "description": r"""Explain Batch Normalisation (BN) mathematically and describe why it improves training.

1. **Forward pass** — for a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$, write the normalisation, scale, and shift operations:
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

2. **Learnable parameters** — explain the role of $\gamma$ and $\beta$ and why they are necessary (i.e., why pure normalisation alone would be harmful).

3. **Training vs inference** — how does BN behave differently at test time? What running statistics are maintained?

4. **Why BN helps** — discuss at least two mechanisms: (a) reducing internal covariate shift; (b) smoothing the loss landscape / allowing higher learning rates.

5. **Layer Norm vs Batch Norm** — write the normalisation formula for Layer Normalisation and explain why it is preferred in sequence models (RNNs, Transformers).""",
        "grading_criteria": r"""- Forward pass formulas correct: mean $\mu_\mathcal{B}$, variance $\sigma^2_\mathcal{B}$ computed over batch; $\epsilon$ for numerical stability; scale $\gamma$ and shift $\beta$
- $\gamma, \beta$: without them, BN would force all pre-activations to have zero mean/unit variance regardless of what the layer learned → loss of representational power; they allow BN to learn the optimal scale/mean
- Inference: uses running (exponential moving) averages of $\mu$ and $\sigma^2$ accumulated during training; not recomputed per-batch
- Why it helps: (a) reduces ICS so each layer trains on a more stable distribution; (b) loss surface becomes smoother with larger regions of consistent gradient direction → allows larger learning rates → faster convergence
- Layer Norm: normalises over feature dimension rather than batch dimension; formula: $\hat{x} = (x - \mu_x)/\sigma_x$; preferred for sequences because batch statistics are unstable for varying-length sequences; works with batch size 1""",
    },
    {
        "title": "[Neural Networks] Convolutional Neural Networks: Architecture & Theory",
        "subject_area": "neural_networks",
        "description": r"""Explain the theoretical foundations of Convolutional Neural Networks (CNNs) for image recognition.

1. **Discrete convolution** — write the 2D convolution operation $(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] K[m,n]$. What does each learned filter detect?

2. **Key properties** — explain translation equivariance (and how pooling makes it approximate invariance) and parameter sharing. Compute the number of parameters in a convolutional layer with $C_{in}$ input channels, $C_{out}$ output channels, and kernel size $k \times k$.

3. **Receptive field** — derive the receptive field of a pixel after $L$ convolutional layers each with kernel size $k$ and stride 1. How do dilated convolutions increase it without adding parameters?

4. **Pooling** — compare max pooling and average pooling: what property does each preserve and when is each preferred?

5. **Modern architectures** — briefly describe the key innovations in ResNets (residual connections and why they solve vanishing gradients) and the difference from VGG-style networks.""",
        "grading_criteria": r"""- Convolution formula correct; filters detect local patterns (edges, textures, object parts at different layers)
- Translation equivariance: shifting input shifts feature map equally; parameter sharing: same filter weights applied at every spatial location → $C_{out}(C_{in} \cdot k^2 + 1)$ params (with bias); contrast with fully-connected layer
- Receptive field after $L$ layers of kernel $k$, stride 1: $RF = 1 + L(k-1)$; dilated conv with rate $d$: effective $k' = k + (k-1)(d-1)$, same parameter count
- Max pooling: preserves dominant activations (good for texture); average pooling: smoother spatial aggregation (better for global pooling at end of network)
- ResNet: $F(\mathbf{x}) + \mathbf{x}$ skip connections; gradient can flow directly through identity → $\nabla \mathcal{L}$ never vanishes completely; enables training of very deep networks (100+ layers); VGG: sequential, no skip connections""",
    },
    {
        "title": "[Neural Networks] Optimisation: SGD, Momentum & Adam",
        "subject_area": "neural_networks",
        "description": r"""Compare optimisation algorithms for training neural networks, focusing on their mathematical update rules and convergence properties.

1. **SGD with mini-batches** — write the update rule. Explain why mini-batch gradient is an unbiased estimator of the full gradient, and the role of batch size on gradient noise and convergence.

2. **Momentum (Heavy Ball)** — write the update equations using velocity $\mathbf{v}_t$. Explain intuitively why momentum accelerates convergence in ravines and dampens oscillations.

3. **RMSProp** — write the update equations. How does the exponential moving average of squared gradients $\mathbf{s}_t$ adapt the learning rate per parameter?

4. **Adam** — combine momentum and RMSProp to derive the Adam update:
$$\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t), \quad \hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t), \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}\hat{\mathbf{m}}_t$$
Explain the bias-correction terms $1-\beta_1^t$ and $1-\beta_2^t$.

5. **Learning rate scheduling** — describe cosine annealing and warm restarts. When would you use them versus a constant learning rate?""",
        "grading_criteria": r"""- SGD update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_\theta \mathcal{L}_\mathcal{B}$; mini-batch gradient unbiased (each sample equally likely to be in batch); larger batch → less noise but poorer generalisation (sharp minima)
- Momentum: $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\nabla\mathcal{L}$, $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\mathbf{v}_t$; exponential moving average of gradients → smooths oscillations in high-curvature directions; accelerates in consistent gradient directions
- RMSProp: $\mathbf{s}_t = \rho\mathbf{s}_{t-1} + (1-\rho)(\nabla\mathcal{L})^2$; update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{\sqrt{\mathbf{s}_t}+\epsilon}\nabla\mathcal{L}$; large gradient history → small effective LR (prevents overshooting)
- Adam: correct formulas; bias correction needed because $\mathbf{m}_0 = \mathbf{v}_0 = 0$ → estimates biased toward 0 initially; correction amplifies updates in early steps
- Cosine annealing: $\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max}-\eta_\text{min})(1+\cos(\pi t/T))$; warm restarts (SGDR) periodically reset LR to escape local minima; useful for long training runs""",
    },

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATIVE AI FOR NLP (5)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "title": "[Generative AI] Transformer Self-Attention Mechanism",
        "subject_area": "generative_ai",
        "description": r"""Derive and explain the scaled dot-product self-attention mechanism at the core of the Transformer architecture.

1. **Query, Key, Value** — given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, define the projections $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, $\mathbf{V} = \mathbf{X}\mathbf{W}^V$ and explain what each represents semantically.

2. **Attention formula** — write and explain:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
Why is the $\sqrt{d_k}$ scaling necessary?

3. **Multi-head attention** — write the formulas for concatenating $H$ heads. What does running multiple heads in parallel achieve?

4. **Computational complexity** — state the time and memory complexity of self-attention with respect to sequence length $n$ and model dimension $d$. Why is this a problem for long sequences?

5. **Causal masking** — how is the attention matrix modified for decoder (autoregressive) models to prevent attending to future tokens?""",
        "grading_criteria": r"""- Q, K, V semantics: Q = what am I looking for; K = what do I contain; V = what I will contribute; projections learned independently
- Attention formula correct; $\sqrt{d_k}$ scaling prevents softmax saturation when $d_k$ is large (dot products grow as $O(d_k)$ → saturated softmax → near-zero gradients)
- Multi-head: $\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\mathbf{W}^O$ where $\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)$; multiple heads learn different relationship types in parallel
- Complexity: $O(n^2 d)$ time and $O(n^2 + nd)$ memory; the $n^2$ term from $\mathbf{Q}\mathbf{K}^T$ is the bottleneck for long sequences
- Causal mask: add $-\infty$ (or large negative) to upper triangle of $\mathbf{Q}\mathbf{K}^T$ before softmax → future positions get zero attention weight""",
    },
    {
        "title": "[Generative AI] Positional Encoding & Tokenisation",
        "subject_area": "generative_ai",
        "description": r"""Explain how Transformers represent position information and how text is tokenised.

1. **Why positional encoding is needed** — show that without it, self-attention is permutation-invariant. Why is this a problem for language modelling?

2. **Sinusoidal positional encoding** — the original Transformer adds:
$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$
Explain why this encoding allows the model to attend to relative positions, and two advantages over learned positional encodings.

3. **Rotary Position Embedding (RoPE)** — describe the key idea: multiplying Q and K by a rotation matrix based on position, so that $\mathbf{q}_m^T\mathbf{k}_n$ depends only on $m - n$. Why is this useful for long-context models?

4. **Byte-Pair Encoding (BPE) tokenisation** — describe the BPE algorithm (merge most frequent pairs iteratively). What is the tradeoff between vocabulary size and sequence length?

5. **Tokenisation challenges** — give two examples of inputs that common tokenisers handle poorly and explain why.""",
        "grading_criteria": r"""- Permutation invariance: attention($\pi(\mathbf{X})$) = $\pi$(attention($\mathbf{X})$); word order is meaningless without PE → can't distinguish "cat bites dog" from "dog bites cat"
- Sinusoidal: relative position representable as linear function of absolute PE; extrapolates to unseen sequence lengths; not a parameter → no extra training; fixed prevents overfitting to training lengths
- RoPE: Q and K rotated by angle proportional to position; inner product depends only on relative offset; benefits: length generalisation, computationally efficient, used in LLaMA/GPT-NeoX
- BPE: start with character vocabulary; iteratively merge most frequent adjacent pair; larger vocab → fewer tokens per sentence but larger embedding table; typical modern vocab: 32k-100k tokens
- Tokenisation issues: (1) numbers split arbitrarily (e.g. "1234" → "12"+"34") → arithmetic is hard; (2) non-English languages over-segmented because training data was mostly English → higher token cost per word""",
    },
    {
        "title": "[Generative AI] BERT vs GPT: Pretraining Objectives",
        "subject_area": "generative_ai",
        "description": r"""Compare the pretraining objectives and architectures of BERT (encoder) and GPT (decoder) models, and explain what each is suited for.

1. **BERT's Masked Language Modelling (MLM)** — describe the objective. 15% of tokens are masked; write the loss function. Why does MLM force the model to learn bidirectional context?

2. **GPT's Causal Language Modelling (CLM)** — write the autoregressive factorisation:
$$p(\mathbf{x}) = \prod_{t=1}^T p(x_t | x_1, \ldots, x_{t-1})$$
Explain the next-token prediction loss.

3. **Architectural differences** — BERT uses a full (bidirectional) attention mask; GPT uses a causal mask. How does this affect what each model can do at inference time?

4. **Task suitability** — give two tasks where BERT-style models excel and two where GPT-style models excel. Explain the reasoning.

5. **T5 and the encoder-decoder paradigm** — how does T5 unify both paradigms using a text-to-text framework? Describe its pretraining objective (span corruption).""",
        "grading_criteria": r"""- MLM loss: $-\sum_{i \in \text{masked}} \log P(x_i | \mathbf{x}_{\text{rest}})$; bidirectionality: model sees tokens from both left and right to predict masked token → learns rich contextual representations
- CLM loss: $-\sum_t \log p(x_t|x_{<t})$; autoregressive factorisation correct; each position only attends to past → causal mask; training signal at every position
- BERT: full attention → deep contextual representations but cannot generate; GPT: causal attention → can generate autoregressively but doesn't have bidirectional context
- BERT excels: classification, NER, QA (SQuAD-style), semantic similarity; GPT excels: text generation, summarisation, translation, few-shot prompting
- T5: all tasks framed as "input text → output text"; span corruption: replace contiguous spans with sentinel tokens, predict spans; encoder processes input, decoder generates output""",
    },
    {
        "title": "[Generative AI] Fine-Tuning Strategies: Full Fine-Tuning vs LoRA/PEFT",
        "subject_area": "generative_ai",
        "description": r"""Compare strategies for adapting large pretrained language models to downstream tasks.

1. **Full fine-tuning** — describe the procedure and its drawbacks at scale (memory, storage, catastrophic forgetting).

2. **Low-Rank Adaptation (LoRA)** — the key idea is to decompose the weight update as:
$$\Delta\mathbf{W} = \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k)$$
Explain why this reduces trainable parameters. How is $\mathbf{A}$ initialised and why?

3. **QLoRA** — describe how quantising the base model to 4-bit (NF4) while keeping LoRA adapters in bfloat16 enables fine-tuning very large models on a single GPU.

4. **Prompt tuning and prefix tuning** — explain these approaches as alternatives to weight modification. What are their parameter counts?

5. **When to use each** — given a task, model size, and hardware budget, describe how you would decide between full fine-tuning, LoRA, prompt tuning, and in-context learning (ICL).""",
        "grading_criteria": r"""- Full fine-tuning: updates all $N$ parameters; requires $N$-parameter gradients and optimiser states (Adam: $3N$ memory) → impractical for 7B+ models; separate copy per task; risk of catastrophic forgetting of pretrained knowledge
- LoRA: parameter count = $r(d+k)$ vs full $dk$; for $r=16$, $d=k=4096$ → 131k vs 16.8M params (< 1%); $\mathbf{A}$ initialised with random Gaussian, $\mathbf{B}$ with zeros → $\Delta\mathbf{W}=0$ at start (no disruption)
- QLoRA: NF4 quantisation reduces memory $4\times$; double quantisation further compresses; bfloat16 adapters maintain gradient precision; allows 65B model on single 48GB GPU
- Prompt tuning: prepend trainable "soft tokens" to input; prefix tuning: prepend trainable vectors to each layer's K,V; both extremely parameter-efficient (thousands of params)
- Decision framework: small model or small data → full FT; large model, task-specific → LoRA; no GPU or API-only → ICL or prompt tuning; memory-constrained → QLoRA""",
    },
    {
        "title": "[Generative AI] Retrieval-Augmented Generation (RAG)",
        "subject_area": "generative_ai",
        "description": r"""Explain the architecture, motivation, and limitations of Retrieval-Augmented Generation (RAG).

1. **Motivation** — why do LLMs need external retrieval? Discuss knowledge cutoffs, hallucinations, and context length constraints.

2. **RAG pipeline** — describe each component: document chunking, embedding model, vector store (approximate nearest neighbour search), retrieval, and generation. Write the augmented prompt structure.

3. **Embedding similarity** — explain cosine similarity $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ as the retrieval criterion and why it is used over Euclidean distance for high-dimensional embeddings.

4. **Naive RAG vs Advanced RAG** — describe at least three improvements in advanced RAG: e.g., re-ranking (cross-encoder), hybrid search (dense + BM25), query expansion, or HyDE (hypothetical document embeddings).

5. **Limitations** — identify at least three failure modes of RAG and describe mitigations for each.""",
        "grading_criteria": r"""- Motivation: parametric knowledge frozen at training cutoff; hallucination when model lacks knowledge; RAG grounds generation in retrieved evidence
- Pipeline: chunk documents (fixed-size or semantic); embed with model (e.g. `text-embedding-3`); store in vector DB (FAISS, Pinecone); at query time: embed query, ANN search, retrieve top-k, prepend to prompt; prompt = [system] + [retrieved context] + [user query]
- Cosine similarity: measures angular similarity; normalisation makes it scale-invariant → better for semantic similarity where magnitude doesn't reflect meaning; Euclidean sensitive to vector norm
- Advanced RAG: re-ranker uses cross-encoder to score (query, doc) pairs more accurately; hybrid search combines sparse (BM25, keyword) and dense (semantic) for recall+precision; HyDE: generate hypothetical answer, embed it, retrieve similar real documents
- Limitations: (1) retrieval can fail → missing evidence → hallucination → mitigate with larger $k$ or query expansion; (2) context window overflow → hierarchical or iterative retrieval; (3) inconsistent/contradictory retrieved chunks → explicit grounding and citation""",
    },
]


class Command(BaseCommand):
    help = "Load ML/AI theory questions (25 total, 5 per topic)."

    def add_arguments(self, parser):
        parser.add_argument("--flush", action="store_true",
                            help="Delete all existing theory problems first.")

    def handle(self, *args, **options):
        if options["flush"]:
            count, _ = Problem.objects.filter(category="theory").delete()
            self.stdout.write(self.style.WARNING(f"Deleted {count} existing theory problem(s)."))

        created = skipped = 0
        for ch in CHALLENGES:
            obj, was_created = Problem.objects.get_or_create(
                title=ch["title"],
                defaults={
                    "category": "theory",
                    "subject_area": ch["subject_area"],
                    "description": ch["description"].strip(),
                    "grading_criteria": ch["grading_criteria"].strip(),
                },
            )
            if was_created:
                created += 1
                self.stdout.write(f"  ✓ {obj.title}")
            else:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"  – Skipped (exists): {obj.title}"))

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! {created} theory question(s) created, {skipped} skipped."
        ))
