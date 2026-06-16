import numpy as np
from numpy.linalg import inv, solve
from scipy.stats import multivariate_normal, chi2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp



class SmoothedEmpiricalDensity1D:
   
    def __init__(self, alpha: float = 1.0):
        assert alpha >= 0.0, "alpha must be >= 0"
        self.alpha = float(alpha)
        self.total: int = 0
        self.M: int = -1
        self.K_obs: int = 0 # number of observed unique values
        self._counts: np.ndarray | None = None
        self._pmf_obs: np.ndarray | None = None
        self._logpmf_obs: np.ndarray | None = None
        self._p_extra: float | None = None # p(M+1)
        self._logp_extra: float | None = None

    def fit(self, data):
        x = np.asarray(data)
        assert x.ndim == 1, "fit() expects a 1D array (n,)"
        if x.size == 0:
            raise ValueError("fit() requires at least one sample")
        if not np.issubdtype(x.dtype, np.integer):
            if np.allclose(x, np.round(x)):
                x = x.astype(int)
            else:
                raise ValueError("data must be integers (counts)")
        if np.any(x < 0):
            raise ValueError("data must be nonnegative counts")

        self.total = int(x.size)
        self.M = int(x.max())
        u = np.unique(x)
        self.K_obs = int(u.size)

        # counts on [0..M]
        self._counts = np.bincount(x, minlength=self.M + 1).astype(float)

        if self.alpha == 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                pmf_obs = self._counts / float(self.total)
                logpmf_obs = np.log(pmf_obs, where=(pmf_obs > 0),
                                    out=np.full_like(pmf_obs, -np.inf))
            self._pmf_obs = pmf_obs
            self._logpmf_obs = logpmf_obs
            self._p_extra = 0.0
            self._logp_extra = -np.inf
        else:
            denom = float(self.total) + self.alpha * float(self.K_obs)
            pmf_obs = (self._counts + self.alpha) / denom
            logpmf_obs = np.log(pmf_obs)
            # single extra bin at M+1 with alpha/denom
            p_extra = self.alpha / denom
            self._pmf_obs = pmf_obs
            self._logpmf_obs = logpmf_obs
            self._p_extra = float(p_extra)
            self._logp_extra = float(np.log(p_extra))

        return self

    def pmf(self, x):
        x = np.asarray(x)
        flat = x.ravel()
        out = np.zeros_like(flat, dtype=float)

        m_int_nonneg = (flat >= 0) & np.equal(flat, np.round(flat))
        idx_obs = m_int_nonneg & (flat <= self.M)
        if np.any(idx_obs):
            out[idx_obs] = self._pmf_obs[flat[idx_obs].astype(int)]

        idx_extra = m_int_nonneg & (flat == self.M + 1)
        if np.any(idx_extra):
            out[idx_extra] = self._p_extra

        return out.reshape(x.shape)

    def logpmf(self, x):
        x = np.asarray(x)
        flat = x.ravel()
        out = np.full_like(flat, -np.inf, dtype=float)

        m_int_nonneg = (flat >= 0) & np.equal(flat, np.round(flat))
        idx_obs = m_int_nonneg & (flat <= self.M)
        if np.any(idx_obs):
            out[idx_obs] = self._logpmf_obs[flat[idx_obs].astype(int)]

        idx_extra = m_int_nonneg & (flat == self.M + 1)
        if np.any(idx_extra):
            out[idx_extra] = self._logp_extra

        return out.reshape(x.shape)

    # sklearn-style helper
    def score_samples(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            assert X.shape[1] == 1, "score_samples expects (n,1) for 2D inputs"
            vals = X[:, 0]
        else:
            vals = X
        return self.logpmf(vals).astype(float)
    
    def clone(self):
        return type(self)(alpha=self.alpha)







import numpy as np


class GoodTuringDensity1D:
    """
    Good--Turing PMF estimator on a truncated nonnegative integer support.

    Support:
        {0, ..., max_val}

    For observed symbols with count r > 0, use

        p_hat(x) = (r + 1) N_{r+1} / (n N_r)

    when this is available. Otherwise fall back to empirical frequency.

    The total unseen mass is

        p_unseen = N_1 / n

    and is spread uniformly over unobserved bins in {0, ..., max_val}.
    """

    def __init__(self, max_val: int, fallback: str = "empirical"):
        if max_val < 0:
            raise ValueError("max_val must be >= 0")
        if fallback not in {"empirical", "zero"}:
            raise ValueError("fallback must be 'empirical' or 'zero'")

        self.max_val = int(max_val)
        self.fallback = fallback

        self.total: int = 0
        self.M: int = self.max_val
        self.K_obs: int = 0

        self._counts: np.ndarray | None = None
        self._pmf_obs: np.ndarray | None = None
        self._logpmf_obs: np.ndarray | None = None
        self._freq_counts: np.ndarray | None = None
        self._p_unseen_total: float | None = None

    def fit(self, data):
        x = np.asarray(data)

        if x.ndim != 1:
            raise ValueError("fit() expects a 1D array (n,)")
        if x.size == 0:
            raise ValueError("fit() requires at least one sample")

        if not np.issubdtype(x.dtype, np.integer):
            if np.allclose(x, np.round(x)):
                x = x.astype(int)
            else:
                raise ValueError("data must be integers")

        if np.any(x < 0):
            raise ValueError("data must be nonnegative counts")

        if np.any(x > self.max_val):
            raise ValueError(
                "data contains values larger than max_val. "
                "Increase max_val or pre-truncate/bin the data."
            )

        self.total = int(x.size)

        # Counts over the full truncated support {0, ..., max_val}
        counts = np.bincount(x, minlength=self.max_val + 1).astype(int)
        counts = counts[: self.max_val + 1]
        self._counts = counts.astype(float)

        positive_counts = counts[counts > 0]
        self.K_obs = int(positive_counts.size)

        if self.K_obs == 0:
            raise ValueError("No observed bins found.")

        max_r = int(positive_counts.max())

        # N_r = number of symbols observed exactly r times
        # Need length max_r + 2 so N_{r+1} is accessible.
        freq_counts = np.bincount(positive_counts, minlength=max_r + 2).astype(float)
        self._freq_counts = freq_counts

        pmf = np.zeros(self.max_val + 1, dtype=float)

        # Good--Turing estimates for observed bins
        for val in range(self.max_val + 1):
            r = counts[val]

            if r == 0:
                continue

            N_r = freq_counts[r]
            N_r_plus_1 = freq_counts[r + 1] if r + 1 < len(freq_counts) else 0.0

            if N_r > 0 and N_r_plus_1 > 0:
                pmf[val] = ((r + 1) * N_r_plus_1) / (self.total * N_r)
            else:
                if self.fallback == "empirical":
                    pmf[val] = r / self.total
                else:
                    pmf[val] = 0.0

        # Good--Turing unseen mass
        N_1 = freq_counts[1] if len(freq_counts) > 1 else 0.0
        p_unseen_total = N_1 / self.total

        unseen = counts == 0
        n_unseen = int(unseen.sum())

        if n_unseen > 0:
            pmf[unseen] = p_unseen_total / n_unseen
        else:
            p_unseen_total = 0.0

        # Renormalise because fallback + raw GT can make total mass != 1
        s = pmf.sum()
        if s <= 0:
            raise RuntimeError("Estimated PMF has zero total mass.")
        pmf = pmf / s

        self._pmf_obs = pmf
        self._p_unseen_total = float(p_unseen_total)

        with np.errstate(divide="ignore"):
            self._logpmf_obs = np.log(
                pmf,
                where=(pmf > 0),
                out=np.full_like(pmf, -np.inf, dtype=float),
            )

        return self

    def pmf(self, x):
        x = np.asarray(x)
        flat = x.ravel()
        out = np.zeros_like(flat, dtype=float)

        valid = (
            (flat >= 0)
            & np.equal(flat, np.round(flat))
            & (flat <= self.max_val)
        )

        if np.any(valid):
            out[valid] = self._pmf_obs[flat[valid].astype(int)]

        return out.reshape(x.shape)

    def logpmf(self, x):
        x = np.asarray(x)
        flat = x.ravel()
        out = np.full_like(flat, -np.inf, dtype=float)

        valid = (
            (flat >= 0)
            & np.equal(flat, np.round(flat))
            & (flat <= self.max_val)
        )

        if np.any(valid):
            out[valid] = self._logpmf_obs[flat[valid].astype(int)]

        return out.reshape(x.shape)

    def score_samples(self, X):
        X = np.asarray(X)

        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError("score_samples expects shape (n, 1) for 2D inputs")
            vals = X[:, 0]
        else:
            vals = X

        return self.logpmf(vals).astype(float)
    
    def clone(self):
        return type(self)(max_val=self.max_val, fallback=self.fallback)



class NPMLEEmpiricalBayesDensity1D:
    """
    Robbins / Kiefer-Wolfowitz NPMLE empirical-Bayes estimator
    for a finite 1D discrete support {0, ..., max_val}.

    Model:
        N_x | lambda_x ~ Pois(lambda_x),
        lambda_x ~ G,

    where N_x is the observed count of symbol x in the sample.

    The NPMLE estimates G as a discrete distribution over a fixed grid
    of lambda values. Then q_hat(x) is obtained from the posterior mean
    E[lambda | N_x], normalized over x.
    """

    def __init__(
        self,
        max_val=None,
        grid_size=300,
        max_iter=2000,
        tol=1e-8,
        eps=1e-300,
        verbose=False,
    ):
        self.max_val = max_val
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.verbose = verbose

    def fit(self, data):
        data = np.asarray(data, dtype=int).ravel()

        if np.any(data < 0):
            raise ValueError("Expected nonnegative integer data.")

        if self.max_val is None:
            self.max_val = int(data.max())

        self.support = np.arange(self.max_val + 1)
        self.K = len(self.support)
        self.n = len(data)

        # Counts N_x for each symbol x in {0, ..., max_val}
        self.counts = np.bincount(data, minlength=self.max_val + 1)[:self.max_val + 1]

        # Grid for lambda = n q_x.
        # Quadratic grid puts more resolution near 0, where rare symbols live.
        t = np.linspace(0.0, 1.0, self.grid_size)
        self.lambda_grid = self.n * t**2

        # Avoid exact duplication if n is small.
        self.lambda_grid = np.unique(self.lambda_grid)
        self.grid_size = len(self.lambda_grid)

        # Log Poisson probabilities:
        # log Pois(c; lambda) = -lambda + c log(lambda) - log(c!)
        c = self.counts[:, None]
        lam = self.lambda_grid[None, :]

        log_pois = np.empty((self.K, self.grid_size))

        positive_lam = lam > 0

        log_pois[:, :] = -np.inf

        # lambda > 0
        log_pois[:, positive_lam.ravel()] = (
            -lam[:, positive_lam.ravel()]
            + c * np.log(lam[:, positive_lam.ravel()])
            - gammaln(c + 1.0)
        )

        # lambda = 0: Pois(0;0)=1 and Pois(c>0;0)=0
        zero_lam_idx = np.where(self.lambda_grid == 0.0)[0]
        if len(zero_lam_idx) > 0:
            j0 = zero_lam_idx[0]
            log_pois[self.counts == 0, j0] = 0.0
            log_pois[self.counts > 0, j0] = -np.inf

        self.log_pois = log_pois

        # EM for mixture weights over lambda_grid
        w = np.ones(self.grid_size) / self.grid_size

        prev_ll = -np.inf

        for it in range(self.max_iter):
            log_w = np.log(np.maximum(w, self.eps))
            log_joint = self.log_pois + log_w[None, :]

            log_denom = logsumexp(log_joint, axis=1, keepdims=True)
            resp = np.exp(log_joint - log_denom)

            w_new = resp.mean(axis=0)
            w_new = np.maximum(w_new, 0.0)
            w_new = w_new / w_new.sum()

            ll = np.sum(log_denom)

            if self.verbose and (it % 100 == 0 or it == self.max_iter - 1):
                print(f"EM iter {it:4d}: log-lik = {ll:.6f}")

            if np.abs(ll - prev_ll) < self.tol * (1.0 + np.abs(prev_ll)):
                w = w_new
                break

            w = w_new
            prev_ll = ll

        self.weights = w
        self.n_iter_ = it + 1
        self.loglik_ = float(prev_ll)

        # Posterior mean E[lambda | N_x]
        log_w = np.log(np.maximum(self.weights, self.eps))
        log_joint = self.log_pois + log_w[None, :]
        log_denom = logsumexp(log_joint, axis=1, keepdims=True)
        resp = np.exp(log_joint - log_denom)

        lambda_post_mean = resp @ self.lambda_grid

        # Convert lambda estimates to q estimates and normalize.
        probs = lambda_post_mean / self.n
        probs = np.maximum(probs, 0.0)

        if probs.sum() <= 0:
            raise RuntimeError("NPMLE produced zero total probability mass.")

        self.probs = probs / probs.sum()

        return self

    def pmf(self, x):
        x = np.asarray(x, dtype=int)
        out = np.zeros_like(x, dtype=float)

        mask = (x >= 0) & (x <= self.max_val)
        out[mask] = self.probs[x[mask]]

        return out

    def logpmf(self, x):
        p = self.pmf(x)
        return np.log(np.maximum(p, self.eps))

    def __call__(self, x):
        return self.pmf(x)
    
    def clone(self):
        return NPMLEEmpiricalBayesDensity1D(
            max_val=self.max_val,
            grid_size=self.grid_size,
            max_iter=self.max_iter,
            tol=self.tol,
            eps=self.eps,
            verbose=self.verbose,
        )



class CMP1D:
    """
    CMP for 1-D count data with matching set M(x) = {x+1}.
    T(x) = [x, -log(x!)] so T(x+1)-T(x) = [1, -log(x+1)].
    """

    def __init__(self, empirical, cutoff: float = -1e10):
        assert hasattr(empirical, "logpmf"), "empirical must have logpmf(x)"
        assert hasattr(empirical, "clone"), "empirical must have clone()"
        assert cutoff < -1e2, "Cutoff is too large."
        self.empirical = empirical
        self.param_size = 2
        self.cutoff = float(cutoff)

    def _Lambda_nu(self, data_1d):
        x = np.asarray(data_1d)
        assert x.ndim == 1, "data must be a 1-D array (n,)"

        if not np.issubdtype(x.dtype, np.integer):
            if np.allclose(x, np.round(x)):
                x = x.astype(int)
            else:
                raise ValueError("data must be integer counts")
        if np.any(x < 0):
            raise ValueError("data must be nonnegative")

        xp1 = x + 1
        t1 = np.ones_like(x, dtype=float)
        t2 = -np.log(xp1.astype(float))

        log_q_x = self.empirical.logpmf(x)
        log_q_xp1 = self.empirical.logpmf(xp1)
        log_ratio = log_q_xp1 - log_q_x

        valid = np.isfinite(log_ratio) & (log_ratio > self.cutoff)

        s1 = float(np.sum(t1[valid]))
        s2 = float(np.sum(t2[valid]))
        s22 = float(np.sum(t2[valid] ** 2))
        lr1 = float(np.sum(log_ratio[valid]))
        lr2 = float(np.sum(t2[valid] * log_ratio[valid]))

        Lambda = np.array([[s1, s2], [s2, s22]], dtype=float)
        nu = np.array([[lr1], [lr2]], dtype=float)
        ignored = int((~valid).sum())

        return Lambda, nu, ignored

    def posterior(self, data, beta: float, mu_prior, Sigma_prior, return_matrices=False):
        assert isinstance(beta, float) and beta > 0.0

        mu_prior = np.asarray(mu_prior, dtype=float).reshape(2, 1)
        Sigma_prior = np.asarray(Sigma_prior, dtype=float).reshape(2, 2)

        Lambda, nu, _ = self._Lambda_nu(np.asarray(data).ravel())

        Sig0_inv = inv(Sigma_prior)
        A = Sig0_inv + 2.0 * beta * Lambda
        Sigma_post = inv(A)
        mu_post = Sigma_post @ (Sig0_inv @ mu_prior + 2.0 * beta * nu)

        post = multivariate_normal(mean=mu_post.ravel(), cov=Sigma_post)
        return (post, Lambda, nu) if return_matrices else post

    def _prepare_bootstrap_cache(self, data, prior_mean, prior_cov, B=200, delta=0.05, seed=12345):
        rng = np.random.default_rng(seed)
        x = np.asarray(data).ravel()
        n = x.shape[0]

        L_hat, nu_hat, _ = self._Lambda_nu(x)
        ridge = 0.0
        if np.linalg.cond(L_hat) > 1e12:
            ridge = 1e-8 * np.trace(L_hat) / L_hat.shape[0]

        theta_hat = solve(L_hat + ridge * np.eye(2), nu_hat)
        q = float(chi2.ppf(1 - delta, df=2))
        boot_idx = rng.integers(0, n, size=(B, n))

        L_list = np.empty((B, 2, 2), dtype=float)
        nu_list = np.empty((B, 2, 1), dtype=float)

        for b in range(B):
            xb = x[boot_idx[b]]
            emp_b = self.empirical.clone().fit(xb)
            model_b = CMP1D(empirical=emp_b, cutoff=self.cutoff)
            L_list[b], nu_list[b], _ = model_b._Lambda_nu(xb)

        return {
            "theta_hat": theta_hat,
            "q": q,
            "boot_L": L_list,
            "boot_nu": nu_list,
            "mu0": np.asarray(prior_mean, float).reshape(2, 1),
            "Sig0": np.asarray(prior_cov, float).reshape(2, 2),
        }

    def _coverage_from_cache(self, beta: float, cache) -> float:
        theta_hat = cache["theta_hat"]
        q = cache["q"]
        mu0 = cache["mu0"]
        Sig0 = cache["Sig0"]
        Sig0_inv = inv(Sig0)

        hits = 0
        for L_b, nu_b in zip(cache["boot_L"], cache["boot_nu"]):
            A = Sig0_inv + 2.0 * beta * L_b
            try:
                A_inv = inv(A)
            except np.linalg.LinAlgError:
                eps = 1e-8 * np.trace(A) / A.shape[0]
                A_inv = inv(A + eps * np.eye(2))

            mu_b = A_inv @ (Sig0_inv @ mu0 + 2.0 * beta * nu_b)
            d = theta_hat - mu_b
            hits += float(d.T @ (A @ d)) <= q

        return hits / cache["boot_L"].shape[0]

    def fit_coverage(self, data, prior_mean, prior_cov, delta=0.05, B=200,
                     beta_low=1e-4, beta_high=10.0, seed=12345, verbose=False):
        cache = self._prepare_bootstrap_cache(
            data=data,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            B=B,
            delta=delta,
            seed=seed,
        )

        target = 1.0 - delta

        def objective(beta):
            if beta < 0:
                return target**2 + 1e6 * (abs(beta) + 1.0)
            return (self._coverage_from_cache(beta, cache) - target) ** 2

        res = minimize_scalar(
            objective,
            bounds=(max(0.0, float(beta_low)), float(beta_high)),
            method="bounded",
            options={"xatol": 1e-3},
        )

        beta_star = float(res.x)
        cov_star = float(self._coverage_from_cache(beta_star, cache))

        if verbose:
            print(
                f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} "
                f"(target {target:.4f}); fun={res.fun:.4g}, success={res.success}"
            )

        return beta_star, cov_star, res




# class CMP1D:
#     """
#     Minimal CMP for 1-D count data with ***specific*** matching set M(x) = {x+1} (non-circular).
#     T(x) = [ x,  -log(x!) ]^T  =>  T(x+1)-T(x) = [ 1,  -log(x+1) ]^T
#     Parameter dimension = 2.
#     """

#     def __init__(self, empirical, cutoff: float = -1e10):
#         if not hasattr(empirical, "logpmf"):
#             raise TypeError("empirical must provide a logpmf(x) method")

#         self.empirical = empirical
#         self.param_size = 2
#         assert cutoff < -1e2, "Cutoff is too large."
#         self.cutoff = float(cutoff)

#     # ---- vectorized Lambda and nu over data ----
#     def _Lambda_nu(self, data_1d):
#         x = np.asarray(data_1d)
#         assert x.ndim == 1, "data must be a 1-D array (n,)"
#         if not np.issubdtype(x.dtype, np.integer):
#             if np.allclose(x, np.round(x)):
#                 x = x.astype(int)
#             else:
#                 raise ValueError("data must be integer counts")
#         if np.any(x < 0):
#             raise ValueError("data must be nonnegative")

#         xp1 = x + 1
#         t1 = np.ones_like(x, dtype=float)
#         t2 = np.zeros_like(x, dtype=float)

#         # With the legacy+one extension, valid neighbors are xp1>=0;
#         # probabilities at xp1==M+1 are handled by the empirical.
#         valid_nonneg = (xp1 >= 0)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             t2[valid_nonneg] = -np.log(xp1[valid_nonneg])

#         log_q_x = self.empirical.logpmf(x)
#         log_q_xp1 = self.empirical.logpmf(xp1)
#         log_ratio = log_q_xp1 - log_q_x

#         valid = valid_nonneg & (log_ratio > self.cutoff)

#         s1 = float(np.sum(t1[valid]))
#         s2 = float(np.sum(t2[valid]))
#         s22 = float(np.sum(t2[valid] * t2[valid]))
#         lr1 = float(np.sum(log_ratio[valid]))
#         lr2 = float(np.sum(t2[valid] * log_ratio[valid]))

#         Lambda = np.array([[s1, s2],
#                            [s2, s22]], dtype=float)
#         nu = np.array([[lr1],
#                        [lr2]], dtype=float)
#         ignored = int((~valid).sum())
#         return Lambda, nu, ignored

#     # ---- posterior ----
#     def posterior(self, data, beta: float, mu_prior, Sigma_prior, return_matrices=False):
#         mu_prior = np.asarray(mu_prior, dtype=float).reshape(2, 1)
#         Sigma_prior = np.asarray(Sigma_prior, dtype=float).reshape(2, 2)
#         assert isinstance(beta, float) and beta > 0.0

#         Lambda, nu, _ = self._Lambda_nu(np.asarray(data).ravel())

#         Sig0_inv = inv(Sigma_prior)
#         A = Sig0_inv + 2.0 * beta * Lambda
#         Sigma_post = inv(A)
#         mu_post = Sigma_post @ (Sig0_inv @ mu_prior + 2.0 * beta * nu)

#         post = multivariate_normal(mean=mu_post.ravel(), cov=Sigma_post)
#         return (post, Lambda, nu) if return_matrices else post

#     # ---- fast coverage with SciPy ----
#     def _prepare_bootstrap_cache(self, data, prior_mean, prior_cov, B=200, delta=0.05, seed=12345):
#         rng = np.random.default_rng(seed)
#         x = np.asarray(data).ravel()
#         n = x.shape[0]

#         L_hat, nu_hat, _ = self._Lambda_nu(x)
#         ridge = 0.0
#         if np.linalg.cond(L_hat) > 1e12:
#             ridge = 1e-8 * np.trace(L_hat) / L_hat.shape[0]
#         theta_hat = solve(L_hat + ridge * np.eye(2), nu_hat)

#         q = float(chi2.ppf(1 - delta, df=2))
#         boot_idx = rng.integers(0, n, size=(B, n))

#         alpha_emp = self.empirical.alpha
#         L_list = np.empty((B, 2, 2), dtype=float)
#         nu_list = np.empty((B, 2, 1), dtype=float)

#         for b in range(B):
#             xb = x[boot_idx[b]]
#             # IMPORTANT: fit the SAME legacy+one empirical on bootstrap sample
#             emp_b = SmoothedEmpiricalDensity1D(alpha=alpha_emp).fit(xb)
#             model_b = CMP1D(empirical=emp_b, cutoff=self.cutoff)
#             L_b, nu_b, _ = model_b._Lambda_nu(xb)
#             L_list[b] = L_b
#             nu_list[b] = nu_b

#         cache = {
#             "theta_hat": theta_hat,
#             "q": q,
#             "boot_L": L_list,
#             "boot_nu": nu_list,
#             "mu0": np.asarray(prior_mean, float).reshape(2, 1),
#             "Sig0": np.asarray(prior_cov, float).reshape(2, 2),
#         }
#         return cache

#     def _coverage_from_cache(self, beta: float, cache) -> float:
#         theta_hat = cache["theta_hat"]
#         q = cache["q"]
#         mu0 = cache["mu0"]
#         Sig0 = cache["Sig0"]
#         Sig0_inv = inv(Sig0)

#         Ls = cache["boot_L"]
#         nus = cache["boot_nu"]

#         hits = 0
#         for L_b, nu_b in zip(Ls, nus):
#             A = Sig0_inv + 2.0 * beta * L_b
#             try:
#                 A_inv = inv(A)
#             except np.linalg.LinAlgError:
#                 eps = 1e-8 * np.trace(A) / A.shape[0]
#                 A_inv = inv(A + eps * np.eye(2))
#             mu_b = A_inv @ (Sig0_inv @ mu0 + 2.0 * beta * nu_b)
#             d = theta_hat - mu_b
#             val = float(d.T @ (A @ d))  # Σ_post^{-1} = A
#             hits += (val <= q)
#         return hits / Ls.shape[0]

#     def fit_coverage(self, data, prior_mean, prior_cov, delta=0.05, B=200,
#                      beta_low=1e-4, beta_high=10.0, seed=12345, verbose=False):
#         cache = self._prepare_bootstrap_cache(
#             data=data, prior_mean=prior_mean, prior_cov=prior_cov,
#             B=B, delta=delta, seed=seed
#         )
#         target = 1.0 - delta

#         def objective(beta):
#             if beta < 0:
#                 return (target - 0.0)**2 + 1e6 * (abs(beta) + 1.0)
#             cov = self._coverage_from_cache(beta, cache)
#             return (cov - target) ** 2

#         bounds = (max(0.0, float(beta_low)), float(beta_high))
#         res = minimize_scalar(objective, bounds=bounds, method="bounded",
#                               options={"xatol": 1e-3})
#         beta_star = float(res.x)
#         cov_star = float(self._coverage_from_cache(beta_star, cache))
#         if verbose:
#             print(f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} "
#                   f"(target {target:.4f}); fun={res.fun:.4g}, success={res.success}")
#         return beta_star, cov_star, res
