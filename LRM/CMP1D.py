import numpy as np
from numpy.linalg import inv, solve
from scipy.stats import multivariate_normal, chi2
from scipy.optimize import minimize_scalar


class SmoothedEmpiricalDensity1D:
   
    def __init__(self, alpha: float = 1.0):
        assert alpha >= 0.0, "alpha must be >= 0"
        self.alpha = float(alpha)
        self.total: int = 0
        self.M: int = -1
        self.K_obs: int = 0                  # number of observed unique values
        self._counts: np.ndarray | None = None
        self._pmf_obs: np.ndarray | None = None
        self._logpmf_obs: np.ndarray | None = None
        self._p_extra: float | None = None   # p(M+1)
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


class CMP1D:
    """
    Minimal CMP for 1-D count data with neighbor x -> x+1 (non-circular).
    T(x) = [ x,  -log(x!) ]^T  =>  T(x+1)-T(x) = [ 1,  -log(x+1) ]^T
    Parameter dimension = 2.
    """

    def __init__(self, empirical: SmoothedEmpiricalDensity1D, cutoff: float = -1e10):
        assert isinstance(empirical, SmoothedEmpiricalDensity1D)
        self.empirical = empirical
        self.param_size = 2
        assert cutoff < -1e2, "Cutoff is too large."
        self.cutoff = float(cutoff)

    # ---- vectorized Λ and ν over data ----
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
        t2 = np.zeros_like(x, dtype=float)

        # With the legacy+one extension, valid neighbors are xp1>=0;
        # probabilities at xp1==M+1 are handled by the empirical.
        valid_nonneg = (xp1 >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t2[valid_nonneg] = -np.log(xp1[valid_nonneg])

        log_q_x = self.empirical.logpmf(x)
        log_q_xp1 = self.empirical.logpmf(xp1)
        log_ratio = log_q_xp1 - log_q_x

        valid = valid_nonneg & (log_ratio > self.cutoff)

        s1 = float(np.sum(t1[valid]))
        s2 = float(np.sum(t2[valid]))
        s22 = float(np.sum(t2[valid] * t2[valid]))
        lr1 = float(np.sum(log_ratio[valid]))
        lr2 = float(np.sum(t2[valid] * log_ratio[valid]))

        Lambda = np.array([[s1, s2],
                           [s2, s22]], dtype=float)
        nu = np.array([[lr1],
                       [lr2]], dtype=float)
        ignored = int((~valid).sum())
        return Lambda, nu, ignored

    # ---- posterior ----
    def posterior(self, data, beta: float, mu_prior, Sigma_prior, return_matrices=False):
        mu_prior = np.asarray(mu_prior, dtype=float).reshape(2, 1)
        Sigma_prior = np.asarray(Sigma_prior, dtype=float).reshape(2, 2)
        assert isinstance(beta, float) and beta > 0.0

        Lambda, nu, _ = self._Lambda_nu(np.asarray(data).ravel())

        Sig0_inv = inv(Sigma_prior)
        A = Sig0_inv + 2.0 * beta * Lambda
        Sigma_post = inv(A)
        mu_post = Sigma_post @ (Sig0_inv @ mu_prior + 2.0 * beta * nu)

        post = multivariate_normal(mean=mu_post.ravel(), cov=Sigma_post)
        return (post, Lambda, nu) if return_matrices else post

    # ---- fast coverage with SciPy ----
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

        alpha_emp = self.empirical.alpha
        L_list = np.empty((B, 2, 2), dtype=float)
        nu_list = np.empty((B, 2, 1), dtype=float)

        for b in range(B):
            xb = x[boot_idx[b]]
            # IMPORTANT: fit the SAME legacy+one empirical on bootstrap sample
            emp_b = SmoothedEmpiricalDensity1D(alpha=alpha_emp).fit(xb)
            model_b = CMP1D(empirical=emp_b, cutoff=self.cutoff)
            L_b, nu_b, _ = model_b._Lambda_nu(xb)
            L_list[b] = L_b
            nu_list[b] = nu_b

        cache = {
            "theta_hat": theta_hat,
            "q": q,
            "boot_L": L_list,
            "boot_nu": nu_list,
            "mu0": np.asarray(prior_mean, float).reshape(2, 1),
            "Sig0": np.asarray(prior_cov, float).reshape(2, 2),
        }
        return cache

    def _coverage_from_cache(self, beta: float, cache) -> float:
        theta_hat = cache["theta_hat"]
        q = cache["q"]
        mu0 = cache["mu0"]
        Sig0 = cache["Sig0"]
        Sig0_inv = inv(Sig0)

        Ls = cache["boot_L"]
        nus = cache["boot_nu"]

        hits = 0
        for L_b, nu_b in zip(Ls, nus):
            A = Sig0_inv + 2.0 * beta * L_b
            try:
                A_inv = inv(A)
            except np.linalg.LinAlgError:
                eps = 1e-8 * np.trace(A) / A.shape[0]
                A_inv = inv(A + eps * np.eye(2))
            mu_b = A_inv @ (Sig0_inv @ mu0 + 2.0 * beta * nu_b)
            d = theta_hat - mu_b
            val = float(d.T @ (A @ d))  # Σ_post^{-1} = A
            hits += (val <= q)
        return hits / Ls.shape[0]

    def fit_coverage(self, data, prior_mean, prior_cov, delta=0.05, B=200,
                     beta_low=1e-4, beta_high=10.0, seed=12345, verbose=False):
        cache = self._prepare_bootstrap_cache(
            data=data, prior_mean=prior_mean, prior_cov=prior_cov,
            B=B, delta=delta, seed=seed
        )
        target = 1.0 - delta

        def objective(beta):
            if beta < 0:
                return (target - 0.0)**2 + 1e6 * (abs(beta) + 1.0)
            cov = self._coverage_from_cache(beta, cache)
            return (cov - target) ** 2

        bounds = (max(0.0, float(beta_low)), float(beta_high))
        res = minimize_scalar(objective, bounds=bounds, method="bounded",
                              options={"xatol": 1e-3})
        beta_star = float(res.x)
        cov_star = float(self._coverage_from_cache(beta_star, cache))
        if verbose:
            print(f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} "
                  f"(target {target:.4f}); fun={res.fun:.4g}, success={res.success}")
        return beta_star, cov_star, res
