import numpy as np
from numpy.linalg import inv, solve, LinAlgError
from scipy.stats import chi2
from scipy.optimize import minimize_scalar




# empirical conditional for the CMP AR model.
class SmoothedEmpiricalConditionalPMF:
    def __init__(self, x_series, k_max=10, alpha=1.0, eps=1e-12):
        self.k_max = k_max
        self.alpha = alpha
        self.eps = float(eps)
        self.cond_pmf = self._compute_smoothed_pmf(x_series)
        with np.errstate(divide='ignore'):
            self.log_cond_pmf = np.log(self.cond_pmf)

    def _compute_smoothed_pmf(self, x_series):
        x_series = np.asarray(x_series)
        cond_counts = np.zeros((self.k_max + 1, self.k_max + 1), dtype=float)
        for t in range(1, len(x_series)):
            x_prev, x_curr = x_series[t - 1], x_series[t]
            if 0 <= x_prev <= self.k_max and 0 <= x_curr <= self.k_max:
                cond_counts[x_prev, x_curr] += 1
        cond_counts += self.alpha  # Laplace smoothing

        # Safe row-normalization: rows with zero total become all-zeros (not NaNs)
        row_sums = cond_counts.sum(axis=1, keepdims=True)
        cond_pmf = np.divide(
            cond_counts, row_sums,
            out=np.zeros_like(cond_counts),
            where=row_sums > 0
        )
        return cond_pmf


    def score_samples(self, x_curr, x_prev):
        if 0 <= x_prev <= self.k_max and 0 <= x_curr <= self.k_max:
            prob = self.cond_pmf[x_prev, x_curr]
            return np.log(prob) if prob > 0 else float("-inf")
        else:
            return float("-inf")

    # Vectorized helper
    def score_batch_log(self, curr, prev):
        ok = (prev >= 0) & (prev <= self.k_max) & (curr >= 0) & (curr <= self.k_max)
        out = np.full(curr.shape, -np.inf, dtype=float)
        if np.any(ok):
            p = self.cond_pmf[prev[ok], curr[ok]]
            mask = p >= self.eps                      # ← drop tiny/zero probs
            # write only entries with p >= eps
            out_sub = np.full(p.shape, -np.inf, dtype=float)
            out_sub[mask] = np.log(p[mask])
            out[ok] = out_sub
        return out


class CMPAR1:
    """
    Inference + calibration tool for the CMP AR(1) model.
    Matches the Ising-style API:
      - estimate_params_full()
      - posterior(beta, prior_mean, prior_cov)
      - coverage(...)
      - fit_coverage(...)
    All compute-matrix logic is encapsulated inside this class.
    """
    def __init__(self, samples, k_max=10, alpha=1.0, phi=0.05, eps=1e-12):
        self.samples = np.asarray(samples, dtype=int)
        self.k_max = int(k_max)
        self.alpha = float(alpha)
        self.phi = float(phi)

        # Transition indices (like "sites" in Ising)
        self._t_idx_all = np.arange(1, len(self.samples), dtype=int)
        self.n_trans = self._t_idx_all.size

        # Empirical conditional from full data
        self.empirical = SmoothedEmpiricalConditionalPMF(
            self.samples, k_max=self.k_max, alpha=self.alpha, eps=eps
        )
        # Precompute prefix weights for vectorized T
        self._S1, self._S2 = self._prefix_weights(self.samples, self.phi)

        # Full-sample matrices
        self._Lambda, self._nu = self._compute_matrices_from_indices(self._t_idx_all, self.empirical)

        # Bootstrap caches
        self._boot_idx = None
        self._boot_Lambdas = None
        self._boot_Nus = None

    # ---------- Core sufficient stats ----------
    @staticmethod
    def _T_components_for_delta(x_curr, s1, s2, delta):
        """
        For move x -> x+delta with delta in {-1,+1}, return dT components as vector.
        Using identities:
          dT1 = delta * s1
          dT2 = delta * s2
          dT3 = -[gammaln(1+x+delta) - gammaln(1+x)]
               = -log(1+x)   if delta=+1
               = +log(x)     if delta=-1 (for x>=1)
        """
        if delta == +1:
            dT1 = s1
            dT2 = s2
            dT3 = -np.log(1.0 + x_curr)
        else:
            # delta == -1 (only valid for x_curr >= 1)
            dT1 = -s1
            dT2 = -s2
            dT3 = np.log(x_curr)
        return dT1, dT2, dT3

    @staticmethod
    def _prefix_weights(samples, phi):
        """
        S1[t] = sum_{n=1..t} phi^(n-1)
        S2[t] = sum_{n=1..t} phi^(n-1) * log(1 + samples[t-n])
        with recursions:
          S1[t] = 1 + phi*S1[t-1],  S1[0]=0
          S2[t] = log(1+s[t-1]) + phi*S2[t-1], S2[0]=0
        """
        s = np.asarray(samples, dtype=int)
        Tn = s.shape[0]
        S1 = np.zeros(Tn, dtype=float)
        S2 = np.zeros(Tn, dtype=float)
        for t in range(1, Tn):
            S1[t] = 1.0 + phi * S1[t-1]
            S2[t] = np.log(1.0 + s[t-1]) + phi * S2[t-1]
        return S1, S2

    def compute_matrices(self):
        """
        Public method (like your original free function), returns full-sample (Lambda, nu).
        Shapes: Lambda (3,3), nu (3,1).
        """
        return self._Lambda.copy(), self._nu.copy()

    def _compute_matrices_from_indices(self, idx, empirical):
        """
        Vectorized build of (Lambda, nu) restricted to transitions at indices idx.
        Terms with probability < eps are excluded from the nu summation.
        """
        s = self.samples
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return np.zeros((3, 3)), np.zeros((3, 1))

        eps = float(getattr(empirical, "eps", 1e-12))  # <- epsilon threshold

        x_prev = s[idx - 1]
        x_curr = s[idx]
        s1 = self._S1[idx]
        s2 = self._S2[idx]

        # +1 branch
        xp = x_curr + 1
        mask_p = (xp <= self.k_max)
        if np.any(mask_p):
            dT1_p, dT2_p, dT3_p = self._T_components_for_delta(
                x_curr[mask_p], s1[mask_p], s2[mask_p], +1
            )
            M_p = np.stack([dT1_p, dT2_p, dT3_p], axis=1)
        else:
            M_p = np.zeros((0, 3))

        # -1 branch
        xm = x_curr - 1
        mask_m = (x_curr > 0) & (xm >= 0)
        if np.any(mask_m):
            dT1_m, dT2_m, dT3_m = self._T_components_for_delta(
                x_curr[mask_m], s1[mask_m], s2[mask_m], -1
            )
            M_m = np.stack([dT1_m, dT2_m, dT3_m], axis=1)
        else:
            M_m = np.zeros((0, 3))

        # Lambda accumulation
        Lambda = M_p.T @ M_p + M_m.T @ M_m

        # ----- nu accumulation with epsilon filtering (exclude invalid terms) -----
        # Work directly with probabilities to avoid -inf arithmetic; then take logs.
        # base probs for the transitions in idx
        base_p = empirical.cond_pmf[x_prev, x_curr]

        # +1 branch contributions
        if np.any(mask_p):
            base_p_p = base_p[mask_p]                          # P(x_curr | x_prev)
            next_p_p = empirical.cond_pmf[x_prev[mask_p], xp[mask_p]]  # P(x_curr+1 | x_prev)
            valid_p = (base_p_p >= eps) & (next_p_p >= eps)    # exclude tiny/zero terms
            if np.any(valid_p):
                diff_p = (np.log(next_p_p[valid_p]) - np.log(base_p_p[valid_p])).reshape(-1, 1)
                nu_p = (M_p[valid_p] * diff_p).sum(axis=0, keepdims=True).T
            else:
                nu_p = np.zeros((3, 1))
        else:
            nu_p = np.zeros((3, 1))

        # -1 branch contributions
        if np.any(mask_m):
            base_p_m = base_p[mask_m]                          # P(x_curr | x_prev)
            next_p_m = empirical.cond_pmf[x_prev[mask_m], xm[mask_m]]  # P(x_curr-1 | x_prev)
            valid_m = (base_p_m >= eps) & (next_p_m >= eps)    # exclude tiny/zero terms
            if np.any(valid_m):
                diff_m = (np.log(next_p_m[valid_m]) - np.log(base_p_m[valid_m])).reshape(-1, 1)
                nu_m = (M_m[valid_m] * diff_m).sum(axis=0, keepdims=True).T
            else:
                nu_m = np.zeros((3, 1))
        else:
            nu_m = np.zeros((3, 1))

        nu = nu_p + nu_m
        return Lambda, nu


    # ---------- Posterior & parameter estimation ----------
    def posterior(self, beta, prior_mean, prior_cov):
        """
        Matches Ising-style signature: returns (mu, Sigma).
        """
        b = float(beta)
        P0_inv = inv(np.asarray(prior_cov, dtype=float).reshape(3,3))
        mu0 = np.asarray(prior_mean, dtype=float).reshape(3,1)
        A = P0_inv + 2.0 * b * self._Lambda
        Sigma = inv(A)
        mu = Sigma @ (P0_inv @ mu0 + 2.0 * b * self._nu)
        return mu, Sigma

    def estimate_params_full(self, ridge=1e-12):
        """
        Solve (Lambda + ridge I) w = nu  -> θ̂ (3,)
        """
        try:
            w = solve(self._Lambda + ridge * np.eye(3), self._nu)
        except LinAlgError:
            w = solve(self._Lambda + 1e-8 * np.eye(3), self._nu)
        return w.ravel()

    # ---------- Bootstrap helpers (Ising-style) ----------
    def _set_bootstrap_weight_cache(self, B=200, seed=12345, scheme="multinomial"):
        n = self.n_trans
        rng = np.random.default_rng(seed)
        if scheme == "multinomial":
            p = np.full(n, 1.0 / n, dtype=float)
            W = rng.multinomial(n, p, size=B).astype(np.int64)
        elif scheme == "poisson":
            W = rng.poisson(1.0, size=(B, n)).astype(np.int64)
        else:
            raise ValueError("scheme must be 'multinomial' or 'poisson'")
        base = self._t_idx_all
        self._boot_idx = [np.repeat(base, w) for w in W]

    def _counts_from_indices(self, idx):
        counts = np.zeros((self.k_max + 1, self.k_max + 1), dtype=float)
        s = self.samples
        if len(idx) == 0:
            return counts
        idx = np.asarray(idx, dtype=int)
        prev = s[idx - 1]
        curr = s[idx]
        ok = (prev >= 0) & (prev <= self.k_max) & (curr >= 0) & (curr <= self.k_max)
        np.add.at(counts, (prev[ok], curr[ok]), 1.0)
        return counts

    def _lookup_from_counts(self, counts):
        counts = counts.astype(float) + self.alpha

        # --- Safe row normalization: avoid division by zero ---
        row_sums = counts.sum(axis=1, keepdims=True)
        pmf = np.divide(
            counts, row_sums,
            out=np.zeros_like(counts),
            where=row_sums > 0
        )

        emp = SmoothedEmpiricalConditionalPMF.__new__(SmoothedEmpiricalConditionalPMF)
        emp.k_max = self.k_max
        emp.alpha = self.alpha
        emp.cond_pmf = pmf
        emp.eps = getattr(self.empirical, "eps", 1e-12)  # preserve epsilon if present
        with np.errstate(divide='ignore', invalid='ignore'):
            emp.log_cond_pmf = np.log(pmf)
        return emp

    def _prepare_bootstrap_stats(self, B=200, seed=12345, scheme="multinomial"):
        self._set_bootstrap_weight_cache(B=B, seed=seed, scheme=scheme)
        Ls, vs = [], []
        for idx in self._boot_idx:
            counts = self._counts_from_indices(idx)
            empirical_b = self._lookup_from_counts(counts)
            L_b, v_b = self._compute_matrices_from_indices(idx, empirical_b)
            Ls.append(L_b); vs.append(v_b)
        self._boot_Lambdas = np.stack(Ls, axis=0)
        self._boot_Nus     = np.stack(vs, axis=0)

    # ---------- Coverage + calibration (Ising-style) ----------
    def coverage(self, beta, prior_mean, prior_cov, B=200, delta=0.05, seed=12345, scheme="multinomial"):
        if (self._boot_Lambdas is None) or (self._boot_Lambdas.shape[0] != B):
            self._prepare_bootstrap_stats(B=B, seed=seed, scheme=scheme)

        theta = self.estimate_params_full().reshape(-1, 1)
        SigInv = inv(np.asarray(prior_cov, dtype=float).reshape(3,3))
        mu0 = np.asarray(prior_mean, dtype=float).reshape(3,1)
        rhs0 = SigInv @ mu0
        q = chi2.ppf(1 - delta, df=theta.shape[0])

        hits = 0
        b = float(beta)
        for k in range(B):
            Lb = self._boot_Lambdas[k]
            nub = self._boot_Nus[k]
            A = SigInv + 2.0 * b * Lb
            mu_b = solve(A, rhs0 + 2.0 * b * nub)
            d = theta - mu_b
            hits += float(d.T @ (A @ d)) <= q
        return hits / B

    def fit_coverage(self, prior_mean, prior_cov, delta=0.05, B=200,
                     beta_low=1e-6, beta_high=10.0, seed=12345, scheme="multinomial",
                     replications=1, verbose=False):
        target = 1.0 - delta

        def make_obj(rep_seed):
            self._set_bootstrap_weight_cache(B=B, seed=rep_seed, scheme=scheme)
            Ls, vs = [], []
            for idx in self._boot_idx:
                counts = self._counts_from_indices(idx)
                empirical_b = self._lookup_from_counts(counts)
                L_b, v_b = self._compute_matrices_from_indices(idx, empirical_b)
                Ls.append(L_b); vs.append(v_b)
            self._boot_Lambdas = np.stack(Ls, axis=0)
            self._boot_Nus     = np.stack(vs, axis=0)

            def obj(beta):
                if beta < 0:
                    return (target - 0.0) ** 2 + 1e3 * abs(beta)
                cov = self.coverage(beta=beta, prior_mean=prior_mean, prior_cov=prior_cov,
                                    B=B, delta=delta, seed=rep_seed, scheme=scheme)
                return (cov - target) ** 2
            return obj

        if replications == 1:
            obj = make_obj(seed)
            res = minimize_scalar(
                obj,
                bounds=(max(0.0, beta_low), beta_high if beta_high is not None else 10.0),
                method="bounded",
                options={"xatol": 1e-3}
            )
        else:
            objs = [make_obj(int(seed + 977 * k)) for k in range(replications)]
            def obj_avg(beta):
                vals = [f(beta) for f in objs]
                return float(np.mean(vals))
            res = minimize_scalar(
                obj_avg,
                bounds=(max(0.0, beta_low), beta_high if beta_high is not None else 10.0),
                method="bounded",
                options={"xatol": 1e-3}
            )

        beta_star = float(res.x)
        cov_star = self.coverage(beta=beta_star, prior_mean=prior_mean, prior_cov=prior_cov,
                                 B=B, delta=delta, seed=seed, scheme=scheme)
        if verbose:
            print(f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} (target {1-delta:.4f}); "
                  f"fun={res.fun:.4g}, success={res.success}")
        return beta_star, cov_star, res
