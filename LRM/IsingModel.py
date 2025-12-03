import numpy as np
from numpy.linalg import inv, solve, LinAlgError
from scipy.optimize import minimize_scalar
from scipy.stats import chi2



class IsingModel:
    class _EmpiricalConditional:
        def __init__(self, grid, alpha=0.0):
            self.alpha = float(alpha)
            self.lookup_table = self._build_lookup(np.asarray(grid))

        @staticmethod
        def _encode_neighbors(neighbors):
            n = np.asarray(neighbors)
            return ((n == 1).astype(np.uint8) @ (1 << np.arange(4)))

        def _build_lookup(self, grid):
            rows, cols = grid.shape
            counts = np.zeros((16, 2), dtype=np.int64)
            top, bottom = np.roll(grid, 1, 0), np.roll(grid, -1, 0)
            left, right = np.roll(grid, 1, 1), np.roll(grid, -1, 1)

            for i in range(rows):
                for j in range(cols):
                    neighbors = np.array([top[i, j], bottom[i, j], left[i, j], right[i, j]])
                    idx = self._encode_neighbors(neighbors)
                    spin_idx = 0 if grid[i, j] == -1 else 1
                    counts[idx, spin_idx] += 1

            counts = counts.astype(np.float64)
            if self.alpha > 0.0:
                counts += self.alpha
            totals = counts.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                probs = counts / totals
                probs[np.isnan(probs)] = 0.0
            return probs

        def vectorized_log_probs(self, spins, neighbor_arrays):
            spins = np.asarray(spins)
            neighbor_arrays = np.asarray(neighbor_arrays)
            indices = self._encode_neighbors(neighbor_arrays)
            spin_indices = (spins == 1).astype(np.uint8)
            probs = self.lookup_table[indices, spin_indices]
            with np.errstate(divide='ignore'):
                log_probs = np.where(probs > 0, np.log(probs), -np.inf)
            return log_probs

    def __init__(self, grid, alpha=0.0):
        self.grid = np.asarray(grid)
        self.alpha = float(alpha)

        self.empirical = IsingModel._EmpiricalConditional(self.grid, alpha=self.alpha)

        s = self.grid
        top, bottom = np.roll(s, 1, 0), np.roll(s, -1, 0)
        left, right = np.roll(s, 1, 1), np.roll(s, -1, 1)
        self._neighbors_full = np.stack([top, bottom, left, right], axis=-1).reshape(-1, 4)
        self._s_flat = s.ravel()
        self._sn_flat = self._neighbors_full.sum(axis=1)

        self._neighbor_idx = self.empirical._encode_neighbors(self._neighbors_full)
        self._spin_idx = (self._s_flat == 1).astype(np.uint8)

        self._Lambda, self._nu = self._compute_matrices_from_indices(
            np.arange(self._s_flat.size), self.empirical.lookup_table
        )

        self._boot_idx = None
        self._boot_Lambdas = None
        self._boot_Nus = None

    def _compute_matrices_from_indices(self, idx, lookup_table):
        idx = np.asarray(idx, dtype=np.int64)
        s_i, sn_i = self._s_flat[idx], self._sn_flat[idx]
        s2_i, sn2_i = s_i * s_i, sn_i * sn_i

        L11 = 4.0 * np.sum(s2_i)
        L12 = 4.0 * np.sum(s2_i * sn_i)
        L22 = 4.0 * np.sum(s2_i * sn2_i)
        Lambda = np.array([[L11, L12], [L12, L22]], dtype=float)

        spin_i = self._spin_idx[idx]
        nidx_i = self._neighbor_idx[idx]
        probs = lookup_table[nidx_i, spin_i]
        probs_comp = lookup_table[nidx_i, 1 - spin_i]

        valid = (probs > 0.0) & (probs_comp > 0.0)
        if not np.any(valid):
            return Lambda, np.zeros((2, 1), dtype=float)

        s_v, sn_v = s_i[valid], sn_i[valid]
        q_log_ratio = np.log(probs_comp[valid]) - np.log(probs[valid])
        v1 = -2.0 * np.sum(s_v * q_log_ratio)
        v2 = -2.0 * np.sum(s_v * sn_v * q_log_ratio)
        nu = np.array([v1, v2], dtype=float).reshape(2, 1)
        return Lambda, nu

    def posterior(self, beta, prior_mean, prior_cov):
        b = float(beta)
        P0_inv = inv(np.asarray(prior_cov, dtype=float).reshape(2, 2))
        mu0 = np.asarray(prior_mean, dtype=float).reshape(2, 1)
        A = P0_inv + 2.0 * b * self._Lambda
        Sigma = inv(A)
        mu = Sigma @ (P0_inv @ mu0 + 2.0 * b * self._nu)
        return mu, Sigma

    def _lookup_from_counts(self, counts):
        counts = counts.astype(np.float64)
        if self.alpha > 0.0:
            counts += self.alpha
        totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            probs = counts / totals
            probs[np.isnan(probs)] = 0.0
        return probs

    def _counts_from_indices(self, idx):
        counts = np.zeros((16, 2), dtype=np.int64)
        nidx, sp = self._neighbor_idx[idx], self._spin_idx[idx]
        np.add.at(counts, (nidx, sp), 1)
        return counts

    def _estimate_params_from_indices(self, idx, lookup_table, ridge=1e-12):
        L, v = self._compute_matrices_from_indices(idx, lookup_table)
        try:
            w = solve(L + ridge * np.eye(2), v)
        except LinAlgError:
            w = solve(L + 1e-8 * np.eye(2), v)
        return w.ravel()

    def estimate_params_full(self, ridge=1e-12):
        N = self._s_flat.size
        idx_all = np.arange(N)
        return self._estimate_params_from_indices(idx_all, self.empirical.lookup_table, ridge=ridge)

    def _set_bootstrap_weight_cache(self, B=200, seed=12345, scheme="multinomial"):
        n = self._s_flat.size
        rng = np.random.default_rng(seed)
        if scheme == "multinomial":
            p = np.full(n, 1.0 / n, dtype=float)
            W = rng.multinomial(n, p, size=B).astype(np.int64)
        elif scheme == "poisson":
            W = rng.poisson(1.0, size=(B, n)).astype(np.int64)
        else:
            raise ValueError("scheme must be 'multinomial' or 'poisson'")
        self._boot_idx = [np.repeat(np.arange(n), w) for w in W]

    def _prepare_bootstrap_stats(self, B=200, seed=12345, scheme="multinomial"):
        self._set_bootstrap_weight_cache(B=B, seed=seed, scheme=scheme)
        Ls, vs = [], []
        for idx in self._boot_idx:
            counts = self._counts_from_indices(idx)
            lookup = self._lookup_from_counts(counts)
            L, v = self._compute_matrices_from_indices(idx, lookup)
            Ls.append(L); vs.append(v)
        self._boot_Lambdas = np.stack(Ls, axis=0)
        self._boot_Nus     = np.stack(vs, axis=0)

    def coverage(self, beta, prior_mean, prior_cov, B=200, delta=0.05, seed=12345, scheme="multinomial"):
        if (self._boot_Lambdas is None) or (self._boot_Lambdas.shape[0] != B):
            self._prepare_bootstrap_stats(B=B, seed=seed, scheme=scheme)

        theta = self.estimate_params_full().reshape(-1, 1)
        SigInv = inv(np.asarray(prior_cov, dtype=float).reshape(2, 2))
        mu0 = np.asarray(prior_mean, dtype=float).reshape(2, 1)
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
                lookup = self._lookup_from_counts(counts)
                L, v = self._compute_matrices_from_indices(idx, lookup)
                Ls.append(L); vs.append(v)
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





