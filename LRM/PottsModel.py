import numpy as np
from collections import defaultdict
from typing import  Optional
from numpy.linalg import inv
from scipy.ndimage import convolve
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln



# ============================================================
# Helpers: 4-neighbor tuples and counts (periodic wrap)
# ============================================================

def _neighbor_tuples_4(grid):
    """Return neighbor tuples (top, bottom, left, right) for each site; shape (N, 4)."""
    top    = np.roll(grid,  1, axis=0)
    bottom = np.roll(grid, -1, axis=0)
    left   = np.roll(grid,  1, axis=1)
    right  = np.roll(grid, -1, axis=1)
    neighbors = np.stack([top, bottom, left, right], axis=-1)  # (L, M, 4)
    return neighbors.reshape(-1, 4)  # (N, 4)


def _neighbor_counts_4(grid, Xi):
    """
    For each site and each state in S, count 4-neighbor matches via convolution.
    Returns a (N, q) integer array where q = len(S), N = L*L.
    """
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    masks = np.array([grid == s for s in Xi])  # (q, L, M)
    counts = np.array([convolve(mask.astype(int), kernel, mode='wrap') for mask in masks])  # (q, L, M)
    return counts.reshape(len(Xi), -1).T  # (N, q)


# ============================================================
# Utilities for packing neighbor 4-tuples into integer keys
# ============================================================

def _factorize_columns(neigh):
    """
    Map each entry of the (N,4) neighbor matrix to a dense 0..K-1 index using
    a global value table 'vals'. Returns (vals, neigh_comp) where neigh_comp has same shape as neigh.
    """
    vals, inv = np.unique(neigh, return_inverse=True)
    return vals, inv.reshape(neigh.shape)


def _pack_keys(neigh_comp, K):
    """
    Pack 4 base-K digits per row into a single integer key (mixed radix).
    neigh_comp: (N,4) with entries in 0..K-1.
    """
    bases = (K ** np.arange(neigh_comp.shape[1], dtype=np.int64))  # [1, K, K^2, K^3]
    return (neigh_comp * bases).sum(axis=1).astype(np.int64)  # (N,)


# ============================================================
# EmpiricalPottsConditional (vectorized; Laplace smoothing)
# ============================================================

class EmpiricalPottsConditional:
    """
    Empirical P(center | 4-neighbor tuple) with Laplace smoothing (alpha).
    Vectorized storage: keep a table of probabilities indexed by packed neighbor keys.
    Unseen tuples -> uniform (if alpha>0) else zeros (if alpha==0).
    """
    def __init__(self, grid, alpha=0.0):
        self.alpha = float(alpha)

        # States and mappings
        self.Xi = np.unique(grid)
        self.q = len(self.Xi)
        self.state_to_index = {s: i for i, s in enumerate(self.Xi)}

        # Build neighbor tuples and centers
        neigh = _neighbor_tuples_4(grid) # (N,4)
        centers = grid.ravel() # (N,)
        cidx = np.searchsorted(self.Xi, centers) # (N,) indices into 0..q-1

        # Factorize neighbor values globally and pack to keys
        self._vals, neigh_comp = _factorize_columns(neigh) # (V,), (N,4) in 0..V-1
        self._K = int(self._vals.size)
        keys = _pack_keys(neigh_comp, self._K) # (N,)

        # Unique keys and inverse mapping
        uniq_keys, inv = np.unique(keys, return_inverse=True) # (U,), (N,)
        self._uniq_keys = uniq_keys
        self._key2row = {int(k): i for i, k in enumerate(uniq_keys)} # key -> row index

        # Counts per unique neighbor tuple and class (vectorized)
        counts = np.zeros((uniq_keys.size, self.q), dtype=np.int64) # (U,q)
        np.add.at(counts, (inv, cidx), 1)

        # Convert counts -> probs with Laplace smoothing
        if self.alpha > 0.0:
            den = counts.sum(axis=1, keepdims=True) + self.q * self.alpha
            self._probs = (counts.astype(np.float64) + self.alpha) / den # (U,q)
        else:
            den = counts.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                self._probs = counts.astype(np.float64) / den
                self._probs[np.isnan(self._probs)] = 0.0 # rows with den=0

        # Convenience default rows
        self._uniform = np.full(self.q, 1.0 / self.q, dtype=np.float64)
        self._zeros   = np.zeros(self.q, dtype=np.float64)

    # ---- Packing helpers for queries ----
    def _pack_tuple(self, neigh_tuple):
        """Pack a single 4-tuple using this object's value table and radix."""
        comp = np.searchsorted(self._vals, np.asarray(neigh_tuple))
        return int(_pack_keys(comp.reshape(1, -1), self._K)[0])

    def _pack_many(self, neigh_matrix):
        """Pack a (m,4) matrix of 4-tuples; returns (m,) int64 keys."""
        comp = np.searchsorted(self._vals, np.asarray(neigh_matrix))
        return _pack_keys(comp, self._K)

    # ---- Public API ----
    def probs_for_tuple(self, neigh_tuple):
        key = self._pack_tuple(neigh_tuple)
        row = self._key2row.get(key, None)
        if row is None:
            return self._uniform.copy() if self.alpha > 0.0 else self._zeros.copy()
        return self._probs[row]

    def prob(self, spin_value, neighbors):
        idx = self.state_to_index.get(spin_value, None)
        if idx is None:
            return 0.0
        return float(self.probs_for_tuple(neighbors)[idx])

    def log_prob(self, spin_value, neighbors):
        p = self.prob(spin_value, neighbors)
        return np.log(p) if p > 0 else -np.inf

    # ---- Vectorized bulk retrieval (for PottsModel) ----
    def probs_for_keys(self, keys):
        """
        Given an array of packed keys (m,), return P(.|neigh) rows (m,q).
        Unseen rows -> uniform (alpha>0) or zeros (alpha==0).
        """
        m = int(len(keys))
        out = (np.full((m, self.q), 1.0 / self.q, dtype=np.float64)
               if self.alpha > 0.0 else np.zeros((m, self.q), dtype=np.float64))
        rows = np.fromiter((self._key2row.get(int(k), -1) for k in keys), count=m, dtype=np.int64)
        found = rows >= 0
        out[found] = self._probs[rows[found]]
        return out

    def pack_keys(self, neigh_matrix):
        """Expose packing for external callers (matches this object's value table)."""
        return self._pack_many(neigh_matrix)

    def summarize_conditionals(self, quantiles=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99)):
        """
        Summarize all q(s | neighbors) values stored in self._probs.
        You can pass any iterable of quantiles in (0,1). Median (0.5) is always included.
        """
        if self._probs is None or self._probs.size == 0:
            print("no conditionals stored — empty _probs.")
            return None

        # sanitize & split quantiles around median
        qs = sorted(float(q) for q in quantiles if 0.0 < q < 1.0 and q != 0.5)
        qs_lo = [q for q in qs if q < 0.5]
        qs_hi = [q for q in qs if q > 0.5]

        vals = self._probs.ravel()
        summary = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        }
        for q in qs:
            summary[q] = float(np.quantile(vals, q))

        print("Summary of q(s | neighbors):")
        print(f"    min: {summary['min']:.6f}")
        for q in qs_lo:
            label = f"q{int(round(q*100)):02d}"
            print(f"{label:>6}: {summary[q]:.6f}")
        print(f" median: {summary['median']:.6f}")
        for q in qs_hi:
            label = f"q{int(round(q*100)):02d}"
            print(f"{label:>6}: {summary[q]:.6f}")
        print(f"    max: {summary['max']:.6f}")
        print(f"   mean: {summary['mean']:.6f}")

        return summary

# ==========================================
# PottsModel 
# ==========================================

class PottsModel:

    def __init__(self, grid, alpha=0.0, log_eps=np.log(1e-6)):
        self.grid  = np.asarray(grid)
        self.alpha = (0.1 if alpha is None else float(alpha))
        self.log_eps = float(log_eps)

        # States and mappings
        self.Xi = np.unique(self.grid)
        self.q  = len(self.Xi)

        # Center indices (0..q-1) in the SAME order as S
        self._center_idx = np.searchsorted(self.Xi, self.grid.ravel()).astype(np.int64)  # (N,)
        self._N = self.grid.size

        # Empirical conditional (vectorized)
        self.empirical = EmpiricalPottsConditional(self.grid, alpha=self.alpha)

        # Precompute neighbor tuples/keys and counts
        self._neighbor_tuples = _neighbor_tuples_4(self.grid)            # (N,4)
        self._neighbor_keys   = self.empirical.pack_keys(self._neighbor_tuples)  # (N,)
        self._neighbor_counts = _neighbor_counts_4(self.grid, self.Xi)   # (N,q)

        # Compute  and  once on full data (vectorized)
        self._Lambda, self._nu = self._compute_matrices_from_indices(np.arange(self._N), self.empirical)

        # Bootstrap cache
        self._boot_idx = None
        self._boot_Lambdas = None
        self._boot_Nus = None


    def _compute_matrices_from_indices(self, idx, empirical):

        idx  = np.asarray(idx, dtype=np.int64)
        ncnt = self._neighbor_counts[idx] # (m, q)
        cidx = self._center_idx[idx] # (m,)
        keys = self._neighbor_keys[idx] # (m,)
        m, q = ncnt.shape
        eps = np.exp(self.log_eps)

        # Empirical probabilities for all sites
        P = empirical.probs_for_keys(keys) # (m, q)
        pc = P[np.arange(m), cidx] # (m,)

        # u(i,.) = n_s - n_c; zero current column
        u = ncnt - ncnt[np.arange(m), cidx][:, None]
        u[np.arange(m), cidx] = 0

        # Per-state gating: include only states with p_s > eps; exclude current column
        mask = (P > eps)
        mask[np.arange(m), cidx] = False           # (m, q)

        # Λ over included pairs
        Lambda = float(np.sum((u * u)[mask], dtype=np.float64))

        #  over included pairs; guard pc == 0 for safety
        valid_pc_rows = pc > 0
        if np.any(valid_pc_rows):
            rows  = np.where(valid_pc_rows)[0]
            u2    = u[rows]
            P2    = P[rows]
            c2    = cidx[rows]
            mask2 = mask[rows]

            # logs only where used
            logP2 = np.zeros_like(P2, dtype=np.float64)
            logP2[mask2] = np.log(P2[mask2])
            logPc2 = np.log(P2[np.arange(rows.size), c2])[:, None]

            contrib = (u2 * (logP2 - logPc2)) * mask2
            nu = float(np.sum(contrib, dtype=np.float64))
        else:
            nu = 0.0

        return Lambda, nu


    # -------------------- posterior (scalar) --------------------
    def posterior(self, beta, prior_mean, prior_var):

        b  = float(beta)
        pv = float(prior_var)
        if pv <= 0:
            raise ValueError("prior_var must be positive.")
        A = (1.0 / pv) + 2.0 * b * self._Lambda
        Sigma = 1.0 / A
        mu = Sigma * ((prior_mean / pv) + 2.0 * b * self._nu)
        return float(mu), float(Sigma)


    # -------------------- point estimate --------------------
    def estimate_param_full(self, ridge=1e-12):
        denom = self._Lambda + float(ridge)
        return 0.0 if denom == 0.0 else float(self._nu / denom)


    # -------------------- bootstrap prep (vectorized) --------------------
    def _set_bootstrap_weight_cache(self, B=200, seed=12345, scheme="multinomial"):
        n = self._N
        rng = np.random.default_rng(seed)
        if scheme == "multinomial":
            p = np.full(n, 1.0 / n, dtype=float)
            W = rng.multinomial(n, p, size=B).astype(np.int64)
        elif scheme == "poisson":
            W = rng.poisson(1.0, size=(B, n)).astype(np.int64)
        else:
            raise ValueError("scheme must be 'multinomial' or 'poisson'")
        self._boot_idx = [np.repeat(np.arange(n, dtype=np.int64), w) for w in W]


    def _counts_from_indices(self, idx):
    
        idx = np.asarray(idx, dtype=np.int64)
        sub_keys = self._neighbor_keys[idx]     # (m,)
        sub_cidx = self._center_idx[idx]        # (m,)
        uniq_sub_keys, inv = np.unique(sub_keys, return_inverse=True)  # (U,), (m,)
        counts = np.zeros((uniq_sub_keys.size, self.q), dtype=np.int64)
        np.add.at(counts, (inv, sub_cidx), 1)
        return uniq_sub_keys, counts


    def _lookup_from_counts(self, uniq_keys, counts):
        
        a = self.alpha
        q = self.q

        if a > 0.0:
            den = counts.sum(axis=1, keepdims=True) + q * a
            probs = (counts.astype(np.float64) + a) / den  # (U,q)
        else:
            den = counts.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                probs = counts.astype(np.float64) / den
                probs[np.isnan(probs)] = 0.0

        key2row = {int(k): i for i, k in enumerate(uniq_keys)}
        uniform = np.full(q, 1.0 / q, dtype=np.float64)
        zeros   = np.zeros(q, dtype=np.float64)
        alpha   = a

        # Reuse the SAME packing as the main empirical (consistent value table)
        vals = self.empirical._vals
        K    = self.empirical._K

        def _pack_many(neigh_matrix):
            comp = np.searchsorted(vals, np.asarray(neigh_matrix))
            return _pack_keys(comp, K)

        class _Lookup:
            def probs_for_keys(self, keys):
                m = int(len(keys))
                out = (np.full((m, q), 1.0 / q) if alpha > 0.0
                       else np.zeros((m, q), dtype=np.float64))
                rows = np.fromiter((key2row.get(int(k), -1) for k in keys), count=m, dtype=np.int64)
                found = rows >= 0
                out[found] = probs[rows[found]]
                return out
            def probs_for_tuple(self, neigh_tuple):
                key = _pack_many(np.asarray(neigh_tuple).reshape(1, -1))[0]
                row = key2row.get(int(key), None)
                if row is None:
                    return uniform.copy() if alpha > 0.0 else zeros.copy()
                return probs[row]

        return _Lookup()


    def _prepare_bootstrap_stats(self, B=200, seed=12345, scheme="multinomial"):
        self._set_bootstrap_weight_cache(B=B, seed=seed, scheme=scheme)
        Ls, vs = [], []
        for idx in self._boot_idx:
            uniq_keys, counts = self._counts_from_indices(idx)
            lookup = self._lookup_from_counts(uniq_keys, counts)  # same alpha
            L, v = self._compute_matrices_from_indices(idx, lookup)
            Ls.append(L); vs.append(v)
        self._boot_Lambdas = np.asarray(Ls, dtype=np.float64)
        self._boot_Nus     = np.asarray(vs, dtype=np.float64)


    # -------------------- coverage (vectorized over B) --------------------
    def coverage(self, beta, prior_mean, prior_var, B=200, delta=0.05, seed=12345, scheme="multinomial"):
        if (self._boot_Lambdas is None) or (self._boot_Lambdas.shape[0] != B):
            self._prepare_bootstrap_stats(B=B, seed=seed, scheme=scheme)

        theta = self.estimate_param_full()
        pv = float(prior_var)
        if pv <= 0:
            raise ValueError("prior_var must be positive.")

        qchi = chi2.ppf(1 - delta, df=1)
        b = float(beta)

        A = (1.0 / pv) + 2.0 * b * self._boot_Lambdas     # (B,)
        mu_b = ( (prior_mean / pv) + 2.0 * b * self._boot_Nus ) / A
        d = theta - mu_b
        hits = np.sum((d * d * A) <= qchi)
        return float(hits) / float(self._boot_Lambdas.size)

    # -------------------- beta calibration (same interface) --------------------
    def fit_coverage(self, prior_mean, prior_var, delta=0.05, B=200,
                     beta_low=1e-6, beta_high=10.0, seed=12345, scheme="multinomial",
                     replications=1, verbose=False):
        target = 1.0 - delta

        def make_obj(rep_seed):
            self._set_bootstrap_weight_cache(B=B, seed=rep_seed, scheme=scheme)
            Ls, vs = [], []
            for idx in self._boot_idx:
                uniq_keys, counts = self._counts_from_indices(idx)
                lookup = self._lookup_from_counts(uniq_keys, counts)
                L, v = self._compute_matrices_from_indices(idx, lookup)
                Ls.append(L); vs.append(v)
            self._boot_Lambdas = np.asarray(Ls, dtype=np.float64)
            self._boot_Nus     = np.asarray(vs, dtype=np.float64)

            def obj(beta):
                if beta < 0:
                    return (target - 0.0) ** 2 + 1e3 * abs(beta)
                cov = self.coverage(beta=beta, prior_mean=prior_mean, prior_var=prior_var,
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
        cov_star = self.coverage(beta=beta_star, prior_mean=prior_mean, prior_var=prior_var,
                                 B=B, delta=delta, seed=seed, scheme=scheme)
        if verbose:
            print(f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} (target {1-delta:.4f}); "
                  f"fun={res.fun:.4g}, success={res.success}")
        return beta_star, cov_star, res
