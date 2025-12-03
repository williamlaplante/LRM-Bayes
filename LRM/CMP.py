import numpy as np
from itertools import product
from collections import Counter
from scipy.special import factorial, logsumexp
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
from scipy.stats import multivariate_normal
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from joblib import Parallel, delayed




def weight_function(x : np.ndarray, med : np.ndarray, mad : np.ndarray):
    """
    Compute the robust weight function:
    
        w(x) = (1 + sum_i ((x_i - med_i)^2 / mad_i^2))^(-1/2)
    
    Parameters:
        x : np.ndarray of shape (d,)
            Input data point.
        med : np.ndarray of shape (d,)
            Median of each dimension.
        mad : np.ndarray of shape (d,)
            MAD (median absolute deviation) of each dimension.
    
    Returns:
        float: The computed weight w(x).
    """
    x = np.asarray(x)
    med = np.asarray(med)
    mad = np.asarray(mad)

    # Avoid division by zero in case of mad == 0
    mad = np.where(mad == 0, 1e-8, mad)

    #squared_terms = ((x - med) / mad) ** 2
    #weight = (1 + np.sum(squared_terms)) ** -1

    #weight = (np.exp(-med) * med**x) / gamma(x + 1) #univariate
    weight = np.exp(-np.sum(med)) * np.prod(med**x / gamma(x + 1)) #multivariate (product of marginals)
    return weight


#NOTE: score_samples takes 1 sample! 

class PMFEstimator(ABC):
    @abstractmethod
    def fit(self, data):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def prob(self, x):
        """Return the probability of the given input x."""
        pass

    @abstractmethod
    def score_samples(self, x):
        """Return the log-probability of the input x."""
        pass



#Empirical density for CMP model. Alpha=0 recovers PMF.
class SmoothedEmpiricalDensity(PMFEstimator):
    def __init__(self, alpha=1.0):
        
        #alpha: Additive smoothing parameter (alpha=0 recovers the empirical distribution).
        
        self.alpha = alpha
        self.counts = None
        self.total = None
        self.dim = None
        self.value_sets = None
        self._num_combinations = None

    def fit(self, data):
        
        #Fit the model to a dataset of shape (n_samples, d), where d >= 1.
        #Each row in `data` is a discrete d-dimensional sample.
        
        data = np.asarray(data)
        assert data.ndim == 2, "fit() expects a 2D array of shape (n_samples, d)"
        
        self.dim = data.shape[1]
        self.total = len(data)

        # Count each unique d-dimensional sample
        tuples = [tuple(row) for row in data]
        self.counts = Counter(tuples)

        # Track observed unique values in each dimension
        self.value_sets = [set() for _ in range(self.dim)]
        for row in data:
            for i, val in enumerate(row):
                self.value_sets[i].add(val)

        # Compute the number of possible d-dimensional combinations
        self._num_combinations = np.prod([len(s) for s in self.value_sets]) if self.alpha > 0 else None

    def prob(self, x):
        
        #Return the (smoothed) probability of a single d-dimensional sample `x`.
        #If alpha=0, this matches the empirical distribution.
        
        x = np.asarray(x).flatten()
        assert len(x) == self.dim, f"Input must have {self.dim} dimensions"

        x_tuple = tuple(x)
        count = self.counts.get(x_tuple, 0)

        if self.alpha == 0:
            return count / self.total if self.total > 0 else 0.0

        numerator = count + self.alpha
        denominator = self.total + self.alpha * self._num_combinations
        return numerator / denominator

    def score_samples(self, x):
        
        #Return the log-probability of a single d-dimensional sample `x`.
        #Returns -inf for unseen x when alpha=0.
        
        p = self.prob(x)
        return np.array([np.log(p)]) if p > 0 else np.array([-np.inf])







class CMP:
    """CMP model for n-dimensional count data.
    """
    def __init__(self, d : int, empirical : PMFEstimator, j_vals : list = [1], cutoff : float = - 1e10, circular : bool = True, robust : bool = False):
        """
        Args:
            d (int): dimension of the count data.
            empirical (PMFEstimator): A PMF estimator with a "score_samples" function.
            j_vals (list, optional): Neighbours for M(x). Must be integers. Defaults to [1]. 
            cutoff (float, optional): _description_. Defaults to -1e10.
            circular (bool, optional): _description_. Defaults to True.
            robust (bool, optional) : Uses the IMQ weight function for robustness when computing the loss, posterior, and point estimates. Defaults to False.
        """
        assert d >=1 and isinstance(d, int), "d must be an integer."
        self.d = d
        self.param_size = int(2*self.d + self.d*(self.d-1)/2)

        self.params = np.ones(self.param_size)
        self.empirical = empirical

        self.params = np.ones((self.param_size, 1))
        
        assert all(isinstance(x, int) for x in j_vals)
        self.j_vals = j_vals
        

        self.domain = None
        self.N_max_list = None

        self.robust = robust

        self.Z = 1.0
        assert cutoff < -1e2, "Cutoff is too large."
        self.cutoff = cutoff

        assert isinstance(circular, bool), "Circular must be boolean."
        self.circular = circular
        return
    
    
    def T(self, sample : np.ndarray) -> np.ndarray:
        """Compute T(x) for a single observation in the CMP model.

        Args:
            sample (np.ndarray): A single data point.

        Returns:
            np.ndarray: T(x) vector.
        """
        assert len(sample.flatten()) == self.d
        i, j = np.triu_indices(self.d, k=1)
        return np.block([sample, - sample[i] * sample[j], - np.log(factorial(sample))]).reshape(-1,1)
    

    def log_prob(self, sample : np.ndarray, params : np.ndarray) -> np.float64:
        """Compute the ***unnormalised*** log probability of a sample given parameters. 

        Args:
            sample (np.ndarray): A single data point.
            params (np.ndarray): A set of parameters.

        Returns:
            np.float64: ***unnormalised*** Log probability.
        """
        assert len(sample.flatten()) == self.d
        assert len(params.flatten()) == self.param_size

        T = self.T(sample)

        return np.dot(params, T)
    

    def compute_Z(self, params : np.ndarray, N_max_list : list = [25]):
        """Compute the normalisation constant of the unnormalised model given a set of parameters and a truncation per dimension.

        Args:
            params (np.ndarray): Set of parameters.
            N_max_list (list): List of values for the truncation.
        """

        assert len(N_max_list) == self.d, "The length of the list of values for the truncation must match the dimension."
        ranges = [range(N) for N in N_max_list]

        assert isinstance(params, np.ndarray), "params must be an array."
        assert params.ndim == 1, f"params must be an array (vector) of one dimension and shape ({self.param_size}, )."
        assert len(params.flatten()) == self.param_size, f"Parameter size must be {self.param_size}. Received {len(params.flatten())}."

        #Compute all combinations within truncation limits.
        combinations = list(product(*ranges))

        #Gather all the unnormalised log probabilities. 
        log_probs = []
        for comb in combinations:
            log_probs.append(self.log_prob(np.array(comb), params))
        
        #Compute and update the normalisation constant.
        self.Z = np.exp(logsumexp(log_probs))

        #Update the truncation.
        self.N_max_list = N_max_list


        #Update the parameters.
        self.params = params

        return
    

    def get_neighbors(self, x: Union[tuple, np.ndarray, list]) -> List[np.ndarray]:
        """
        Get the neighbours of a given sample, using j_vals selected.

        neighbour((x1, x2, ... , xd)) = (x1, x2, ..., x_i + j, x_{i+1}, ... , xd)
        for each dimension i and each j in self.j_vals.

        If self.circular is True, wrap into the per-dimension domain via modular arithmetic.
        Assumes each dimension's domain is a contiguous integer range [min(domain[i]), max(domain[i])].

        Args:
            x: Given sample (tuple/list/ndarray).

        Returns:
            List[np.ndarray]: Axis-aligned neighbors (change exactly one coordinate).
        """
        x = np.asarray(x).ravel()
        assert x.size == self.d, "Length of sample should match dimension."

        neighbors: List[np.ndarray] = []

        if self.circular:
            # precompute per-dimension lower bound and width (no need for hi explicitly)
            lows   = [int(np.min(self.domain[i])) for i in range(self.d)]
            widths = [int(np.max(self.domain[i])) - lows[i] + 1 for i in range(self.d)]

            for i in range(self.d):
                lo, width = lows[i], widths[i]
                for j in self.j_vals:
                    nbr = x.copy()
                    new_val = nbr[i] + j                  # candidate move
                    nbr[i] = ((new_val - lo) % width) + lo  # wrap into [lo, lo+width-1]
                    neighbors.append(nbr)
        else:
            # non-circular: just add j (may go outside domain)
            for i in range(self.d):
                for j in self.j_vals:
                    nbr = x.copy()
                    nbr[i] = nbr[i] + j
                    neighbors.append(nbr)

        return neighbors


    def compute_matrices(self, data : np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute the Lambda_n and nu_n matrices. 

        Args:
            data (np.ndarray): data of size (n_samples, d).

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: _description_
        """
        assert len(data.shape) == 2 and data.shape[1] == self.d, f"Data must be a 2D array with shape (n, {self.d}). Got shape {data.shape}."

        #compute the domain for the neighbours to perform circular conditions if needed.
        self.domain = [np.unique(data[:, i]) for i in range(self.d)]

        #if robust, compute median and mad.
        if self.robust:
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)

        #Setup matrices and count.
        Lambda = np.zeros((self.param_size, self.param_size))
        nu = np.zeros((self.param_size, 1))
        count = 0

        #Compute the matrices.
        for sample in data:
            #get neighbour per sample and compute T
            neighbors = self.get_neighbors(sample)
            neighbors_len = len(neighbors)

            T = self.T(sample)

            #If robust, compute weights.
            if self.robust:
                weight = weight_function(x=sample, med=med, mad=mad)
            else:
                weight = 1.0
            
            log_q_sample = self.empirical.score_samples(sample.reshape(1,-1)).item()

            #for each neighbour x', compute T(x') - T(x)
            for neighbor in neighbors:

                #if neighbour is in the domain only.
                if np.all(neighbor >= 0):

                    T_p = self.T(neighbor)
                    T_diff = (T_p - T)

                    log_q_neighbor = self.empirical.score_samples(neighbor.reshape(1,-1)).item()

                    log_q_ratio = log_q_neighbor - log_q_sample

                    #if log_q_ratio is bigger than cutoff only (catches -np.inf as well, which comes from p(x) = 0).
                    if  log_q_ratio > self.cutoff:
                        
                        #scale by weights.
                        Lambda += (1/neighbors_len) * T_diff @ T_diff.T * weight
                        nu += (1/neighbors_len) * T_diff * log_q_ratio * weight

                    #keep track of the number of points we ignore.
                    else:
                        count+=1
                else:
                    count+=1

        return Lambda, nu, count


    def compute_matrices_weights(
    self,
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    empirical=None
) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute (Lambda, nu, count) given data, (optional) weights, and an empirical distribution.
        """

        # ---------- ENSURE SHAPE ----------
        data = np.asarray(data, dtype=float)
        assert data.ndim == 2 and data.shape[1] == self.d

        # ---------- SMALL HELPER: safe scoring for batch or single ----------
        def _score_batch(emp, X):
            X = np.asarray(X)
            if X.ndim == 1:
                return float(emp.score_samples(X))
            out = np.empty(X.shape[0], dtype=float)
            for i in range(X.shape[0]):
                out[i] = float(emp.score_samples(X[i]))
            return out

        # ---------- STRUCTURAL CACHE (no dependence on empirical) ----------
        if not hasattr(self, "_cmf_struct"):
            self._cmf_struct = {}

        S = self._cmf_struct
        need_struct = (
            S.get("shape") != tuple(data.shape)
            or S.get("robust") != bool(getattr(self, "robust", False))
            or S.get("param_size") != self.param_size
        )
        if need_struct:
            # Domain once
            self.domain = [np.unique(data[:, i]) for i in range(self.d)]

            # Unique rows (robust, version-independent)
            uniq_rows, inv_idx = np.unique(data, axis=0, return_inverse=True)
            uniq_keys = [tuple(row.tolist()) for row in uniq_rows]
            U, p = len(uniq_keys), self.param_size

            # Robust stats once
            if self.robust:
                med = np.median(data, axis=0)
                mad = np.median(np.abs(data - med), axis=0)

            # Per-unique structural pieces (neighbors, T_x, T_diff, outer)
            neighs    = [None]*U
            T_x_list  = [None]*U
            Tdiff     = [None]*U   # list of arrays (k_u, p, 1)
            Outer     = [None]*U   # list of arrays (k_u, p, p)
            ks        = np.zeros(U, dtype=int)
            w_robust  = np.ones(U, dtype=float)

            for u, key in enumerate(uniq_keys):
                x = np.asarray(key, float)
                if x.ndim != 1 or x.size != self.d:
                    raise ValueError(f"Bad support key {key} (shape {x.shape}), expected ({self.d},).")
                x = x.reshape(self.d,)

                N = self.get_neighbors(x)
                if len(N) == 0:
                    ks[u] = 0
                    neighs[u] = np.empty((0, self.d))
                    T_x_list[u] = np.zeros((p,1))
                    Tdiff[u] = np.empty((0, p, 1))
                    Outer[u] = np.empty((0, p, p))
                    continue

                neigh = np.asarray(N, float).reshape(-1, self.d)
                k_u = neigh.shape[0]
                ks[u] = k_u
                neighs[u] = neigh

                Tx = self.T(x);  Tx = Tx if Tx.ndim == 2 else Tx.reshape(-1,1)
                T_x_list[u] = Tx

                Ts = []
                for j in range(k_u):
                    nbr = neigh[j]
                    if np.any(nbr < 0):
                        Ts.append(Tx)  # dummy; never used because valid mask excludes this nbr
                    else:
                        Ts.append(self.T(nbr.reshape(self.d,)))
                        
                Ts = [t if t.ndim == 2 else t.reshape(-1,1) for t in Ts]
                Tn = np.stack(Ts, axis=0).reshape(k_u, p, 1)
                D  = Tn - Tx.reshape(1, p, 1)
                Tdiff[u] = D
                Outer[u] = (D @ np.transpose(D, (0,2,1)))  # (k_u, p, p)

                if self.robust:
                    w_robust[u] = float(weight_function(x=x, med=med, mad=mad))

            self._cmf_struct = dict(
                shape=tuple(data.shape),
                robust=bool(getattr(self, "robust", False)),
                param_size=self.param_size,
                uniq_keys=uniq_keys,
                inv_idx=inv_idx,
                neighs=neighs,
                T_x=T_x_list,
                Tdiff=Tdiff,
                Outer=Outer,
                ks=ks,
                w_robust=w_robust,
            )
            S = self._cmf_struct

        # ---------- AGGREGATION PER BOOTSTRAP (depends on empirical + weights) ----------
        if weights is None:
            weights = np.ones(data.shape[0], float)
        weights = np.asarray(weights, float).reshape(-1)

        inv_idx = np.asarray(S["inv_idx"], np.int64).reshape(-1)
        U, p = len(S["uniq_keys"]), self.param_size

        # Build / use the empirical for THIS bootstrap
        if empirical is None:
            # If integer weights, do exact row replication
            if np.all(np.equal(np.mod(weights, 1), 0)):
                idx = np.repeat(np.arange(data.shape[0]), weights.astype(int))
                emp = SmoothedEmpiricalDensity(alpha=self.empirical.alpha)
                emp.fit(data[idx])
            else:
                # fallback: approximate by normalized rounding
                w_int = np.rint(weights / weights.mean()).astype(int)
                idx = np.repeat(np.arange(data.shape[0]), np.maximum(w_int, 0))
                emp = SmoothedEmpiricalDensity(alpha=self.empirical.alpha)
                emp.fit(data[idx])
        else:
            emp = empirical

        # collapse weights to unique rows
        w_unique = np.bincount(inv_idx, weights=weights, minlength=U).astype(float)

        Lambda = np.zeros((p,p))
        nu     = np.zeros((p,1))
        count  = 0
        cutoff = getattr(self, "cutoff", -np.inf)

        for u in range(U):
            if w_unique[u] == 0.0 or S["ks"][u] == 0:
                continue

            neigh = S["neighs"][u]           # (k_u, d)
            k_u   = neigh.shape[0]

            # Single x (1D), neighbors possibly many (2D)
            key = S["uniq_keys"][u]
            x_vec = np.asarray(key, float).reshape(self.d,)
            log_q_x = float(emp.score_samples(x_vec))
            log_q_neigh = _score_batch(emp, neigh)     # (k_u,)
            log_q_ratio = log_q_neigh - log_q_x

            # domain + cutoff
            valid = (log_q_ratio > cutoff) & np.all(neigh >= 0, axis=1)
            count += int((~valid).sum())
            if not np.any(valid):
                continue

            D = S["Tdiff"][u][valid]         # (k_v, p, 1)
            O = S["Outer"][u][valid]         # (k_v, p, p)

            scale = (w_unique[u] * S["w_robust"][u]) / float(k_u)
            Lambda += scale * O.sum(axis=0)
            nu     += scale * (D * log_q_ratio[valid].reshape(-1,1,1)).sum(axis=0)

        return Lambda, nu, count


    def point_estimate(self, data : np.ndarray) -> np.ndarray:
        """Compute point estimate from matrices.

        Args:
            data (np.ndarray): data of size (n_samples, d).

        Returns:
            np.ndarray: Point estimate (hat{theta}).
        """
        Lambda, nu, _ = self.compute_matrices(data=data)
        theta_hat = np.linalg.inv(Lambda) @ nu
        return theta_hat
    

    def posterior(self, data : np.ndarray, beta : float, mu_prior : np.ndarray, Sigma_prior : np.ndarray, return_matrices : bool = False) -> rv_frozen:
        """
        Computes the conjugate posterior for the CMP model. 

        Args:
            data (np.ndarray): data of size (n_samples, d).
            beta (float): learning rate, >0.
            mu_prior (np.ndarray): prior mean.
            Sigma_prior (np.ndarray): prior covariance.

        Returns:
            rv_frozen: multivariate normal posterior.
        """
        assert isinstance(beta, float) and beta > 0, "Beta must be a float and greater than 0."
        assert mu_prior.shape == (self.param_size, 1), f"mu_prior must be of shape {(self.param_size, 1)}. Received {mu_prior.shape}."
        assert Sigma_prior.shape == (self.param_size, self.param_size), f"Sigma_prior must be of shape {(self.param_size, self.param_size)}. Received {Sigma_prior.shape}."

        #Compute Lambda, nu matrices.
        Lambda, nu, _ = self.compute_matrices(data=data)

        #Compute posterior mean and covariance.
        Sigma_posterior = np.linalg.inv( (np.linalg.inv(Sigma_prior) + 2 * beta * Lambda) )
        mu_posterior = Sigma_posterior @ (np.linalg.inv(Sigma_prior) @ mu_prior + 2 * beta * nu)

        #Return posterior object. 
        posterior = multivariate_normal(mean = mu_posterior.flatten(), cov = Sigma_posterior)

        if return_matrices:
            return posterior, Lambda, nu
        
        else:
            return posterior
    

    def compute_divergence(self, data : np.ndarray, params : np.ndarray) -> Tuple[np.float64, int]:
        """Compute the divergence given data and a set of parameters. Can be minimised to obtain parameter estimate. 

        Args:
            data (np.ndarray): data of size (n_samples, d).
            params (np.ndarray): Set of parameters.

        Returns:
            Tuple[np.float64, int]: Returns divergence estimate and number of ignored terms.
        """
        
        #compute domain for the neighbours.
        self.domain = [np.unique(data[:, i]) for i in range(self.d)]

        assert isinstance(params, np.ndarray), "params must be an array."
        assert params.ndim == 1, f"params must be an array (vector) of one dimension and shape ({self.param_size}, )."
        assert len(params.flatten()) == self.param_size, f"Parameter size must be {self.param_size}. Received {len(params.flatten())}."

        #initialize.
        total = 0.0
        count = 0

        for sample in data:
            neighbors = self.get_neighbors(sample)

            for neighbor in neighbors:
                
                #If neighbour is in the domain.
                if np.all(neighbor >= 0):
                    #compute the estimate ratio and the model ratio.
                    log_p_ratio = self.log_prob(neighbor, params) - self.log_prob(sample, params)
                    log_q_ratio = (self.empirical.score_samples(neighbor.reshape(1,-1)).item() - self.empirical.score_samples(sample.reshape(1,-1)).item())

                    #If them PMF estimate ratio respects cutoff condition.
                    if  log_q_ratio > self.cutoff:
                        total += log_p_ratio**2 - 2 * log_p_ratio * log_q_ratio

                    else:
                        count+=1

                else:
                    count+=1    

        return total, count
    


    def marginal(self, x : Union[np.ndarray, tuple, list], i : int) -> np.float64:
        """Compute the ***normalised*** marginal p(X_i = x_i). NOTE: Must run compute_Z with the right set of parameters beforehand!

        Args:
            x (Union[np.ndarray, tuple, list]): marginal evaluated at x.
            i (int): ith marginal

        Returns:
            np.float64: marginal probability.
        """
        assert i < self.d, "marginal of x_i must have i < d. Note that i starts at 0."

        assert self.N_max_list is not None, "Must run compute_Z before computing the marginal."

        ranges = [range(N) for N in self.N_max_list]
        ranges[i] = [x]  # fix x_i = x

        # Get combinations with x_i = x
        combinations = list(product(*ranges))


        total = 0.0
        for comb in combinations:
            total += np.exp(self.log_prob(np.array(comb), self.params))

        return (total / self.Z).item()
    

    def _set_bootstrap_weight_cache(self, data, B=200, seed=12345, scheme="multinomial"):
        """
        Cache B bootstrap weight vectors of length n for reuse across beta evaluations.
        scheme: 'multinomial' (exact nonparametric bootstrap) or 'poisson'.
        """
        n = int(data.shape[0])
        rng = np.random.default_rng(seed)

        if scheme == "multinomial":
            # shape (B, n); each row sums to n
            p = np.full(n, 1.0 / n, dtype=float)
            W = rng.multinomial(n, p, size=B).astype(float)
        elif scheme == "poisson":
            # shape (B, n); E[w_i]=1, Var[w_i]=1
            W = rng.poisson(1.0, size=(B, n)).astype(float)
        else:
            raise ValueError(f"unknown scheme '{scheme}' (use 'multinomial' or 'poisson')")

        self._boot_weights = W  # (B, n)

    def _prepare_bootstrap_stats(self, data, B=200, seed=12345, scheme="multinomial", n_jobs=-1):
        n = data.shape[0]
        rng = np.random.default_rng(seed)
        if scheme == "multinomial":
            p = np.full(n, 1.0 / n)
            W = np.vstack([rng.multinomial(n, p) for _ in range(B)]).astype(float)
        else:
            W = rng.poisson(1.0, size=(B, n)).astype(float)

        # --- NEW: build an empirical on the *original* data and pass it in ---
        alpha = getattr(self.empirical, "alpha", None)
        emp0 = SmoothedEmpiricalDensity(alpha=alpha)
        emp0.fit(data)

        # Ensure the structural cache is built once (neighbors, T, etc.)
        _ = self.compute_matrices_weights(data=data, weights=np.ones(n, float), empirical=emp0)

        def one(b):
            w = W[b].ravel()
            # exact multinomial bootstrap by row replication for the *empirical*
            idx = np.repeat(np.arange(n), w.astype(int))
            emp_b = SmoothedEmpiricalDensity(alpha=alpha)
            emp_b.fit(data[idx])
            L, nu, _ = self.compute_matrices_weights(data=data, weights=w, empirical=emp_b)
            return L, nu

        out = Parallel(n_jobs=n_jobs, backend="threading", batch_size="auto")(delayed(one)(b) for b in range(B))
        Lambdas, Nus = zip(*out)

        self._boot_weights = W
        self._boot_Lambdas = np.array(Lambdas)
        self._boot_Nus     = np.array(Nus)



    
    def coverage(self, beta, data, prior_mean, prior_cov, B=200, delta=0.05, n_jobs=-1, verbose=False):
        # prepare caches once (weights + all (Lambda_b, nu_b))
        if (getattr(self, "_boot_Lambdas", None) is None
            or self._boot_Lambdas.shape[0] != B
            or self._boot_weights.shape != (B, data.shape[0])):
            self._prepare_bootstrap_stats(data, B=B, n_jobs=n_jobs)

        theta = self.point_estimate(data).reshape(-1, 1)
        SigInv = np.linalg.inv(prior_cov)
        mu0 = prior_mean.reshape(-1, 1)
        rhs0 = SigInv @ mu0
        q = chi2.ppf(1 - delta, df=mu0.shape[0])

        hits = 0
        for b in range(B):
            Lb = self._boot_Lambdas[b]
            nub = self._boot_Nus[b]
            A = SigInv + 2.0 * beta * Lb
            mu = np.linalg.solve(A, rhs0 + 2.0 * beta * nub)
            d = theta - mu
            hits += float(d.T @ (A @ d)) <= q
        return hits / B

    

    def fit_coverage(self, data, prior_mean, prior_cov, delta=0.05, B=200, beta_low=0.0001, beta_high=10.0, seed=12345,
            n_jobs=-1, verbose=False, replications=1):
        """
        Find beta by minimizing (coverage(beta) - (1-delta))^2 with SciPy.
        - Uses fixed bootstrap weights (common random numbers) for smoothness.
        - Works even if coverage(beta) is non-monotone.
        - If 'replications'>1, averages the objective across multiple independent bootstrap caches
        (slower, but reduces noise).

        Returns: beta_star, cov_star, result  (result is the SciPy OptimizeResult)
        """
        target = 1.0 - delta

        # one replication of the objective with a fixed bootstrap cache
        def make_obj(rep_seed):
            # fix bootstrap WEIGHTS once for this replication
            self._set_bootstrap_weight_cache(data, B=B, seed=rep_seed, scheme="multinomial")

            def obj(beta):
                # enforce beta >= 0 softly
                if beta < 0:
                    return (target - 0.0)**2 + (abs(beta) * 1e3)
                cov = self.coverage(beta=beta, data=data, prior_mean=prior_mean,
                                    prior_cov=prior_cov, B=B, delta=delta, n_jobs=n_jobs, verbose=False)
                return (cov - target) ** 2
            return obj

        if replications == 1:
            obj = make_obj(seed)
            method = "bounded" if beta_high is not None else "Brent"
            bounds = (max(0.0, beta_low), beta_high if beta_high is not None else None)

            if method == "bounded":
                res = minimize_scalar(obj, bounds=bounds, method="bounded", options={"xatol":1e-3})
            else:
                # Brent needs a bracket; try a loose one
                res = minimize_scalar(obj, bracket=(max(0.0, beta_low), max(0.0, beta_low)+1.0, (beta_high or 10.0)))
        else:
            # Average K independent replications of the objective to reduce noise
            objs = [make_obj(int(seed + k*9973)) for k in range(replications)]
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
        # final coverage using the same cache used in the last evaluation (re-seed for determinism)
        self._set_bootstrap_weight_cache(data, B=B, seed=seed, scheme="multinomial")
        cov_star = self.coverage(beta=beta_star, data=data, prior_mean=prior_mean,
                                prior_cov=prior_cov, B=B, delta=delta, n_jobs=n_jobs, verbose=False)

        if verbose:
            print(f"[scipy] beta*: {beta_star:.6g}, coverage: {cov_star:.4f} (target {target:.4f}); fun={res.fun:.4g}, success={res.success}")

        return beta_star, cov_star, res