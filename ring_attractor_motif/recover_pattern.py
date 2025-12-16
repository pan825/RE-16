import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, quadratic_assignment

# -----------------------------
# Utility: correlation safely
# -----------------------------
def _corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float((a @ b) / denom)

def _mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))

# -----------------------------
# Signature feature (order-invariant)
# -----------------------------
def _node_signature(W, i, m=12):
    # sort -> order-invariant
    out = np.sort(W[i, :])[-m:]
    inn = np.sort(W[:, i])[-m:]
    return np.concatenate([out, inn]).astype(float)

def _template_signature(T, j, m=12):
    out = np.sort(T[j, :])[-m:]
    inn = np.sort(T[:, j])[-m:]
    return np.concatenate([out, inn]).astype(float)

def _cos_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float((a @ b) / denom)

# -----------------------------
# Step A: pick K nodes from N using Hungarian on signatures
# -----------------------------
def select_k_nodes_by_template(W_large, T, m=12):
    """
    Returns selected indices (in W_large index space), length K.
    """
    N = W_large.shape[0]
    K = T.shape[0]
    cost = np.zeros((N, K), dtype=float)

    Tfeat = [_template_signature(T, j, m=m) for j in range(K)]
    Lfeat = [_node_signature(W_large, i, m=m) for i in range(N)]

    for i in range(N):
        for j in range(K):
            # maximize similarity -> minimize negative similarity
            cost[i, j] = -_cos_sim(Lfeat[i], Tfeat[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind gives K matched rows (selected neurons), but note: N>=K
    # linear_sum_assignment on (N,K) returns K pairs.
    return row_ind  # length K

# -----------------------------
# Step B: reorder selected K nodes to match template using QAP (FAQ)
# -----------------------------
from scipy.optimize import quadratic_assignment
import numpy as np

def reorder_to_template_by_qap(W_sub, T, maximize=True, seed=0):
    """
    Compatible with older SciPy (no maximize=, maybe no rng=).
    Returns:
      P: permutation within selected K that makes W_sub[P][:,P] align with T
      res: raw result object/dict from scipy
    """
    # Old SciPy FAQ: typically minimizes trace(A^T P B P^T)
    # If we want to maximize similarity, minimize the negative similarity:
    A = T
    B = W_sub
    if maximize:
        A = -A  # turn maximize into minimize

    # Some older SciPy doesn't accept rng= either, so use seed kw if present, else ignore
    try:
        res = quadratic_assignment(A, B, method="faq", seed=seed)
    except TypeError:
        res = quadratic_assignment(A, B, method="faq")

    # SciPy returns either OptimizeResult with attribute `col_ind` or a dict-like
    P = res["col_ind"] if isinstance(res, dict) else res.col_ind
    return P, res

def reorder_to_template_by_qap_multiseed(W_sub, T, n_trials=30):
    best = None
    best_score = -np.inf

    for seed in range(n_trials):
        P, _ = reorder_to_template_by_qap(W_sub, T, maximize=True, seed=seed)
        W_try = W_sub[np.ix_(P, P)]
        score = _corr(W_try, T)

        if score > best_score:
            best_score = score
            best = (P, W_try, score)

    return best  # (perm, recovered_matrix, corr)
    
def masked_qap_matrix(W, T, q=0.85):
    thr = np.quantile(T, q)
    mask = (T >= thr).astype(float)
    return W * mask

# -----------------------------
# Scoring: includes "strong-edge mask" alignment score
# -----------------------------
def score_recovery(W_rec, T, strong_q=0.90):
    """
    strong_q: quantile threshold (e.g. 0.9) on template entries (excluding diagonal optionally)
    """
    K = T.shape[0]
    # Basic scores
    corr = _corr(W_rec, T)
    mse = _mse(W_rec, T)

    # Strong-edge alignment score: do the large entries of template land on large entries of recovered?
    T_flat = T.copy().astype(float)
    # include diagonal too because your template has strong diag; keep it simple:
    thr = np.quantile(T_flat.ravel(), strong_q)
    mask = (T_flat >= thr)

    # Compare rank-like: average recovered value on those mask positions vs elsewhere
    in_mean = float(W_rec[mask].mean()) if np.any(mask) else float("nan")
    out_mean = float(W_rec[~mask].mean()) if np.any(~mask) else float("nan")
    sep = in_mean - out_mean  # bigger is better

    return {
        "corr": corr,
        "mse": mse,
        "template_strong_thr": float(thr),
        "mean_on_template_strong": in_mean,
        "mean_off_template_strong": out_mean,
        "strong_sep": sep,
    }

# -----------------------------
# Main pipeline (optional blind-sort integration)
# -----------------------------
def recover_pattern_full(
    W_shuffled,
    T,
    K=None,
    use_blind_sort=False,
    blind_sorter=None,
    m_signature=12,
    qap_maximize=True,
    seed=0,
    plot=True,
    title_prefix=""
):
    """
    Full pipeline:
      1) (optional) blind sort W_shuffled -> W_work, with perm mapping
      2) select K nodes from W_work using signature + Hungarian
      3) map selected indices back to original W_shuffled if blind sort was used
      4) build W_sub from original W_shuffled
      5) reorder W_sub via QAP to match template T
      6) return dict with indices, permutations, scores, and optionally plots

    Parameters
    ----------
    W_shuffled : (N,N)
    T          : (K,K) template
    use_blind_sort : bool
    blind_sorter   : an object with .sort(W) -> (W_sorted, perm) where W_sorted = W[perm][:,perm]
                     (e.g., your MatrixSorter(method='blind'))
    """
    N = W_shuffled.shape[0]
    if K is None:
        K = T.shape[0]

    # ---- Step 1: optional blind sort ----
    if use_bl_toggle := use_blind_sort:
        if blind_sorter is None:
            raise ValueError("use_blind_sort=True but blind_sorter is None.")
        W_work, perm = blind_sorter.sort(W_shuffled)
        # perm: indices such that W_work = W_shuffled[perm][:,perm]
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
    else:
        W_work = W_shuffled
        perm = None
        inv_perm = None

    # ---- Step 2: select K nodes in W_work space ----
    selected_in_work = select_k_nodes_by_template(W_work, T, m=m_signature)

    # ---- Step 3: map to original W_shuffled indices ----
    if use_bl_toggle:
        selected_in_orig = perm[selected_in_work]
    else:
        selected_in_orig = selected_in_work

    # ---- Step 4: extract submatrix from original W_shuffled ----
    W_sub = W_shuffled[np.ix_(selected_in_orig, selected_in_orig)]

    # ---- Step 5: reorder submatrix to match template using QAP ----
    P, W_rec, best_corr = reorder_to_template_by_qap_multiseed(
        W_sub, T, n_trials=40
    )


    # ---- Step 6: scoring ----
    scores = score_recovery(W_rec, T, strong_q=0.90)

    out = {
        "selected_indices_orig": selected_in_orig,   # length K (which neurons in W_shuffled)
        "selected_submatrix": W_sub,                 # KxK before ordering
        "recovered_matrix": W_rec,                   # KxK ordered to match template
        "scores": scores,
        "blind_sort_perm": perm,                     # None if not used
    }

    # ---- Visualization ----
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        ax = axes

        ax[0, 0].imshow(W_shuffled)
        ax[0, 0].set_title(f"{title_prefix}W_shuffled")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(T)
        ax[0, 1].set_title(f"{title_prefix}Template T")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(W_sub)
        ax[0, 2].set_title(f"{title_prefix}Selected KxK (unordered)")
        ax[0, 2].axis("off")

        if use_bl_toggle:
            ax[1, 0].imshow(W_work)
            ax[1, 0].set_title(f"{title_prefix}Blind-sorted W (work space)")
            ax[1, 0].axis("off")
        else:
            ax[1, 0].axis("off")

        ax[1, 1].imshow(W_rec)
        ax[1, 1].set_title(f"{title_prefix}Recovered (ordered to match T)")
        ax[1, 1].axis("off")

        # Difference map
        diff = (W_rec.astype(float) - T.astype(float))
        ax[1, 2].imshow(diff)
        ax[1, 2].set_title(
            f"{title_prefix}Diff (Recovered - T)\n"
            f"corr={scores['corr']:.3f}, mse={scores['mse']:.2f}, strong_sep={scores['strong_sep']:.2f}"
        )
        ax[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    return out
