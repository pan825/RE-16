import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import SpectralCoclustering
import numpy as np
import scipy.cluster.hierarchy as sch

class MatrixSorter:
    """
    A class that provides multiple algorithms for sorting/reordering matrices
    to reveal underlying structure (e.g., diagonal patterns, clusters).

    Methods:
    - 'blind': Hierarchical clustering based on node features
    - 'spectral': Spectral seriation using Fiedler vector
    - 'degree': Sort by node degree/strength
    - 'robust': Separates signal from noise, then applies spectral sorting
    - 'blind_improved': Improved blind sorting
    - 'spectral_co_clustering': Spectral co-clustering
    - 'hierarchical_clustering': Hierarchical clustering


    """
    
    def __init__(self, method='spectral'):
        """
        Initialize the MatrixSorter.
        
        Parameters:
        -----------
        method : str, optional (default='spectral')
            The sorting method to use. Options:
            - 'blind': Hierarchical clustering based on node features
            - 'spectral': Spectral seriation using Fiedler vector
            - 'degree': Sort by node degree/strength
            - 'robust': Separates signal from noise, then applies spectral sorting
            - 'blind_improved': Improved blind sorting
            - 'spectral_co_clustering': Spectral co-clustering
            - 'hierarchical_clustering': Hierarchical clustering
        """
        self.method = method
        self.available_methods = ['blind', 'blind_improved', 'spectral', 'degree', 'robust', 'spectral_co_clustering', 'hierarchical_clustering']
        
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not recognized. "
                           f"Available methods: {self.available_methods}")
    
    def sort(self, W):
        """
        Sort the matrix W using the selected method.
        
        Parameters:
        -----------
        W : np.ndarray
            (N, N) adjacency/weight matrix
        
        Returns:
        --------
        W_sorted : np.ndarray
            Reordered matrix
        perm : np.ndarray
            Permutation indices used for sorting
        """
        if self.method == 'blind':
            return self._blind_sort(W)
        elif self.method == 'blind_improved':
            return self._blind_sort_improved(W)
        elif self.method == 'spectral':
            return self._spectral_sort(W)
        elif self.method == 'degree':
            return self._degree_sort(W)
        elif self.method == 'robust':
            return self._robust_sort(W)
        elif self.method == 'spectral_co_clustering':
            return self._spectral_co_clustering_sort(W)
        elif self.method == 'hierarchical_clustering':
            return self._hierarchical_clustering_sort(W)
    
    def _blind_sort(self, W):
        """
        Hierarchical clustering based on outgoing and incoming connections.
        """
        N = W.shape[0]
        
        # Use both outgoing and incoming connections as features
        outgoing = W              # shape (N, N)
        incoming = W.T            # shape (N, N)
        
        # Concatenate into a 2N dimensional vector
        features = np.concatenate([outgoing, incoming], axis=1)  # (N, 2N)
        
        # Normalize to avoid scale differences
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_norm = features / norm
        
        # Compute pairwise distances and perform hierarchical clustering
        distance_vec = pdist(features_norm, metric='cosine')
        Z = linkage(distance_vec, method='ward')
        
        # Get the leaf order from the dendrogram
        perm = leaves_list(Z)   # shape (N,)
        
        W_sorted = W[perm][:, perm]
        
        return W_sorted, perm

    def _blind_sort_improved(self, W):
        """
        Improved blind sorting using:
        - Row-normalized outgoing and incoming connection profiles
        - Degree features (in/out) to stabilize ordering
        - Dimensionality reduction via SVD for denoising
        - Cosine distance + average-linkage hierarchical clustering
        - Optimal leaf ordering to minimize adjacent dissimilarity
        """
        N = W.shape[0]
        if N <= 1:
            perm = np.arange(N)
            return W, perm
        
        # Build feature matrix that captures both directions
        # features: [outgoing | incoming | out_degree | in_degree]
        outgoing = W.astype(float)
        incoming = W.T.astype(float)
        features = np.concatenate([outgoing, incoming], axis=1)  # (N, 2N)
        
        # Row-wise L2 normalization for scale invariance
        row_norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        features = features / row_norm
        
        # Append degree features (normalized)
        out_degree = np.sum(W, axis=1, keepdims=True).astype(float)
        in_degree = np.sum(W, axis=0, keepdims=True).T.astype(float)
        deg = np.concatenate([out_degree, in_degree], axis=1)
        deg /= (np.linalg.norm(deg, axis=1, keepdims=True) + 1e-12)
        
        features = np.concatenate([features, deg], axis=1)
        
        # Optional denoising / compression via SVD (no external deps)
        # Keep up to 32 components or the rank, whichever is smaller
        try:
            U, S, Vt = np.linalg.svd(features, full_matrices=False)
            k = min(32, U.shape[1])
            features_reduced = U[:, :k] * S[:k]
        except Exception:
            features_reduced = features
        
        # Normalize again before cosine distances
        f_norm = np.linalg.norm(features_reduced, axis=1, keepdims=True) + 1e-12
        X = features_reduced / f_norm
        
        # Cosine distance between node feature vectors
        distance_vec = pdist(X, metric='cosine')
        
        # Degenerate case: if all distances ~ 0, fall back to degree-based sort
        if np.allclose(distance_vec, 0.0):
            degrees_total = (out_degree + in_degree).ravel()
            perm = np.argsort(degrees_total)[::-1]
            W_sorted = W[perm][:, perm]
            return W_sorted, perm
        
        # Average-linkage is appropriate for cosine distances (Ward expects Euclidean)
        Z = linkage(distance_vec, method='average')
        
        # Improve the 1D seriation with optimal leaf ordering
        try:
            Z_opt = optimal_leaf_ordering(Z, distance_vec)
            perm = leaves_list(Z_opt)
        except Exception:
            perm = leaves_list(Z)
        
        W_sorted = W[perm][:, perm]
        return W_sorted, perm

    
    def _spectral_sort(self, W):
        """
        Uses the Fiedler vector of the Graph Laplacian to seriate the matrix.
        This puts connected nodes next to each other, recovering diagonal structures.
        """
        N = W.shape[0]
        
        # Symmetrize the matrix (make it an undirected graph for Laplacian)
        W_sym = 0.5 * (W + W.T)
        
        # Compute the Graph Laplacian
        L = laplacian(W_sym, normed=False)
        
        # Compute Eigenvalues and Eigenvectors
        # We want the eigenvector associated with the 2nd smallest eigenvalue (Fiedler vector)
        evals, evecs = eigh(L)
        
        # The 0th eigenvalue is approx 0. The 1st is the Fiedler value.
        fiedler_vec = evecs[:, 1]
        
        # The values in the Fiedler vector define the 'position' on the line
        perm = np.argsort(fiedler_vec)
        
        W_sorted = W[perm][:, perm]
        
        return W_sorted, perm
    

    def _spectral_co_clustering_sort(self, W):
        # 假設 W 是非負的
        # 使用 Spectral Co-clustering 找回結構
        model = SpectralCoclustering(n_clusters=2, random_state=0)
        model.fit(W)
        
        # 取得排序後的索引
        fit_idx = np.argsort(model.row_labels_)
        W_sorted = W[fit_idx, :][:, fit_idx]
        
        return W_sorted, fit_idx

    def _hierarchical_clustering_sort(self, W):
        # 計算相關性距離矩陣
        # 我們將每一列視為一個神經元的「特徵向量」
        d = sch.distance.pdist(W)
        L = sch.linkage(d, method='ward')
        
        # 取得葉節點排序
        ind = sch.leaves_list(L)
        
        # 重新排列矩陣
        W_sorted = W[ind, :][:, ind]
        return W_sorted, ind


    def _degree_sort(self, W):
        """
        Sorts the matrix based on total node degree (sum of incoming + outgoing weights).
        This pushes the 'dense' cluster to the top-left.
        """
        # Calculate total strength (in-degree + out-degree)
        degrees = np.sum(W, axis=1) + np.sum(W, axis=0)
        
        # Sort indices in descending order (Strongest nodes first)
        perm = np.argsort(degrees)[::-1]
        
        W_sorted = W[perm][:, perm]
        return W_sorted, perm
    
    def _robust_sort(self, W):
        """
        Recovers the initial structure by separating the signal from the noise
        before sorting.
        
        1. Separates 'Active' neurons (signal) from 'Silent' ones (noise) using degree.
        2. Uses Spectral Seriation (Fiedler vector) ONLY on the Active block to recover the diagonal.
        3. Stacks them back together.
        """
        N = W.shape[0]
        
        # --- Step 1: Isolate the Active Core ---
        # Calculate node strength (degree)
        degrees = np.sum(W, axis=1) + np.sum(W, axis=0)
        
        # Threshold: Define "Active" as nodes with above-average connectivity
        threshold = np.mean(degrees)
        
        active_idx = np.where(degrees > threshold)[0]
        silent_idx = np.where(degrees <= threshold)[0]
        
        # --- Step 2: Spectral Sort the Active Core ONLY ---
        if len(active_idx) > 1:
            # Extract the submatrix of active neurons
            W_core = W[np.ix_(active_idx, active_idx)]
            
            # Symmetrize for Laplacian
            W_sym = 0.5 * (W_core + W_core.T)
            
            # Compute Laplacian and Fiedler Vector (2nd smallest eigenvector)
            L = laplacian(W_sym, normed=True)  # Normed often works better for seriation
            evals, evecs = eigh(L)
            
            # The Fiedler vector is usually the second eigenvector (index 1)
            fiedler_vec = evecs[:, 1]
            
            # Sort the active indices based on their position in the Fiedler vector
            core_perm = np.argsort(fiedler_vec)
            sorted_active_idx = active_idx[core_perm]
            
            # OPTIONAL: Flip if the diagonal is "anti-diagonal" (running / instead of \)
            W_test = W_core[core_perm][:, core_perm]
            if np.trace(W_test) < np.trace(np.flip(W_test, axis=0)):
                sorted_active_idx = sorted_active_idx[::-1]
                
        else:
            sorted_active_idx = active_idx
        
        # --- Step 3: Combine ---
        # Put Active nodes first (Top-Left), Silent nodes last
        final_perm = np.concatenate([sorted_active_idx, silent_idx])
        
        W_sorted = W[final_perm][:, final_perm]
        
        return W_sorted, final_perm


# Example usage
if __name__ == "__main__":
    # Create a synthetic block-diagonal matrix with noise
    np.random.seed(42)
    N = 100
    
    # Create two blocks
    block1_size = 30
    block2_size = 25
    noise_size = N - block1_size - block2_size
    
    # Initialize empty matrix
    W = np.zeros((N, N))
    
    # Block 1: Dense connections
    W[:block1_size, :block1_size] = np.random.rand(block1_size, block1_size) * 0.8
    
    # Block 2: Dense connections
    W[block1_size:block1_size+block2_size, 
      block1_size:block1_size+block2_size] = np.random.rand(block2_size, block2_size) * 0.7
    
    # Add sparse noise throughout
    W += np.random.rand(N, N) * 0.05
    
    # Make it non-negative
    W = np.abs(W)
    
    print("Original matrix shape:", W.shape)
    print("Original matrix trace:", np.trace(W))
    
    # Try different sorting methods
    for method in ['spectral', 'robust', 'degree', 'blind']:
        print(f"\n--- Using {method} method ---")
        sorter = MatrixSorter(method=method)
        W_sorted, perm = sorter.sort(W)
        
        print(f"Sorted matrix trace: {np.trace(W_sorted):.4f}")
        print(f"Permutation shape: {perm.shape}")
        print(f"First 10 indices in permutation: {perm[:10]}")

