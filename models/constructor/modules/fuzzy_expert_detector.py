
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, csr_matrix
import os

from ...utils import logger


class SubGraphInfo:
    """SubGraph info data class."""
    def __init__(self, members: List[str], keywords: List[str] = None,
                 description: str = "", membership_scores: Dict[str, float] = None):
        self.members = members
        self.keywords = keywords or []
        self.description = description
        self.membership_scores = membership_scores or {}


class FuzzyExpertDetector:

    def __init__(self, config, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.config = config
        self.embedding_model = embedding_model

        self.min_experts = 10
        self.max_experts = 100
        self.fuzziness_m = 1.5
        self.membership_threshold = 0.3
        self.min_nodes_in_expert = 10

        self.use_sparse = True
        self.batch_size = 1000
        self.max_fcm_iterations = 100

        logger.warning("FuzzyExpertDetector initialized")
        logger.warning(f"  Expert range: {self.min_experts}-{self.max_experts}")
        logger.warning(f"  Fuzziness m: {self.fuzziness_m}")
        logger.warning(f"  Membership threshold: {self.membership_threshold}")
        logger.warning(f"  Memory optimization: use_sparse={self.use_sparse}, batch_size={self.batch_size}")

    def detect_fuzzy_experts(
        self,
        expert_processing_nodes: List[str],
        node_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        logger.warning("=" * 80)
        logger.warning("FUZZY EXPERT DETECTION")
        logger.warning("=" * 80)

        total_unassigned = len(expert_processing_nodes)
        logger.warning(f"Processing {total_unassigned} unassigned nodes")

        logger.warning("\n[Step 1] Preparing embedding matrix")

        embeddings = []
        valid_nodes = []

        for node in expert_processing_nodes:
            if node in node_embeddings:
                embeddings.append(node_embeddings[node])
                valid_nodes.append(node)

        embeddings = np.array(embeddings).astype('float32')
        logger.info(f"  Valid nodes with embeddings: {len(valid_nodes)}")
        logger.info(f"  Embedding matrix shape: {embeddings.shape}")

        logger.warning("\n[Step 3] Running Fuzzy C-Means clustering")

        membership_matrix, centroids = self._fuzzy_c_means_sparse(
            embeddings, num_clusters=self.max_experts
        )
        logger.info(f"  Using sparse storage (CSR format)")

        logger.info(f"  Clustering completed")

        logger.warning("\n[Step 4] Building expert subGraphs (overlapping)")

        expert_subGraphs = self._build_expert_subGraphs_from_sparse(
            valid_nodes, membership_matrix
        )

        logger.warning(f"  Created {len(expert_subGraphs)} expert subGraphs")

        logger.warning("\n[Step 5] Analyzing experts overlap statistics")

        overlap_stats = self._compute_expert_overlap_statistics(
            expert_subGraphs, valid_nodes
        )

        result = {
            'expert_subGraphs': expert_subGraphs,
            'membership_matrix': membership_matrix,
            'expert_centroids': {
                f'Expert_{i+1}': centroids[i]
                for i in range(len(centroids))
            },
            'valid_nodes': valid_nodes,
            'expert_overlap_statistics': overlap_stats
        }

        logger.warning("=" * 80)
        logger.warning(" Fuzzy Expert Detection Completed")
        logger.warning("=" * 80)

        return result

    def _fuzzy_c_means_sparse(
        self,
        data: np.ndarray,
        num_clusters: int,
        max_iter: int = 100,
        epsilon: float = 1e-5
    ) -> Tuple[csr_matrix, np.ndarray]:
        n_samples, n_features = data.shape
        m = self.fuzziness_m

        logger.info(f"  Starting Corrected FCM: N={n_samples}, K={num_clusters}, m={m}")

        centroids = self._initialize_centers_kmeanspp(data, num_clusters)

        membership = np.random.rand(n_samples, num_clusters)
        membership = membership / membership.sum(axis=1, keepdims=True)

        for iteration in range(max_iter):
            distances_sq = np.zeros((n_samples, num_clusters))

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_data = data[start_idx:end_idx]
                batch_distances_sq = self._calculate_distances(batch_data, centroids)
                distances_sq[start_idx:end_idx] = batch_distances_sq

            distances_sq = np.fmax(distances_sq, 1e-10)

            power = 1.0 / (m - 1)
            new_membership = np.zeros((n_samples, num_clusters))

            for i in range(n_samples):
                d_sq_i = distances_sq[i]

                ratios = d_sq_i[:, np.newaxis] / d_sq_i[np.newaxis, :]
                ratios = ratios ** power
                row_sums = ratios.sum(axis=1)

                new_membership[i] = 1.0 / row_sums

            new_membership = new_membership / new_membership.sum(axis=1, keepdims=True)

            u_powered = new_membership ** m

            new_centroids = np.zeros((num_clusters, n_features))
            for j in range(num_clusters):
                weights = u_powered[:, j:j + 1]  # N×1
                weighted_sum = (weights * data).sum(axis=0)
                weight_sum = weights.sum()

                if weight_sum > 1e-10:
                    new_centroid = weighted_sum / weight_sum
                    centroid_norm = np.linalg.norm(new_centroid)
                    if centroid_norm > 1e-10:
                        new_centroids[j] = new_centroid / centroid_norm
                    else:
                        new_centroids[j] = data[np.random.randint(n_samples)]
                        new_centroids[j] = new_centroids[j] / np.linalg.norm(new_centroids[j])
                else:
                    new_centroids[j] = data[np.random.randint(n_samples)]
                    new_centroids[j] = new_centroids[j] / np.linalg.norm(new_centroids[j])

            membership_change = np.linalg.norm(new_membership - membership)
            centroid_change = np.linalg.norm(new_centroids - centroids)

            membership = new_membership
            centroids = new_centroids

            if membership_change < epsilon and centroid_change < epsilon:
                logger.info(f"  Converged at iteration {iteration + 1}")
                logger.info(f"    Membership change: {membership_change:.6f}")
                logger.info(f"    Centroid change: {centroid_change:.6f}")
                break

        return self._sparsify_membership_matrix(membership, centroids)

    def _initialize_centers_kmeanspp(self, data, num_clusters):
        n_samples = data.shape[0]

        data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)

        centers = np.zeros((num_clusters, data.shape[1]))

        first_idx = np.random.randint(n_samples)
        centers[0] = data_norm[first_idx]

        for i in range(1, num_clusters):
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                similarities = np.dot(data_norm[j], centers[:i].T)
                dists_sq = (1 - similarities) ** 2
                distances[j] = np.min(dists_sq)

            probabilities = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers[i] = data_norm[next_idx]

        return centers
    def _calculate_distances(self, X, centers):
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)  # N×D
        centers_norm = centers / np.linalg.norm(centers, axis=1, keepdims=True)  # K×D

        cosine_sim = np.dot(X_norm, centers_norm.T)  # N×K
        cosine_dist_sq = (1 - cosine_sim) ** 2

        cosine_dist_sq = np.fmax(cosine_dist_sq, 1e-10)

        return cosine_dist_sq

    def _sparsify_membership_matrix(self, membership, centroids):
        n_samples, num_clusters = membership.shape

        self._debug_membership_distribution(membership)

        threshold = self._select_adaptive_threshold(membership)

        logger.info(f"  Using adaptive threshold: {threshold:.3f}")

        membership_sparse = lil_matrix(membership.shape)
        nnz_count = 0
        covered_nodes = 0

        for i in range(n_samples):
            node_covered = False
            for j in range(num_clusters):
                if membership[i, j] > threshold:
                    membership_sparse[i, j] = membership[i, j]
                    nnz_count += 1
                    node_covered = True

            if node_covered:
                covered_nodes += 1

        membership_sparse = membership_sparse.tocsr()

        total_elements = n_samples * num_clusters
        sparsity = 1 - (nnz_count / total_elements)

        logger.info(f"  Sparsification completed:")
        logger.info(f"    - Non-zero elements: {nnz_count:,} / {total_elements:,}")
        logger.info(f"    - Sparsity: {sparsity * 100:.1f}%")
        logger.info(f"    - Coverage: {covered_nodes / n_samples * 100:.1f}%")

        return membership_sparse, centroids

    def _debug_membership_distribution(self, membership):
        logger.info(f"  [DEBUG] Membership distribution:")
        logger.info(f"    Min: {membership.min():.6f}")
        logger.info(f"    Max: {membership.max():.6f}")
        logger.info(f"    Mean: {membership.mean():.6f}")
        logger.info(f"    Std: {membership.std():.6f}")

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for thresh in thresholds:
            covered = np.any(membership > thresh, axis=1).sum()
            coverage = covered / membership.shape[0] * 100
            logger.info(f"    > {thresh}: {coverage:.1f}%")

    def _select_adaptive_threshold(self, membership):
        n_samples, num_clusters = membership.shape
        candidate_thresholds = np.linspace(0.05, 0.5, 20)

        for threshold in candidate_thresholds:
            covered_nodes = np.any(membership > threshold, axis=1).sum()
            coverage = covered_nodes / n_samples

            if coverage >= 0.7:
                return threshold

        best_threshold = 0.1
        best_coverage = 0

        for threshold in candidate_thresholds:
            covered_nodes = np.any(membership > threshold, axis=1).sum()
            coverage = covered_nodes / n_samples

            if coverage > best_coverage:
                best_coverage = coverage
                best_threshold = threshold

        logger.warning(
            f"  best coverage, using best: {best_threshold:.3f} ({best_coverage * 100:.1f}%)")
        return best_threshold

    def _build_expert_subGraphs_from_sparse(
            self,
            nodes: List[str],
            membership_sparse: csr_matrix
    ) -> Dict[str, SubGraphInfo]:
        num_clusters = membership_sparse.shape[1]
        initial_expert_subGraphs = {}
        valid_expert_subGraphs = {}

        logger.info("  Building expert subGraphs from sparse matrix...")

        for k in range(num_clusters):
            expert_id = f'Expert_{k + 1}'

            col_indices = membership_sparse.getcol(k).nonzero()[0]

            members = []
            membership_scores = {}

            for idx in col_indices:
                node = nodes[idx]
                score = membership_sparse[idx, k]

                members.append(node)
                membership_scores[node] = float(score)

            if members:
                initial_expert_subGraphs[expert_id] = SubGraphInfo(
                    members=members,
                    keywords=[],
                    description="",
                    membership_scores=membership_scores
                )

                avg_score = np.mean(list(membership_scores.values()))
                logger.info(f"    {expert_id}: {len(members)} members "
                            f"(avg membership={avg_score:.3f})")

        logger.info("  Filtering small expert subGraphs...")
        logger.info(f"    Minimum nodes per expert: {self.min_nodes_in_expert}")

        small_subGraphs = []
        for expert_id, subGraph_info in initial_expert_subGraphs.items():
            if len(subGraph_info.members) < self.min_nodes_in_expert:
                small_subGraphs.append((expert_id, len(subGraph_info.members)))
            else:
                valid_expert_subGraphs[expert_id] = subGraph_info

        if small_subGraphs:
            logger.info(f"    Removed {len(small_subGraphs)} small subGraphs:")
            for expert_id, member_count in small_subGraphs:
                logger.info(f"      - {expert_id}: {member_count} nodes (below threshold {self.min_nodes_in_expert})")
        else:
            logger.info("    No subGraphs removed (all meet minimum size requirement)")

        logger.info("  Renumbering expert subGraphs...")

        sorted_keys = sorted(valid_expert_subGraphs.keys(),
                             key=lambda x: int(x.split('_')[1]))

        expert_subGraphs = {}
        for new_idx, old_key in enumerate(sorted_keys, start=1):
            new_expert_id = f'Expert_{new_idx}'
            expert_subGraphs[new_expert_id] = valid_expert_subGraphs[old_key]

            old_size = len(valid_expert_subGraphs[old_key].members)
            logger.info(f"    {old_key} -> {new_expert_id}: {old_size} members")

        logger.info("  Expert subGraph filtering and renumbering completed:")
        logger.info(f"    Initial subGraphs: {num_clusters}")
        logger.info(f"    Removed (too small): {len(small_subGraphs)}")
        logger.info(f"    Final subGraphs: {len(expert_subGraphs)}")
        logger.info(
            f"    Total members in final subGraphs: {sum(len(c.members) for c in expert_subGraphs.values())}")

        return expert_subGraphs


    def _compute_expert_overlap_statistics(
        self,
        expert_subGraphs: Dict[str, SubGraphInfo],
        all_nodes: List[str]
    ) -> Dict:
        node_membership_count = Counter()

        for expert_info in expert_subGraphs.values():
            for node in expert_info.members:
                node_membership_count[node] += 1

        membership_distribution = Counter(node_membership_count.values())

        logger.info("  Overlap statistics:")
        for count in sorted(membership_distribution.keys()):
            num_nodes = membership_distribution[count]
            percentage = num_nodes / len(all_nodes) * 100
            logger.info(f"    Nodes in {count} expert(s): {num_nodes} ({percentage:.1f}%)")

        if node_membership_count:
            avg_membership = np.mean(list(node_membership_count.values()))
        else:
            avg_membership = 0.0

        logger.info(f"    Average memberships per node: {avg_membership:.2f}")

        uncovered_nodes = [n for n in all_nodes if node_membership_count[n] == 0]
        logger.info(f"    Uncovered nodes: {len(uncovered_nodes)} ({len(uncovered_nodes)/len(all_nodes)*100:.1f}%)")

        overlap_stats = {
            'avg_memberships_per_node': avg_membership,
            'membership_distribution': dict(membership_distribution),
            'uncovered_count': len(uncovered_nodes),
            'coverage': 1 - len(uncovered_nodes) / len(all_nodes)
        }

        return overlap_stats

    def _create_empty_result(self) -> Dict:
        return {
            'expert_subGraphs': {},
            'membership_matrix': np.array([[]]),
            'expert_centroids': {},
            'valid_nodes': [],
            'expert_overlap_statistics': {
                'avg_memberships_per_node': 0.0,
                'membership_distribution': {},
                'uncovered_count': 0,
                'coverage': 0.0
            }
        }

    def set_parameters(
        self,
        min_experts: int = None,
        max_experts: int = None,
        fuzziness_m: float = None,
        membership_threshold: float = None,
    ):
        if min_experts is not None:
            self.min_experts = min_experts
        if max_experts is not None:
            self.max_experts = max_experts
        if fuzziness_m is not None:
            self.fuzziness_m = fuzziness_m
        if membership_threshold is not None:
            self.membership_threshold = membership_threshold

        logger.info("Fuzzy Expert parameters updated")

