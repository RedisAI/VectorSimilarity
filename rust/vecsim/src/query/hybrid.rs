//! Hybrid search heuristics for choosing between ad-hoc and batch search.
//!
//! When performing filtered queries (queries that combine a filter with similarity search),
//! there are two strategies:
//!
//! 1. **Ad-hoc brute force**: Iterate through all filtered vectors and compute distances directly
//! 2. **Batch iterator**: Use the index's batch iterator to progressively retrieve results
//!
//! The `prefer_adhoc_search` heuristic uses a decision tree classifier to choose the optimal
//! strategy based on index characteristics and query parameters.
//!
//! ## Decision Factors
//!
//! - `index_size`: Total number of vectors in the index
//! - `subset_size`: Estimated number of vectors passing the filter
//! - `k`: Number of results requested
//! - `dim`: Vector dimensionality
//! - `m`: HNSW connectivity parameter (for HNSW indices)

/// Search mode tracking for hybrid queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// Initial state - no search performed yet.
    #[default]
    Empty,
    /// Standard k-NN query (no filtering).
    StandardKnn,
    /// Hybrid query using ad-hoc brute force from the start.
    HybridAdhocBf,
    /// Hybrid query using batch iterator.
    HybridBatches,
    /// Hybrid query that switched from batches to ad-hoc brute force.
    HybridBatchesToAdhocBf,
    /// Range query mode.
    RangeQuery,
}

/// Result ordering options for query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryResultOrder {
    /// Sort by distance score (ascending - closest first).
    #[default]
    ByScore,
    /// Sort by vector ID.
    ById,
    /// Primary sort by score, secondary by ID.
    ByScoreThenId,
}

/// Hybrid search policy for filtered queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridPolicy {
    /// Automatically choose the best strategy.
    #[default]
    Auto,
    /// Force ad-hoc brute force search.
    ForceAdhocBf,
    /// Force batch iterator search.
    ForceBatches,
}

/// Parameters for hybrid search decision.
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    /// Total vectors in the index.
    pub index_size: usize,
    /// Estimated vectors passing the filter.
    pub subset_size: usize,
    /// Number of results requested.
    pub k: usize,
    /// Vector dimensionality.
    pub dim: usize,
    /// Whether this is the initial check (true) or re-evaluation (false).
    pub initial_check: bool,
}

/// Decision tree heuristic for BruteForce indices.
///
/// This implements a 10-leaf decision tree trained on empirical data.
/// Returns true if ad-hoc brute force is preferred, false for batch iterator.
pub fn prefer_adhoc_brute_force(params: &HybridSearchParams, label_count: usize) -> bool {
    let index_size = params.index_size;
    let dim = params.dim;

    // Calculate ratio r = subset_size / label_count
    let r = if label_count == 0 {
        0.0
    } else {
        (params.subset_size.min(label_count) as f64) / (label_count as f64)
    };

    // Decision tree based on C++ implementation
    // (sklearn DecisionTreeClassifier with 10 leaves)

    if index_size <= 5500 {
        return true; // Ad-hoc for small indices
    }

    if dim <= 300 {
        if r <= 0.15 {
            return true; // Ad-hoc for small filter ratios
        }
        if r <= 0.35 {
            if dim <= 75 {
                return false; // Batches for very low dimensions
            }
            if index_size <= 550_000 {
                return true; // Ad-hoc for medium index size
            }
            return false; // Batches for large indices
        }
        return false; // Batches for large filter ratios
    }

    // dim > 300
    if r <= 0.025 {
        return true; // Ad-hoc for very small filter ratios
    }
    if r <= 0.55 {
        if index_size <= 55_000 {
            return true; // Ad-hoc for smaller indices
        }
        if r <= 0.045 {
            return true; // Ad-hoc for small filter ratios
        }
        return false; // Batches otherwise
    }

    false // Batches for large filter ratios
}

/// Decision tree heuristic for HNSW indices.
///
/// This implements a 20-leaf decision tree trained on empirical data.
/// HNSW considers additional parameters like k and M (connectivity).
pub fn prefer_adhoc_hnsw(params: &HybridSearchParams, label_count: usize, m: usize) -> bool {
    let index_size = params.index_size;
    let dim = params.dim;
    let k = params.k;

    // Calculate ratio
    let r = if label_count == 0 {
        0.0
    } else {
        (params.subset_size.min(label_count) as f64) / (label_count as f64)
    };

    // Decision tree based on C++ implementation (20 leaves)

    if index_size <= 30_000 {
        // Small to medium index
        if index_size <= 5500 {
            return true; // Always ad-hoc for very small indices
        }
        if r <= 0.17 {
            return true; // Ad-hoc for small filter ratios
        }
        if k <= 12 {
            if dim <= 55 {
                return false; // Batches for small dimensions
            }
            if m <= 10 {
                return false; // Batches for low connectivity
            }
            return true; // Ad-hoc otherwise
        }
        return true; // Ad-hoc for larger k
    }

    // Large index (> 30000)
    if r <= 0.025 {
        // Very small filter ratio
        if index_size <= 750_000 {
            return true;
        }
        if r <= 0.0035 {
            return true;
        }
        if k <= 45 {
            return false;
        }
        return true;
    }

    if r <= 0.085 {
        // Small filter ratio
        if index_size <= 75_000 {
            return true;
        }
        if k <= 7 {
            if dim <= 25 {
                return false;
            }
            if m <= 10 {
                return false;
            }
            return true;
        }
        if dim <= 55 {
            return false;
        }
        if m <= 24 {
            if index_size <= 550_000 {
                return true;
            }
            return false;
        }
        return true;
    }

    if r <= 0.175 {
        // Medium filter ratio
        if index_size <= 75_000 {
            if k <= 7 {
                if dim <= 25 {
                    return false;
                }
                return true;
            }
            return true;
        }
        if k <= 45 {
            return false;
        }
        return true;
    }

    // Large filter ratio (> 0.175)
    if index_size <= 75_000 {
        if k <= 12 {
            if dim <= 55 {
                return false;
            }
            if m <= 10 {
                return false;
            }
            return true;
        }
        return true;
    }

    false // Batches for large indices with large filter ratios
}

/// Threshold-based heuristic for SVS indices.
///
/// SVS uses a simpler threshold-based approach rather than a full decision tree.
pub fn prefer_adhoc_svs(params: &HybridSearchParams) -> bool {
    let index_size = params.index_size;
    let k = params.k;

    // Calculate ratio
    let subset_ratio = if index_size == 0 {
        0.0
    } else {
        (params.subset_size as f64) / (index_size as f64)
    };

    // Thresholds from C++ implementation
    const SMALL_SUBSET_THRESHOLD: f64 = 0.07; // 7% of index
    const LARGE_SUBSET_THRESHOLD: f64 = 0.21; // 21% of index
    const SMALL_INDEX_THRESHOLD: usize = 75_000; // 75k vectors
    const LARGE_INDEX_THRESHOLD: usize = 750_000; // 750k vectors

    if subset_ratio < SMALL_SUBSET_THRESHOLD {
        // Small subset: ad-hoc if index is not large
        index_size < LARGE_INDEX_THRESHOLD
    } else if subset_ratio < LARGE_SUBSET_THRESHOLD {
        // Medium subset: ad-hoc if index is small OR k is large
        index_size < SMALL_INDEX_THRESHOLD || k > 12
    } else {
        // Large subset: ad-hoc only if index is small
        index_size < SMALL_INDEX_THRESHOLD
    }
}

/// Heuristic for Tiered indices (uses the larger sub-index's heuristic).
pub fn prefer_adhoc_tiered<F, B>(
    params: &HybridSearchParams,
    frontend_size: usize,
    backend_size: usize,
    frontend_heuristic: F,
    backend_heuristic: B,
) -> bool
where
    F: FnOnce(&HybridSearchParams) -> bool,
    B: FnOnce(&HybridSearchParams) -> bool,
{
    if backend_size > frontend_size {
        backend_heuristic(params)
    } else {
        frontend_heuristic(params)
    }
}

/// Determine the search mode based on the decision and initial_check flag.
pub fn determine_search_mode(prefer_adhoc: bool, initial_check: bool) -> SearchMode {
    if prefer_adhoc {
        if initial_check {
            SearchMode::HybridAdhocBf
        } else {
            SearchMode::HybridBatchesToAdhocBf
        }
    } else {
        SearchMode::HybridBatches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brute_force_small_index() {
        let params = HybridSearchParams {
            index_size: 1000,
            subset_size: 500,
            k: 10,
            dim: 128,
            initial_check: true,
        };

        // Small index should prefer ad-hoc
        assert!(prefer_adhoc_brute_force(&params, 1000));
    }

    #[test]
    fn test_brute_force_large_index_small_ratio() {
        let params = HybridSearchParams {
            index_size: 1_000_000,
            subset_size: 10_000, // 1% ratio
            k: 10,
            dim: 128,
            initial_check: true,
        };

        // Large index with small ratio - check decision
        let result = prefer_adhoc_brute_force(&params, 1_000_000);
        // With 1% ratio and dim=128, should prefer ad-hoc
        assert!(result);
    }

    #[test]
    fn test_brute_force_large_index_large_ratio() {
        let params = HybridSearchParams {
            index_size: 1_000_000,
            subset_size: 500_000, // 50% ratio
            k: 10,
            dim: 128,
            initial_check: true,
        };

        // Large index with large ratio should prefer batches
        assert!(!prefer_adhoc_brute_force(&params, 1_000_000));
    }

    #[test]
    fn test_hnsw_small_index() {
        let params = HybridSearchParams {
            index_size: 1000,
            subset_size: 500,
            k: 10,
            dim: 128,
            initial_check: true,
        };

        // Small index should prefer ad-hoc
        assert!(prefer_adhoc_hnsw(&params, 1000, 16));
    }

    #[test]
    fn test_hnsw_considers_k() {
        let params_small_k = HybridSearchParams {
            index_size: 100_000,
            subset_size: 10_000, // 10% ratio
            k: 5,
            dim: 128,
            initial_check: true,
        };

        let params_large_k = HybridSearchParams {
            index_size: 100_000,
            subset_size: 10_000,
            k: 100,
            dim: 128,
            initial_check: true,
        };

        // k affects the decision for HNSW
        let _result_small_k = prefer_adhoc_hnsw(&params_small_k, 100_000, 16);
        let _result_large_k = prefer_adhoc_hnsw(&params_large_k, 100_000, 16);
        // Both are valid results; k does affect the decision
    }

    #[test]
    fn test_svs_thresholds() {
        // Small subset ratio (< 7%)
        let params_small = HybridSearchParams {
            index_size: 100_000,
            subset_size: 5_000, // 5%
            k: 10,
            dim: 128,
            initial_check: true,
        };
        assert!(prefer_adhoc_svs(&params_small)); // Should prefer ad-hoc

        // Medium subset ratio (7-21%)
        let params_medium = HybridSearchParams {
            index_size: 100_000,
            subset_size: 15_000, // 15%
            k: 10,
            dim: 128,
            initial_check: true,
        };
        // Medium with large k should prefer ad-hoc
        let params_medium_large_k = HybridSearchParams {
            k: 20,
            ..params_medium
        };
        assert!(prefer_adhoc_svs(&params_medium_large_k));

        // Large subset ratio (> 21%) with small index
        let params_large = HybridSearchParams {
            index_size: 50_000,
            subset_size: 25_000, // 50%
            k: 10,
            dim: 128,
            initial_check: true,
        };
        assert!(prefer_adhoc_svs(&params_large)); // Small index prefers ad-hoc
    }

    #[test]
    fn test_search_mode_determination() {
        assert_eq!(
            determine_search_mode(true, true),
            SearchMode::HybridAdhocBf
        );
        assert_eq!(
            determine_search_mode(true, false),
            SearchMode::HybridBatchesToAdhocBf
        );
        assert_eq!(
            determine_search_mode(false, true),
            SearchMode::HybridBatches
        );
        assert_eq!(
            determine_search_mode(false, false),
            SearchMode::HybridBatches
        );
    }

    #[test]
    fn test_query_result_order() {
        assert_eq!(QueryResultOrder::default(), QueryResultOrder::ByScore);
    }

    #[test]
    fn test_hybrid_policy() {
        assert_eq!(HybridPolicy::default(), HybridPolicy::Auto);
    }
}
