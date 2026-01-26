//! Batch distance computation for improved SIMD efficiency.
//!
//! This module provides batch distance computation functions that compute
//! distances from a single query to multiple candidate vectors in one call.
//! This improves cache utilization and allows for better SIMD pipelining.

use crate::distance::DistanceFunction;
use crate::types::{DistanceType, IdType, VectorElement};

/// Compute distances from a query to multiple vectors.
///
/// This function computes distances in batch, which can be more efficient
/// than computing them one at a time due to better cache behavior and
/// SIMD pipelining.
///
/// Returns a vector of (id, distance) pairs in the same order as input.
#[inline]
pub fn compute_distances_batch<'a, T, D, F>(
    query: &[T],
    candidates: &[IdType],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> Vec<(IdType, D)>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    let mut results = Vec::with_capacity(candidates.len());

    // Process candidates - prefetching is handled separately
    for &id in candidates {
        if let Some(data) = data_getter(id) {
            let dist = dist_fn.compute(data, query, dim);
            results.push((id, dist));
        }
    }

    results
}

/// Compute distances from a query to multiple vectors with prefetching.
///
/// This version prefetches the next vector while computing the current distance,
/// which helps hide memory latency.
#[inline]
pub fn compute_distances_batch_prefetch<'a, T, D, F>(
    query: &[T],
    candidates: &[IdType],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> Vec<(IdType, D)>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    use crate::utils::prefetch::prefetch_slice;

    let mut results = Vec::with_capacity(candidates.len());
    let n = candidates.len();

    if n == 0 {
        return results;
    }

    // Prefetch first candidate
    if let Some(first_data) = data_getter(candidates[0]) {
        prefetch_slice(first_data);
    }

    for i in 0..n {
        let id = candidates[i];

        // Prefetch next candidate while computing current
        if i + 1 < n {
            if let Some(next_data) = data_getter(candidates[i + 1]) {
                prefetch_slice(next_data);
            }
        }

        if let Some(data) = data_getter(id) {
            let dist = dist_fn.compute(data, query, dim);
            results.push((id, dist));
        }
    }

    results
}

/// Compute distances between pairs of vectors (for neighbor selection heuristic).
///
/// Given a candidate and a list of selected neighbors, compute the distance
/// from the candidate to each selected neighbor. This is used in the
/// diversity-aware neighbor selection heuristic.
#[inline]
pub fn compute_pairwise_distances<'a, T, D, F>(
    candidate_id: IdType,
    selected_ids: &[IdType],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> Vec<D>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    let candidate_data = match data_getter(candidate_id) {
        Some(d) => d,
        None => return vec![D::infinity(); selected_ids.len()],
    };

    selected_ids
        .iter()
        .map(|&sel_id| {
            data_getter(sel_id)
                .map(|sel_data| dist_fn.compute(candidate_data, sel_data, dim))
                .unwrap_or_else(D::infinity)
        })
        .collect()
}

/// Check if a candidate is dominated by any selected neighbor.
///
/// A candidate is "dominated" (and thus should be pruned) if it's closer
/// to any selected neighbor than to the target. This is used in the
/// HNSW heuristic neighbor selection.
///
/// Returns (is_good, distances) where distances can be reused if needed.
#[inline]
pub fn check_candidate_diversity<'a, T, D, F>(
    candidate_id: IdType,
    candidate_dist_to_target: D,
    selected_ids: &[IdType],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> bool
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    let candidate_data = match data_getter(candidate_id) {
        Some(d) => d,
        None => return true, // Keep if we can't check
    };

    // Check against each selected neighbor
    for &sel_id in selected_ids {
        if let Some(sel_data) = data_getter(sel_id) {
            let dist_to_selected = dist_fn.compute(candidate_data, sel_data, dim);
            if dist_to_selected < candidate_dist_to_target {
                return false; // Candidate is dominated
            }
        }
    }

    true // Candidate provides good diversity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::L2Distance;
    use crate::distance::DistanceFunction;

    fn make_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0], // id 0
            vec![1.0, 0.0, 0.0, 0.0], // id 1
            vec![0.0, 1.0, 0.0, 0.0], // id 2
            vec![0.0, 0.0, 1.0, 0.0], // id 3
            vec![2.0, 0.0, 0.0, 0.0], // id 4
        ]
    }

    #[test]
    fn test_compute_distances_batch() {
        let vectors = make_test_vectors();
        let dist_fn = L2Distance::<f32>::new(4);
        let query = vec![0.0f32, 0.0, 0.0, 0.0];
        let candidates = vec![1, 2, 3, 4];

        let data_getter = |id: IdType| -> Option<&[f32]> {
            vectors.get(id as usize).map(|v| v.as_slice())
        };

        let results = compute_distances_batch(
            &query,
            &candidates,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 0.001); // L2^2 to (1,0,0,0)
        assert_eq!(results[3].0, 4);
        assert!((results[3].1 - 4.0).abs() < 0.001); // L2^2 to (2,0,0,0)
    }

    #[test]
    fn test_compute_distances_batch_prefetch() {
        let vectors = make_test_vectors();
        let dist_fn = L2Distance::<f32>::new(4);
        let query = vec![0.0f32, 0.0, 0.0, 0.0];
        let candidates = vec![1, 2, 3];

        let data_getter = |id: IdType| -> Option<&[f32]> {
            vectors.get(id as usize).map(|v| v.as_slice())
        };

        let results = compute_distances_batch_prefetch(
            &query,
            &candidates,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert_eq!(results.len(), 3);
        // All should have distance 1.0 (L2^2 of unit vectors)
        for (_, dist) in &results {
            assert!((dist - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_check_candidate_diversity_good() {
        let vectors = vec![
            vec![0.0f32, 0.0], // target (id 0)
            vec![1.0, 0.0],   // selected (id 1)
            vec![0.0, 1.0],   // candidate (id 2) - diverse direction
        ];
        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            vectors.get(id as usize).map(|v| v.as_slice())
        };

        // Candidate 2 has distance 1.0 to target, and distance 2.0 to selected[1]
        // Since 2.0 > 1.0, candidate provides good diversity
        let is_good = check_candidate_diversity(
            2,   // candidate
            1.0, // distance to target
            &[1],
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        assert!(is_good);
    }

    #[test]
    fn test_check_candidate_diversity_bad() {
        let vectors = vec![
            vec![0.0f32, 0.0], // target (id 0)
            vec![1.0, 0.0],   // selected (id 1)
            vec![1.1, 0.0],   // candidate (id 2) - very close to selected[1]
        ];
        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            vectors.get(id as usize).map(|v| v.as_slice())
        };

        // Candidate 2 has distance ~1.21 to target, but only ~0.01 to selected[1]
        // Since 0.01 < 1.21, candidate is dominated (not diverse)
        let is_good = check_candidate_diversity(
            2,
            1.21, // distance to target
            &[1],
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        assert!(!is_good);
    }

    #[test]
    fn test_compute_pairwise_distances() {
        let vectors = make_test_vectors();
        let dist_fn = L2Distance::<f32>::new(4);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            vectors.get(id as usize).map(|v| v.as_slice())
        };

        // Compute distances from candidate 0 to selected [1, 2, 3]
        let dists = compute_pairwise_distances(
            0,
            &[1, 2, 3],
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert_eq!(dists.len(), 3);
        // All distances should be 1.0 (L2^2 from origin to unit vectors)
        for d in &dists {
            assert!((d - 1.0).abs() < 0.001);
        }
    }
}

