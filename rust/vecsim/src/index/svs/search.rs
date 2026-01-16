//! Search algorithms for Vamana graph traversal.
//!
//! This module provides:
//! - `greedy_beam_search`: Beam search from entry point
//! - `robust_prune`: Alpha-based diverse neighbor selection

use super::graph::VamanaGraph;
use crate::distance::DistanceFunction;
use crate::index::hnsw::VisitedNodesHandler;
use crate::types::{DistanceType, IdType, VectorElement};
use crate::utils::{MaxHeap, MinHeap};

/// Result of a search: (id, distance) pairs sorted by distance.
pub type SearchResult<D> = Vec<(IdType, D)>;

/// Greedy beam search from entry point.
///
/// Explores the graph using a beam of candidates, maintaining the
/// best candidates found so far.
#[allow(clippy::too_many_arguments)]
pub fn greedy_beam_search<'a, T, D, F>(
    entry_point: IdType,
    query: &[T],
    beam_width: usize,
    graph: &VamanaGraph,
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
    visited: &VisitedNodesHandler,
) -> SearchResult<D>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    greedy_beam_search_filtered::<T, D, F, fn(IdType) -> bool>(
        entry_point,
        query,
        beam_width,
        graph,
        data_getter,
        dist_fn,
        dim,
        visited,
        None,
    )
}

/// Greedy beam search with optional filter.
#[allow(clippy::too_many_arguments)]
pub fn greedy_beam_search_filtered<'a, T, D, F, P>(
    entry_point: IdType,
    query: &[T],
    beam_width: usize,
    graph: &VamanaGraph,
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
    visited: &VisitedNodesHandler,
    filter: Option<&P>,
) -> SearchResult<D>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
    P: Fn(IdType) -> bool + ?Sized,
{
    // Candidates to explore (min-heap: closest first)
    let mut candidates = MinHeap::<D>::with_capacity(beam_width * 2);

    // Results (max-heap: keeps L closest, largest at top)
    let mut results = MaxHeap::<D>::new(beam_width);

    // Initialize with entry point
    visited.visit(entry_point);

    if let Some(entry_data) = data_getter(entry_point) {
        let dist = dist_fn.compute(entry_data, query, dim);
        candidates.push(entry_point, dist);

        if !graph.is_deleted(entry_point) {
            let passes = filter.is_none_or(|f| f(entry_point));
            if passes {
                results.insert(entry_point, dist);
            }
        }
    }

    // Explore candidates
    while let Some(candidate) = candidates.pop() {
        // Check if we can stop early
        if results.is_full() {
            if let Some(worst_dist) = results.top_distance() {
                if candidate.distance.to_f64() > worst_dist.to_f64() {
                    break;
                }
            }
        }

        // Skip deleted nodes
        if graph.is_deleted(candidate.id) {
            continue;
        }

        // Explore neighbors
        for neighbor in graph.get_neighbors(candidate.id) {
            if visited.visit(neighbor) {
                continue; // Already visited
            }

            // Skip deleted neighbors
            if graph.is_deleted(neighbor) {
                continue;
            }

            // Compute distance
            if let Some(neighbor_data) = data_getter(neighbor) {
                let dist = dist_fn.compute(neighbor_data, query, dim);

                // Check filter
                let passes = filter.is_none_or(|f| f(neighbor));

                // Add to results if it passes filter and is close enough
                if passes
                    && (!results.is_full()
                        || dist.to_f64() < results.top_distance().unwrap().to_f64())
                {
                    results.try_insert(neighbor, dist);
                }

                // Add to candidates for exploration if close enough
                if !results.is_full() || dist.to_f64() < results.top_distance().unwrap().to_f64() {
                    candidates.push(neighbor, dist);
                }
            }
        }
    }

    // Convert to sorted vector
    results
        .into_sorted_vec()
        .into_iter()
        .map(|e| (e.id, e.distance))
        .collect()
}

/// Robust pruning for diverse neighbor selection.
///
/// Implements the alpha-based pruning from the Vamana paper:
/// A neighbor is selected only if for all already-selected neighbors s:
///   dist(neighbor, s) * alpha >= dist(neighbor, target)
///
/// This ensures neighbors are diverse and not all clustered together.
#[allow(clippy::too_many_arguments)]
pub fn robust_prune<'a, T, D, F>(
    target: IdType,
    candidates: &[(IdType, D)],
    max_degree: usize,
    alpha: f32,
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> Vec<IdType>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    let _target_data = match data_getter(target) {
        Some(d) => d,
        None => return select_closest(candidates, max_degree),
    };

    // Sort candidates by distance
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.1.to_f64()
            .partial_cmp(&b.1.to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected: Vec<IdType> = Vec::with_capacity(max_degree);
    let alpha_f64 = alpha as f64;

    for &(candidate_id, candidate_dist) in &sorted {
        if selected.len() >= max_degree {
            break;
        }

        // Skip if candidate is the target itself
        if candidate_id == target {
            continue;
        }

        let candidate_data = match data_getter(candidate_id) {
            Some(d) => d,
            None => continue,
        };

        // Check alpha condition against all selected neighbors
        let mut is_diverse = true;
        let candidate_dist_f64 = candidate_dist.to_f64();

        for &selected_id in &selected {
            if let Some(selected_data) = data_getter(selected_id) {
                let dist_to_selected = dist_fn.compute(candidate_data, selected_data, dim);
                // If candidate is closer to a selected neighbor than to target * alpha,
                // it's not diverse enough
                if dist_to_selected.to_f64() * alpha_f64 < candidate_dist_f64 {
                    is_diverse = false;
                    break;
                }
            }
        }

        if is_diverse {
            selected.push(candidate_id);
        }
    }

    // If we couldn't fill to max_degree due to alpha pruning,
    // add remaining closest candidates from sorted list (fallback)
    if selected.len() < max_degree {
        for &(candidate_id, _) in &sorted {
            if selected.len() >= max_degree {
                break;
            }
            if !selected.contains(&candidate_id) && candidate_id != target {
                selected.push(candidate_id);
            }
        }
    }

    selected
}

/// Simple selection of closest candidates (no diversity).
fn select_closest<D: DistanceType>(candidates: &[(IdType, D)], max_degree: usize) -> Vec<IdType> {
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.1.to_f64()
            .partial_cmp(&b.1.to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.into_iter().take(max_degree).map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_closest() {
        let candidates = vec![(1u32, 1.0f32), (2, 0.5), (3, 2.0), (4, 0.3)];
        let selected = select_closest(&candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 4); // Closest
        assert_eq!(selected[1], 2);
    }

    #[test]
    fn test_select_closest_max() {
        let candidates = vec![(1u32, 1.0f32), (2, 0.5)];
        let selected = select_closest(&candidates, 10);
        assert_eq!(selected.len(), 2);
    }
}
