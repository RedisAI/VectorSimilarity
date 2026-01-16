//! Search algorithms for HNSW graph traversal.
//!
//! This module provides the core search algorithms:
//! - `greedy_search`: Single entry point search (for upper layers)
//! - `search_layer`: Full layer search with candidate exploration

use super::graph::ElementGraphData;
use super::visited::VisitedNodesHandler;
use crate::distance::DistanceFunction;
use crate::types::{DistanceType, IdType, VectorElement};
use crate::utils::{MaxHeap, MinHeap};

/// Result of a layer search: (id, distance) pairs.
pub type SearchResult<D> = Vec<(IdType, D)>;

/// Greedy search to find the single closest element at a given layer.
///
/// This is used to traverse upper layers where we just need to find
/// the best entry point for the next layer.
pub fn greedy_search<'a, T, D, F>(
    entry_point: IdType,
    query: &[T],
    level: usize,
    graph: &[Option<ElementGraphData>],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> (IdType, D)
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
{
    let mut current = entry_point;
    let mut current_dist = if let Some(data) = data_getter(entry_point) {
        dist_fn.compute(data, query, dim)
    } else {
        D::infinity()
    };

    loop {
        let mut changed = false;

        if let Some(Some(element)) = graph.get(current as usize) {
            for neighbor in element.get_neighbors(level) {
                if let Some(data) = data_getter(neighbor) {
                    let dist = dist_fn.compute(data, query, dim);
                    if dist.to_f64() < current_dist.to_f64() {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    (current, current_dist)
}

/// Search a layer to find the ef closest elements.
///
/// This is the main search algorithm for finding nearest neighbors
/// at a given layer.
#[allow(clippy::too_many_arguments)]
pub fn search_layer<'a, T, D, F, P>(
    entry_points: &[(IdType, D)],
    query: &[T],
    level: usize,
    ef: usize,
    graph: &[Option<ElementGraphData>],
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
    let mut candidates = MinHeap::<D>::with_capacity(ef * 2);

    // Results (max-heap: keeps k closest, largest at top)
    let mut results = MaxHeap::<D>::new(ef);

    // Initialize with entry points
    for &(id, dist) in entry_points {
        if !visited.visit(id) {
            candidates.push(id, dist);

            // Check filter for results
            let passes = filter.is_none_or(|f| f(id));
            if passes {
                results.insert(id, dist);
            }
        }
    }

    // Explore candidates
    while let Some(candidate) = candidates.pop() {
        // Check if we can stop (candidate is further than worst result)
        if results.is_full() {
            if let Some(worst_dist) = results.top_distance() {
                if candidate.distance.to_f64() > worst_dist.to_f64() {
                    break;
                }
            }
        }

        // Get neighbors of this candidate
        if let Some(Some(element)) = graph.get(candidate.id as usize) {
            if element.meta.deleted {
                continue;
            }

            for neighbor in element.get_neighbors(level) {
                if visited.visit(neighbor) {
                    continue; // Already visited
                }

                // Check if neighbor is valid
                if let Some(Some(neighbor_element)) = graph.get(neighbor as usize) {
                    if neighbor_element.meta.deleted {
                        continue;
                    }
                }

                // Compute distance to neighbor
                if let Some(data) = data_getter(neighbor) {
                    let dist = dist_fn.compute(data, query, dim);

                    // Add to results if it passes filter and is close enough
                    let passes = filter.is_none_or(|f| f(neighbor));

                    if passes
                        && (!results.is_full()
                            || dist.to_f64() < results.top_distance().unwrap().to_f64())
                        {
                            results.try_insert(neighbor, dist);
                        }

                    // Add to candidates for exploration
                    if !results.is_full()
                        || dist.to_f64() < results.top_distance().unwrap().to_f64()
                    {
                        candidates.push(neighbor, dist);
                    }
                }
            }
        }
    }

    // Convert results to vector
    results
        .into_sorted_vec()
        .into_iter()
        .map(|e| (e.id, e.distance))
        .collect()
}

/// Select neighbors using the simple heuristic (just keep closest).
pub fn select_neighbors_simple<D: DistanceType>(candidates: &[(IdType, D)], m: usize) -> Vec<IdType> {
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.1.to_f64()
            .partial_cmp(&b.1.to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.into_iter().take(m).map(|(id, _)| id).collect()
}

/// Select neighbors using the heuristic from the HNSW paper.
///
/// This heuristic ensures diversity in the selected neighbors.
#[allow(clippy::too_many_arguments)]
pub fn select_neighbors_heuristic<'a, T, D, F>(
    target: IdType,
    candidates: &[(IdType, D)],
    m: usize,
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
    _extend_candidates: bool,
    keep_pruned: bool,
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
        None => return select_neighbors_simple(candidates, m),
    };

    // Sort candidates by distance
    let mut working: Vec<_> = candidates.to_vec();
    working.sort_by(|a, b| {
        a.1.to_f64()
            .partial_cmp(&b.1.to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected: Vec<IdType> = Vec::with_capacity(m);
    let mut pruned: Vec<(IdType, D)> = Vec::new();

    for (candidate_id, candidate_dist) in working {
        if selected.len() >= m {
            if keep_pruned {
                pruned.push((candidate_id, candidate_dist));
            }
            continue;
        }

        // Check if this candidate is closer to target than to any selected neighbor
        let mut is_good = true;

        if let Some(candidate_data) = data_getter(candidate_id) {
            for &selected_id in &selected {
                if let Some(selected_data) = data_getter(selected_id) {
                    let dist_to_selected = dist_fn.compute(candidate_data, selected_data, dim);
                    if dist_to_selected.to_f64() < candidate_dist.to_f64() {
                        is_good = false;
                        break;
                    }
                }
            }
        }

        if is_good {
            selected.push(candidate_id);
        } else if keep_pruned {
            pruned.push((candidate_id, candidate_dist));
        }
    }

    // Fill remaining slots with pruned candidates if needed
    if keep_pruned && selected.len() < m {
        for (id, _) in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(id);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_neighbors_simple() {
        let candidates = vec![(1, 1.0f32), (2, 0.5), (3, 2.0), (4, 0.3)];

        let selected = select_neighbors_simple(&candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 4); // Closest
        assert_eq!(selected[1], 2);
    }
}
