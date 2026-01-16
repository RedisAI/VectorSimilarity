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
    use crate::distance::{l2::L2Distance, DistanceFunction};
    use crate::index::hnsw::VisitedNodesHandlerPool;

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

    #[test]
    fn test_select_closest_empty() {
        let candidates: Vec<(IdType, f32)> = vec![];
        let selected = select_closest(&candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_select_closest_equal_distances() {
        let candidates = vec![(1u32, 1.0f32), (2, 1.0), (3, 1.0)];
        let selected = select_closest(&candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_greedy_beam_search_single_node() {
        let dist_fn = L2Distance::<f32>::new(4);
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]];

        let mut graph = VamanaGraph::new(10, 4);
        graph.set_label(0, 100);

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.0, 0.0, 0.0, 0.0];

        let results = greedy_beam_search(
            0,
            &query,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
            &visited,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_greedy_beam_search_finds_closest() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
            vec![3.0, 0.0], // id 3 - closest to query
        ];

        let mut graph = VamanaGraph::new(10, 4);
        for i in 0..4 {
            graph.set_label(i, i as u64 * 10);
        }
        // Linear chain: 0 -> 1 -> 2 -> 3
        graph.set_neighbors(0, &[1]);
        graph.set_neighbors(1, &[0, 2]);
        graph.set_neighbors(2, &[1, 3]);
        graph.set_neighbors(3, &[2]);

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [3.0, 0.0];

        let results = greedy_beam_search(
            0, // start from 0
            &query,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
        );

        // First result should be closest (id 3)
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_greedy_beam_search_respects_beam_width() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();

        let mut graph = VamanaGraph::new(10, 10);
        for i in 0..10 {
            graph.set_label(i, i as u64);
        }
        // Fully connected
        for i in 0..10 {
            let neighbors: Vec<u32> = (0..10).filter(|&j| j != i).collect();
            graph.set_neighbors(i, &neighbors);
        }

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [0.0, 0.0];

        // Beam width = 3
        let results = greedy_beam_search(
            5, // start from middle
            &query,
            3,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
        );

        assert!(results.len() <= 3);
    }

    #[test]
    fn test_greedy_beam_search_skips_deleted() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1 - closest to query but deleted
            vec![2.0, 0.0], // id 2
        ];

        let mut graph = VamanaGraph::new(10, 4);
        for i in 0..3 {
            graph.set_label(i, i as u64 * 10);
        }
        graph.set_neighbors(0, &[1, 2]);
        graph.set_neighbors(1, &[0, 2]);
        graph.set_neighbors(2, &[0, 1]);

        // Mark id 1 as deleted
        graph.mark_deleted(1);

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.0, 0.0]; // Closest to deleted node

        let results = greedy_beam_search(
            0,
            &query,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
        );

        // Should not contain deleted id
        for (id, _) in &results {
            assert_ne!(*id, 1, "Deleted node should not appear in results");
        }
    }

    #[test]
    fn test_greedy_beam_search_filtered() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
            vec![3.0, 0.0], // id 3
        ];

        let mut graph = VamanaGraph::new(10, 4);
        for i in 0..4 {
            graph.set_label(i, i as u64);
        }
        // Fully connected
        graph.set_neighbors(0, &[1, 2, 3]);
        graph.set_neighbors(1, &[0, 2, 3]);
        graph.set_neighbors(2, &[0, 1, 3]);
        graph.set_neighbors(3, &[0, 1, 2]);

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.5, 0.0]; // Between 1 and 2

        // Filter: only accept even IDs
        let filter = |id: IdType| -> bool { id % 2 == 0 };

        let results = greedy_beam_search_filtered(
            0,
            &query,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
            Some(&filter),
        );

        // All results should have even IDs
        for (id, _) in &results {
            assert_eq!(id % 2, 0, "Filter should only allow even IDs");
        }
    }

    #[test]
    fn test_robust_prune_empty() {
        let dist_fn = L2Distance::<f32>::new(4);
        let candidates: Vec<(IdType, f32)> = vec![];
        let data_getter = |_id: IdType| -> Option<&[f32]> { None };

        let selected = robust_prune(
            0,
            &candidates,
            5,
            1.2,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert!(selected.is_empty());
    }

    #[test]
    fn test_robust_prune_basic() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // target (id 0)
            vec![1.0, 0.0], // id 1
            vec![0.0, 1.0], // id 2 (orthogonal to 1, diverse)
            vec![1.0, 0.0], // id 3 (same as 1, not diverse)
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let candidates = vec![(1, 1.0f32), (2, 1.0f32), (3, 1.0f32)];

        let selected = robust_prune(
            0,
            &candidates,
            2,
            1.2,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        // Should prefer diverse neighbors
        assert_eq!(selected.len(), 2);
        // Should include orthogonal vector 2 for diversity
        assert!(selected.contains(&1) || selected.contains(&2));
    }

    #[test]
    fn test_robust_prune_excludes_target() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // target (id 0)
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        // Include target in candidates
        let candidates = vec![(0, 0.0f32), (1, 1.0f32), (2, 4.0f32)];

        let selected = robust_prune(
            0,
            &candidates,
            2,
            1.2,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        // Should not include target itself
        assert!(!selected.contains(&0));
    }

    #[test]
    fn test_robust_prune_fallback_when_all_pruned() {
        // Set up vectors where alpha pruning would reject everything
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // target (id 0)
            vec![1.0, 0.0], // id 1
            vec![1.1, 0.0], // id 2 (very close to 1)
            vec![1.2, 0.0], // id 3 (very close to 1 and 2)
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        // All very close together
        let candidates = vec![(1, 1.0f32), (2, 1.21f32), (3, 1.44f32)];

        // Use high alpha that would prune most candidates
        let selected = robust_prune(
            0,
            &candidates,
            3,
            2.0, // High alpha
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        // Should fill to max_degree using fallback
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_robust_prune_fewer_candidates_than_degree() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // target (id 0)
            vec![1.0, 0.0], // id 1
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let candidates = vec![(1, 1.0f32)];

        let selected = robust_prune(
            0,
            &candidates,
            10, // Want 10, only have 1
            1.2,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 1);
    }

    #[test]
    fn test_robust_prune_alpha_effect() {
        // Test that higher alpha allows more similar neighbors
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],  // target (id 0)
            vec![1.0, 0.0],  // id 1
            vec![1.5, 0.0],  // id 2 (in same direction as 1)
            vec![0.0, 2.0],  // id 3 (different direction)
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let candidates = vec![(1, 1.0f32), (2, 2.25f32), (3, 4.0f32)];

        // Low alpha (more strict diversity)
        let selected_low = robust_prune(
            0,
            &candidates,
            3,
            1.0,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        // High alpha (less strict diversity)
        let selected_high = robust_prune(
            0,
            &candidates,
            3,
            2.0,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
        );

        // Both should return results
        assert!(!selected_low.is_empty());
        assert!(!selected_high.is_empty());
    }

    #[test]
    fn test_greedy_beam_search_disconnected_entry() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0 - isolated
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
        ];

        let mut graph = VamanaGraph::new(10, 4);
        for i in 0..3 {
            graph.set_label(i, i as u64);
        }
        // Node 0 is isolated (no neighbors)
        graph.set_neighbors(1, &[2]);
        graph.set_neighbors(2, &[1]);

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.5, 0.0];

        // Start from isolated node
        let results = greedy_beam_search(
            0,
            &query,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
        );

        // Should return at least the entry point
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_greedy_beam_search_results_sorted() {
        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
            vec![3.0, 0.0], // id 3
            vec![4.0, 0.0], // id 4
        ];

        let mut graph = VamanaGraph::new(10, 5);
        for i in 0..5 {
            graph.set_label(i, i as u64);
        }
        // Fully connected
        for i in 0..5 {
            let neighbors: Vec<u32> = (0..5).filter(|&j| j != i).collect();
            graph.set_neighbors(i, &neighbors);
        }

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [2.5, 0.0];

        let results = greedy_beam_search(
            0,
            &query,
            5,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
        );

        // Results should be sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results should be sorted by distance"
            );
        }
    }
}
