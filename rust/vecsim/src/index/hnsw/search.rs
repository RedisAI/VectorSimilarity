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
            for neighbor in element.iter_neighbors(level) {
                if let Some(data) = data_getter(neighbor) {
                    let dist = dist_fn.compute(data, query, dim);
                    if dist < current_dist {
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
                if candidate.distance > worst_dist {
                    break;
                }
            }
        }

        // Get neighbors of this candidate
        if let Some(Some(element)) = graph.get(candidate.id as usize) {
            if element.meta.deleted {
                continue;
            }

            for neighbor in element.iter_neighbors(level) {
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

                    // Check if close enough to consider
                    let dominated = results.is_full()
                        && dist >= results.top_distance().unwrap();

                    if !dominated {
                        // Add to results if it passes filter
                        let passes = filter.is_none_or(|f| f(neighbor));
                        if passes {
                            results.try_insert(neighbor, dist);
                        }
                        // Add to candidates for exploration
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

use crate::types::LabelType;
use std::collections::HashMap;

/// Search result for multi-value indices: (label, distance) pairs.
pub type MultiSearchResult<D> = Vec<(LabelType, D)>;

/// Search a layer to find the k closest unique labels.
///
/// This is a label-aware search for multi-value indices. It explores the graph
/// like standard HNSW but tracks labels separately. Vectors from already-seen
/// labels are still explored (their neighbors might lead to new labels) but
/// only count once in the label results. This prevents early termination from
/// cutting off exploration when vectors cluster by label.
#[allow(clippy::too_many_arguments)]
pub fn search_layer_multi<'a, T, D, F, P>(
    entry_points: &[(IdType, D)],
    query: &[T],
    level: usize,
    k: usize,
    ef: usize,
    graph: &[Option<ElementGraphData>],
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
    visited: &VisitedNodesHandler,
    id_to_label: &HashMap<IdType, LabelType>,
    filter: Option<&P>,
) -> MultiSearchResult<D>
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
    P: Fn(LabelType) -> bool + ?Sized,
{
    // Track labels we've found and their best distances
    let mut label_best: HashMap<LabelType, D> = HashMap::with_capacity(k * 2);

    // Candidates to explore (min-heap: closest first)
    let mut candidates = MinHeap::<D>::with_capacity(ef * 2);

    // Helper to get the worst (largest) distance among the top-ef labels
    let get_ef_worst_dist = |label_best: &HashMap<LabelType, D>, ef: usize| -> Option<D> {
        if label_best.len() < ef {
            return None;
        }
        let mut dists: Vec<D> = label_best.values().copied().collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists.get(ef - 1).copied()
    };

    // Initialize with entry points
    for &(id, dist) in entry_points {
        if !visited.visit(id) {
            candidates.push(id, dist);

            // Check filter and update label tracking
            if let Some(&label) = id_to_label.get(&id) {
                let passes = filter.is_none_or(|f| f(label));
                if passes {
                    label_best
                        .entry(label)
                        .and_modify(|best| {
                            if dist < *best {
                                *best = dist;
                            }
                        })
                        .or_insert(dist);
                }
            }
        }
    }

    // Explore candidates with label-aware early termination
    while let Some(candidate) = candidates.pop() {
        // Early termination: stop when we have ef labels AND
        // candidate is further than the ef-th best label distance
        if label_best.len() >= ef {
            if let Some(ef_worst) = get_ef_worst_dist(&label_best, ef) {
                if candidate.distance > ef_worst {
                    break;
                }
            }
        }

        // Get neighbors of this candidate
        if let Some(Some(element)) = graph.get(candidate.id as usize) {
            if element.meta.deleted {
                continue;
            }

            for neighbor in element.iter_neighbors(level) {
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

                    // Less aggressive pruning: only prune if significantly worse
                    // Always explore if we haven't found enough labels yet
                    let should_explore = if label_best.len() >= ef {
                        get_ef_worst_dist(&label_best, ef)
                            .map_or(true, |worst| dist < worst)
                    } else {
                        true
                    };

                    if should_explore {
                        candidates.push(neighbor, dist);
                    }

                    // Always update label tracking regardless of pruning
                    if let Some(&label) = id_to_label.get(&neighbor) {
                        let passes = filter.is_none_or(|f| f(label));
                        if passes {
                            label_best
                                .entry(label)
                                .and_modify(|best| {
                                    if dist < *best {
                                        *best = dist;
                                    }
                                })
                                .or_insert(dist);
                        }
                    }
                }
            }
        }
    }

    // Convert to sorted vector of (label, distance)
    let mut results_vec: Vec<_> = label_best.into_iter().collect();
    results_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results_vec.truncate(k);
    results_vec
}

/// Select neighbors using the simple heuristic (just keep closest).
pub fn select_neighbors_simple<D: DistanceType>(candidates: &[(IdType, D)], m: usize) -> Vec<IdType> {
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
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
        a.1.partial_cmp(&b.1)
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
                    if dist_to_selected < candidate_dist {
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
    use crate::distance::{l2::L2Distance, DistanceFunction};

    #[test]
    fn test_select_neighbors_simple() {
        let candidates = vec![(1, 1.0f32), (2, 0.5), (3, 2.0), (4, 0.3)];

        let selected = select_neighbors_simple(&candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 4); // Closest
        assert_eq!(selected[1], 2);
    }

    #[test]
    fn test_select_neighbors_simple_empty() {
        let candidates: Vec<(IdType, f32)> = vec![];
        let selected = select_neighbors_simple(&candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_select_neighbors_simple_fewer_than_m() {
        let candidates = vec![(1, 1.0f32), (2, 0.5)];
        let selected = select_neighbors_simple(&candidates, 10);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_neighbors_simple_equal_distances() {
        let candidates = vec![(1, 1.0f32), (2, 1.0), (3, 1.0)];
        let selected = select_neighbors_simple(&candidates, 2);
        assert_eq!(selected.len(), 2);
        // All have same distance, so first two in sorted order
    }

    #[test]
    fn test_select_neighbors_heuristic_empty() {
        let candidates: Vec<(IdType, f32)> = vec![];
        let dist_fn = L2Distance::<f32>::new(4);
        let data_getter = |_id: IdType| -> Option<&[f32]> { None };

        let selected = select_neighbors_heuristic(
            0,
            &candidates,
            5,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
            false,
            false,
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn test_select_neighbors_heuristic_basic() {
        // Create simple test vectors
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0], // target (id 0)
            vec![1.0, 0.0, 0.0, 0.0], // id 1
            vec![0.0, 1.0, 0.0, 0.0], // id 2
            vec![0.0, 0.0, 1.0, 0.0], // id 3
            vec![0.5, 0.5, 0.0, 0.0], // id 4 (between 1 and 2)
        ];

        let dist_fn = L2Distance::<f32>::new(4);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        // Distances from target (0,0,0,0):
        // id 1: 1.0 (1^2)
        // id 2: 1.0
        // id 3: 1.0
        // id 4: 0.5 (0.5^2 + 0.5^2 = 0.5)
        let candidates = vec![(1, 1.0f32), (2, 1.0f32), (3, 1.0f32), (4, 0.5f32)];

        let selected = select_neighbors_heuristic(
            0,
            &candidates,
            3,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
            false,
            false,
        );

        // Should select 4 (closest) and some diverse set
        assert!(!selected.is_empty());
        assert!(selected.len() <= 3);
        assert_eq!(selected[0], 4); // Closest first
    }

    #[test]
    fn test_select_neighbors_heuristic_with_keep_pruned() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // target
            vec![1.0, 0.0], // id 1
            vec![1.1, 0.0], // id 2 (very close to 1)
            vec![0.0, 1.0], // id 3 (diverse direction)
        ];

        let dist_fn = L2Distance::<f32>::new(2);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        // id 2 is closer to id 1 than to target, so might be pruned
        let candidates = vec![(1, 1.0f32), (2, 1.21f32), (3, 1.0f32)];

        let selected = select_neighbors_heuristic(
            0,
            &candidates,
            3,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            false,
            true, // keep_pruned = true
        );

        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_greedy_search_single_node() {
        let dist_fn = L2Distance::<f32>::new(4);
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]];

        // Create single-node graph
        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        graph.push(Some(ElementGraphData::new(1, 0, 4, 2)));

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.0, 0.0, 0.0, 0.0];

        let (result_id, result_dist) = greedy_search(
            0, // entry point
            &query,
            0, // level
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert_eq!(result_id, 0);
        assert!(result_dist < 0.001); // Should be very close to 0
    }

    #[test]
    fn test_greedy_search_finds_closest() {
        let dist_fn = L2Distance::<f32>::new(4);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0], // id 0
            vec![1.0, 0.0, 0.0, 0.0], // id 1
            vec![2.0, 0.0, 0.0, 0.0], // id 2
            vec![3.0, 0.0, 0.0, 0.0], // id 3
        ];

        // Create linear graph: 0 -> 1 -> 2 -> 3
        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        for i in 0..4 {
            graph.push(Some(ElementGraphData::new(i as u64, 0, 4, 2)));
        }
        graph[0].as_ref().unwrap().set_neighbors(0, &[1]);
        graph[1].as_ref().unwrap().set_neighbors(0, &[0, 2]);
        graph[2].as_ref().unwrap().set_neighbors(0, &[1, 3]);
        graph[3].as_ref().unwrap().set_neighbors(0, &[2]);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        // Query close to id 3
        let query = [3.0, 0.0, 0.0, 0.0];

        let (result_id, result_dist) = greedy_search(
            0, // start from entry point 0
            &query,
            0,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        assert_eq!(result_id, 3);
        assert!(result_dist < 0.001);
    }

    #[test]
    fn test_greedy_search_invalid_entry_point() {
        let dist_fn = L2Distance::<f32>::new(4);
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]];

        let graph: Vec<Option<ElementGraphData>> = vec![Some(ElementGraphData::new(1, 0, 4, 2))];

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let query = [1.0, 0.0, 0.0, 0.0];

        // Entry point 99 doesn't exist
        let (result_id, result_dist) = greedy_search(
            99,
            &query,
            0,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            4,
        );

        // Should stay at entry point with infinity distance
        assert_eq!(result_id, 99);
        assert!(result_dist.is_infinite());
    }

    #[test]
    fn test_search_layer_basic() {
        use crate::index::hnsw::VisitedNodesHandlerPool;

        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1 - closest to query
            vec![2.0, 0.0], // id 2
            vec![3.0, 0.0], // id 3
        ];

        // Create connected graph
        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        for i in 0..4 {
            graph.push(Some(ElementGraphData::new(i as u64, 0, 4, 2)));
        }
        // Full connectivity at level 0
        graph[0].as_ref().unwrap().set_neighbors(0, &[1, 2, 3]);
        graph[1].as_ref().unwrap().set_neighbors(0, &[0, 2, 3]);
        graph[2].as_ref().unwrap().set_neighbors(0, &[0, 1, 3]);
        graph[3].as_ref().unwrap().set_neighbors(0, &[0, 1, 2]);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let query = [1.0, 0.0]; // Closest to id 1
        let entry_points = vec![(0u32, 1.0f32)]; // Start from id 0

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool>(
            &entry_points,
            &query,
            0,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
            None,
        );

        assert!(!results.is_empty());
        // First result should be id 1 (exact match)
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_search_layer_with_filter() {
        use crate::index::hnsw::VisitedNodesHandlerPool;

        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1
            vec![2.0, 0.0], // id 2
            vec![3.0, 0.0], // id 3
        ];

        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        for i in 0..4 {
            graph.push(Some(ElementGraphData::new(i as u64, 0, 4, 2)));
        }
        graph[0].as_ref().unwrap().set_neighbors(0, &[1, 2, 3]);
        graph[1].as_ref().unwrap().set_neighbors(0, &[0, 2, 3]);
        graph[2].as_ref().unwrap().set_neighbors(0, &[0, 1, 3]);
        graph[3].as_ref().unwrap().set_neighbors(0, &[0, 1, 2]);

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let query = [1.0, 0.0];
        let entry_points = vec![(0u32, 1.0f32)];

        // Filter: only accept even IDs
        let filter = |id: IdType| -> bool { id % 2 == 0 };

        let results = search_layer(
            &entry_points,
            &query,
            0,
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
    fn test_search_layer_skips_deleted() {
        use crate::index::hnsw::VisitedNodesHandlerPool;

        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], // id 0
            vec![1.0, 0.0], // id 1 - closest but deleted
            vec![2.0, 0.0], // id 2
        ];

        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        for i in 0..3 {
            graph.push(Some(ElementGraphData::new(i as u64, 0, 4, 2)));
        }
        graph[0].as_ref().unwrap().set_neighbors(0, &[1, 2]);
        graph[1].as_ref().unwrap().set_neighbors(0, &[0, 2]);
        graph[2].as_ref().unwrap().set_neighbors(0, &[0, 1]);

        // Mark id 1 as deleted
        graph[1].as_mut().unwrap().meta.deleted = true;

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let query = [1.0, 0.0]; // Closest to id 1 (which is deleted)
        let entry_points = vec![(0u32, 1.0f32)];

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool>(
            &entry_points,
            &query,
            0,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
            None,
        );

        // Should not include deleted id 1
        for (id, _) in &results {
            assert_ne!(*id, 1, "Deleted node should not appear in results");
        }
    }

    #[test]
    fn test_search_layer_empty_graph() {
        use crate::index::hnsw::VisitedNodesHandlerPool;

        let dist_fn = L2Distance::<f32>::new(2);
        let graph: Vec<Option<ElementGraphData>> = Vec::new();

        let data_getter = |_id: IdType| -> Option<&[f32]> { None };

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let query = [1.0, 0.0];
        let entry_points: Vec<(IdType, f32)> = vec![];

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool>(
            &entry_points,
            &query,
            0,
            10,
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
            None,
        );

        assert!(results.is_empty());
    }

    #[test]
    fn test_search_layer_respects_ef() {
        use crate::index::hnsw::VisitedNodesHandlerPool;

        let dist_fn = L2Distance::<f32>::new(2);
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();

        let mut graph: Vec<Option<ElementGraphData>> = Vec::new();
        for i in 0..10 {
            graph.push(Some(ElementGraphData::new(i as u64, 0, 10, 5)));
        }

        // Fully connected
        for i in 0..10 {
            let neighbors: Vec<u32> = (0..10).filter(|&j| j != i).map(|j| j as u32).collect();
            graph[i].as_ref().unwrap().set_neighbors(0, &neighbors);
        }

        let data_getter = |id: IdType| -> Option<&[f32]> {
            if (id as usize) < vectors.len() {
                Some(&vectors[id as usize])
            } else {
                None
            }
        };

        let pool = VisitedNodesHandlerPool::new(100);
        let visited = pool.get();

        let query = [0.0, 0.0];
        let entry_points = vec![(5u32, 25.0f32)];

        // Set ef = 3, should return at most 3 results
        let results = search_layer::<f32, f32, _, fn(IdType) -> bool>(
            &entry_points,
            &query,
            0,
            3, // ef = 3
            &graph,
            data_getter,
            &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
            2,
            &visited,
            None,
        );

        assert!(results.len() <= 3);
    }
}
