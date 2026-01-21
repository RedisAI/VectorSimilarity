//! Search algorithms for HNSW graph traversal.
//!
//! This module provides the core search algorithms:
//! - `greedy_search`: Single entry point search (for upper layers)
//! - `search_layer`: Full layer search with candidate exploration

use super::concurrent_graph::ConcurrentGraph;
use super::graph::ElementGraphData;
use super::visited::VisitedNodesHandler;
use crate::distance::DistanceFunction;
use crate::types::{DistanceType, IdType, VectorElement};
use crate::utils::prefetch::prefetch_slice;
use crate::utils::{MaxHeap, MinHeap};

/// Trait for graph access abstraction.
///
/// This allows search functions to work with both slice-based graphs
/// (for tests) and the concurrent graph structure (for production).
pub trait GraphAccess {
    /// Get an element by ID.
    fn get(&self, id: IdType) -> Option<&ElementGraphData>;
}

/// Implementation for slice-based graphs (used in tests).
impl GraphAccess for [Option<ElementGraphData>] {
    #[inline]
    fn get(&self, id: IdType) -> Option<&ElementGraphData> {
        <[Option<ElementGraphData>]>::get(self, id as usize).and_then(|x| x.as_ref())
    }
}

/// Implementation for Vec-based graphs (used in tests).
impl GraphAccess for Vec<Option<ElementGraphData>> {
    #[inline]
    fn get(&self, id: IdType) -> Option<&ElementGraphData> {
        self.as_slice().get(id as usize).and_then(|x| x.as_ref())
    }
}

/// Implementation for ConcurrentGraph.
impl GraphAccess for ConcurrentGraph {
    #[inline]
    fn get(&self, id: IdType) -> Option<&ElementGraphData> {
        ConcurrentGraph::get(self, id)
    }
}

/// Result of a layer search: (id, distance) pairs.
pub type SearchResult<D> = Vec<(IdType, D)>;



/// Greedy search to find the single closest element at a given layer.
///
/// This is used to traverse upper layers where we just need to find
/// the best entry point for the next layer.
pub fn greedy_search<'a, T, D, F, G>(
    entry_point: IdType,
    query: &[T],
    level: usize,
    graph: &G,
    data_getter: F,
    dist_fn: &dyn DistanceFunction<T, Output = D>,
    dim: usize,
) -> (IdType, D)
where
    T: VectorElement,
    D: DistanceType,
    F: Fn(IdType) -> Option<&'a [T]>,
    G: GraphAccess + ?Sized,
{
    let mut current = entry_point;
    let mut current_dist = if let Some(data) = data_getter(entry_point) {
        dist_fn.compute(data, query, dim)
    } else {
        D::infinity()
    };

    loop {
        let mut changed = false;

        if let Some(element) = graph.get(current) {
            // Get neighbor count without allocation
            let neighbor_count = element.neighbor_count(level);

            // Prefetch first neighbor's data
            if neighbor_count > 0 {
                if let Some(first_neighbor) = element.get_neighbor_at(level, 0) {
                    if let Some(first_data) = data_getter(first_neighbor) {
                        prefetch_slice(first_data);
                    }
                }
            }

            // Iterate using indexed access to avoid Vec allocation
            for i in 0..neighbor_count {
                let neighbor = match element.get_neighbor_at(level, i) {
                    Some(n) => n,
                    None => continue,
                };

                // Prefetch next neighbor's data while processing current
                if i + 1 < neighbor_count {
                    if let Some(next_neighbor) = element.get_neighbor_at(level, i + 1) {
                        if let Some(next_data) = data_getter(next_neighbor) {
                            prefetch_slice(next_data);
                        }
                    }
                }

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
pub fn search_layer<'a, T, D, F, P, G>(
    entry_points: &[(IdType, D)],
    query: &[T],
    level: usize,
    ef: usize,
    graph: &G,
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
    G: GraphAccess + ?Sized,
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
        if let Some(element) = graph.get(candidate.id) {
            if element.meta.deleted {
                continue;
            }

            // Get neighbor count without allocation
            let neighbor_count = element.neighbor_count(level);

            // Prefetch first neighbor's data
            if neighbor_count > 0 {
                if let Some(first_neighbor) = element.get_neighbor_at(level, 0) {
                    if let Some(first_data) = data_getter(first_neighbor) {
                        prefetch_slice(first_data);
                    }
                }
            }

            // Iterate using indexed access to avoid Vec allocation
            for i in 0..neighbor_count {
                // Get current neighbor
                let neighbor = match element.get_neighbor_at(level, i) {
                    Some(n) => n,
                    None => continue,
                };

                // Prefetch next neighbor's data while processing current
                if i + 1 < neighbor_count {
                    if let Some(next_neighbor) = element.get_neighbor_at(level, i + 1) {
                        if let Some(next_data) = data_getter(next_neighbor) {
                            prefetch_slice(next_data);
                        }
                    }
                }

                if visited.visit(neighbor) {
                    continue; // Already visited
                }

                // Check if neighbor is valid
                if let Some(neighbor_element) = graph.get(neighbor) {
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
pub fn search_layer_multi<'a, T, D, F, P, G>(
    entry_points: &[(IdType, D)],
    query: &[T],
    level: usize,
    k: usize,
    ef: usize,
    graph: &G,
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
    G: GraphAccess + ?Sized,
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
        if let Some(element) = graph.get(candidate.id) {
            if element.meta.deleted {
                continue;
            }

            // Get neighbor count without allocation
            let neighbor_count = element.neighbor_count(level);

            // Prefetch first neighbor's data
            if neighbor_count > 0 {
                if let Some(first_neighbor) = element.get_neighbor_at(level, 0) {
                    if let Some(first_data) = data_getter(first_neighbor) {
                        prefetch_slice(first_data);
                    }
                }
            }

            // Iterate using indexed access to avoid Vec allocation
            for i in 0..neighbor_count {
                let neighbor = match element.get_neighbor_at(level, i) {
                    Some(n) => n,
                    None => continue,
                };

                // Prefetch next neighbor's data while processing current
                if i + 1 < neighbor_count {
                    if let Some(next_neighbor) = element.get_neighbor_at(level, i + 1) {
                        if let Some(next_data) = data_getter(next_neighbor) {
                            prefetch_slice(next_data);
                        }
                    }
                }

                if visited.visit(neighbor) {
                    continue; // Already visited
                }

                // Check if neighbor is valid
                if let Some(neighbor_element) = graph.get(neighbor) {
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

/// Range search on a layer to find all elements within a radius.
///
/// This implements the epsilon-neighborhood search algorithm from the C++ implementation:
/// - Uses a dynamic range that shrinks as closer candidates are found
/// - Terminates early when no candidates are within range * (1 + epsilon)
/// - Returns all vectors within the radius
///
/// # Arguments
/// * `entry_points` - Initial entry points with their distances
/// * `query` - Query vector
/// * `level` - Graph level to search (typically 0 for range queries)
/// * `radius` - Maximum distance for results
/// * `epsilon` - Expansion factor for search boundaries (e.g., 0.01 = 1% expansion)
/// * `graph` - Graph structure
/// * `data_getter` - Function to get vector data by ID
/// * `dist_fn` - Distance function
/// * `dim` - Vector dimension
/// * `visited` - Visited nodes handler for deduplication
/// * `filter` - Optional filter function for results
#[allow(clippy::too_many_arguments)]
pub fn search_layer_range<'a, T, D, F, P, G>(
    entry_points: &[(IdType, D)],
    query: &[T],
    level: usize,
    radius: D,
    epsilon: f64,
    graph: &G,
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
    G: GraphAccess + ?Sized,
{
    // Results container
    let mut results: Vec<(IdType, D)> = Vec::new();

    // Candidates to explore (min-heap: closest first, stored as negative for max-heap behavior)
    // We use MinHeap but negate distances to get max-heap behavior (pop largest = closest)
    let mut candidates = MinHeap::<D>::with_capacity(256);

    // Initialize dynamic range based on entry point
    let mut dynamic_range = D::infinity();

    // Initialize with entry points
    for &(id, dist) in entry_points {
        if !visited.visit(id) {
            candidates.push(id, dist);

            // Check if entry point is within radius
            let passes_filter = filter.is_none_or(|f| f(id));
            if passes_filter && dist.to_f64() <= radius.to_f64() {
                results.push((id, dist));
            }

            // Update dynamic range
            if dist.to_f64() < dynamic_range.to_f64() {
                dynamic_range = dist;
            }
        }
    }

    // Ensure dynamic_range >= radius (we need to explore at least to the radius)
    if dynamic_range.to_f64() < radius.to_f64() {
        dynamic_range = radius;
    }

    // Search boundary includes epsilon expansion
    let compute_boundary = |range: D| -> f64 { range.to_f64() * (1.0 + epsilon) };

    // Explore candidates
    // Compute initial boundary (matching C++ behavior: boundary is computed BEFORE shrinking)
    let mut current_boundary = compute_boundary(dynamic_range);

    while let Some(candidate) = candidates.pop() {
        // Early termination: stop if best candidate is outside the dynamic search boundary
        if candidate.distance.to_f64() > current_boundary {
            break;
        }

        // Shrink dynamic range if this candidate is closer but still >= radius
        // Update boundary AFTER shrinking (matching C++ behavior)
        let cand_dist = candidate.distance.to_f64();
        if cand_dist < dynamic_range.to_f64() && cand_dist >= radius.to_f64() {
            dynamic_range = candidate.distance;
            current_boundary = compute_boundary(dynamic_range);
        }

        // Get neighbors of this candidate
        if let Some(element) = graph.get(candidate.id) {
            if element.meta.deleted {
                continue;
            }

            // Get neighbor count without allocation
            let neighbor_count = element.neighbor_count(level);

            // Prefetch first neighbor's data
            if neighbor_count > 0 {
                if let Some(first_neighbor) = element.get_neighbor_at(level, 0) {
                    if let Some(first_data) = data_getter(first_neighbor) {
                        prefetch_slice(first_data);
                    }
                }
            }

            // Iterate using indexed access to avoid Vec allocation
            for i in 0..neighbor_count {
                let neighbor = match element.get_neighbor_at(level, i) {
                    Some(n) => n,
                    None => continue,
                };

                // Prefetch next neighbor's data while processing current
                if i + 1 < neighbor_count {
                    if let Some(next_neighbor) = element.get_neighbor_at(level, i + 1) {
                        if let Some(next_data) = data_getter(next_neighbor) {
                            prefetch_slice(next_data);
                        }
                    }
                }

                if visited.visit(neighbor) {
                    continue; // Already visited
                }

                // Check if neighbor is valid
                if let Some(neighbor_element) = graph.get(neighbor) {
                    if neighbor_element.meta.deleted {
                        continue;
                    }
                }

                // Compute distance to neighbor
                if let Some(data) = data_getter(neighbor) {
                    let dist = dist_fn.compute(data, query, dim);
                    let dist_f64 = dist.to_f64();

                    // Add to candidates if within dynamic search boundary
                    if dist_f64 < current_boundary {
                        candidates.push(neighbor, dist);

                        // Add to results if within radius and passes filter
                        if dist_f64 <= radius.to_f64() {
                            let passes = filter.is_none_or(|f| f(neighbor));
                            if passes {
                                results.push((neighbor, dist));
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort results by distance
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
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
/// This heuristic ensures diversity in the selected neighbors by checking
/// that each candidate is closer to the target than to any already-selected
/// neighbor. Uses batch distance computation for better cache efficiency.
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
    use crate::distance::batch::check_candidate_diversity;

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

        // Use batch diversity check - this computes distances from candidate
        // to all selected neighbors and checks if any is closer than candidate_dist
        let is_good = check_candidate_diversity(
            candidate_id,
            candidate_dist,
            &selected,
            &data_getter,
            dist_fn,
            dim,
        );

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

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool, _>(
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

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool, _>(
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

        let results = search_layer::<f32, f32, _, fn(IdType) -> bool, _>(
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
        let results = search_layer::<f32, f32, _, fn(IdType) -> bool, _>(
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
