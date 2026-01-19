//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! HNSW is an approximate nearest neighbor algorithm that provides
//! logarithmic query complexity with high recall. It constructs a
//! multi-layer graph where each layer is a proximity graph.
//!
//! Key parameters:
//! - `M`: Maximum number of connections per element per layer
//! - `ef_construction`: Size of dynamic candidate list during construction
//! - `ef_runtime`: Size of dynamic candidate list during search (runtime)

pub mod batch_iterator;
pub mod concurrent_graph;
pub mod graph;
pub mod multi;
pub mod search;
pub mod single;
pub mod visited;

pub use batch_iterator::{HnswSingleBatchIterator, HnswMultiBatchIterator};
/// Type alias for HNSW batch iterator.
pub type HnswBatchIterator<'a, T> = HnswSingleBatchIterator<'a, T>;
pub use graph::{ElementGraphData, DEFAULT_M, DEFAULT_M_MAX, DEFAULT_M_MAX_0};
pub use multi::HnswMulti;
pub use single::{HnswSingle, HnswStats};
pub use visited::{VisitedNodesHandler, VisitedNodesHandlerPool};

use crate::containers::DataBlocks;
use crate::distance::{create_distance_function, DistanceFunction, Metric};
use crate::types::{DistanceType, IdType, LabelType, VectorElement, INVALID_ID};
use concurrent_graph::ConcurrentGraph;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

// Profiling support
#[cfg(feature = "profile")]
use std::cell::RefCell;
#[cfg(feature = "profile")]
use std::time::Instant;

#[cfg(feature = "profile")]
thread_local! {
    pub static PROFILE_STATS: RefCell<ProfileStats> = RefCell::new(ProfileStats::default());
}

#[cfg(feature = "profile")]
#[derive(Default)]
pub struct ProfileStats {
    pub search_layer_ns: u64,
    pub select_neighbors_ns: u64,
    pub add_links_ns: u64,
    pub visited_pool_ns: u64,
    pub greedy_search_ns: u64,
    pub calls: u64,
    // Detailed add_links breakdown
    pub add_links_lock_ns: u64,
    pub add_links_get_neighbors_ns: u64,
    pub add_links_contains_ns: u64,
    pub add_links_prune_ns: u64,
    pub add_links_set_ns: u64,
    pub add_links_prune_count: u64,
}

#[cfg(feature = "profile")]
impl ProfileStats {
    pub fn print_and_reset(&mut self) {
        if self.calls > 0 {
            let total = self.search_layer_ns + self.select_neighbors_ns + self.add_links_ns
                + self.visited_pool_ns + self.greedy_search_ns;
            println!("Profile stats ({} insertions):", self.calls);
            println!("  search_layer:      {:>8.2}ms ({:>5.1}%)",
                     self.search_layer_ns as f64 / 1_000_000.0,
                     100.0 * self.search_layer_ns as f64 / total as f64);
            println!("  select_neighbors:  {:>8.2}ms ({:>5.1}%)",
                     self.select_neighbors_ns as f64 / 1_000_000.0,
                     100.0 * self.select_neighbors_ns as f64 / total as f64);
            println!("  add_links:         {:>8.2}ms ({:>5.1}%)",
                     self.add_links_ns as f64 / 1_000_000.0,
                     100.0 * self.add_links_ns as f64 / total as f64);
            println!("  visited_pool:      {:>8.2}ms ({:>5.1}%)",
                     self.visited_pool_ns as f64 / 1_000_000.0,
                     100.0 * self.visited_pool_ns as f64 / total as f64);
            println!("  greedy_search:     {:>8.2}ms ({:>5.1}%)",
                     self.greedy_search_ns as f64 / 1_000_000.0,
                     100.0 * self.greedy_search_ns as f64 / total as f64);

            // Detailed add_links breakdown
            if self.add_links_ns > 0 {
                println!("\n  add_links breakdown:");
                println!("    lock:            {:>8.2}ms ({:>5.1}%)",
                         self.add_links_lock_ns as f64 / 1_000_000.0,
                         100.0 * self.add_links_lock_ns as f64 / self.add_links_ns as f64);
                println!("    get_neighbors:   {:>8.2}ms ({:>5.1}%)",
                         self.add_links_get_neighbors_ns as f64 / 1_000_000.0,
                         100.0 * self.add_links_get_neighbors_ns as f64 / self.add_links_ns as f64);
                println!("    contains:        {:>8.2}ms ({:>5.1}%)",
                         self.add_links_contains_ns as f64 / 1_000_000.0,
                         100.0 * self.add_links_contains_ns as f64 / self.add_links_ns as f64);
                println!("    prune:           {:>8.2}ms ({:>5.1}%) [{} calls]",
                         self.add_links_prune_ns as f64 / 1_000_000.0,
                         100.0 * self.add_links_prune_ns as f64 / self.add_links_ns as f64,
                         self.add_links_prune_count);
                println!("    set_neighbors:   {:>8.2}ms ({:>5.1}%)",
                         self.add_links_set_ns as f64 / 1_000_000.0,
                         100.0 * self.add_links_set_ns as f64 / self.add_links_ns as f64);
            }
        }
        *self = ProfileStats::default();
    }
}

/// Parameters for creating an HNSW index.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Maximum number of connections per element (default: 16).
    pub m: usize,
    /// Maximum connections at level 0 (default: 2*M).
    pub m_max_0: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (default value).
    pub ef_runtime: usize,
    /// Initial capacity (number of vectors).
    pub initial_capacity: usize,
    /// Enable diverse neighbor selection heuristic.
    pub enable_heuristic: bool,
    /// Random seed for reproducible level generation (None = random).
    pub seed: Option<u64>,
}

impl HnswParams {
    /// Create new parameters with required fields.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            m: DEFAULT_M,
            m_max_0: DEFAULT_M_MAX_0,
            ef_construction: 200,
            ef_runtime: 10,
            initial_capacity: 1024,
            enable_heuristic: true,
            seed: None,
        }
    }

    /// Set M parameter.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max_0 = m * 2;
        self
    }

    /// Set M_max_0 parameter independently (overrides 2*M default).
    pub fn with_m_max_0(mut self, m_max_0: usize) -> Self {
        self.m_max_0 = m_max_0;
        self
    }

    /// Set ef_construction.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_runtime.
    pub fn with_ef_runtime(mut self, ef: usize) -> Self {
        self.ef_runtime = ef;
        self
    }

    /// Set initial capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Enable/disable heuristic neighbor selection.
    pub fn with_heuristic(mut self, enable: bool) -> Self {
        self.enable_heuristic = enable;
        self
    }

    /// Set random seed for reproducible level generation.
    ///
    /// When set, the same sequence of insertions will produce
    /// the same graph structure, useful for testing and debugging.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Core HNSW implementation shared between single and multi variants.
///
/// This structure supports concurrent access for parallel insertion using
/// a lock-free graph structure. Only per-node locks are used for neighbor
/// modifications, matching the C++ implementation approach.
pub(crate) struct HnswCore<T: VectorElement> {
    /// Vector storage (thread-safe with interior mutability).
    pub data: DataBlocks<T>,
    /// Graph structure for each element (lock-free concurrent access).
    pub graph: ConcurrentGraph,
    /// Distance function.
    pub dist_fn: Box<dyn DistanceFunction<T, Output = T::DistanceType>>,
    /// Entry point to the graph (top level).
    pub entry_point: AtomicU32,
    /// Current maximum level in the graph.
    pub max_level: AtomicU32,
    /// Pool of visited handlers for concurrent searches.
    pub visited_pool: VisitedNodesHandlerPool,
    /// Parameters.
    pub params: HnswParams,
    /// Multiplier for random level generation (1/ln(M)).
    pub level_mult: f64,
    /// Random number generator for level selection.
    rng: parking_lot::Mutex<rand::rngs::StdRng>,
}

impl<T: VectorElement> HnswCore<T> {
    /// Create a new HNSW core.
    pub fn new(params: HnswParams) -> Self {
        use rand::SeedableRng;

        let data = DataBlocks::new(params.dim, params.initial_capacity);
        let dist_fn = create_distance_function(params.metric, params.dim);
        let visited_pool = VisitedNodesHandlerPool::new(params.initial_capacity);

        // Level multiplier: 1/ln(M)
        let level_mult = 1.0 / (params.m as f64).ln();

        // Initialize RNG with seed if provided, otherwise use entropy
        let rng = match params.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self {
            data,
            graph: ConcurrentGraph::new(params.initial_capacity),
            dist_fn,
            entry_point: AtomicU32::new(INVALID_ID),
            max_level: AtomicU32::new(0),
            visited_pool,
            level_mult,
            rng: parking_lot::Mutex::new(rng),
            params,
        }
    }

    /// Generate a random level for a new element.
    pub fn generate_random_level(&self) -> u8 {
        let mut rng = self.rng.lock();
        let r: f64 = rng.gen();
        let level = (-r.ln() * self.level_mult).floor() as u8;
        level.min(32) // Cap at reasonable level
    }

    /// Add a vector and return its internal ID.
    ///
    /// Returns `None` if the vector dimension doesn't match.
    pub fn add_vector(&mut self, vector: &[T]) -> Option<IdType> {
        self.add_vector_concurrent(vector)
    }

    /// Add a vector concurrently and return its internal ID.
    ///
    /// This method is thread-safe and can be called from multiple threads.
    /// Returns `None` if the vector dimension doesn't match.
    pub fn add_vector_concurrent(&self, vector: &[T]) -> Option<IdType> {
        let processed = self.dist_fn.preprocess(vector, self.params.dim);
        self.data.add_concurrent(&processed)
    }

    /// Get vector data by ID.
    #[inline]
    pub fn get_vector(&self, id: IdType) -> Option<&[T]> {
        self.data.get(id)
    }

    /// Insert a new element into the graph.
    #[cfg(not(feature = "profile"))]
    pub fn insert(&mut self, id: IdType, label: LabelType) {
        self.insert_concurrent(id, label);
    }

    /// Insert a new element into the graph (profiled version).
    #[cfg(feature = "profile")]
    pub fn insert(&mut self, id: IdType, label: LabelType) {
        self.insert_concurrent(id, label);
        PROFILE_STATS.with(|s| s.borrow_mut().calls += 1);
    }

    /// Insert a new element into the graph concurrently.
    ///
    /// This method is thread-safe and can be called from multiple threads.
    /// It uses fine-grained locking to allow concurrent insertions while
    /// maintaining graph consistency.
    #[cfg(not(feature = "profile"))]
    pub fn insert_concurrent(&self, id: IdType, label: LabelType) {
        self.insert_concurrent_impl(id, label);
    }

    /// Insert a new element into the graph concurrently (profiled version).
    #[cfg(feature = "profile")]
    pub fn insert_concurrent(&self, id: IdType, label: LabelType) {
        self.insert_concurrent_impl(id, label);
        PROFILE_STATS.with(|s| s.borrow_mut().calls += 1);
    }

    /// Concurrent insert implementation.
    fn insert_concurrent_impl(&self, id: IdType, label: LabelType) {
        let level = self.generate_random_level();

        // Create graph data for this element
        let graph_data = ElementGraphData::new(
            label,
            level,
            self.params.m_max_0,
            self.params.m,
        );

        // Set the graph data (ConcurrentGraph handles capacity automatically)
        let id_usize = id as usize;
        self.graph.set(id, graph_data);

        // Update visited pool if needed
        if id_usize >= self.visited_pool.current_capacity() {
            self.visited_pool.resize(id_usize + 1024);
        }

        let entry_point = self.entry_point.load(Ordering::Acquire);

        if entry_point == INVALID_ID {
            // First element - use CAS to avoid race
            if self.entry_point.compare_exchange(
                INVALID_ID, id,
                Ordering::AcqRel, Ordering::Relaxed
            ).is_ok() {
                self.max_level.store(level as u32, Ordering::Release);
                return;
            }
            // Another thread beat us, continue with normal insertion
        }

        // Get query vector
        let query = match self.get_vector(id) {
            Some(v) => v,
            None => return,
        };

        // Search from entry point to find insertion point
        let current_max = self.max_level.load(Ordering::Acquire) as usize;
        let mut current_entry = self.entry_point.load(Ordering::Acquire);

        // Traverse upper layers with greedy search
        #[cfg(feature = "profile")]
        let greedy_start = Instant::now();

        for l in (level as usize + 1..=current_max).rev() {
            let (new_entry, _) = search::greedy_search(
                current_entry,
                query,
                l,
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
            );
            current_entry = new_entry;
        }

        #[cfg(feature = "profile")]
        PROFILE_STATS.with(|s| s.borrow_mut().greedy_search_ns += greedy_start.elapsed().as_nanos() as u64);

        // Insert at each level from min(level, max_level) down to 0
        let start_level = level.min(current_max as u8);
        let mut entry_points = vec![(current_entry, self.compute_distance(current_entry, query))];

        for l in (0..=start_level as usize).rev() {
            #[cfg(feature = "profile")]
            let pool_start = Instant::now();

            let mut visited = self.visited_pool.get();
            visited.reset();

            #[cfg(feature = "profile")]
            PROFILE_STATS.with(|s| s.borrow_mut().visited_pool_ns += pool_start.elapsed().as_nanos() as u64);

            // Search this layer
            #[cfg(feature = "profile")]
            let search_start = Instant::now();

            let neighbors = search::search_layer::<T, T::DistanceType, _, fn(IdType) -> bool, _>(
                &entry_points,
                query,
                l,
                self.params.ef_construction,
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                None,
            );

            #[cfg(feature = "profile")]
            PROFILE_STATS.with(|s| s.borrow_mut().search_layer_ns += search_start.elapsed().as_nanos() as u64);

            // Select neighbors
            #[cfg(feature = "profile")]
            let select_start = Instant::now();

            let m = if l == 0 { self.params.m_max_0 } else { self.params.m };
            let selected = if self.params.enable_heuristic {
                search::select_neighbors_heuristic(
                    id,
                    &neighbors,
                    m,
                    |id| self.data.get(id),
                    self.dist_fn.as_ref(),
                    self.params.dim,
                    false,
                    true,
                )
            } else {
                search::select_neighbors_simple(&neighbors, m)
            };

            #[cfg(feature = "profile")]
            PROFILE_STATS.with(|s| s.borrow_mut().select_neighbors_ns += select_start.elapsed().as_nanos() as u64);

            // Build selected neighbors with distances for mutually_connect_new_element
            let selected_with_distances: Vec<(IdType, T::DistanceType)> = selected
                .iter()
                .filter_map(|&neighbor_id| {
                    // Find the distance from the search results
                    neighbors.iter()
                        .find(|&&(nid, _)| nid == neighbor_id)
                        .map(|&(nid, dist)| (nid, dist))
                })
                .collect();

            // Mutually connect new element with its neighbors using lock ordering
            #[cfg(feature = "profile")]
            let links_start = Instant::now();

            self.mutually_connect_new_element(id, &selected_with_distances, l);

            #[cfg(feature = "profile")]
            PROFILE_STATS.with(|s| s.borrow_mut().add_links_ns += links_start.elapsed().as_nanos() as u64);

            // Use neighbors as entry points for next level
            if !neighbors.is_empty() {
                entry_points = neighbors;
            }
        }

        // Update entry point and max level if needed using CAS
        self.maybe_update_entry_point(id, level);
    }

    /// Maybe update entry point and max level using CAS for thread safety.
    fn maybe_update_entry_point(&self, new_id: IdType, new_level: u8) {
        loop {
            let current_max = self.max_level.load(Ordering::Acquire);
            if (new_level as u32) <= current_max {
                return;
            }
            if self.max_level.compare_exchange(
                current_max, new_level as u32,
                Ordering::AcqRel, Ordering::Relaxed
            ).is_ok() {
                self.entry_point.store(new_id, Ordering::Release);
                return;
            }
        }
    }

    /// Mutually connect a new element with its selected neighbors at a given level.
    ///
    /// This method implements the C++ lock ordering strategy:
    /// 1. Lock nodes in sorted ID order to prevent deadlocks
    /// 2. Fast path: if neighbor has space, append link directly
    /// 3. Slow path: if neighbor is full, call revisit_neighbor_connections
    ///
    /// This is the key function for achieving true parallel insertion with high recall.
    fn mutually_connect_new_element(
        &self,
        new_node_id: IdType,
        selected_neighbors: &[(IdType, T::DistanceType)],
        level: usize,
    ) {
        let max_m = if level == 0 { self.params.m_max_0 } else { self.params.m };

        for &(neighbor_id, dist_to_neighbor) in selected_neighbors {
            // Get both elements
            let (new_element, neighbor_element) = match (
                self.graph.get(new_node_id),
                self.graph.get(neighbor_id),
            ) {
                (Some(n), Some(e)) => (n, e),
                _ => continue,
            };

            // Check level bounds
            if level >= new_element.levels.len() || level >= neighbor_element.levels.len() {
                continue;
            }

            // Lock in sorted ID order to prevent deadlocks (C++ pattern)
            let (_lock1, _lock2) = if new_node_id < neighbor_id {
                (new_element.lock.lock(), neighbor_element.lock.lock())
            } else {
                (neighbor_element.lock.lock(), new_element.lock.lock())
            };

            // Check if new node can still add neighbors (may have changed between iterations)
            let new_node_neighbors = new_element.get_neighbors(level);
            if new_node_neighbors.len() >= max_m {
                // New node is full, skip remaining neighbors
                break;
            }

            // Check if connection already exists
            if new_node_neighbors.contains(&neighbor_id) {
                continue;
            }

            // Check if neighbor has space for the new node
            let neighbor_neighbors = neighbor_element.get_neighbors(level);
            if neighbor_neighbors.len() < max_m {
                // Fast path: neighbor has space, make bidirectional connection
                let mut new_neighbors = new_node_neighbors;
                new_neighbors.push(neighbor_id);
                new_element.set_neighbors(level, &new_neighbors);

                let mut updated_neighbor_neighbors = neighbor_neighbors;
                updated_neighbor_neighbors.push(new_node_id);
                neighbor_element.set_neighbors(level, &updated_neighbor_neighbors);
            } else {
                // Slow path: neighbor is full, need to revisit its connections
                // First add new_node -> neighbor (new node has space, we checked above)
                let mut new_neighbors = new_node_neighbors;
                new_neighbors.push(neighbor_id);
                new_element.set_neighbors(level, &new_neighbors);

                // Now revisit neighbor's connections to possibly include new_node
                self.revisit_neighbor_connections_locked(
                    new_node_id,
                    neighbor_id,
                    dist_to_neighbor,
                    neighbor_element,
                    level,
                    max_m,
                );
            }
        }
    }

    /// Revisit a neighbor's connections when it's at capacity.
    ///
    /// This is called while holding both the new_node and neighbor locks.
    /// It re-evaluates which neighbors the neighbor should keep, possibly
    /// including the new node if it's closer than existing neighbors.
    fn revisit_neighbor_connections_locked(
        &self,
        new_node_id: IdType,
        neighbor_id: IdType,
        dist_to_new_node: T::DistanceType,
        neighbor_element: &ElementGraphData,
        level: usize,
        max_m: usize,
    ) {
        // Collect all candidates: existing neighbors + new node
        let neighbor_data = match self.data.get(neighbor_id) {
            Some(d) => d,
            None => return,
        };

        let current_neighbors = neighbor_element.get_neighbors(level);
        let mut candidates: Vec<(IdType, T::DistanceType)> = Vec::with_capacity(current_neighbors.len() + 1);

        // Add new node as candidate with its pre-computed distance
        candidates.push((new_node_id, dist_to_new_node));

        // Add existing neighbors with their distances to the neighbor
        for &existing_neighbor_id in &current_neighbors {
            if let Some(existing_data) = self.data.get(existing_neighbor_id) {
                let dist = self.dist_fn.compute(existing_data, neighbor_data, self.params.dim);
                candidates.push((existing_neighbor_id, dist));
            }
        }

        // Select best M neighbors using simple selection (M closest)
        if candidates.len() > max_m {
            candidates.select_nth_unstable_by(max_m - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(max_m);
        }

        let selected: Vec<IdType> = candidates.iter().map(|&(id, _)| id).collect();

        // Update neighbor's connections
        neighbor_element.set_neighbors(level, &selected);
    }

    /// Add a bidirectional link between two elements at a given level (concurrent version).
    ///
    /// This method uses the per-node lock in ElementGraphData for thread safety.
    /// NOTE: This is the legacy single-lock version, kept for compatibility.
    /// For parallel insertion, use mutually_connect_new_element instead.
    #[allow(dead_code)]
    fn add_bidirectional_link_concurrent(&self, from: IdType, to: IdType, level: usize) {
        if let Some(from_element) = self.graph.get(from) {
            if level < from_element.levels.len() {
                #[cfg(feature = "profile")]
                let lock_start = Instant::now();

                let _lock = from_element.lock.lock();

                #[cfg(feature = "profile")]
                PROFILE_STATS.with(|s| s.borrow_mut().add_links_lock_ns += lock_start.elapsed().as_nanos() as u64);

                #[cfg(feature = "profile")]
                let get_start = Instant::now();

                let mut current_neighbors = from_element.get_neighbors(level);

                #[cfg(feature = "profile")]
                PROFILE_STATS.with(|s| s.borrow_mut().add_links_get_neighbors_ns += get_start.elapsed().as_nanos() as u64);

                #[cfg(feature = "profile")]
                let contains_start = Instant::now();

                if current_neighbors.contains(&to) {
                    #[cfg(feature = "profile")]
                    PROFILE_STATS.with(|s| s.borrow_mut().add_links_contains_ns += contains_start.elapsed().as_nanos() as u64);
                    return;
                }

                #[cfg(feature = "profile")]
                PROFILE_STATS.with(|s| s.borrow_mut().add_links_contains_ns += contains_start.elapsed().as_nanos() as u64);

                current_neighbors.push(to);

                // Check if we need to prune
                let m = if level == 0 { self.params.m_max_0 } else { self.params.m };

                if current_neighbors.len() > m {
                    #[cfg(feature = "profile")]
                    let prune_start = Instant::now();

                    // Need to select best neighbors - use simple selection (M closest)
                    // This is faster than the heuristic and still maintains good graph quality
                    let query = match self.data.get(from) {
                        Some(v) => v,
                        None => return,
                    };

                    let mut candidates: Vec<_> = current_neighbors
                        .iter()
                        .filter_map(|&n| {
                            self.data.get(n).map(|data| {
                                let dist = self.dist_fn.compute(data, query, self.params.dim);
                                (n, dist)
                            })
                        })
                        .collect();

                    // Use partial sort to find M closest - O(n) instead of O(n log n)
                    // select_nth_unstable partitions so elements [0..m] are the m smallest
                    if candidates.len() > m {
                        candidates.select_nth_unstable_by(m - 1, |a, b| {
                            a.1.partial_cmp(&b.1).unwrap()
                        });
                        candidates.truncate(m);
                    }
                    let selected: Vec<_> = candidates.iter().map(|&(id, _)| id).collect();

                    #[cfg(feature = "profile")]
                    {
                        PROFILE_STATS.with(|s| {
                            let mut stats = s.borrow_mut();
                            stats.add_links_prune_ns += prune_start.elapsed().as_nanos() as u64;
                            stats.add_links_prune_count += 1;
                        });
                    }

                    #[cfg(feature = "profile")]
                    let set_start = Instant::now();

                    from_element.set_neighbors(level, &selected);

                    #[cfg(feature = "profile")]
                    PROFILE_STATS.with(|s| s.borrow_mut().add_links_set_ns += set_start.elapsed().as_nanos() as u64);
                } else {
                    #[cfg(feature = "profile")]
                    let set_start = Instant::now();

                    from_element.set_neighbors(level, &current_neighbors);

                    #[cfg(feature = "profile")]
                    PROFILE_STATS.with(|s| s.borrow_mut().add_links_set_ns += set_start.elapsed().as_nanos() as u64);
                }
            }
        }
    }

    /// Compute distance between two elements.
    #[inline]
    fn compute_distance(&self, id: IdType, query: &[T]) -> T::DistanceType {
        if let Some(data) = self.data.get(id) {
            self.dist_fn.compute(data, query, self.params.dim)
        } else {
            T::DistanceType::infinity()
        }
    }

    /// Mark an element as deleted.
    pub fn mark_deleted(&mut self, id: IdType) {
        self.mark_deleted_concurrent(id);
    }

    /// Mark an element as deleted (concurrent version).
    pub fn mark_deleted_concurrent(&self, id: IdType) {
        if let Some(element) = self.graph.get(id) {
            // ElementMetaData.deleted is not atomic, but this is a best-effort
            // tombstone - reads may see stale state briefly, which is acceptable
            // for deletion semantics. Use a separate lock if stronger guarantees needed.
            let _lock = element.lock.lock();
            // SAFETY: We hold the element's lock, so this is the only writer
            unsafe {
                let meta = &element.meta as *const _ as *mut graph::ElementMetaData;
                (*meta).deleted = true;
            }
        }
        self.data.mark_deleted_concurrent(id);
    }

    /// Search for nearest neighbors.
    pub fn search(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(IdType) -> bool>,
    ) -> Vec<(IdType, T::DistanceType)> {
        let entry_point = self.entry_point.load(Ordering::Acquire);
        if entry_point == INVALID_ID {
            return Vec::new();
        }

        let current_max = self.max_level.load(Ordering::Acquire) as usize;
        let mut current_entry = entry_point;

        // Greedy search through upper layers
        for l in (1..=current_max).rev() {
            let (new_entry, _) = search::greedy_search(
                current_entry,
                query,
                l,
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
            );
            current_entry = new_entry;
        }

        // Search layer 0 with full ef
        let mut visited = self.visited_pool.get();
        visited.reset();

        let entry_dist = self.compute_distance(current_entry, query);
        let entry_points = vec![(current_entry, entry_dist)];

        let results = if let Some(f) = filter {
            search::search_layer(
                &entry_points,
                query,
                0,
                ef.max(k),
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                Some(f),
            )
        } else {
            search::search_layer::<T, T::DistanceType, _, fn(IdType) -> bool, _>(
                &entry_points,
                query,
                0,
                ef.max(k),
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                None,
            )
        };

        // Return top k
        results.into_iter().take(k).collect()
    }

    /// Search for k nearest unique labels (for multi-value indices).
    ///
    /// This method does label-aware search during graph traversal,
    /// ensuring we find the k closest unique labels rather than the
    /// k closest vectors (which might share labels).
    pub fn search_multi(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        id_to_label: &HashMap<IdType, LabelType>,
        filter: Option<&dyn Fn(LabelType) -> bool>,
    ) -> Vec<(LabelType, T::DistanceType)> {
        let entry_point = self.entry_point.load(Ordering::Acquire);
        if entry_point == INVALID_ID {
            return Vec::new();
        }

        let current_max = self.max_level.load(Ordering::Acquire) as usize;
        let mut current_entry = entry_point;

        // Greedy search through upper layers
        for l in (1..=current_max).rev() {
            let (new_entry, _) = search::greedy_search(
                current_entry,
                query,
                l,
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
            );
            current_entry = new_entry;
        }

        // Search layer 0 with label-aware search
        let mut visited = self.visited_pool.get();
        visited.reset();

        let entry_dist = self.compute_distance(current_entry, query);
        let entry_points = vec![(current_entry, entry_dist)];

        if let Some(f) = filter {
            search::search_layer_multi(
                &entry_points,
                query,
                0,
                k,
                ef.max(k),
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                id_to_label,
                Some(f),
            )
        } else {
            search::search_layer_multi::<T, T::DistanceType, _, fn(LabelType) -> bool, _>(
                &entry_points,
                query,
                0,
                k,
                ef.max(k),
                &self.graph,
                |id| self.data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                id_to_label,
                None,
            )
        }
    }
}
