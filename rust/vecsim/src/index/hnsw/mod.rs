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
use rand::Rng;
use std::sync::atomic::{AtomicU32, Ordering};

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
pub(crate) struct HnswCore<T: VectorElement> {
    /// Vector storage.
    pub data: DataBlocks<T>,
    /// Graph structure for each element.
    pub graph: Vec<Option<ElementGraphData>>,
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
            graph: Vec::with_capacity(params.initial_capacity),
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
        let processed = self.dist_fn.preprocess(vector, self.params.dim);
        self.data.add(&processed)
    }

    /// Get vector data by ID.
    #[inline]
    pub fn get_vector(&self, id: IdType) -> Option<&[T]> {
        self.data.get(id)
    }

    /// Insert a new element into the graph.
    pub fn insert(&mut self, id: IdType, label: LabelType) {
        let level = self.generate_random_level();

        // Create graph data for this element
        let graph_data = ElementGraphData::new(
            label,
            level,
            self.params.m_max_0,
            self.params.m,
        );

        // Ensure graph vector is large enough
        let id_usize = id as usize;
        if id_usize >= self.graph.len() {
            self.graph.resize_with(id_usize + 1, || None);
        }
        self.graph[id_usize] = Some(graph_data);

        // Update visited pool if needed
        if id_usize >= self.visited_pool.current_capacity() {
            self.visited_pool.resize(id_usize + 1024);
        }

        let entry_point = self.entry_point.load(Ordering::Acquire);

        if entry_point == INVALID_ID {
            // First element
            self.entry_point.store(id, Ordering::Release);
            self.max_level.store(level as u32, Ordering::Release);
            return;
        }

        // Get query vector
        let query = match self.get_vector(id) {
            Some(v) => v,
            None => return,
        };

        // Search from entry point to find insertion point
        let current_max = self.max_level.load(Ordering::Acquire) as usize;
        let mut current_entry = entry_point;

        // Traverse upper layers with greedy search
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

        // Insert at each level from min(level, max_level) down to 0
        let start_level = level.min(current_max as u8);
        let mut entry_points = vec![(current_entry, self.compute_distance(current_entry, query))];

        for l in (0..=start_level as usize).rev() {
            let mut visited = self.visited_pool.get();
            visited.reset();

            // Search this layer
            let neighbors = search::search_layer::<T, T::DistanceType, _, fn(IdType) -> bool>(
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

            // Select neighbors
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

            // Set outgoing edges for new element
            if let Some(Some(element)) = self.graph.get(id as usize) {
                element.set_neighbors(l, &selected);
            }

            // Add incoming edges from selected neighbors
            for &neighbor_id in &selected {
                self.add_bidirectional_link(neighbor_id, id, l);
            }

            // Use neighbors as entry points for next level
            if !neighbors.is_empty() {
                entry_points = neighbors;
            }
        }

        // Update entry point and max level if needed
        if level as u32 > self.max_level.load(Ordering::Acquire) {
            self.max_level.store(level as u32, Ordering::Release);
            self.entry_point.store(id, Ordering::Release);
        }
    }

    /// Add a bidirectional link between two elements at a given level.
    fn add_bidirectional_link(&self, from: IdType, to: IdType, level: usize) {
        if let Some(Some(from_element)) = self.graph.get(from as usize) {
            if level < from_element.levels.len() {
                let _lock = from_element.lock.lock();

                let mut current_neighbors = from_element.get_neighbors(level);
                if current_neighbors.contains(&to) {
                    return;
                }

                current_neighbors.push(to);

                // Check if we need to prune
                let m = if level == 0 { self.params.m_max_0 } else { self.params.m };

                if current_neighbors.len() > m {
                    // Need to select best neighbors
                    let query = match self.data.get(from) {
                        Some(v) => v,
                        None => return,
                    };

                    let candidates: Vec<_> = current_neighbors
                        .iter()
                        .filter_map(|&n| {
                            self.data.get(n).map(|data| {
                                let dist = self.dist_fn.compute(data, query, self.params.dim);
                                (n, dist)
                            })
                        })
                        .collect();

                    let selected = if self.params.enable_heuristic {
                        search::select_neighbors_heuristic(
                            from,
                            &candidates,
                            m,
                            |id| self.data.get(id),
                            self.dist_fn.as_ref(),
                            self.params.dim,
                            false,
                            true,
                        )
                    } else {
                        search::select_neighbors_simple(&candidates, m)
                    };

                    from_element.set_neighbors(level, &selected);
                } else {
                    from_element.set_neighbors(level, &current_neighbors);
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
        if let Some(Some(element)) = self.graph.get_mut(id as usize) {
            element.meta.deleted = true;
        }
        self.data.mark_deleted(id);
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
            search::search_layer::<T, T::DistanceType, _, fn(IdType) -> bool>(
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
}
