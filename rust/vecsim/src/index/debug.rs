//! Debug and introspection API for indices.
//!
//! This module provides debugging and introspection capabilities:
//! - Graph structure inspection (HNSW, SVS)
//! - Search mode tracking
//! - Index statistics and metadata
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecsim::index::debug::{DebugInfo, GraphInspector};
//!
//! // Get debug info from an HNSW index
//! let info = index.debug_info();
//! println!("Index size: {}", info.basic.index_size);
//!
//! // Inspect graph neighbors
//! if let Some(inspector) = index.graph_inspector() {
//!     let neighbors = inspector.get_neighbors(0, 0);
//!     println!("Level 0 neighbors of node 0: {:?}", neighbors);
//! }
//! ```

use crate::distance::Metric;
use crate::query::SearchMode;
use crate::types::{IdType, LabelType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Basic immutable index information.
#[derive(Debug, Clone)]
pub struct BasicIndexInfo {
    /// Type of the index (BruteForce, HNSW, SVS, etc.).
    pub index_type: IndexType,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Whether this is a multi-value index.
    pub is_multi: bool,
    /// Block size used for storage.
    pub block_size: usize,
}

/// Mutable index statistics.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Total number of vectors in the index.
    pub index_size: usize,
    /// Number of unique labels.
    pub label_count: usize,
    /// Memory usage in bytes (if tracked).
    pub memory_usage: Option<u64>,
    /// Number of deleted vectors pending cleanup.
    pub deleted_count: usize,
    /// Resize count (number of times the index grew).
    pub resize_count: usize,
}

/// HNSW-specific debug information.
#[derive(Debug, Clone)]
pub struct HnswDebugInfo {
    /// M parameter (max connections per node).
    pub m: usize,
    /// Max M for level 0.
    pub m_max: usize,
    /// Max M for higher levels.
    pub m_max0: usize,
    /// ef_construction parameter.
    pub ef_construction: usize,
    /// Current ef_runtime parameter.
    pub ef_runtime: usize,
    /// Entry point node ID.
    pub entry_point: Option<IdType>,
    /// Maximum level in the graph.
    pub max_level: usize,
    /// Distribution of nodes per level.
    pub level_distribution: Vec<usize>,
}

/// SVS/Vamana-specific debug information.
#[derive(Debug, Clone)]
pub struct SvsDebugInfo {
    /// Alpha parameter for graph construction.
    pub alpha: f32,
    /// Maximum neighbors per node (R).
    pub max_neighbors: usize,
    /// Search window size (L).
    pub search_window_size: usize,
    /// Medoid (entry point) node ID.
    pub medoid: Option<IdType>,
    /// Average degree in the graph.
    pub avg_degree: f32,
}

/// Combined debug information for an index.
#[derive(Debug, Clone)]
pub struct DebugInfo {
    /// Basic immutable info.
    pub basic: BasicIndexInfo,
    /// Mutable statistics.
    pub stats: IndexStats,
    /// HNSW-specific info (if applicable).
    pub hnsw: Option<HnswDebugInfo>,
    /// SVS-specific info (if applicable).
    pub svs: Option<SvsDebugInfo>,
    /// Last search mode used.
    pub last_search_mode: SearchMode,
}

/// Index type enumeration for debug info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BruteForceSingle,
    BruteForceMulti,
    HnswSingle,
    HnswMulti,
    SvsSingle,
    SvsMulti,
    TieredSingle,
    TieredMulti,
    TieredSvsSingle,
    TieredSvsMulti,
    DiskIndex,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::BruteForceSingle => write!(f, "BruteForceSingle"),
            IndexType::BruteForceMulti => write!(f, "BruteForceMulti"),
            IndexType::HnswSingle => write!(f, "HnswSingle"),
            IndexType::HnswMulti => write!(f, "HnswMulti"),
            IndexType::SvsSingle => write!(f, "SvsSingle"),
            IndexType::SvsMulti => write!(f, "SvsMulti"),
            IndexType::TieredSingle => write!(f, "TieredSingle"),
            IndexType::TieredMulti => write!(f, "TieredMulti"),
            IndexType::TieredSvsSingle => write!(f, "TieredSvsSingle"),
            IndexType::TieredSvsMulti => write!(f, "TieredSvsMulti"),
            IndexType::DiskIndex => write!(f, "DiskIndex"),
        }
    }
}

/// Trait for indices that support debug inspection.
pub trait Debuggable {
    /// Get comprehensive debug information.
    fn debug_info(&self) -> DebugInfo;

    /// Get basic immutable information.
    fn basic_info(&self) -> BasicIndexInfo;

    /// Get current statistics.
    fn stats(&self) -> IndexStats;

    /// Get the last search mode used.
    fn last_search_mode(&self) -> SearchMode;
}

/// Trait for inspecting graph structure (HNSW, SVS).
pub trait GraphInspector {
    /// Get neighbors of a node at a specific level.
    ///
    /// For HNSW, level 0 is the base level.
    /// For SVS, there's only one level (use level 0).
    fn get_neighbors(&self, node_id: IdType, level: usize) -> Vec<IdType>;

    /// Get the number of levels for a node (HNSW only).
    fn get_node_level(&self, node_id: IdType) -> usize;

    /// Get the entry point of the graph.
    fn entry_point(&self) -> Option<IdType>;

    /// Get the maximum level in the graph.
    fn max_level(&self) -> usize;

    /// Get the label for a node ID.
    fn get_label(&self, node_id: IdType) -> Option<LabelType>;

    /// Check if a node is deleted.
    fn is_deleted(&self, node_id: IdType) -> bool;
}

/// Search mode tracker for hybrid queries.
///
/// This tracks the search strategy used across queries.
#[derive(Debug, Default)]
pub struct SearchModeTracker {
    last_mode: std::sync::atomic::AtomicU8,
    adhoc_count: AtomicUsize,
    batch_count: AtomicUsize,
    switch_count: AtomicUsize,
}

impl SearchModeTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a search mode.
    pub fn record(&self, mode: SearchMode) {
        let mode_u8 = match mode {
            SearchMode::Empty => 0,
            SearchMode::StandardKnn => 1,
            SearchMode::HybridAdhocBf => 2,
            SearchMode::HybridBatches => 3,
            SearchMode::HybridBatchesToAdhocBf => 4,
            SearchMode::RangeQuery => 5,
        };
        self.last_mode.store(mode_u8, Ordering::Relaxed);

        match mode {
            SearchMode::HybridAdhocBf => {
                self.adhoc_count.fetch_add(1, Ordering::Relaxed);
            }
            SearchMode::HybridBatches => {
                self.batch_count.fetch_add(1, Ordering::Relaxed);
            }
            SearchMode::HybridBatchesToAdhocBf => {
                self.switch_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Get the last search mode.
    pub fn last_mode(&self) -> SearchMode {
        match self.last_mode.load(Ordering::Relaxed) {
            0 => SearchMode::Empty,
            1 => SearchMode::StandardKnn,
            2 => SearchMode::HybridAdhocBf,
            3 => SearchMode::HybridBatches,
            4 => SearchMode::HybridBatchesToAdhocBf,
            5 => SearchMode::RangeQuery,
            _ => SearchMode::Empty,
        }
    }

    /// Get count of ad-hoc searches.
    pub fn adhoc_count(&self) -> usize {
        self.adhoc_count.load(Ordering::Relaxed)
    }

    /// Get count of batch searches.
    pub fn batch_count(&self) -> usize {
        self.batch_count.load(Ordering::Relaxed)
    }

    /// Get count of switches from batch to ad-hoc.
    pub fn switch_count(&self) -> usize {
        self.switch_count.load(Ordering::Relaxed)
    }

    /// Get total hybrid search count.
    pub fn total_hybrid_count(&self) -> usize {
        self.adhoc_count() + self.batch_count() + self.switch_count()
    }
}

/// Iterator over debug information entries.
pub struct DebugInfoIterator<'a> {
    entries: std::vec::IntoIter<(&'a str, String)>,
}

impl<'a> DebugInfoIterator<'a> {
    /// Create from debug info.
    pub fn from_debug_info(info: &'a DebugInfo) -> Self {
        let mut entries = Vec::new();

        // Basic info
        entries.push(("index_type", info.basic.index_type.to_string()));
        entries.push(("dim", info.basic.dim.to_string()));
        entries.push(("metric", format!("{:?}", info.basic.metric)));
        entries.push(("is_multi", info.basic.is_multi.to_string()));
        entries.push(("block_size", info.basic.block_size.to_string()));

        // Stats
        entries.push(("index_size", info.stats.index_size.to_string()));
        entries.push(("label_count", info.stats.label_count.to_string()));
        if let Some(mem) = info.stats.memory_usage {
            entries.push(("memory_usage", mem.to_string()));
        }
        entries.push(("deleted_count", info.stats.deleted_count.to_string()));

        // HNSW-specific
        if let Some(ref hnsw) = info.hnsw {
            entries.push(("hnsw_m", hnsw.m.to_string()));
            entries.push(("hnsw_ef_construction", hnsw.ef_construction.to_string()));
            entries.push(("hnsw_ef_runtime", hnsw.ef_runtime.to_string()));
            entries.push(("hnsw_max_level", hnsw.max_level.to_string()));
            if let Some(ep) = hnsw.entry_point {
                entries.push(("hnsw_entry_point", ep.to_string()));
            }
        }

        // SVS-specific
        if let Some(ref svs) = info.svs {
            entries.push(("svs_alpha", svs.alpha.to_string()));
            entries.push(("svs_max_neighbors", svs.max_neighbors.to_string()));
            entries.push(("svs_search_window_size", svs.search_window_size.to_string()));
            entries.push(("svs_avg_degree", svs.avg_degree.to_string()));
            if let Some(medoid) = svs.medoid {
                entries.push(("svs_medoid", medoid.to_string()));
            }
        }

        // Search mode
        entries.push(("last_search_mode", format!("{:?}", info.last_search_mode)));

        Self {
            entries: entries.into_iter(),
        }
    }
}

impl<'a> Iterator for DebugInfoIterator<'a> {
    type Item = (&'a str, String);

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next()
    }
}

/// Element neighbors structure for graph inspection.
#[derive(Debug, Clone)]
pub struct ElementNeighbors {
    /// Node ID.
    pub node_id: IdType,
    /// Label of the node.
    pub label: LabelType,
    /// Neighbors at each level (index 0 = level 0).
    pub neighbors_per_level: Vec<Vec<IdType>>,
}

/// Mapping from internal ID to label.
#[derive(Debug, Clone, Default)]
pub struct IdLabelMapping {
    id_to_label: HashMap<IdType, LabelType>,
    label_to_ids: HashMap<LabelType, Vec<IdType>>,
}

impl IdLabelMapping {
    /// Create a new empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a mapping.
    pub fn insert(&mut self, id: IdType, label: LabelType) {
        self.id_to_label.insert(id, label);
        self.label_to_ids.entry(label).or_default().push(id);
    }

    /// Get label for an ID.
    pub fn get_label(&self, id: IdType) -> Option<LabelType> {
        self.id_to_label.get(&id).copied()
    }

    /// Get IDs for a label.
    pub fn get_ids(&self, label: LabelType) -> Option<&[IdType]> {
        self.label_to_ids.get(&label).map(|v| v.as_slice())
    }

    /// Get total number of mappings.
    pub fn len(&self) -> usize {
        self.id_to_label.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_label.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_mode_tracker() {
        let tracker = SearchModeTracker::new();

        tracker.record(SearchMode::HybridAdhocBf);
        tracker.record(SearchMode::HybridBatches);
        tracker.record(SearchMode::HybridAdhocBf);

        assert_eq!(tracker.adhoc_count(), 2);
        assert_eq!(tracker.batch_count(), 1);
        assert_eq!(tracker.total_hybrid_count(), 3);
    }

    #[test]
    fn test_id_label_mapping() {
        let mut mapping = IdLabelMapping::new();
        mapping.insert(0, 100);
        mapping.insert(1, 100);
        mapping.insert(2, 200);

        assert_eq!(mapping.get_label(0), Some(100));
        assert_eq!(mapping.get_label(2), Some(200));
        assert_eq!(mapping.get_ids(100), Some(&[0, 1][..]));
        assert_eq!(mapping.len(), 3);
    }

    #[test]
    fn test_index_type_display() {
        assert_eq!(IndexType::HnswSingle.to_string(), "HnswSingle");
        assert_eq!(IndexType::BruteForceMulti.to_string(), "BruteForceMulti");
    }

    #[test]
    fn test_debug_info_iterator() {
        let info = DebugInfo {
            basic: BasicIndexInfo {
                index_type: IndexType::HnswSingle,
                dim: 128,
                metric: Metric::L2,
                is_multi: false,
                block_size: 1024,
            },
            stats: IndexStats {
                index_size: 1000,
                label_count: 1000,
                memory_usage: Some(1024 * 1024),
                deleted_count: 0,
                resize_count: 2,
            },
            hnsw: Some(HnswDebugInfo {
                m: 16,
                m_max: 16,
                m_max0: 32,
                ef_construction: 200,
                ef_runtime: 10,
                entry_point: Some(0),
                max_level: 3,
                level_distribution: vec![1000, 50, 5, 1],
            }),
            svs: None,
            last_search_mode: SearchMode::StandardKnn,
        };

        let iter = DebugInfoIterator::from_debug_info(&info);
        let entries: Vec<_> = iter.collect();

        assert!(entries.iter().any(|(k, _)| *k == "index_type"));
        assert!(entries.iter().any(|(k, _)| *k == "hnsw_m"));
        assert!(entries.iter().any(|(k, v)| *k == "dim" && v == "128"));
    }
}
