//! Flat graph data structure for Vamana index.
//!
//! Unlike HNSW's multi-layer graph, Vamana uses a single flat layer.

use crate::types::{IdType, LabelType};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Data for a single node in the Vamana graph.
pub struct VamanaGraphData {
    /// External label.
    pub label: AtomicU64,
    /// Whether this node has been deleted.
    pub deleted: AtomicBool,
    /// Neighbor IDs (protected by RwLock for concurrent access).
    neighbors: RwLock<Vec<IdType>>,
    /// Maximum number of neighbors.
    max_neighbors: usize,
}

impl VamanaGraphData {
    /// Create a new graph node.
    pub fn new(max_neighbors: usize) -> Self {
        Self {
            label: AtomicU64::new(0),
            deleted: AtomicBool::new(false),
            neighbors: RwLock::new(Vec::with_capacity(max_neighbors)),
            max_neighbors,
        }
    }

    /// Get the label.
    #[inline]
    pub fn get_label(&self) -> LabelType {
        self.label.load(Ordering::Acquire)
    }

    /// Set the label.
    #[inline]
    pub fn set_label(&self, label: LabelType) {
        self.label.store(label, Ordering::Release);
    }

    /// Check if deleted.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::Acquire)
    }

    /// Mark as deleted.
    #[inline]
    pub fn mark_deleted(&self) {
        self.deleted.store(true, Ordering::Release);
    }

    /// Get all neighbors.
    pub fn get_neighbors(&self) -> Vec<IdType> {
        self.neighbors.read().clone()
    }

    /// Set neighbors (replaces existing).
    pub fn set_neighbors(&self, new_neighbors: &[IdType]) {
        let mut guard = self.neighbors.write();
        guard.clear();
        guard.extend(new_neighbors.iter().take(self.max_neighbors).copied());
    }

    /// Clear all neighbors.
    pub fn clear_neighbors(&self) {
        self.neighbors.write().clear();
    }

    /// Get the number of neighbors.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.read().len()
    }
}

impl std::fmt::Debug for VamanaGraphData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VamanaGraphData")
            .field("label", &self.get_label())
            .field("deleted", &self.is_deleted())
            .field("neighbor_count", &self.neighbor_count())
            .finish()
    }
}

/// Flat graph for Vamana index.
pub struct VamanaGraph {
    /// Per-node data.
    nodes: Vec<Option<VamanaGraphData>>,
    /// Maximum neighbors per node.
    max_neighbors: usize,
}

impl VamanaGraph {
    /// Create a new graph with given capacity.
    pub fn new(initial_capacity: usize, max_neighbors: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(initial_capacity),
            max_neighbors,
        }
    }

    /// Ensure capacity for at least n nodes.
    pub fn ensure_capacity(&mut self, n: usize) {
        if n > self.nodes.len() {
            self.nodes.resize_with(n, || None);
        }
    }

    /// Get or create node data for an ID.
    fn get_or_create(&mut self, id: IdType) -> &VamanaGraphData {
        let idx = id as usize;
        if idx >= self.nodes.len() {
            self.ensure_capacity(idx + 1);
        }
        if self.nodes[idx].is_none() {
            self.nodes[idx] = Some(VamanaGraphData::new(self.max_neighbors));
        }
        self.nodes[idx].as_ref().unwrap()
    }

    /// Get node data (immutable).
    pub fn get(&self, id: IdType) -> Option<&VamanaGraphData> {
        let idx = id as usize;
        if idx < self.nodes.len() {
            self.nodes[idx].as_ref()
        } else {
            None
        }
    }

    /// Get label for an ID.
    pub fn get_label(&self, id: IdType) -> LabelType {
        self.get(id).map(|n| n.get_label()).unwrap_or(0)
    }

    /// Set label for an ID.
    pub fn set_label(&mut self, id: IdType, label: LabelType) {
        self.get_or_create(id).set_label(label);
    }

    /// Check if a node is deleted.
    pub fn is_deleted(&self, id: IdType) -> bool {
        self.get(id).map(|n| n.is_deleted()).unwrap_or(true)
    }

    /// Mark a node as deleted.
    pub fn mark_deleted(&mut self, id: IdType) {
        if let Some(Some(node)) = self.nodes.get_mut(id as usize) {
            node.mark_deleted();
        }
    }

    /// Get neighbors of a node.
    pub fn get_neighbors(&self, id: IdType) -> Vec<IdType> {
        self.get(id)
            .map(|n| n.get_neighbors())
            .unwrap_or_default()
    }

    /// Set neighbors of a node.
    pub fn set_neighbors(&mut self, id: IdType, neighbors: &[IdType]) {
        self.get_or_create(id).set_neighbors(neighbors);
    }

    /// Clear neighbors of a node.
    pub fn clear_neighbors(&mut self, id: IdType) {
        if let Some(Some(node)) = self.nodes.get_mut(id as usize) {
            node.clear_neighbors();
        }
    }

    /// Get the total number of nodes (including deleted).
    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get average out-degree (for stats).
    pub fn average_degree(&self) -> f64 {
        let active_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .filter(|n| !n.is_deleted())
            .collect();

        if active_nodes.is_empty() {
            return 0.0;
        }

        let total_edges: usize = active_nodes.iter().map(|n| n.neighbor_count()).sum();
        total_edges as f64 / active_nodes.len() as f64
    }

    /// Get maximum out-degree.
    pub fn max_degree(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .filter(|n| !n.is_deleted())
            .map(|n| n.neighbor_count())
            .max()
            .unwrap_or(0)
    }

    /// Get minimum out-degree.
    pub fn min_degree(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .filter(|n| !n.is_deleted())
            .map(|n| n.neighbor_count())
            .min()
            .unwrap_or(0)
    }
}

impl std::fmt::Debug for VamanaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VamanaGraph")
            .field("node_count", &self.len())
            .field("max_neighbors", &self.max_neighbors)
            .field("avg_degree", &self.average_degree())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vamana_graph_basic() {
        let mut graph = VamanaGraph::new(10, 4);

        graph.set_label(0, 100);
        graph.set_label(1, 101);
        graph.set_label(2, 102);

        assert_eq!(graph.get_label(0), 100);
        assert_eq!(graph.get_label(1), 101);
        assert_eq!(graph.get_label(2), 102);
    }

    #[test]
    fn test_vamana_graph_neighbors() {
        let mut graph = VamanaGraph::new(10, 4);

        graph.set_neighbors(0, &[1, 2, 3]);
        graph.set_neighbors(1, &[0, 2]);

        assert_eq!(graph.get_neighbors(0), vec![1, 2, 3]);
        assert_eq!(graph.get_neighbors(1), vec![0, 2]);
        assert!(graph.get_neighbors(2).is_empty());
    }

    #[test]
    fn test_vamana_graph_max_neighbors() {
        let mut graph = VamanaGraph::new(10, 3);

        // Try to set more neighbors than max
        graph.set_neighbors(0, &[1, 2, 3, 4, 5]);

        // Should be truncated
        assert_eq!(graph.get_neighbors(0).len(), 3);
    }

    #[test]
    fn test_vamana_graph_delete() {
        let mut graph = VamanaGraph::new(10, 4);

        graph.set_label(0, 100);
        assert!(!graph.is_deleted(0));

        graph.mark_deleted(0);
        assert!(graph.is_deleted(0));
    }

    #[test]
    fn test_vamana_graph_stats() {
        let mut graph = VamanaGraph::new(10, 4);

        graph.set_neighbors(0, &[1, 2]);
        graph.set_neighbors(1, &[0, 2, 3]);
        graph.set_neighbors(2, &[0]);

        assert!((graph.average_degree() - 2.0).abs() < 0.01);
        assert_eq!(graph.max_degree(), 3);
        assert_eq!(graph.min_degree(), 1);
    }
}
