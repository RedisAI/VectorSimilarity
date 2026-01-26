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

    #[test]
    fn test_vamana_graph_data_new() {
        let node = VamanaGraphData::new(8);
        assert_eq!(node.get_label(), 0);
        assert!(!node.is_deleted());
        assert_eq!(node.neighbor_count(), 0);
        assert!(node.get_neighbors().is_empty());
    }

    #[test]
    fn test_vamana_graph_data_label() {
        let node = VamanaGraphData::new(4);

        node.set_label(12345);
        assert_eq!(node.get_label(), 12345);

        node.set_label(u64::MAX);
        assert_eq!(node.get_label(), u64::MAX);
    }

    #[test]
    fn test_vamana_graph_data_neighbors() {
        let node = VamanaGraphData::new(4);

        node.set_neighbors(&[1, 2, 3]);
        assert_eq!(node.neighbor_count(), 3);
        assert_eq!(node.get_neighbors(), vec![1, 2, 3]);

        // Clear and set new neighbors
        node.clear_neighbors();
        assert_eq!(node.neighbor_count(), 0);

        node.set_neighbors(&[10, 20]);
        assert_eq!(node.get_neighbors(), vec![10, 20]);
    }

    #[test]
    fn test_vamana_graph_data_debug() {
        let node = VamanaGraphData::new(4);
        node.set_label(42);
        node.set_neighbors(&[1, 2]);

        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("VamanaGraphData"));
        assert!(debug_str.contains("label"));
        assert!(debug_str.contains("deleted"));
        assert!(debug_str.contains("neighbor_count"));
    }

    #[test]
    fn test_vamana_graph_empty() {
        let graph = VamanaGraph::new(10, 4);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.average_degree(), 0.0);
        assert_eq!(graph.max_degree(), 0);
        assert_eq!(graph.min_degree(), 0);
    }

    #[test]
    fn test_vamana_graph_get_nonexistent() {
        let graph = VamanaGraph::new(10, 4);

        assert!(graph.get(0).is_none());
        assert!(graph.get(100).is_none());
        assert_eq!(graph.get_label(0), 0);
        assert!(graph.is_deleted(100)); // Non-existent treated as deleted
        assert!(graph.get_neighbors(50).is_empty());
    }

    #[test]
    fn test_vamana_graph_clear_neighbors_nonexistent() {
        let mut graph = VamanaGraph::new(10, 4);
        // Should not panic on non-existent node
        graph.clear_neighbors(0);
        graph.clear_neighbors(100);
    }

    #[test]
    fn test_vamana_graph_mark_deleted_nonexistent() {
        let mut graph = VamanaGraph::new(10, 4);
        // Should not panic on non-existent node
        graph.mark_deleted(0);
        graph.mark_deleted(100);
    }

    #[test]
    fn test_vamana_graph_ensure_capacity() {
        let mut graph = VamanaGraph::new(5, 4);

        // Initially capacity is 5
        graph.set_label(10, 100);

        // Should have auto-expanded
        assert_eq!(graph.get_label(10), 100);
    }

    #[test]
    fn test_vamana_graph_deleted_not_counted_in_stats() {
        let mut graph = VamanaGraph::new(10, 4);

        graph.set_neighbors(0, &[1, 2, 3, 4]);
        graph.set_neighbors(1, &[0]);
        graph.set_neighbors(2, &[0, 1]);

        // Before deletion
        let avg_before = graph.average_degree();

        // Delete node 0 with 4 neighbors
        graph.mark_deleted(0);

        // After deletion, stats should only count non-deleted nodes
        let avg_after = graph.average_degree();

        // With node 0 deleted, we have nodes 1 and 2 with degrees 1 and 2
        assert!((avg_after - 1.5).abs() < 0.01);
        assert!(avg_before > avg_after); // Should be lower after removing high-degree node
    }

    #[test]
    fn test_vamana_graph_debug() {
        let mut graph = VamanaGraph::new(10, 4);
        graph.set_neighbors(0, &[1, 2]);
        graph.set_neighbors(1, &[0]);

        let debug_str = format!("{:?}", graph);
        assert!(debug_str.contains("VamanaGraph"));
        assert!(debug_str.contains("node_count"));
        assert!(debug_str.contains("max_neighbors"));
        assert!(debug_str.contains("avg_degree"));
    }

    #[test]
    fn test_vamana_graph_concurrent_read() {
        use std::sync::Arc;
        use std::thread;

        let mut graph = VamanaGraph::new(100, 10);

        // Populate graph
        for i in 0..100 {
            graph.set_label(i, i as u64 * 10);
            let neighbors: Vec<u32> = (0..10).filter(|&j| j != i).take(5).collect();
            graph.set_neighbors(i, &neighbors);
        }

        let graph = Arc::new(graph);

        // Concurrent reads
        let mut handles = vec![];
        for _ in 0..4 {
            let graph_clone = Arc::clone(&graph);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = graph_clone.get_label(i);
                    let _ = graph_clone.get_neighbors(i);
                    let _ = graph_clone.is_deleted(i);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_vamana_graph_len_with_gaps() {
        let mut graph = VamanaGraph::new(10, 4);

        // Create nodes with gaps
        graph.set_label(0, 100);
        graph.set_label(5, 500);
        graph.set_label(9, 900);

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_vamana_graph_data_neighbors_replace() {
        let node = VamanaGraphData::new(4);

        node.set_neighbors(&[1, 2, 3, 4]);
        assert_eq!(node.get_neighbors(), vec![1, 2, 3, 4]);

        // Replace with different neighbors
        node.set_neighbors(&[10, 20]);
        assert_eq!(node.get_neighbors(), vec![10, 20]);
        assert_eq!(node.neighbor_count(), 2);
    }
}
