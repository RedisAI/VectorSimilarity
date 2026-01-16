//! Graph data structures for HNSW index.
//!
//! This module provides the core data structures for storing the HNSW graph:
//! - `ElementMetaData`: Metadata for each element (label, level)
//! - `LevelLinks`: Neighbor connections for a single level
//! - `ElementGraphData`: Complete graph data for an element across all levels

use crate::types::{IdType, LabelType, INVALID_ID};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of neighbors at level 0.
pub const DEFAULT_M: usize = 16;

/// Maximum number of neighbors at levels > 0.
pub const DEFAULT_M_MAX: usize = 16;

/// Maximum number of neighbors at level 0 (typically 2*M).
pub const DEFAULT_M_MAX_0: usize = 32;

/// Metadata for a single element in the index.
#[derive(Debug)]
pub struct ElementMetaData {
    /// External label.
    pub label: LabelType,
    /// Maximum level this element appears in.
    pub level: u8,
    /// Whether this element has been deleted (tombstone).
    pub deleted: bool,
}

impl ElementMetaData {
    pub fn new(label: LabelType, level: u8) -> Self {
        Self {
            label,
            level,
            deleted: false,
        }
    }
}

/// Neighbor connections for a single level.
///
/// Uses a fixed-size array for cache efficiency.
pub struct LevelLinks {
    /// Neighbor IDs. INVALID_ID marks empty slots.
    neighbors: Vec<AtomicU32>,
    /// Current number of neighbors.
    count: AtomicU32,
    /// Maximum capacity.
    capacity: usize,
}

impl LevelLinks {
    /// Create new level links with given capacity.
    pub fn new(capacity: usize) -> Self {
        let neighbors: Vec<_> = (0..capacity).map(|_| AtomicU32::new(INVALID_ID)).collect();
        Self {
            neighbors,
            count: AtomicU32::new(0),
            capacity,
        }
    }

    /// Get the number of neighbors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get all neighbor IDs.
    pub fn get_neighbors(&self) -> Vec<IdType> {
        let count = self.len();
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let id = self.neighbors[i].load(Ordering::Acquire);
            if id != INVALID_ID {
                result.push(id);
            }
        }
        result
    }

    /// Add a neighbor if there's space.
    /// Returns true if added, false if full.
    pub fn try_add(&self, neighbor: IdType) -> bool {
        let mut current = self.count.load(Ordering::Acquire);
        loop {
            if current as usize >= self.capacity {
                return false;
            }
            match self.count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.neighbors[current as usize].store(neighbor, Ordering::Release);
                    return true;
                }
                Err(c) => current = c,
            }
        }
    }

    /// Set neighbors from a slice. Clears existing neighbors.
    pub fn set_neighbors(&self, neighbors: &[IdType]) {
        // Clear existing
        for i in 0..self.capacity {
            self.neighbors[i].store(INVALID_ID, Ordering::Release);
        }

        let count = neighbors.len().min(self.capacity);
        for (i, &n) in neighbors.iter().take(count).enumerate() {
            self.neighbors[i].store(n, Ordering::Release);
        }
        self.count.store(count as u32, Ordering::Release);
    }

    /// Remove a neighbor by ID.
    pub fn remove(&self, neighbor: IdType) -> bool {
        let count = self.len();
        for i in 0..count {
            if self.neighbors[i].load(Ordering::Acquire) == neighbor {
                // Swap with last and decrement count
                let last_idx = count - 1;
                if i < last_idx {
                    let last = self.neighbors[last_idx].load(Ordering::Acquire);
                    self.neighbors[i].store(last, Ordering::Release);
                }
                self.neighbors[last_idx].store(INVALID_ID, Ordering::Release);
                self.count.fetch_sub(1, Ordering::AcqRel);
                return true;
            }
        }
        false
    }

    /// Check if a neighbor exists.
    pub fn contains(&self, neighbor: IdType) -> bool {
        let count = self.len();
        for i in 0..count {
            if self.neighbors[i].load(Ordering::Acquire) == neighbor {
                return true;
            }
        }
        false
    }
}

impl std::fmt::Debug for LevelLinks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevelLinks")
            .field("count", &self.len())
            .field("capacity", &self.capacity)
            .field("neighbors", &self.get_neighbors())
            .finish()
    }
}

/// Complete graph data for an element across all levels.
pub struct ElementGraphData {
    /// Metadata (label, level, deleted flag).
    pub meta: ElementMetaData,
    /// Neighbor links for each level (level 0 to max_level).
    pub levels: Vec<LevelLinks>,
    /// Lock for modifying this element's neighbors.
    pub lock: Mutex<()>,
}

impl ElementGraphData {
    /// Create new graph data for an element.
    pub fn new(label: LabelType, level: u8, m_max_0: usize, m_max: usize) -> Self {
        let mut levels = Vec::with_capacity(level as usize + 1);

        // Level 0 has m_max_0 capacity
        levels.push(LevelLinks::new(m_max_0));

        // Higher levels have m_max capacity
        for _ in 1..=level {
            levels.push(LevelLinks::new(m_max));
        }

        Self {
            meta: ElementMetaData::new(label, level),
            levels,
            lock: Mutex::new(()),
        }
    }

    /// Get neighbors at a specific level.
    pub fn get_neighbors(&self, level: usize) -> Vec<IdType> {
        if level < self.levels.len() {
            self.levels[level].get_neighbors()
        } else {
            Vec::new()
        }
    }

    /// Set neighbors at a specific level.
    pub fn set_neighbors(&self, level: usize, neighbors: &[IdType]) {
        if level < self.levels.len() {
            self.levels[level].set_neighbors(neighbors);
        }
    }

    /// Get the maximum level for this element.
    #[inline]
    pub fn max_level(&self) -> u8 {
        self.meta.level
    }
}

impl std::fmt::Debug for ElementGraphData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementGraphData")
            .field("meta", &self.meta)
            .field("levels", &self.levels)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_links() {
        let links = LevelLinks::new(4);
        assert!(links.is_empty());

        assert!(links.try_add(1));
        assert!(links.try_add(2));
        assert!(links.try_add(3));
        assert!(links.try_add(4));
        assert!(!links.try_add(5)); // Full

        assert_eq!(links.len(), 4);
        assert!(links.contains(2));

        assert!(links.remove(2));
        assert_eq!(links.len(), 3);
        assert!(!links.contains(2));
    }

    #[test]
    fn test_element_graph_data() {
        let data = ElementGraphData::new(42, 2, 32, 16);

        assert_eq!(data.meta.label, 42);
        assert_eq!(data.max_level(), 2);
        assert_eq!(data.levels.len(), 3);
        assert_eq!(data.levels[0].capacity(), 32);
        assert_eq!(data.levels[1].capacity(), 16);
        assert_eq!(data.levels[2].capacity(), 16);
    }
}
