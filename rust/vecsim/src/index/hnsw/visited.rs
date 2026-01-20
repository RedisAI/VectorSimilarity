//! Visited nodes tracking for HNSW graph traversal.
//!
//! This module provides efficient tracking of visited nodes during graph search:
//! - `VisitedNodesHandler`: Tag-based visited tracking (no clearing needed)
//! - `VisitedNodesHandlerPool`: Pool of handlers for concurrent searches

use crate::types::IdType;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// A handler for tracking visited nodes during graph traversal.
///
/// Uses a tag-based approach: instead of clearing the visited array between
/// searches, we increment a tag. A node is considered visited if its stored
/// tag matches the current tag.
pub struct VisitedNodesHandler {
    /// Tag for each node (node is visited if tag matches current_tag).
    tags: Vec<AtomicU32>,
    /// Current search tag.
    current_tag: u32,
    /// Capacity (maximum number of nodes).
    capacity: usize,
}

impl VisitedNodesHandler {
    /// Create a new handler with given capacity.
    pub fn new(capacity: usize) -> Self {
        let tags = (0..capacity).map(|_| AtomicU32::new(0)).collect();
        Self {
            tags,
            current_tag: 1,
            capacity,
        }
    }

    /// Reset for a new search. O(1) operation.
    #[inline]
    pub fn reset(&mut self) {
        self.current_tag = self.current_tag.wrapping_add(1);
        // Handle wrap-around by clearing all tags
        if self.current_tag == 0 {
            for tag in &self.tags {
                tag.store(0, Ordering::Relaxed);
            }
            self.current_tag = 1;
        }
    }

    /// Mark a node as visited. Returns true if it was already visited.
    #[inline]
    pub fn visit(&self, id: IdType) -> bool {
        let idx = id as usize;
        if idx >= self.capacity {
            return false;
        }

        let old = self.tags[idx].swap(self.current_tag, Ordering::AcqRel);
        old == self.current_tag
    }

    /// Check if a node has been visited without marking it.
    #[inline]
    pub fn is_visited(&self, id: IdType) -> bool {
        let idx = id as usize;
        if idx >= self.capacity {
            return false;
        }
        self.tags[idx].load(Ordering::Acquire) == self.current_tag
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize the handler to accommodate more nodes.
    pub fn resize(&mut self, new_capacity: usize) {
        if new_capacity > self.capacity {
            self.tags
                .resize_with(new_capacity, || AtomicU32::new(0));
            self.capacity = new_capacity;
        }
    }
}

/// Pool of visited nodes handlers for concurrent searches.
///
/// Allows multiple threads to perform searches simultaneously by
/// checking out handlers from the pool.
pub struct VisitedNodesHandlerPool {
    /// Available handlers.
    handlers: Mutex<Vec<VisitedNodesHandler>>,
    /// Default capacity for new handlers.
    default_capacity: std::sync::atomic::AtomicUsize,
}

impl VisitedNodesHandlerPool {
    /// Create a new pool.
    pub fn new(default_capacity: usize) -> Self {
        Self {
            handlers: Mutex::new(Vec::new()),
            default_capacity: std::sync::atomic::AtomicUsize::new(default_capacity),
        }
    }

    /// Get a handler from the pool, creating one if necessary.
    pub fn get(&self) -> PooledHandler<'_> {
        let cap = self.default_capacity.load(std::sync::atomic::Ordering::Acquire);
        let handler = self.handlers.lock().pop().unwrap_or_else(|| {
            VisitedNodesHandler::new(cap)
        });
        PooledHandler {
            handler: Some(handler),
            pool: self,
        }
    }

    /// Return a handler to the pool.
    fn return_handler(&self, mut handler: VisitedNodesHandler) {
        handler.reset();
        self.handlers.lock().push(handler);
    }

    /// Resize all handlers in the pool and update default capacity.
    pub fn resize(&self, new_capacity: usize) {
        self.default_capacity.store(new_capacity, std::sync::atomic::Ordering::Release);
        let mut handlers = self.handlers.lock();
        for handler in handlers.iter_mut() {
            handler.resize(new_capacity);
        }
    }

    /// Get the current default capacity.
    pub fn current_capacity(&self) -> usize {
        self.default_capacity.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A handler checked out from the pool.
///
/// Automatically returns the handler when dropped.
pub struct PooledHandler<'a> {
    handler: Option<VisitedNodesHandler>,
    pool: &'a VisitedNodesHandlerPool,
}

impl<'a> PooledHandler<'a> {
    /// Get a mutable reference to the handler.
    #[inline]
    pub fn get_mut(&mut self) -> &mut VisitedNodesHandler {
        self.handler.as_mut().expect("Handler already returned")
    }

    /// Get an immutable reference to the handler.
    #[inline]
    pub fn get(&self) -> &VisitedNodesHandler {
        self.handler.as_ref().expect("Handler already returned")
    }
}

impl<'a> Drop for PooledHandler<'a> {
    fn drop(&mut self) {
        if let Some(handler) = self.handler.take() {
            self.pool.return_handler(handler);
        }
    }
}

impl<'a> std::ops::Deref for PooledHandler<'a> {
    type Target = VisitedNodesHandler;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a> std::ops::DerefMut for PooledHandler<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_nodes_handler() {
        let mut handler = VisitedNodesHandler::new(100);

        // First visit should return false
        assert!(!handler.visit(5));
        assert!(handler.is_visited(5));

        // Second visit should return true
        assert!(handler.visit(5));

        // Reset
        handler.reset();
        assert!(!handler.is_visited(5));

        // Can visit again
        assert!(!handler.visit(5));
    }

    #[test]
    fn test_pool() {
        let pool = VisitedNodesHandlerPool::new(100);

        {
            let h1 = pool.get();
            h1.visit(10);
            assert!(h1.is_visited(10));
        }

        // Handler should be returned to pool and reset
        {
            let h2 = pool.get();
            // After reset, node should not be visited
            // (unless tag wrapped around, which is unlikely)
            assert!(!h2.is_visited(10));
        }
    }
}
