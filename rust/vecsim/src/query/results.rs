//! Query result types.

use crate::types::{DistanceType, LabelType};
use std::cmp::Ordering;

/// A single query result containing a label and its distance to the query.
#[derive(Debug, Clone, Copy)]
pub struct QueryResult<D: DistanceType> {
    /// The external label of the matching vector.
    pub label: LabelType,
    /// The distance from the query vector to this result.
    pub distance: D,
}

impl<D: DistanceType> QueryResult<D> {
    /// Create a new query result.
    #[inline]
    pub fn new(label: LabelType, distance: D) -> Self {
        Self { label, distance }
    }
}

impl<D: DistanceType> PartialEq for QueryResult<D> {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.distance.to_f64() == other.distance.to_f64()
    }
}

impl<D: DistanceType> Eq for QueryResult<D> {}

impl<D: DistanceType> PartialOrd for QueryResult<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: DistanceType> Ord for QueryResult<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by distance first, then by label for tie-breaking
        match self.distance.to_f64().partial_cmp(&other.distance.to_f64()) {
            Some(Ordering::Equal) | None => self.label.cmp(&other.label),
            Some(ord) => ord,
        }
    }
}

/// A collection of query results.
#[derive(Debug, Clone)]
pub struct QueryReply<D: DistanceType> {
    /// The results, sorted by distance (ascending for most metrics).
    pub results: Vec<QueryResult<D>>,
}

impl<D: DistanceType> QueryReply<D> {
    /// Create a new empty query reply.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Create a query reply with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            results: Vec::with_capacity(capacity),
        }
    }

    /// Create a query reply from a vector of results.
    pub fn from_results(results: Vec<QueryResult<D>>) -> Self {
        Self { results }
    }

    /// Add a result to the reply.
    #[inline]
    pub fn push(&mut self, result: QueryResult<D>) {
        self.results.push(result);
    }

    /// Get the number of results.
    #[inline]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if there are no results.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Sort results by distance (ascending).
    pub fn sort_by_distance(&mut self) {
        self.results.sort();
    }

    /// Sort results by distance (descending) - useful for inner product.
    pub fn sort_by_distance_desc(&mut self) {
        self.results.sort_by(|a, b| b.cmp(a));
    }

    /// Truncate to at most k results.
    pub fn truncate(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Iterate over results.
    pub fn iter(&self) -> impl Iterator<Item = &QueryResult<D>> {
        self.results.iter()
    }

    /// Get the best (closest) result, if any.
    pub fn best(&self) -> Option<&QueryResult<D>> {
        self.results.first()
    }

    /// Sort results by label (ascending).
    pub fn sort_by_label(&mut self) {
        self.results.sort_by_key(|r| r.label);
    }

    /// Sort results by distance, then by label for ties.
    ///
    /// This is the default sort behavior but provided explicitly.
    pub fn sort_by_distance_then_label(&mut self) {
        self.sort_by_distance();
    }

    /// Remove duplicate labels, keeping only the best (closest) result for each label.
    ///
    /// Results should be sorted by distance first for deterministic behavior.
    pub fn deduplicate_by_label(&mut self) {
        if self.results.is_empty() {
            return;
        }

        // Sort by distance first to ensure we keep the best per label
        self.sort_by_distance();

        let mut seen = std::collections::HashSet::new();
        self.results.retain(|r| seen.insert(r.label));
    }

    /// Filter results to only include those within the given distance threshold.
    pub fn filter_by_distance(&mut self, max_distance: D) {
        let threshold = max_distance.to_f64();
        self.results.retain(|r| r.distance.to_f64() <= threshold);
    }

    /// Get top-k results (sorts by distance and truncates).
    pub fn top_k(&mut self, k: usize) {
        self.sort_by_distance();
        self.truncate(k);
    }

    /// Skip the first n results (for pagination).
    pub fn skip(&mut self, n: usize) {
        if n >= self.results.len() {
            self.results.clear();
        } else {
            self.results = self.results.split_off(n);
        }
    }

    /// Convert distances to similarity scores.
    ///
    /// For metrics where lower distance = more similar (L2, Cosine),
    /// this returns `1 / (1 + distance)` giving values in (0, 1].
    ///
    /// Returns a vector of (label, similarity) pairs.
    pub fn to_similarities(&self) -> Vec<(LabelType, f64)> {
        self.results
            .iter()
            .map(|r| {
                let dist = r.distance.to_f64();
                let similarity = 1.0 / (1.0 + dist);
                (r.label, similarity)
            })
            .collect()
    }

    /// Convert a single distance to a similarity score.
    pub fn distance_to_similarity(distance: D) -> f64 {
        1.0 / (1.0 + distance.to_f64())
    }

    /// Get results within a percentage of the best distance.
    ///
    /// For example, `threshold_percent = 0.2` keeps results within 20% of the best.
    pub fn filter_by_relative_distance(&mut self, threshold_percent: f64) {
        if self.results.is_empty() {
            return;
        }

        self.sort_by_distance();
        let best_dist = self.results[0].distance.to_f64();
        let threshold = best_dist * (1.0 + threshold_percent);
        self.results.retain(|r| r.distance.to_f64() <= threshold);
    }
}

impl<D: DistanceType> Default for QueryReply<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: DistanceType> IntoIterator for QueryReply<D> {
    type Item = QueryResult<D>;
    type IntoIter = std::vec::IntoIter<QueryResult<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.results.into_iter()
    }
}

impl<'a, D: DistanceType> IntoIterator for &'a QueryReply<D> {
    type Item = &'a QueryResult<D>;
    type IntoIter = std::slice::Iter<'a, QueryResult<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.results.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_result_ordering() {
        let r1 = QueryResult::<f32>::new(1, 0.5);
        let r2 = QueryResult::<f32>::new(2, 1.0);
        let r3 = QueryResult::<f32>::new(3, 0.5);

        assert!(r1 < r2);
        assert!(r1 < r3); // Same distance, but label 1 < 3
    }

    #[test]
    fn test_query_reply_sort() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 1.0));
        reply.push(QueryResult::new(2, 0.5));
        reply.push(QueryResult::new(3, 0.75));

        reply.sort_by_distance();

        assert_eq!(reply.results[0].label, 2);
        assert_eq!(reply.results[1].label, 3);
        assert_eq!(reply.results[2].label, 1);
    }
}
