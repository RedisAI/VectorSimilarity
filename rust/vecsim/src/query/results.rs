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

    #[test]
    fn test_query_result_new() {
        let result = QueryResult::<f32>::new(42, 1.5);
        assert_eq!(result.label, 42);
        assert!((result.distance - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_result_equality() {
        let r1 = QueryResult::<f32>::new(1, 0.5);
        let r2 = QueryResult::<f32>::new(1, 0.5);
        let r3 = QueryResult::<f32>::new(1, 0.6);
        let r4 = QueryResult::<f32>::new(2, 0.5);

        assert_eq!(r1, r2);
        assert_ne!(r1, r3); // Different distance
        assert_ne!(r1, r4); // Different label
    }

    #[test]
    fn test_query_result_clone() {
        let r1 = QueryResult::<f32>::new(42, 1.5);
        let r2 = r1;
        assert_eq!(r1.label, r2.label);
        assert!((r1.distance - r2.distance).abs() < f32::EPSILON);
    }

    #[test]
    fn test_query_result_ordering_tie_breaking() {
        let r1 = QueryResult::<f32>::new(1, 0.5);
        let r2 = QueryResult::<f32>::new(2, 0.5);
        let r3 = QueryResult::<f32>::new(3, 0.5);

        // Same distance, so compare by label
        assert!(r1 < r2);
        assert!(r2 < r3);
        assert!(r1 < r3);
    }

    #[test]
    fn test_query_reply_new() {
        let reply = QueryReply::<f32>::new();
        assert!(reply.is_empty());
        assert_eq!(reply.len(), 0);
    }

    #[test]
    fn test_query_reply_with_capacity() {
        let reply = QueryReply::<f32>::with_capacity(10);
        assert!(reply.is_empty());
        assert!(reply.results.capacity() >= 10);
    }

    #[test]
    fn test_query_reply_from_results() {
        let results = vec![
            QueryResult::new(1, 0.5),
            QueryResult::new(2, 1.0),
        ];
        let reply = QueryReply::from_results(results);
        assert_eq!(reply.len(), 2);
    }

    #[test]
    fn test_query_reply_push() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));
        assert_eq!(reply.len(), 2);
    }

    #[test]
    fn test_query_reply_sort_by_distance_desc() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));
        reply.push(QueryResult::new(3, 0.75));

        reply.sort_by_distance_desc();

        assert_eq!(reply.results[0].label, 2); // 1.0
        assert_eq!(reply.results[1].label, 3); // 0.75
        assert_eq!(reply.results[2].label, 1); // 0.5
    }

    #[test]
    fn test_query_reply_truncate() {
        let mut reply = QueryReply::<f32>::new();
        for i in 0..10 {
            reply.push(QueryResult::new(i, i as f32 * 0.1));
        }

        reply.truncate(5);
        assert_eq!(reply.len(), 5);

        // Truncate to larger than current size does nothing
        reply.truncate(100);
        assert_eq!(reply.len(), 5);
    }

    #[test]
    fn test_query_reply_best() {
        let mut reply = QueryReply::<f32>::new();
        assert!(reply.best().is_none());

        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 0.3));

        reply.sort_by_distance();

        let best = reply.best().unwrap();
        assert_eq!(best.label, 2);
    }

    #[test]
    fn test_query_reply_sort_by_label() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(3, 0.5));
        reply.push(QueryResult::new(1, 1.0));
        reply.push(QueryResult::new(2, 0.75));

        reply.sort_by_label();

        assert_eq!(reply.results[0].label, 1);
        assert_eq!(reply.results[1].label, 2);
        assert_eq!(reply.results[2].label, 3);
    }

    #[test]
    fn test_query_reply_deduplicate_by_label() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(1, 0.3)); // Same label, closer
        reply.push(QueryResult::new(2, 1.0));
        reply.push(QueryResult::new(2, 1.5)); // Same label, farther

        reply.deduplicate_by_label();

        assert_eq!(reply.len(), 2);
        // Should keep the closer one for each label
        let labels: Vec<_> = reply.results.iter().map(|r| r.label).collect();
        assert!(labels.contains(&1));
        assert!(labels.contains(&2));
    }

    #[test]
    fn test_query_reply_filter_by_distance() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));
        reply.push(QueryResult::new(3, 1.5));
        reply.push(QueryResult::new(4, 2.0));

        reply.filter_by_distance(1.0);

        assert_eq!(reply.len(), 2);
        assert!(reply.results.iter().all(|r| r.distance <= 1.0));
    }

    #[test]
    fn test_query_reply_top_k() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(3, 1.5));
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(4, 2.0));
        reply.push(QueryResult::new(2, 1.0));

        reply.top_k(2);

        assert_eq!(reply.len(), 2);
        assert_eq!(reply.results[0].label, 1); // 0.5
        assert_eq!(reply.results[1].label, 2); // 1.0
    }

    #[test]
    fn test_query_reply_skip() {
        let mut reply = QueryReply::<f32>::new();
        for i in 0..5 {
            reply.push(QueryResult::new(i, i as f32));
        }

        reply.skip(2);

        assert_eq!(reply.len(), 3);
        assert_eq!(reply.results[0].label, 2);
    }

    #[test]
    fn test_query_reply_skip_all() {
        let mut reply = QueryReply::<f32>::new();
        for i in 0..5 {
            reply.push(QueryResult::new(i, i as f32));
        }

        reply.skip(10);

        assert!(reply.is_empty());
    }

    #[test]
    fn test_query_reply_skip_exact() {
        let mut reply = QueryReply::<f32>::new();
        for i in 0..5 {
            reply.push(QueryResult::new(i, i as f32));
        }

        reply.skip(5);

        assert!(reply.is_empty());
    }

    #[test]
    fn test_query_reply_to_similarities() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.0)); // similarity = 1.0
        reply.push(QueryResult::new(2, 1.0)); // similarity = 0.5

        let sims = reply.to_similarities();

        assert_eq!(sims.len(), 2);
        assert!((sims[0].1 - 1.0).abs() < 0.001);
        assert!((sims[1].1 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_query_reply_distance_to_similarity() {
        assert!((QueryReply::distance_to_similarity(0.0f32) - 1.0).abs() < 0.001);
        assert!((QueryReply::distance_to_similarity(1.0f32) - 0.5).abs() < 0.001);
        assert!((QueryReply::distance_to_similarity(3.0f32) - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_query_reply_filter_by_relative_distance() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 1.0));
        reply.push(QueryResult::new(2, 1.1)); // Within 20%
        reply.push(QueryResult::new(3, 1.15)); // Within 20% (threshold is 1.2)
        reply.push(QueryResult::new(4, 2.0)); // Beyond 20%

        reply.filter_by_relative_distance(0.2);

        assert_eq!(reply.len(), 3);
        assert!(reply.results.iter().all(|r| r.distance <= 1.2 + 0.001));
    }

    #[test]
    fn test_query_reply_filter_by_relative_distance_empty() {
        let mut reply = QueryReply::<f32>::new();
        reply.filter_by_relative_distance(0.2);
        assert!(reply.is_empty());
    }

    #[test]
    fn test_query_reply_default() {
        let reply: QueryReply<f32> = QueryReply::default();
        assert!(reply.is_empty());
    }

    #[test]
    fn test_query_reply_into_iterator() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        let collected: Vec<_> = reply.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_query_reply_ref_iterator() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        let mut count = 0;
        for _ in &reply {
            count += 1;
        }
        assert_eq!(count, 2);
        // reply is still valid
        assert_eq!(reply.len(), 2);
    }

    #[test]
    fn test_query_reply_iter() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        let labels: Vec<_> = reply.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec![1, 2]);
    }

    #[test]
    fn test_query_reply_clone() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        let cloned = reply.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.results[0].label, 1);
        assert_eq!(cloned.results[1].label, 2);
    }

    #[test]
    fn test_query_reply_debug() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));

        let debug_str = format!("{:?}", reply);
        assert!(debug_str.contains("QueryReply"));
        assert!(debug_str.contains("results"));
    }

    #[test]
    fn test_query_result_debug() {
        let result = QueryResult::<f32>::new(42, 1.5);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("QueryResult"));
        assert!(debug_str.contains("label"));
        assert!(debug_str.contains("distance"));
    }

    #[test]
    fn test_query_reply_sort_by_distance_then_label() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(3, 0.5));
        reply.push(QueryResult::new(1, 0.5)); // Same distance
        reply.push(QueryResult::new(2, 1.0));

        reply.sort_by_distance_then_label();

        assert_eq!(reply.results[0].label, 1); // 0.5, label 1
        assert_eq!(reply.results[1].label, 3); // 0.5, label 3
        assert_eq!(reply.results[2].label, 2); // 1.0
    }

    #[test]
    fn test_query_reply_deduplicate_empty() {
        let mut reply = QueryReply::<f32>::new();
        reply.deduplicate_by_label();
        assert!(reply.is_empty());
    }

    #[test]
    fn test_query_reply_deduplicate_single() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.deduplicate_by_label();
        assert_eq!(reply.len(), 1);
    }

    #[test]
    fn test_query_result_with_f64() {
        let r1 = QueryResult::<f64>::new(1, 0.5);
        let r2 = QueryResult::<f64>::new(2, 1.0);

        assert!(r1 < r2);
        assert_eq!(r1.label, 1);
        assert!((r1.distance - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_query_reply_with_f64() {
        let mut reply = QueryReply::<f64>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        reply.sort_by_distance();

        assert_eq!(reply.results[0].label, 1);
        assert_eq!(reply.len(), 2);
    }

    #[test]
    fn test_query_reply_top_k_more_than_available() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.5));
        reply.push(QueryResult::new(2, 1.0));

        reply.top_k(10);

        // Should have all results, sorted
        assert_eq!(reply.len(), 2);
        assert_eq!(reply.results[0].label, 1);
    }

    #[test]
    fn test_query_reply_filter_by_distance_zero() {
        let mut reply = QueryReply::<f32>::new();
        reply.push(QueryResult::new(1, 0.0));
        reply.push(QueryResult::new(2, 0.5));

        reply.filter_by_distance(0.0);

        assert_eq!(reply.len(), 1);
        assert_eq!(reply.results[0].label, 1);
    }
}
