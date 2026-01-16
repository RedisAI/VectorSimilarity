//! Distance metric implementations for vector similarity.
//!
//! This module provides various distance/similarity metrics:
//! - L2 (Euclidean) distance
//! - Inner product (dot product) similarity
//! - Cosine similarity/distance
//!
//! Each metric has scalar and SIMD-optimized implementations.

pub mod cosine;
pub mod ip;
pub mod l2;
pub mod simd;

use crate::types::{DistanceType, VectorElement};

/// Distance/similarity metric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// L2 (Euclidean) squared distance.
    /// Lower values indicate more similar vectors.
    L2,
    /// Inner product (dot product).
    /// For normalized vectors, higher values indicate more similar vectors.
    /// Note: This returns the negative inner product for use as a distance.
    InnerProduct,
    /// Cosine similarity converted to distance.
    /// Returns 1 - cosine_similarity, so lower values indicate more similar vectors.
    Cosine,
}

impl Metric {
    /// Check if this metric uses lower-is-better ordering.
    ///
    /// L2 and Cosine use lower-is-better (distance).
    /// Inner product uses higher-is-better (similarity), but we negate it internally.
    pub fn lower_is_better(&self) -> bool {
        true // All metrics are converted to distances internally
    }

    /// Get a human-readable name for the metric.
    pub fn name(&self) -> &'static str {
        match self {
            Metric::L2 => "L2",
            Metric::InnerProduct => "IP",
            Metric::Cosine => "Cosine",
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for distance computation functions.
pub trait DistanceFunction<T: VectorElement>: Send + Sync {
    /// The output distance type.
    type Output: DistanceType;

    /// Compute the distance between two vectors.
    ///
    /// # Safety
    /// Both vectors must have the same length as specified in `dim`.
    fn compute(&self, a: &[T], b: &[T], dim: usize) -> Self::Output;

    /// Get the metric type.
    fn metric(&self) -> Metric;

    /// Pre-process a vector before storage (e.g., normalize for cosine).
    /// Returns the processed vector.
    fn preprocess(&self, vector: &[T], _dim: usize) -> Vec<T> {
        // Default implementation: no preprocessing
        vector.to_vec()
    }

    /// Compute distance from a pre-processed vector to a raw query.
    /// Default uses regular compute.
    fn compute_from_preprocessed(&self, stored: &[T], query: &[T], dim: usize) -> Self::Output {
        self.compute(stored, query, dim)
    }
}

/// Create a distance function for the given metric and element type.
pub fn create_distance_function<T: VectorElement>(
    metric: Metric,
    dim: usize,
) -> Box<dyn DistanceFunction<T, Output = T::DistanceType>>
where
    T::DistanceType: DistanceType,
{
    // Select the best implementation based on available SIMD features
    let use_simd = simd::is_simd_available();

    match metric {
        Metric::L2 => {
            if use_simd {
                Box::new(l2::L2Distance::<T>::with_simd(dim))
            } else {
                Box::new(l2::L2Distance::<T>::scalar(dim))
            }
        }
        Metric::InnerProduct => {
            if use_simd {
                Box::new(ip::InnerProductDistance::<T>::with_simd(dim))
            } else {
                Box::new(ip::InnerProductDistance::<T>::scalar(dim))
            }
        }
        Metric::Cosine => {
            if use_simd {
                Box::new(cosine::CosineDistance::<T>::with_simd(dim))
            } else {
                Box::new(cosine::CosineDistance::<T>::scalar(dim))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_properties() {
        assert!(Metric::L2.lower_is_better());
        assert!(Metric::InnerProduct.lower_is_better());
        assert!(Metric::Cosine.lower_is_better());
    }

    #[test]
    fn test_create_distance_functions() {
        let l2: Box<dyn DistanceFunction<f32, Output = f32>> =
            create_distance_function(Metric::L2, 128);
        assert_eq!(l2.metric(), Metric::L2);

        let ip: Box<dyn DistanceFunction<f32, Output = f32>> =
            create_distance_function(Metric::InnerProduct, 128);
        assert_eq!(ip.metric(), Metric::InnerProduct);

        let cos: Box<dyn DistanceFunction<f32, Output = f32>> =
            create_distance_function(Metric::Cosine, 128);
        assert_eq!(cos.metric(), Metric::Cosine);
    }
}
