//! Cosine distance implementation.
//!
//! Cosine similarity is defined as: dot(a, b) / (||a|| * ||b||)
//! Cosine distance is: 1 - cosine_similarity
//!
//! For efficiency, vectors can be pre-normalized during insertion,
//! reducing cosine distance computation to inner product distance.

use super::simd::{self, SimdCapability};
use super::{DistanceFunction, Metric};
use crate::types::{DistanceType, VectorElement};
use std::marker::PhantomData;

/// Cosine distance calculator.
///
/// Returns 1 - cosine_similarity, so values range from 0 (identical direction)
/// to 2 (opposite direction).
pub struct CosineDistance<T: VectorElement> {
    dim: usize,
    simd_capability: SimdCapability,
    _phantom: PhantomData<T>,
}

impl<T: VectorElement> CosineDistance<T> {
    /// Create a new cosine distance calculator with automatic SIMD detection.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            simd_capability: simd::detect_simd_capability(),
            _phantom: PhantomData,
        }
    }

    /// Create with SIMD enabled (if available).
    pub fn with_simd(dim: usize) -> Self {
        Self::new(dim)
    }

    /// Create with scalar-only implementation.
    pub fn scalar(dim: usize) -> Self {
        Self {
            dim,
            simd_capability: SimdCapability::None,
            _phantom: PhantomData,
        }
    }
}

impl<T: VectorElement> DistanceFunction<T> for CosineDistance<T> {
    type Output = T::DistanceType;

    #[inline]
    fn compute(&self, a: &[T], b: &[T], dim: usize) -> Self::Output {
        debug_assert_eq!(a.len(), dim);
        debug_assert_eq!(b.len(), dim);
        debug_assert_eq!(dim, self.dim);

        // For pre-normalized vectors, this reduces to inner product
        // For raw vectors, we need to compute the full cosine distance
        match self.simd_capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx512 => {
                simd::avx512::cosine_distance_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx2 => {
                simd::avx2::cosine_distance_f32(a, b, dim)
            }
            #[cfg(target_arch = "aarch64")]
            SimdCapability::Neon => {
                simd::neon::cosine_distance_f32(a, b, dim)
            }
            #[allow(unreachable_patterns)]
            SimdCapability::None | _ => {
                cosine_distance_scalar(a, b, dim)
            }
        }
    }

    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    fn preprocess(&self, vector: &[T], dim: usize) -> Vec<T> {
        // Normalize the vector during preprocessing
        normalize_vector(vector, dim)
    }

    fn compute_from_preprocessed(&self, stored: &[T], query: &[T], dim: usize) -> Self::Output {
        // When stored vectors are pre-normalized, we need to normalize query too
        // and then it's just 1 - inner_product
        let query_normalized = normalize_vector(query, dim);
        let ip = inner_product_raw(stored, &query_normalized, dim);
        T::DistanceType::from_f64(1.0 - ip)
    }
}

/// Normalize a vector to unit length.
pub fn normalize_vector<T: VectorElement>(vector: &[T], dim: usize) -> Vec<T> {
    let norm = compute_norm(vector, dim);
    if norm < 1e-30 {
        // Avoid division by zero for zero vectors
        return vector.to_vec();
    }
    let inv_norm = 1.0 / norm;
    vector
        .iter()
        .map(|&x| T::from_f32(x.to_f32() * inv_norm as f32))
        .collect()
}

/// Compute the L2 norm of a vector.
#[inline]
pub fn compute_norm<T: VectorElement>(vector: &[T], dim: usize) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..dim {
        let v = vector[i].to_f32() as f64;
        sum += v * v;
    }
    sum.sqrt()
}

/// Raw inner product without conversion to distance.
#[inline]
fn inner_product_raw<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..dim {
        sum += a[i].to_f32() as f64 * b[i].to_f32() as f64;
    }
    sum
}

/// Scalar implementation of cosine distance.
#[inline]
pub fn cosine_distance_scalar<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..dim {
        let va = a[i].to_f32() as f64;
        let vb = b[i].to_f32() as f64;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        // Handle zero vectors
        return T::DistanceType::from_f64(1.0);
    }

    let cosine_sim = dot / denom;
    // Clamp to [-1, 1] to handle floating point errors
    let cosine_sim = cosine_sim.max(-1.0).min(1.0);
    T::DistanceType::from_f64(1.0 - cosine_sim)
}

/// Optimized scalar with loop unrolling for f32.
#[inline]
pub fn cosine_distance_scalar_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut norm_a0 = 0.0f32;
    let mut norm_a1 = 0.0f32;
    let mut norm_b0 = 0.0f32;
    let mut norm_b1 = 0.0f32;

    let unroll = dim / 2 * 2;
    let mut i = 0;

    while i < unroll {
        let va0 = a[i];
        let va1 = a[i + 1];
        let vb0 = b[i];
        let vb1 = b[i + 1];

        dot0 += va0 * vb0;
        dot1 += va1 * vb1;
        norm_a0 += va0 * va0;
        norm_a1 += va1 * va1;
        norm_b0 += vb0 * vb0;
        norm_b1 += vb1 * vb1;

        i += 2;
    }

    // Handle remaining elements
    while i < dim {
        let va = a[i];
        let vb = b[i];
        dot0 += va * vb;
        norm_a0 += va * va;
        norm_b0 += vb * vb;
        i += 1;
    }

    let dot = dot0 + dot1;
    let norm_a = norm_a0 + norm_a1;
    let norm_b = norm_b0 + norm_b1;

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (dot / denom).max(-1.0).min(1.0);
    1.0 - cosine_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = cosine_distance_scalar(&a, &a, 4);
        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let dist = cosine_distance_scalar(&a, &b, 3);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let dist = cosine_distance_scalar(&a, &b, 3);
        assert!((dist - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_vector() {
        let a = vec![3.0f32, 4.0, 0.0];
        let normalized = normalize_vector(&a, 3);
        // Norm should be 5, so normalized is [0.6, 0.8, 0.0]
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
        assert!(normalized[2].abs() < 0.001);

        // Verify unit length
        let norm = compute_norm(&normalized, 3);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_distance_function() {
        let dist_fn = CosineDistance::<f32>::scalar(3);

        // Same direction should have distance 0
        let a = vec![1.0f32, 1.0, 1.0];
        let b = vec![2.0f32, 2.0, 2.0];
        let dist = dist_fn.compute(&a, &b, 3);
        assert!(dist.abs() < 0.001);
    }
}
