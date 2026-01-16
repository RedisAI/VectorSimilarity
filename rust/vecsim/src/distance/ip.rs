//! Inner product (dot product) distance implementation.
//!
//! Inner product is defined as: sum(a[i] * b[i])
//! For use as a distance metric (lower is better), we return the negative
//! inner product: -sum(a[i] * b[i])
//!
//! This metric is particularly useful for normalized vectors where it
//! corresponds to cosine similarity.

use super::simd::{self, SimdCapability};
use super::{DistanceFunction, Metric};
use crate::types::{DistanceType, VectorElement};
use std::marker::PhantomData;

/// Inner product distance calculator.
///
/// Returns the negative inner product so that lower values indicate
/// more similar vectors (consistent with other distance metrics).
pub struct InnerProductDistance<T: VectorElement> {
    dim: usize,
    simd_capability: SimdCapability,
    _phantom: PhantomData<T>,
}

impl<T: VectorElement> InnerProductDistance<T> {
    /// Create a new inner product distance calculator with automatic SIMD detection.
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

impl<T: VectorElement> DistanceFunction<T> for InnerProductDistance<T> {
    type Output = T::DistanceType;

    #[inline]
    fn compute(&self, a: &[T], b: &[T], dim: usize) -> Self::Output {
        debug_assert_eq!(a.len(), dim);
        debug_assert_eq!(b.len(), dim);
        debug_assert_eq!(dim, self.dim);

        // Compute inner product and negate for distance
        let ip = match self.simd_capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx512 => {
                simd::avx512::inner_product_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx2 => {
                simd::avx2::inner_product_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx => {
                simd::avx::inner_product_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Sse => {
                simd::sse::inner_product_f32(a, b, dim)
            }
            #[cfg(target_arch = "aarch64")]
            SimdCapability::Neon => {
                simd::neon::inner_product_f32(a, b, dim)
            }
            #[allow(unreachable_patterns)]
            _ => {
                inner_product_scalar(a, b, dim)
            }
        };

        // Return 1 - ip to convert similarity to distance
        // For normalized vectors: 1 - cos(θ) ranges from 0 (identical) to 2 (opposite)
        T::DistanceType::from_f64(1.0 - ip.to_f64())
    }

    fn metric(&self) -> Metric {
        Metric::InnerProduct
    }
}

/// Scalar implementation of inner product.
#[inline]
pub fn inner_product_scalar<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let mut sum = 0.0f64;
    for i in 0..dim {
        sum += a[i].to_f32() as f64 * b[i].to_f32() as f64;
    }
    T::DistanceType::from_f64(sum)
}

/// Optimized scalar with loop unrolling for f32.
#[inline]
pub fn inner_product_scalar_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let unroll = dim / 4 * 4;
    let mut i = 0;

    while i < unroll {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < dim {
        sum0 += a[i] * b[i];
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Optimized scalar with loop unrolling for f64.
#[inline]
pub fn inner_product_scalar_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let unroll = dim / 4 * 4;
    let mut i = 0;

    while i < unroll {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < dim {
        sum0 += a[i] * b[i];
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_product_scalar() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];

        let ip = inner_product_scalar(&a, &b, 4);
        // 1*1 + 2*1 + 3*1 + 4*1 = 10
        assert!((ip - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_inner_product_normalized() {
        // Two normalized vectors
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];

        let ip = inner_product_scalar(&a, &b, 3);
        // Should be 1.0 for identical normalized vectors
        assert!((ip - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];

        let ip = inner_product_scalar(&a, &b, 3);
        // Should be 0.0 for orthogonal vectors
        assert!(ip.abs() < 0.001);
    }

    #[test]
    fn test_inner_product_distance() {
        let dist_fn = InnerProductDistance::<f32>::scalar(3);

        // Identical normalized vectors should have distance close to 0
        let a = vec![1.0f32, 0.0, 0.0];
        let dist = dist_fn.compute(&a, &a, 3);
        assert!(dist.abs() < 0.001);

        // Orthogonal vectors should have distance 1
        let b = vec![0.0f32, 1.0, 0.0];
        let dist = dist_fn.compute(&a, &b, 3);
        assert!((dist - 1.0).abs() < 0.001);
    }
}
