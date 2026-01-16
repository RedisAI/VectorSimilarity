//! L2 (Euclidean) squared distance implementation.
//!
//! L2 distance is defined as: sqrt(sum((a[i] - b[i])^2))
//! For efficiency, we compute the squared L2 distance (without sqrt)
//! since the ordering is preserved and sqrt is expensive.

use super::simd::{self, SimdCapability};
use super::{DistanceFunction, Metric};
use crate::types::{DistanceType, VectorElement};
use std::marker::PhantomData;

/// L2 (Euclidean) squared distance calculator.
pub struct L2Distance<T: VectorElement> {
    dim: usize,
    simd_capability: SimdCapability,
    _phantom: PhantomData<T>,
}

impl<T: VectorElement> L2Distance<T> {
    /// Create a new L2 distance calculator with automatic SIMD detection.
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

impl<T: VectorElement> DistanceFunction<T> for L2Distance<T> {
    type Output = T::DistanceType;

    #[inline]
    fn compute(&self, a: &[T], b: &[T], dim: usize) -> Self::Output {
        debug_assert_eq!(a.len(), dim);
        debug_assert_eq!(b.len(), dim);
        debug_assert_eq!(dim, self.dim);

        // Dispatch to appropriate implementation
        match self.simd_capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx512 => {
                simd::avx512::l2_squared_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Avx2 => {
                simd::avx2::l2_squared_f32(a, b, dim)
            }
            #[cfg(target_arch = "x86_64")]
            SimdCapability::Sse => {
                simd::sse::l2_squared_f32(a, b, dim)
            }
            #[cfg(target_arch = "aarch64")]
            SimdCapability::Neon => {
                simd::neon::l2_squared_f32(a, b, dim)
            }
            #[allow(unreachable_patterns)]
            _ => {
                l2_squared_scalar(a, b, dim)
            }
        }
    }

    fn metric(&self) -> Metric {
        Metric::L2
    }
}

/// Scalar implementation of L2 squared distance.
#[inline]
pub fn l2_squared_scalar<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let mut sum = 0.0f64;
    for i in 0..dim {
        let diff = a[i].to_f32() as f64 - b[i].to_f32() as f64;
        sum += diff * diff;
    }
    T::DistanceType::from_f64(sum)
}

/// Optimized scalar with loop unrolling for f32.
#[inline]
pub fn l2_squared_scalar_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let unroll = dim / 4 * 4;
    let mut i = 0;

    while i < unroll {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];

        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;

        i += 4;
    }

    // Handle remaining elements
    while i < dim {
        let d = a[i] - b[i];
        sum0 += d * d;
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Optimized scalar with loop unrolling for f64.
#[inline]
pub fn l2_squared_scalar_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let unroll = dim / 4 * 4;
    let mut i = 0;

    while i < unroll {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];

        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;

        i += 4;
    }

    // Handle remaining elements
    while i < dim {
        let d = a[i] - b[i];
        sum0 += d * d;
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_scalar() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let dist = l2_squared_scalar(&a, &b, 4);
        // (5-1)^2 + (6-2)^2 + (7-3)^2 + (8-4)^2 = 16 + 16 + 16 + 16 = 64
        assert!((dist - 64.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_scalar_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];

        let dist = l2_squared_scalar_f32(&a, &b, 4);
        assert!((dist - 64.0).abs() < 0.001);
    }

    #[test]
    fn test_l2_identical_vectors() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let dist = l2_squared_scalar(&a, &a, 4);
        assert!(dist.abs() < 0.0001);
    }

    #[test]
    fn test_l2_distance_function() {
        let dist_fn = L2Distance::<f32>::scalar(4);
        let a = vec![0.0f32, 0.0, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0, 0.0];

        let dist = dist_fn.compute(&a, &b, 4);
        // 3^2 + 4^2 = 9 + 16 = 25
        assert!((dist - 25.0).abs() < 0.001);
    }
}
