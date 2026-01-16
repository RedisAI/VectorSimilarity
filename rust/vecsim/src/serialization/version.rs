//! Versioning support for serialization.
//!
//! This module defines version constants and compatibility checking
//! for index serialization formats.

/// Current serialization version.
pub const CURRENT_VERSION: u32 = 1;

/// Minimum supported version for reading.
#[allow(dead_code)]
pub const MIN_SUPPORTED_VERSION: u32 = 1;

/// Serialization version information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerializationVersion {
    pub major: u16,
    pub minor: u16,
}

impl SerializationVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    pub const fn current() -> Self {
        Self::new(1, 0)
    }

    pub fn to_u32(self) -> u32 {
        ((self.major as u32) << 16) | (self.minor as u32)
    }

    pub fn from_u32(value: u32) -> Self {
        Self {
            major: (value >> 16) as u16,
            minor: (value & 0xFFFF) as u16,
        }
    }

    pub fn is_compatible(self, other: Self) -> bool {
        // Major version must match for compatibility
        self.major == other.major
    }
}

impl Default for SerializationVersion {
    fn default() -> Self {
        Self::current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_roundtrip() {
        let version = SerializationVersion::new(1, 5);
        let encoded = version.to_u32();
        let decoded = SerializationVersion::from_u32(encoded);
        assert_eq!(version, decoded);
    }

    #[test]
    fn test_version_compatibility() {
        let v1_0 = SerializationVersion::new(1, 0);
        let v1_5 = SerializationVersion::new(1, 5);
        let v2_0 = SerializationVersion::new(2, 0);

        assert!(v1_0.is_compatible(v1_5));
        assert!(!v1_0.is_compatible(v2_0));
    }
}
