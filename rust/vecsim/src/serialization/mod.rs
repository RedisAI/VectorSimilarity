//! Serialization and deserialization for vector indices.
//!
//! This module provides functionality to save and load indices to/from disk.
//! Versioned encoding is used to support backward compatibility.

mod version;

pub use version::{SerializationVersion, CURRENT_VERSION};

use crate::distance::Metric;
use std::io::{self, Read, Write};
use thiserror::Error;

/// Magic number for vecsim index files.
pub const MAGIC_NUMBER: u32 = 0x5645_4353; // "VECS" in hex

/// Errors that can occur during serialization.
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid magic number: expected {expected:#x}, got {got:#x}")]
    InvalidMagicNumber { expected: u32, got: u32 },

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Index type mismatch: expected {expected}, got {got}")]
    IndexTypeMismatch { expected: String, got: String },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Metric mismatch: expected {expected:?}, got {got:?}")]
    MetricMismatch { expected: Metric, got: Metric },

    #[error("Data corruption: {0}")]
    DataCorruption(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Result type for serialization operations.
pub type SerializationResult<T> = Result<T, SerializationError>;

/// Index type identifier for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexTypeId {
    BruteForceSingle = 1,
    BruteForceMulti = 2,
    HnswSingle = 3,
    HnswMulti = 4,
}

impl IndexTypeId {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(IndexTypeId::BruteForceSingle),
            2 => Some(IndexTypeId::BruteForceMulti),
            3 => Some(IndexTypeId::HnswSingle),
            4 => Some(IndexTypeId::HnswMulti),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            IndexTypeId::BruteForceSingle => "BruteForceSingle",
            IndexTypeId::BruteForceMulti => "BruteForceMulti",
            IndexTypeId::HnswSingle => "HnswSingle",
            IndexTypeId::HnswMulti => "HnswMulti",
        }
    }
}

/// Data type identifier for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataTypeId {
    F32 = 1,
    F64 = 2,
    Float16 = 3,
    BFloat16 = 4,
}

impl DataTypeId {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(DataTypeId::F32),
            2 => Some(DataTypeId::F64),
            3 => Some(DataTypeId::Float16),
            4 => Some(DataTypeId::BFloat16),
            _ => None,
        }
    }
}

/// Header for serialized index files.
#[derive(Debug, Clone)]
pub struct IndexHeader {
    pub magic: u32,
    pub version: u32,
    pub index_type: IndexTypeId,
    pub data_type: DataTypeId,
    pub metric: Metric,
    pub dimension: usize,
    pub count: usize,
}

impl IndexHeader {
    pub fn new(
        index_type: IndexTypeId,
        data_type: DataTypeId,
        metric: Metric,
        dimension: usize,
        count: usize,
    ) -> Self {
        Self {
            magic: MAGIC_NUMBER,
            version: CURRENT_VERSION,
            index_type,
            data_type,
            metric,
            dimension,
            count,
        }
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        write_u32(writer, self.magic)?;
        write_u32(writer, self.version)?;
        write_u8(writer, self.index_type as u8)?;
        write_u8(writer, self.data_type as u8)?;
        write_u8(writer, metric_to_u8(self.metric))?;
        write_usize(writer, self.dimension)?;
        write_usize(writer, self.count)?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let magic = read_u32(reader)?;
        if magic != MAGIC_NUMBER {
            return Err(SerializationError::InvalidMagicNumber {
                expected: MAGIC_NUMBER,
                got: magic,
            });
        }

        let version = read_u32(reader)?;
        if version > CURRENT_VERSION {
            return Err(SerializationError::UnsupportedVersion(version));
        }

        let index_type = IndexTypeId::from_u8(read_u8(reader)?)
            .ok_or_else(|| SerializationError::InvalidData("Invalid index type".to_string()))?;

        let data_type = DataTypeId::from_u8(read_u8(reader)?)
            .ok_or_else(|| SerializationError::InvalidData("Invalid data type".to_string()))?;

        let metric = metric_from_u8(read_u8(reader)?)?;
        let dimension = read_usize(reader)?;
        let count = read_usize(reader)?;

        Ok(Self {
            magic,
            version,
            index_type,
            data_type,
            metric,
            dimension,
            count,
        })
    }
}

// Helper functions for binary I/O

#[inline]
pub fn write_u8<W: Write>(writer: &mut W, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

#[inline]
pub fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
pub fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[inline]
pub fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[inline]
pub fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[inline]
pub fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[inline]
pub fn write_usize<W: Write>(writer: &mut W, value: usize) -> io::Result<()> {
    write_u64(writer, value as u64)
}

#[inline]
pub fn read_usize<R: Read>(reader: &mut R) -> io::Result<usize> {
    Ok(read_u64(reader)? as usize)
}

#[inline]
pub fn write_f32<W: Write>(writer: &mut W, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[inline]
pub fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[inline]
pub fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[inline]
pub fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Write a vector of f32 values.
pub fn write_f32_slice<W: Write>(writer: &mut W, data: &[f32]) -> io::Result<()> {
    write_usize(writer, data.len())?;
    for &value in data {
        write_f32(writer, value)?;
    }
    Ok(())
}

/// Read a vector of f32 values.
pub fn read_f32_vec<R: Read>(reader: &mut R) -> io::Result<Vec<f32>> {
    let len = read_usize(reader)?;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f32(reader)?);
    }
    Ok(data)
}

/// Write a vector of f64 values.
pub fn write_f64_slice<W: Write>(writer: &mut W, data: &[f64]) -> io::Result<()> {
    write_usize(writer, data.len())?;
    for &value in data {
        write_f64(writer, value)?;
    }
    Ok(())
}

/// Read a vector of f64 values.
pub fn read_f64_vec<R: Read>(reader: &mut R) -> io::Result<Vec<f64>> {
    let len = read_usize(reader)?;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f64(reader)?);
    }
    Ok(data)
}

fn metric_to_u8(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 1,
        Metric::InnerProduct => 2,
        Metric::Cosine => 3,
    }
}

fn metric_from_u8(value: u8) -> SerializationResult<Metric> {
    match value {
        1 => Ok(Metric::L2),
        2 => Ok(Metric::InnerProduct),
        3 => Ok(Metric::Cosine),
        _ => Err(SerializationError::InvalidData(format!(
            "Invalid metric value: {value}"
        ))),
    }
}

/// Trait for serializable indices.
pub trait Serializable {
    /// Save the index to a writer.
    fn save<W: Write>(&self, writer: &mut W) -> SerializationResult<()>;

    /// Get the size in bytes when serialized.
    fn serialized_size(&self) -> usize;
}

/// Trait for deserializable indices.
pub trait Deserializable: Sized {
    /// Load the index from a reader.
    fn load<R: Read>(reader: &mut R) -> SerializationResult<Self>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_roundtrip() {
        let header = IndexHeader::new(
            IndexTypeId::HnswSingle,
            DataTypeId::F32,
            Metric::L2,
            128,
            1000,
        );

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded = IndexHeader::read(&mut cursor).unwrap();

        assert_eq!(loaded.magic, MAGIC_NUMBER);
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert_eq!(loaded.index_type, IndexTypeId::HnswSingle);
        assert_eq!(loaded.data_type, DataTypeId::F32);
        assert_eq!(loaded.metric, Metric::L2);
        assert_eq!(loaded.dimension, 128);
        assert_eq!(loaded.count, 1000);
    }
}
