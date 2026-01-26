//! Memory-mapped block storage for disk-based vector indices.
//!
//! This module provides persistent storage of vectors using memory-mapped files,
//! allowing indices to work with datasets larger than available RAM.

use crate::types::{IdType, VectorElement};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// Header size in bytes (magic + version + dim + count + high_water_mark).
const HEADER_SIZE: usize = 32;

/// Magic number for file format identification.
const MAGIC: u64 = 0x56454353494D4D41; // "VECSIMMA"

/// File format version.
const FORMAT_VERSION: u32 = 1;

/// Memory-mapped block storage for vectors.
///
/// Provides persistent storage using memory-mapped files.
/// The file format is:
/// ```text
/// [Header: 32 bytes]
///   - magic: u64
///   - version: u32
///   - dim: u32
///   - count: u64
///   - high_water_mark: u64
/// [Vector data: dim * sizeof(T) * capacity]
/// [Deleted bitmap: ceil(capacity / 8) bytes]
/// ```
pub struct MmapDataBlocks<T: VectorElement> {
    /// Memory-mapped file.
    mmap: MmapMut,
    /// Path to the data file.
    path: PathBuf,
    /// Vector dimension.
    dim: usize,
    /// Number of active (non-deleted) vectors.
    count: usize,
    /// Highest ID ever allocated (not counting reuse).
    high_water_mark: usize,
    /// Current capacity.
    capacity: usize,
    /// Set of deleted slot IDs.
    free_slots: HashSet<IdType>,
    /// Phantom marker for element type.
    _marker: PhantomData<T>,
}

impl<T: VectorElement> MmapDataBlocks<T> {
    /// Create a new memory-mapped storage at the given path.
    ///
    /// If the file doesn't exist, it will be created.
    /// If it exists, it will be opened and validated.
    pub fn new<P: AsRef<Path>>(path: P, dim: usize, initial_capacity: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let capacity = initial_capacity.max(1);
        let file_size = Self::calculate_file_size(dim, capacity);

        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path, dim, capacity, file_size)
        }
    }

    /// Create a new storage file.
    fn create(path: PathBuf, dim: usize, capacity: usize, file_size: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(file_size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Write header
        Self::write_header(&mut mmap, dim, 0, 0);

        Ok(Self {
            mmap,
            path,
            dim,
            count: 0,
            high_water_mark: 0,
            capacity,
            free_slots: HashSet::new(),
            _marker: PhantomData,
        })
    }

    /// Open an existing storage file.
    fn open(path: PathBuf) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Validate and read header
        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for header",
            ));
        }

        let magic = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {version}"),
            ));
        }

        let dim = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let high_water_mark = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        // Calculate capacity from file size
        let vector_size = dim * std::mem::size_of::<T>();
        let data_area_size = mmap.len() - HEADER_SIZE;
        // Approximate capacity (ignoring deleted bitmap for now)
        let capacity = if vector_size > 0 {
            data_area_size / vector_size
        } else {
            0
        };

        // Load deleted slots from bitmap
        let free_slots = Self::load_deleted_bitmap(&mmap, dim, capacity, high_water_mark);

        Ok(Self {
            mmap,
            path,
            dim,
            count,
            high_water_mark,
            capacity,
            free_slots,
            _marker: PhantomData,
        })
    }

    /// Calculate the file size needed for the given capacity.
    fn calculate_file_size(dim: usize, capacity: usize) -> usize {
        let vector_size = dim * std::mem::size_of::<T>();
        let data_size = vector_size * capacity;
        let bitmap_size = capacity.div_ceil(8);
        HEADER_SIZE + data_size + bitmap_size
    }

    /// Write the header to the mmap.
    fn write_header(mmap: &mut MmapMut, dim: usize, count: usize, high_water_mark: usize) {
        mmap[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        mmap[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        mmap[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
        mmap[16..24].copy_from_slice(&(count as u64).to_le_bytes());
        mmap[24..32].copy_from_slice(&(high_water_mark as u64).to_le_bytes());
    }

    /// Load the deleted bitmap from the file.
    fn load_deleted_bitmap(
        mmap: &MmapMut,
        dim: usize,
        capacity: usize,
        high_water_mark: usize,
    ) -> HashSet<IdType> {
        let mut free_slots = HashSet::new();
        let vector_size = dim * std::mem::size_of::<T>();
        let bitmap_offset = HEADER_SIZE + vector_size * capacity;

        if bitmap_offset >= mmap.len() {
            return free_slots;
        }

        for id in 0..high_water_mark {
            let byte_idx = bitmap_offset + id / 8;
            let bit_idx = id % 8;

            if byte_idx < mmap.len() && (mmap[byte_idx] & (1 << bit_idx)) != 0 {
                free_slots.insert(id as IdType);
            }
        }

        free_slots
    }

    /// Save the deleted bitmap to the file.
    fn save_deleted_bitmap(&mut self) {
        let vector_size = self.dim * std::mem::size_of::<T>();
        let bitmap_offset = HEADER_SIZE + vector_size * self.capacity;

        // Clear bitmap
        let bitmap_size = self.capacity.div_ceil(8);
        if bitmap_offset + bitmap_size <= self.mmap.len() {
            for i in 0..bitmap_size {
                self.mmap[bitmap_offset + i] = 0;
            }

            // Set deleted bits
            for &id in &self.free_slots {
                let id = id as usize;
                let byte_idx = bitmap_offset + id / 8;
                let bit_idx = id % 8;

                if byte_idx < self.mmap.len() {
                    self.mmap[byte_idx] |= 1 << bit_idx;
                }
            }
        }
    }

    /// Get the vector dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the number of active vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the data path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the offset of a vector in the file.
    #[inline]
    fn vector_offset(&self, id: IdType) -> usize {
        let vector_size = self.dim * std::mem::size_of::<T>();
        HEADER_SIZE + (id as usize) * vector_size
    }

    /// Check if an ID is valid.
    #[inline]
    pub fn is_valid(&self, id: IdType) -> bool {
        let id_usize = id as usize;
        id_usize < self.high_water_mark && !self.free_slots.contains(&id)
    }

    /// Add a vector and return its ID.
    pub fn add(&mut self, vector: &[T]) -> Option<IdType> {
        if vector.len() != self.dim {
            return None;
        }

        // Check if we need to grow
        if self.high_water_mark >= self.capacity && !self.grow() {
            return None;
        }

        let id = self.high_water_mark as IdType;
        let offset = self.vector_offset(id);
        let vector_size = self.dim * std::mem::size_of::<T>();

        // Write vector data
        let src = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector_size)
        };
        self.mmap[offset..offset + vector_size].copy_from_slice(src);

        self.high_water_mark += 1;
        self.count += 1;

        // Update header
        Self::write_header(&mut self.mmap, self.dim, self.count, self.high_water_mark);

        Some(id)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: IdType) -> Option<&[T]> {
        if !self.is_valid(id) {
            return None;
        }

        let offset = self.vector_offset(id);
        let vector_size = self.dim * std::mem::size_of::<T>();

        if offset + vector_size > self.mmap.len() {
            return None;
        }

        let ptr = self.mmap[offset..].as_ptr() as *const T;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.dim) })
    }

    /// Mark a vector as deleted.
    pub fn mark_deleted(&mut self, id: IdType) -> bool {
        let id_usize = id as usize;
        if id_usize >= self.high_water_mark || self.free_slots.contains(&id) {
            return false;
        }

        self.free_slots.insert(id);
        self.count = self.count.saturating_sub(1);

        // Update header and bitmap
        Self::write_header(&mut self.mmap, self.dim, self.count, self.high_water_mark);
        self.save_deleted_bitmap();

        true
    }

    /// Grow the storage capacity.
    fn grow(&mut self) -> bool {
        let new_capacity = self.capacity * 2;
        let new_size = Self::calculate_file_size(self.dim, new_capacity);

        // Reopen file and resize
        let file = match OpenOptions::new().read(true).write(true).open(&self.path) {
            Ok(f) => f,
            Err(_) => return false,
        };

        if file.set_len(new_size as u64).is_err() {
            return false;
        }

        // Remap
        match unsafe { MmapOptions::new().map_mut(&file) } {
            Ok(new_mmap) => {
                self.mmap = new_mmap;
                self.capacity = new_capacity;
                true
            }
            Err(_) => false,
        }
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.count = 0;
        self.high_water_mark = 0;
        self.free_slots.clear();

        Self::write_header(&mut self.mmap, self.dim, 0, 0);
        self.save_deleted_bitmap();
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }

    /// Get fragmentation ratio.
    pub fn fragmentation(&self) -> f64 {
        if self.high_water_mark == 0 {
            0.0
        } else {
            self.free_slots.len() as f64 / self.high_water_mark as f64
        }
    }

    /// Iterate over valid IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = IdType> + '_ {
        (0..self.high_water_mark as IdType).filter(move |&id| !self.free_slots.contains(&id))
    }
}

impl<T: VectorElement> Drop for MmapDataBlocks<T> {
    fn drop(&mut self) {
        // Ensure data is flushed on drop
        let _ = self.mmap.flush();
    }
}

// Safety: MmapDataBlocks can be sent between threads
unsafe impl<T: VectorElement> Send for MmapDataBlocks<T> {}

// Note: Sync is NOT implemented because mmap requires careful synchronization
// for concurrent access. Use external locking.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("mmap_test_{}.dat", rand::random::<u64>()));
        path
    }

    #[test]
    fn test_mmap_blocks_basic() {
        let path = temp_path();
        {
            let mut blocks = MmapDataBlocks::<f32>::new(&path, 4, 10).unwrap();

            let v1 = vec![1.0f32, 2.0, 3.0, 4.0];
            let v2 = vec![5.0f32, 6.0, 7.0, 8.0];

            let id1 = blocks.add(&v1).unwrap();
            let id2 = blocks.add(&v2).unwrap();

            assert_eq!(id1, 0);
            assert_eq!(id2, 1);
            assert_eq!(blocks.len(), 2);

            let retrieved = blocks.get(id1).unwrap();
            assert_eq!(retrieved, &v1[..]);
        }

        // Cleanup
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_blocks_persistence() {
        let path = temp_path();
        let v1 = vec![1.0f32, 2.0, 3.0, 4.0];

        // Create and add
        {
            let mut blocks = MmapDataBlocks::<f32>::new(&path, 4, 10).unwrap();
            blocks.add(&v1).unwrap();
            blocks.flush().unwrap();
        }

        // Reopen and verify
        {
            let blocks = MmapDataBlocks::<f32>::new(&path, 4, 10).unwrap();
            assert_eq!(blocks.len(), 1);
            let retrieved = blocks.get(0).unwrap();
            assert_eq!(retrieved, &v1[..]);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_blocks_delete() {
        let path = temp_path();
        {
            let mut blocks = MmapDataBlocks::<f32>::new(&path, 4, 10).unwrap();

            let id1 = blocks.add(&[1.0, 2.0, 3.0, 4.0]).unwrap();
            let id2 = blocks.add(&[5.0, 6.0, 7.0, 8.0]).unwrap();

            assert_eq!(blocks.len(), 2);

            blocks.mark_deleted(id1);
            assert_eq!(blocks.len(), 1);
            assert!(blocks.get(id1).is_none());
            assert!(blocks.get(id2).is_some());
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mmap_blocks_grow() {
        let path = temp_path();
        {
            let mut blocks = MmapDataBlocks::<f32>::new(&path, 4, 2).unwrap();

            // Add more than initial capacity
            for i in 0..10 {
                let v = vec![i as f32; 4];
                blocks.add(&v).unwrap();
            }

            assert_eq!(blocks.len(), 10);
            assert!(blocks.capacity() >= 10);
        }

        fs::remove_file(&path).ok();
    }
}
