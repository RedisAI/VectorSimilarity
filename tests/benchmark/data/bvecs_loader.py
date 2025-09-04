#!/usr/bin/env python3
"""
BVECS file loader and converter for ANN_SIFT1B dataset
Based on the format specification from http://corpus-texmex.irisa.fr/

BVECS format:
- Each vector takes 4 + d bytes where d is the dimensionality
- First 4 bytes: int32 dimension
- Next d bytes: unsigned char vector components
"""

import numpy as np
import struct
import argparse
import os
from pathlib import Path


def read_bvecs(filename, max_vectors=None):
    """
    Read BVECS file and return vectors as numpy array
    
    Args:
        filename: Path to .bvecs file
        max_vectors: Maximum number of vectors to read (None for all)
    
    Returns:
        numpy.ndarray: Array of shape (n_vectors, dimension)
    """
    vectors = []
    dimensions = []
    
    with open(filename, 'rb') as f:
        vector_count = 0
        
        while True:
            # Read dimension (4 bytes, little endian int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
                
            dim = struct.unpack('<i', dim_bytes)[0]
            dimensions.append(dim)
            
            # Read vector data (d bytes, unsigned chars)
            vector_bytes = f.read(dim)
            if len(vector_bytes) != dim:
                break
                
            # Convert to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.uint8).astype(np.float32)
            vectors.append(vector)
            
            vector_count += 1
            
            # Check if we've reached the limit
            if max_vectors and vector_count >= max_vectors:
                break
    
    if not vectors:
        raise ValueError(f"No vectors found in {filename}")
    
    # Convert to numpy array
    vectors_array = np.array(vectors, dtype=np.float32)
    
    print(f"Loaded {len(vectors)} vectors with dimension {vectors_array.shape[1]}")
    print(f"Vector data type: {vectors_array.dtype}")
    print(f"Memory usage: {vectors_array.nbytes / (1024**3):.2f} GB")
    
    return vectors_array


def save_as_fvecs(vectors, output_filename):
    """
    Save vectors in FVECS format (float32)
    
    Args:
        vectors: numpy array of shape (n_vectors, dimension)
        output_filename: Output file path
    """
    with open(output_filename, 'wb') as f:
        for vector in vectors:
            # Write dimension as int32
            f.write(struct.pack('<i', vector.shape[0]))
            # Write vector data as float32
            f.write(vector.astype(np.float32).tobytes())
    
    print(f"Saved {len(vectors)} vectors to {output_filename}")


def save_as_raw(vectors, output_filename):
    """
    Save vectors as raw binary file
    
    Args:
        vectors: numpy array of shape (n_vectors, dimension)
        output_filename: Output file path
    """
    vectors.astype(np.float32).tofile(output_filename)
    print(f"Saved {len(vectors)} vectors to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description='Load and convert BVECS files')
    parser.add_argument('input_file', help='Input .bvecs file')
    parser.add_argument('--max-vectors', type=int, help='Maximum number of vectors to read')
    parser.add_argument('--output-format', choices=['fvecs', 'raw'], default='fvecs',
                       help='Output format (default: fvecs)')
    parser.add_argument('--output-file', help='Output file path (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        return 1
    
    # Load vectors
    try:
        vectors = read_bvecs(args.input_file, args.max_vectors)
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return 1
    
    # Determine output filename
    if not args.output_file:
        input_path = Path(args.input_file)
        if args.output_format == 'fvecs':
            args.output_file = input_path.with_suffix('.fvecs')
        else:
            args.output_file = input_path.with_suffix('.raw')
    
    # Save in requested format
    try:
        if args.output_format == 'fvecs':
            save_as_fvecs(vectors, args.output_file)
        else:
            save_as_raw(vectors, args.output_file)
    except Exception as e:
        print(f"Error saving vectors: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
