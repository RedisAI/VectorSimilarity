#!/bin/bash
# Serialization Benchmark Script
# Runs HNSW disk serialization benchmarks for different dataset sizes and thread counts

set -e

# Configuration
SERIALIZER="./bin/Linux-x86_64-release/hnsw_disk_serializer/hnsw_disk_serializer"
DATA_DIR="tests/benchmark/data"
OUTPUT_DIR="tests/benchmark/data/serialization_benchmarks"
RESULTS_FILE="$OUTPUT_DIR/results.csv"

# Dataset parameters
DIM=96
METRIC="L2"
DATA_TYPE="FLOAT32"
M=32
EFC=200
BATCH_SIZE=1000

# Datasets and thread counts
DATASETS=("100K")
THREADS=(4 8)

# Get branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BRANCH_SAFE=$(echo "$BRANCH" | tr '/' '-')

mkdir -p "$OUTPUT_DIR"
# Build
make -j8
# Initialize results file if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "branch,dataset,threads,time_seconds,vectors_per_second" > "$RESULTS_FILE"
fi

echo "=========================================="
echo "Serialization Benchmark"
echo "Branch: $BRANCH"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    INPUT_FILE="$DATA_DIR/deep.base.${dataset}.fbin"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "ERROR: Input file not found: $INPUT_FILE"
        continue
    fi
    
    for threads in "${THREADS[@]}"; do
        OUTPUT_NAME="$OUTPUT_DIR/deep-${dataset}-L2-dim${DIM}-M${M}-efc${EFC}-${BRANCH_SAFE}-${threads}t"
        
        echo ""
        echo "----------------------------------------"
        echo "Dataset: $dataset, Threads: $threads"
        echo "Output: $OUTPUT_NAME"
        echo "----------------------------------------"
        
        # Remove existing output if present
        rm -rf "$OUTPUT_NAME"
        
        # Run benchmark and capture time
        START_TIME=$(date +%s.%N)
        
        "$SERIALIZER" \
            "$INPUT_FILE" \
            "$OUTPUT_NAME" \
            "$DIM" "$METRIC" "$DATA_TYPE" \
            "$M" "$EFC" "$threads" "$BATCH_SIZE"
        
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        
        # Calculate vectors per second
        if [ "$dataset" = "100K" ]; then
            NUM_VECTORS=100000
        elif [ "$dataset" = "1M" ]; then
            NUM_VECTORS=1000000
        fi
        
        VPS=$(echo "scale=2; $NUM_VECTORS / $ELAPSED" | bc)
        
        echo ""
        echo "Time: ${ELAPSED}s"
        echo "Vectors/sec: $VPS"
        
        # Append to results
        echo "$BRANCH,$dataset,$threads,$ELAPSED,$VPS" >> "$RESULTS_FILE"
    done
done

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="

# Display results
echo ""
echo "Results:"
cat "$RESULTS_FILE"

