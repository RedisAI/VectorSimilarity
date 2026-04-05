set -euo pipefail

BM_TYPE=${1:-}
alg="hnsw"

S3_BUCKET="dev.cto.redis"
S3_PREFIX="VectorSimilarity"
DEST_DIR="tests/benchmark/data"

# Download a file from S3 given its full HTTPS URL.
# Extracts the object key from the URL and uses aws s3 cp.
download_s3() {
    local url="$1"
    local filename
    filename=$(basename "$url")
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${filename}" "${DEST_DIR}/${filename}"
}
export -f download_s3
export S3_BUCKET S3_PREFIX DEST_DIR

if [ -z "$BM_TYPE"  ] || [ "$BM_TYPE" = "benchmarks-all" ]; then
    cat tests/benchmark/data/hnsw_indices/*.txt tests/benchmark/data/svs_indices/*.txt | grep -v '^$' | sort -u | xargs -n 1 -P 0 -I {} bash -c 'download_s3 "$@"' _ {}
    exit 0
elif [ "$BM_TYPE" = "benchmarks-default" ] \
|| [ "$BM_TYPE" = "bm-basics-fp32-single" ] \
|| [ "$BM_TYPE" = "bm-basics-fp32-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp32-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp32-multi" ]
then
    file_name="basic_fp32"
elif [ "$BM_TYPE" = "bm-basics-fp64-single" ] \
|| [ "$BM_TYPE" = "bm-basics-fp64-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp64-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp64-multi" ]
then
    file_name="basic_fp64"
elif [ "$BM_TYPE" = "bm-basics-bf16-single" ] \
|| [ "$BM_TYPE" = "bm-basics-bf16-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-bf16-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-bf16-multi" ]
then
    file_name="basic_bf16"
elif [ "$BM_TYPE" = "bm-basics-fp16-single" ] \
|| [ "$BM_TYPE" = "bm-basics-fp16-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp16-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-fp16-multi" ]
then
    file_name="basic_fp16"
elif [ "$BM_TYPE" = "bm-basics-int8-single" ] \
|| [ "$BM_TYPE" = "bm-basics-int8-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-int8-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-int8-multi" ]
then
    file_name="basic_int8"
elif [ "$BM_TYPE" = "bm-basics-uint8-single" ] \
|| [ "$BM_TYPE" = "bm-basics-uint8-multi" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-uint8-single" ] \
|| [ "$BM_TYPE" = "bm-batch-iter-uint8-multi" ] \
|| [ "$BM_TYPE" = "benchmarks-uint8" ]
then
    file_name="basic_uint8"
elif [ "$BM_TYPE" = "bm-updated-fp32-single" ]; then
    file_name="updated"
elif [ "$BM_TYPE" = "bm-svs-train-fp32" ] \
|| [ "$BM_TYPE" = "bm-svs-train-fp16" ]
then
    file_name="training"
    alg="svs"
elif [ "$BM_TYPE" = "bm-basics-svs-fp32-single" ]; then
    file_name="basic_fp32"
    alg="svs"
else
    echo "No files to download for BM_TYPE=$BM_TYPE"
    exit 0
fi

cat tests/benchmark/data/${alg}_indices/${alg}_indices_$file_name.txt | grep -v '^$' | xargs -n 1 -P 0 -I {} bash -c 'download_s3 "$@"' _ {}
