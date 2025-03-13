BM_TYPE=$1
if [ -z "$BM_TYPE"  ] || [ "$BM_TYPE" = "benchmarks-all" ]; then
    file_name="all"
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
elif [ "$BM_TYPE" = "bm-updated-fp32-single" ]; then
    file_name="updated"
else
    echo "No files to download for BM_TYPE=$BM_TYPE"
    exit 0
fi

cat tests/benchmark/data/hnsw_indices/hnsw_indices_$file_name.txt | xargs -n 1 -P 0 wget --no-check-certificate -P tests/benchmark/data
