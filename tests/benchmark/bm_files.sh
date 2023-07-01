BM_TYPE=$1
if [ -z "$BM_TYPE"  ] || [ "$BM_TYPE" = "benchmarks-all" ]; then
    file_name="all"
elif [ "$BM_TYPE" = "bm-spaces" ]; then
    :
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
elif [ "$BM_TYPE" = "bm-updated-fp32-single" ]; then
    file_name="updated"
fi

wget --no-check-certificate -q -i tests/benchmark/data/hnsw_indices/hnsw_indices_$file_name.txt -P tests/benchmark/data
