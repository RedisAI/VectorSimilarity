GIT_LABEL=$1
if [ -z "$GIT_LABEL"  ] || [ "$GIT_LABEL" = "benchmarks-all" ]; then 
    file_name="all"
elif [ "$GIT_LABEL" = "bm-spaces" ]; then
    :
elif [ "$GIT_LABEL" = "benchmarks-default" ] \
|| [ "$GIT_LABEL" = "bm-basics-fp32-single" ] \
|| [ "$GIT_LABEL" = "bm-basics-fp32-multi" ] \
|| [ "$GIT_LABEL" = "bm-batch-iter-fp32-single" ] \
|| [ "$GIT_LABEL" = "bm-batch-iter-fp32-multi" ] 
then
    file_name="basic_fp32"
elif [ "$GIT_LABEL" = "bm-basics-fp64-single" ] \
|| [ "$GIT_LABEL" = "bm-basics-fp64-multi" ] \
|| [ "$GIT_LABEL" = "bm-batch-iter-fp64-single" ] \
|| [ "$GIT_LABEL" = "bm-batch-iter-fp64-multi" ] 
then
    file_name="basic_fp64"
elif [ "$GIT_LABEL" = "bm-updated-fp32-single" ]; then
    file_name="updated"
fi

wget --no-check-certificate -q -i tests/benchmark/data/hnsw_indices/hnsw_indices_$file_name.txt -P tests/benchmark/data
