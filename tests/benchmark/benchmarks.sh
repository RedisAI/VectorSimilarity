BM_TYPE=$1;
# default label run basics fp32 single + multi, and spaces
if [ -z "$BM_TYPE"  ] || [ "$BM_TYPE" = "benchmarks-all" ]; then
    for bm_class in basics batch_iterator; do
        for type in single multi; do
            for data_type in fp32 fp64; do
                echo ${bm_class}_${type}_${data_type};
            done
        done
    done
    echo updated_index_single_fp32
    echo spaces_fp32
    echo spaces_fp64
elif [ "$BM_TYPE" = "benchmarks-default" ]; then
    echo basics_single_fp32
    echo basics_multi_fp32
    echo spaces_fp32
    echo spaces_fp64
# Basic benchmarks
elif [ "$BM_TYPE" = "bm-basics-fp32-single" ] ; then
    echo basics_single_fp32
elif [ "$BM_TYPE" = "bm-basics-fp32-multi" ] ; then
    echo basics_multi_fp32
elif [ "$BM_TYPE" = "bm-basics-fp64-single" ] ; then
    echo basics_single_fp64
elif [ "$BM_TYPE" = "bm-basics-fp64-multi" ] ; then
    echo basics_multi_fp64

# Batch iterator benchmarks
elif [ "$BM_TYPE" = "bm-batch-iter-fp32-single" ] ; then
    echo batch_iterator_single_fp32
elif [ "$BM_TYPE" = "bm-batch-iter-fp32-multi" ] ; then
    echo batch_iterator_multi_fp32
elif [ "$BM_TYPE" = "bm-batch-iter-fp64-single" ] ; then
    echo batch_iterator_single_fp64
elif [ "$BM_TYPE" = "bm-batch-iter-fp64-multi" ] ; then
    echo batch_iterator_multi_fp64

# Updated index benchmarks
elif [ "$BM_TYPE" = "bm-updated-fp32-single" ] ; then
    echo updated_index_single_fp32

# Spaces benchmarks
elif [ "$BM_TYPE" = "bm-spaces" ] ; then
    echo spaces_fp32
    echo spaces_fp64
fi
