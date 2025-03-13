BM_TYPE=$1;
# default label run basics fp32 single + multi, and spaces
if [ -z "$BM_TYPE"  ] || [ "$BM_TYPE" = "benchmarks-all" ]; then
    for bm_class in basics batch_iterator; do
        for type in single multi; do
            for data_type in fp32 fp64 bf16 fp16 int8; do
                echo ${bm_class}_${type}_${data_type};
            done
        done
    done
    echo updated_index_single_fp32
    echo spaces_fp32
    echo spaces_fp64
    echo spaces_bf16
    echo spaces_fp16
    echo spaces_int8
    echo spaces_uint8

elif [ "$BM_TYPE" = "benchmarks-default" ]; then
    echo basics_single_fp32
    echo basics_multi_fp32
    echo spaces_fp32
    echo spaces_fp64
    echo spaces_bf16
    echo spaces_fp16
    echo spaces_int8
    echo spaces_uint8

# Basic benchmarks
elif [ "$BM_TYPE" = "bm-basics-fp32-single" ] ; then
    echo basics_single_fp32
elif [ "$BM_TYPE" = "bm-basics-fp32-multi" ] ; then
    echo basics_multi_fp32
elif [ "$BM_TYPE" = "bm-basics-fp64-single" ] ; then
    echo basics_single_fp64
elif [ "$BM_TYPE" = "bm-basics-fp64-multi" ] ; then
    echo basics_multi_fp64
elif [ "$BM_TYPE" = "bm-basics-bf16-single" ] ; then
    echo basics_single_bf16
elif [ "$BM_TYPE" = "bm-basics-bf16-multi" ] ; then
    echo basics_multi_bf16
elif [ "$BM_TYPE" = "bm-basics-fp16-single" ] ; then
    echo basics_single_fp16
elif [ "$BM_TYPE" = "bm-basics-fp16-multi" ] ; then
    echo basics_multi_fp16
elif [ "$BM_TYPE" = "bm-basics-int8-single" ] ; then
    echo basics_single_int8
elif [ "$BM_TYPE" = "bm-basics-int8-multi" ] ; then
    echo basics_multi_int8

# Batch iterator benchmarks
elif [ "$BM_TYPE" = "bm-batch-iter-fp32-single" ] ; then
    echo batch_iterator_single_fp32
elif [ "$BM_TYPE" = "bm-batch-iter-fp32-multi" ] ; then
    echo batch_iterator_multi_fp32
elif [ "$BM_TYPE" = "bm-batch-iter-fp64-single" ] ; then
    echo batch_iterator_single_fp64
elif [ "$BM_TYPE" = "bm-batch-iter-fp64-multi" ] ; then
    echo batch_iterator_multi_fp64
elif [ "$BM_TYPE" = "bm-batch-iter-bf16-single" ] ; then
    echo batch_iterator_single_bf16
elif [ "$BM_TYPE" = "bm-batch-iter-bf16-multi" ] ; then
    echo batch_iterator_multi_bf16
elif [ "$BM_TYPE" = "bm-batch-iter-fp16-single" ] ; then
    echo batch_iterator_single_fp16
elif [ "$BM_TYPE" = "bm-batch-iter-fp16-multi" ] ; then
    echo batch_iterator_multi_fp16
elif [ "$BM_TYPE" = "bm-batch-iter-int8-single" ] ; then
    echo batch_iterator_single_int8
elif [ "$BM_TYPE" = "bm-batch-iter-int8-multi" ] ; then
    echo batch_iterator_multi_int8

# Updated index benchmarks
elif [ "$BM_TYPE" = "bm-updated-fp32-single" ] ; then
    echo updated_index_single_fp32

# Spaces benchmarks
elif [ "$BM_TYPE" = "bm-spaces" ] ; then
    echo spaces_fp32
    echo spaces_fp16
    echo spaces_fp64
    echo spaces_bf16
    echo spaces_int8
    echo spaces_uint8

elif [ "$BM_TYPE" = "bm-spaces-fp32" ] ; then
    echo spaces_fp32
elif [ "$BM_TYPE" = "bm-spaces-fp64" ] ; then
    echo spaces_fp64
elif [ "$BM_TYPE" = "bm-spaces-bf16" ] ; then
    echo spaces_bf16
elif [ "$BM_TYPE" = "bm-spaces-fp16" ] ; then
    echo spaces_fp16
elif [ "$BM_TYPE" = "bm-spaces-int8" ] ; then
    echo spaces_int8
elif [ "$BM_TYPE" = "bm-spaces-uint8" ] ; then
    echo spaces_uint8
fi
