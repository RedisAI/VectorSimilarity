BM_FILTER=$1;
# default label run basics fp32 single + multi, and spaces
if [ -z "$BM_FILTER"  ] || [ "$BM_FILTER" = "ALL" ]; then 
    for bm_class in basics batch_iterator; do 
        for type in single multi; do 
            for data_type in fp32; do 
                echo ${bm_class}_${type}_${data_type}; 
            done 
        done 
    done
    echo updated_index_single_fp32
    echo spaces_fp32
    echo spaces_fp64
elif [ "$BM_FILTER" = "BASICS_FP32_S+M_SPACES" ]; then 
    echo basics_single_fp32
    echo basics_multi_fp32
    echo spaces_fp32
    echo spaces_fp64
# Basic benchmarks
elif [ "$BM_FILTER" = "BASICS_FP32_S" ] ; then 
    echo basics_single_fp32
elif [ "$BM_FILTER" = "BASICS_FP32_M" ] ; then 
    echo basics_multi_fp32
elif [ "$BM_FILTER" = "BASICS_FP64_S" ] ; then 
    echo basics_multi_fp64
elif [ "$BM_FILTER" = "BASICS_FP64_M" ] ; then 
    echo basics_multi_fp64

# Batch iterator benchmarks
elif [ "$BM_FILTER" = "BI_FP32_S" ] ; then 
    echo batch_iterator_single_fp32
elif [ "$BM_FILTER" = "BI_FP32_M" ] ; then 
    echo batch_iterator_multi_fp32
elif [ "$BM_FILTER" = "BI_FP64_S" ] ; then 
    echo batch_iterator_multi_fp64
elif [ "$BM_FILTER" = "BI_FP64_M" ] ; then 
    echo batch_iterator_multi_fp64

# Updated index benchmarks
elif [ "$BM_FILTER" = "UPDATED_FP32_S" ] ; then 
    echo updated_index_single_fp32

# Spaces benchmarks
elif [ "$BM_FILTER" = "SPACES" ] ; then 
    echo spaces_fp32
    echo spaces_fp64
fi