
for bm_class in basics batch_iterator; do \
    for type in single multi; do \
        for data_type in fp32; do \
            BM_TEST_NAME=${bm_class}_${type}_${data_type}; \
            echo $BM_TEST_NAME; \
        done \
    done \
done

echo updated_index_single_fp32
echo spaces_fp32
echo spaces_fp64
