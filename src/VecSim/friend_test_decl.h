/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#ifndef INDEX_TEST_FRIEND_CLASS
#define INDEX_TEST_FRIEND_CLASS(class_name)                                                        \
    template <typename>                                                                            \
    friend class class_name;
#endif
