//! Query operations and result handling for C FFI.

use crate::index::{BatchIteratorWrapper, IndexHandle};
use crate::params::VecSimQueryParams;
use crate::types::{
    QueryReplyInternal, QueryReplyIteratorInternal, QueryResultInternal, VecSimQueryReply_Code,
    VecSimQueryReply_Order,
};
use std::ffi::c_void;

/// Query reply handle that owns the results.
pub struct QueryReplyHandle {
    pub reply: QueryReplyInternal,
}

impl QueryReplyHandle {
    pub fn new(reply: QueryReplyInternal) -> Self {
        Self { reply }
    }

    pub fn len(&self) -> usize {
        self.reply.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reply.is_empty()
    }

    pub fn sort_by_order(&mut self, order: VecSimQueryReply_Order) {
        match order {
            VecSimQueryReply_Order::BY_SCORE => self.reply.sort_by_score(),
            VecSimQueryReply_Order::BY_ID => self.reply.sort_by_id(),
        }
    }

    pub fn get_iterator(&self) -> QueryReplyIteratorHandle {
        QueryReplyIteratorHandle::new(&self.reply.results)
    }

    pub fn code(&self) -> VecSimQueryReply_Code {
        self.reply.code
    }
}

/// Iterator handle over query results.
pub struct QueryReplyIteratorHandle<'a> {
    iter: QueryReplyIteratorInternal<'a>,
}

impl<'a> QueryReplyIteratorHandle<'a> {
    pub fn new(results: &'a [QueryResultInternal]) -> Self {
        Self {
            iter: QueryReplyIteratorInternal::new(results),
        }
    }

    pub fn next(&mut self) -> Option<&'a QueryResultInternal> {
        self.iter.next()
    }

    pub fn has_next(&self) -> bool {
        self.iter.has_next()
    }

    pub fn reset(&mut self) {
        self.iter.reset();
    }
}

/// Batch iterator handle.
pub struct BatchIteratorHandle {
    pub inner: Box<dyn BatchIteratorWrapper>,
    pub query_copy: Vec<u8>,
}

impl BatchIteratorHandle {
    pub fn new(inner: Box<dyn BatchIteratorWrapper>, query_copy: Vec<u8>) -> Self {
        Self { inner, query_copy }
    }

    pub fn has_next(&self) -> bool {
        self.inner.has_next()
    }

    pub fn next(&mut self, n: usize, order: VecSimQueryReply_Order) -> QueryReplyHandle {
        let reply = self.inner.next_batch(n, order);
        QueryReplyHandle::new(reply)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Perform a top-k query on an index.
pub fn top_k_query(
    handle: &IndexHandle,
    query: *const c_void,
    k: usize,
    params: Option<&VecSimQueryParams>,
    order: VecSimQueryReply_Order,
) -> QueryReplyHandle {
    let mut reply = handle.wrapper.top_k_query(query, k, params);

    // Sort by requested order
    match order {
        VecSimQueryReply_Order::BY_SCORE => reply.sort_by_score(),
        VecSimQueryReply_Order::BY_ID => reply.sort_by_id(),
    }

    QueryReplyHandle::new(reply)
}

/// Perform a range query on an index.
pub fn range_query(
    handle: &IndexHandle,
    query: *const c_void,
    radius: f64,
    params: Option<&VecSimQueryParams>,
    order: VecSimQueryReply_Order,
) -> QueryReplyHandle {
    let mut reply = handle.wrapper.range_query(query, radius, params);

    // Sort by requested order
    match order {
        VecSimQueryReply_Order::BY_SCORE => reply.sort_by_score(),
        VecSimQueryReply_Order::BY_ID => reply.sort_by_id(),
    }

    QueryReplyHandle::new(reply)
}

/// Create a batch iterator for the given index and query.
pub fn create_batch_iterator(
    handle: &IndexHandle,
    query: *const c_void,
    params: Option<&VecSimQueryParams>,
) -> Option<BatchIteratorHandle> {
    // Copy the query data since we need to own it
    let dim = handle.wrapper.dimension();
    let elem_size = handle.data_type.element_size();
    let query_bytes = dim * elem_size;

    let query_copy = unsafe {
        let slice = std::slice::from_raw_parts(query as *const u8, query_bytes);
        slice.to_vec()
    };

    handle
        .wrapper
        .create_batch_iterator(query, params)
        .map(|inner| BatchIteratorHandle::new(inner, query_copy))
}
