#pragma once

#include "VecSim/memory/vecsim_malloc.h"

#include <mutex>
#include <deque>
#include <string.h>

namespace hnswlib {

typedef unsigned short int vl_type;

class VisitedList : public VecsimBaseObject {
public:
    vl_type curTag;
    vl_type *visitedElements;
    unsigned int numElements;

    VisitedList(int num_elements, std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {
        curTag = 0;
        numElements = num_elements;
        visitedElements = new vl_type[numElements];
        memset(visitedElements, 0, sizeof(vl_type) * numElements);
    }

    void reset() {
        curTag++;
    };

    ~VisitedList() { delete[] visitedElements; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool : public VecsimBaseObject {
    std::deque<VisitedList *, VecsimSTLAllocator<VisitedList *>> pool;
    std::mutex poolGuard;
    int numElements;

public:
    VisitedListPool(int init_max_pools, int num_elements, std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), pool(allocator) {
        numElements = num_elements;
        for (int i = 0; i < init_max_pools; i++)
            pool.push_front(new (allocator) VisitedList(numElements, allocator));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *vl;
#ifdef ENABLE_PARALLELIZATION
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (!pool.empty()) {
                vl = pool.front();
                pool.pop_front();
            } else {
                vl = new VisitedList(numelements);
            }
        }
#else
        vl = pool.front();
#endif
        vl->reset();
        return vl;
    };

    void returnVisitedListToPool(VisitedList *vl) {
        std::unique_lock<std::mutex> lock(poolGuard);
        pool.push_front(vl);
    };

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    };
};

} // namespace hnswlib
