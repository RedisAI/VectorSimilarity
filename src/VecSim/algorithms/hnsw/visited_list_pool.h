#pragma once

#include <mutex>
#include <string.h>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
  public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    };

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

  public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
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
        std::unique_lock<std::mutex> lock(poolguard);
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
