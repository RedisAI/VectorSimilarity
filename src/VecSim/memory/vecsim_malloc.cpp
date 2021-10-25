#include "vecsim_malloc.h"

void *VecsimBaseObject::operator new(size_t size) {
    void *p = vecsim_malloc(size);
    return p;
}

void VecsimBaseObject::operator delete(void *p) { vecsim_free(p); }