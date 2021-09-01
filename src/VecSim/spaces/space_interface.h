#pragma once

template <typename TYPE> using DISTFUNC = TYPE (*)(const void *, const void *, const void *);

template <typename TYPE> class SpaceInterface {
  public:
    virtual size_t get_data_size() const = 0;

    virtual DISTFUNC<TYPE> get_dist_func() const = 0;

    virtual void *get_data_dim() = 0;

    virtual ~SpaceInterface() = default;
};
