#pragma once

typedef enum {
    RAW_DATA_CONTAINER_OK = 0,
    RAW_DATA_CONTAINER_ID_ALREADY_EXIST,
    RAW_DATA_CONTAINER_ID_NOT_EXIST,
    RAW_DATA_CONTAINER_ERR
} RawDataContainer_Status;

struct RawDataContainer {
    /**
     * This is an abstract interface, constructor/destructor should be implemented by the derived
     * classes
     */
    RawDataContainer() = default;
    virtual ~RawDataContainer() = default;

    /**
     * @return number of elements in the container
     */
    virtual size_t size() const = 0;
    /**
     * @param element element's raw data to be added into the container
     * @param id of the new element
     * @return status
     */
    virtual RawDataContainer_Status addElement(const void *element, size_t id) = 0;
    /**
     * @param id of the element to return
     * @return Immutable reference to the element's data, NULL if id doesn't exist
     */
    virtual const char *getElement(size_t id) const = 0;
    /**
     * @param id of the element to remove
     * @return status
     */
    virtual RawDataContainer_Status removeElement(size_t id) = 0;
    /**
     * @param id to change its asociated data
     * @param element the new raw data to associate with id
     * @return status
     */
    virtual RawDataContainer_Status updateElement(size_t id, const void *element) = 0;

    struct Iterator {
        /**
         * This is an abstract interface, constructor/destructor should be implemented by the
         * derived classes
         */
        Iterator() = default;
        virtual ~Iterator() = default;

        /**
         * The basic iterator operations API
         */
        virtual bool hasNext() = 0;
        virtual const char *next() = 0;
        virtual void reset() = 0;
    };

    /**
     * Create a new iterator. Should be freed by the iterator's destroctor.
     */
    virtual std::unique_ptr<Iterator> getIterator() = 0;
};
