#pragma once

#include "VecSim/containers/raw_data_container_interface.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

class RocksDataContainer : public RawDataContainer {
    std::shared_ptr<rocksdb::DB> db;
    std::unique_ptr<rocksdb::ColumnFamilyHandle> cf;
    size_t element_bytes_count;
    size_t count;

public:
    RocksDataContainer(std::shared_ptr<rocksdb::DB> db_, size_t elementBytesCount,
                       std::shared_ptr<VecSimAllocator> allocator)
        : RawDataContainer(), db(db_), element_bytes_count(elementBytesCount), count(0) {
        rocksdb::ColumnFamilyOptions cf_options;
        rocksdb::ColumnFamilyHandle *cf_;
        rocksdb::Status status = db->CreateColumnFamily(cf_options, "vectors", &cf_);
        if (!status.ok()) {
            throw std::runtime_error("VecSim create column family 'vectors' error");
        }
        cf.reset(cf_);
    }
    ~RocksDataContainer() override = default;

    size_t size() const override { return count; }

    Status addElement(const void *element, size_t id) override {
        rocksdb::Slice value(static_cast<const char *>(element), element_bytes_count);
        auto status =
            db->Put(rocksdb::WriteOptions(), cf.get(), std::to_string(id), value);
        if (status.ok()) {
            count++;
            return Status::OK;
        }
        return Status::ERR;
    }

    const char *getElement(size_t id) const override {
        std::string value;
        auto status =
            db->Get(rocksdb::ReadOptions(), cf.get(), std::to_string(id), &value);
        if (status.ok()) {
            // Copy and return the value
            assert(value.size() == element_bytes_count);
            auto result = new char[element_bytes_count];
            memcpy(result, value.c_str(), element_bytes_count);
            return result;
        }
        return nullptr;
    }

    Status removeElement(size_t id) override {
        rocksdb::Slice key(std::to_string(id));
        rocksdb::Status status = db->Delete(rocksdb::WriteOptions{}, cf.get(), key);
        if (status.ok()) {
            count--;
            return Status::OK;
        }
        return Status::ERR;
    }

    Status updateElement(size_t id, const void *element) override {
        return addElement(element, id);
    }

    std::unique_ptr<RawDataContainer::Iterator> getIterator() const override {
        return std::make_unique<RocksDataContainer::Iterator>();
    }

    // // DUMMY IMPLEMENTATION
    struct Iterator : RawDataContainer::Iterator {
        /**
         * This is an abstract interface, constructor/destructor should be implemented by the
         * derived classes
         */
        Iterator() = default;
        virtual ~Iterator() = default;

        /**
         * The basic iterator operations API
         */
        virtual bool hasNext() const { return false; }
        virtual const char *next() { return nullptr; }
        virtual void reset() {}
    };
};
