#include "abs_data_store.h"
#include "write_batch.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/comparator.h" 
#include "rocksdb/merge_operator.h" 
#include "rocksdb/table.h"

class EdgesComparator : public ROCKSDB_NAMESPACE::Comparator {
public:
	~EdgesComparator() {};
	int
	Compare(const ROCKSDB_NAMESPACE::Slice& a,
			const ROCKSDB_NAMESPACE::Slice& b) const override{
		auto gn1 = extractKey(a.data());
		auto gn2 = extractKey(b.data());
		return (gn1.second > gn2.second) ? 1 :
			(gn1.second < gn2.second) ? -1 :
			(int64_t) (gn1.first) - (int64_t) (gn2.first);
	}
	const char* Name() const override {return "EDGES_COMAPRE";};

	void FindShortestSeparator(std::string* ,
							   const ROCKSDB_NAMESPACE::Slice& ) const override {}
	
	void FindShortSuccessor(std::string* ) const override {};
	
			
};
class EdgesMergeOp : public ROCKSDB_NAMESPACE::MergeOperator {
public:
	~EdgesMergeOp() {};
	
	const char* Name() const override {return "EDGES_MERGE";};

#if 0
	bool Merge(
        const ROCKSDB_NAMESPACE::Slice&,
        const ROCKSDB_NAMESPACE::Slice* existing_value,
        const ROCKSDB_NAMESPACE::Slice& value,
        std::string* new_value,
        ROCKSDB_NAMESPACE::Logger* ) const override {
		ROCKSDB_NAMESPACE::Slice existing_slice;		
		
		if (existing_value) {
			existing_slice = *existing_value;
		}
		std::vector<idType> cur_value(existing_slice.size()/sizeof(idType));		
		auto ch = ObjectChange::NewObjectChangeFromString(value.data());
		ch->Apply(cur_value);
		*new_value = std::string(						
			(char *)cur_value.data(), cur_value.size() * sizeof(idType));
        return true;        
      }
#endif
	bool FullMergeV2(const MergeOperationInput& merge_in,
					 MergeOperationOutput* merge_out) const override {
		ROCKSDB_NAMESPACE::Slice existing_slice;		
		
		if (merge_in.existing_value) {
			existing_slice = *merge_in.existing_value;
		}
		std::vector<idType> cur_value(existing_slice.size()/sizeof(idType));
		std::memcpy(cur_value.data(), existing_slice.data(), existing_slice.size());

		for (auto op : merge_in.operand_list) {
			auto ch = ObjectChange::NewObjectChangeFromString(op.data());
			ch->Apply(cur_value);
			delete ch;
		}
		merge_out->new_value = std::string(						
			(char *)cur_value.data(), cur_value.size() * sizeof(idType));
		return true;
	}
};


class SpdbDataStore : public EdgesDataStore {
public:
	SpdbDataStore(std::shared_ptr<VecSimAllocator> allocator,
				  const char *dbPath);
	void Get(std::list<GetParams> &get_params_list) override;
	void Put(const std::list<StageParams> &stage_params_list) override;
	void Flush() override {
		db_->Flush(ROCKSDB_NAMESPACE::FlushOptions(), cfs_);
	}


	
private:
	void OpenDb(const char *dbPath);

				
private:
	ROCKSDB_NAMESPACE::DB *db_;
	std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle*>cfs_;
	std::shared_ptr<VecSimAllocator> allocator_;
};

SpdbDataStore::SpdbDataStore(std::shared_ptr<VecSimAllocator> ,
							 const char *dbPath)
{
	cfs_.resize(LAST_CF);
	OpenDb(dbPath);
}


void SpdbDataStore::OpenDb(const char *dbPath) {
	
	ROCKSDB_NAMESPACE::Options options;
	options.create_if_missing = true;
	options.use_direct_reads = true; 
	// make sure we are working on a new database
	DestroyDB(dbPath, options);

	ROCKSDB_NAMESPACE::Status s = ROCKSDB_NAMESPACE::DB::Open(options, dbPath, &db_);
	assert(s.ok());
	cfs_[0] = db_->DefaultColumnFamily();
	// create column families
	ROCKSDB_NAMESPACE::ColumnFamilyOptions cf_options;
	//cf_options.comparator = new EdgesComparator;	
	cf_options.merge_operator.reset(new EdgesMergeOp);
	cf_options.compression = ROCKSDB_NAMESPACE::kNoCompression;

	ROCKSDB_NAMESPACE::BlockBasedTableOptions bs_options;
	bs_options.block_align = true;
	bs_options.no_block_cache = true;
	bs_options.data_block_index_type = ROCKSDB_NAMESPACE::BlockBasedTableOptions::kDataBlockBinaryAndHash;
	
	cf_options.table_factory.reset(
		NewBlockBasedTableFactory(bs_options));
	
	s = db_->CreateColumnFamily(cf_options, "outgoing_edges", &cfs_[OUTGOING_CF]);
	assert(s.ok());
	s = db_->CreateColumnFamily(cf_options, "incoming_edges", &cfs_[INCOMING_CF]);
	assert(s.ok());
}	

	

void SpdbDataStore::Put(const std::list<StageParams> &stage_params_list) 
{
	ROCKSDB_NAMESPACE::WriteBatch wb;
	for (auto &stage_param : stage_params_list) {
		switch (stage_param.op) {
		case StageParams::WRITE_DATA:
			wb.Put(cfs_[stage_param.cf_id], stage_param.key, stage_param.data);
			break;
		case StageParams::UPDATE_DATA:
			wb.Merge(cfs_[stage_param.cf_id], stage_param.key, stage_param.data);
			break;
		case StageParams::DELETE_DATA:
			wb.Delete(cfs_[stage_param.cf_id], stage_param.key);
			break;
		default:
			assert(0);
		}
	}
	ROCKSDB_NAMESPACE::WriteOptions options;
	options.disableWAL = true;
	auto s = db_->Write(options, &wb);
	assert(s.ok());
 }

void SpdbDataStore::Get(std::list<GetParams> &get_params_list) {
	std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle*> cfs(get_params_list.size());
	std::vector<ROCKSDB_NAMESPACE::Slice> keys(get_params_list.size());
	std::vector<std::string> values;
	int index = 0;
	for (auto &get_param : get_params_list) {
		cfs[index] = cfs_[get_param.cf_id];
		keys[index] = get_param.key;
		index++;
	}
	db_->MultiGet(ROCKSDB_NAMESPACE::ReadOptions(),
				  cfs,
				  keys,
				  &values);
	index = 0;
	for (auto &get_param : get_params_list) {
		get_param.data = std::vector<idType>(values[index].size() / sizeof(idType));
		std::memcpy(get_param.data.data(), values[index].data(), values[index].size());
	}
			 
}			 
			 
		 

	 


EdgesDataStore *NewSpeedbDataStore(std::shared_ptr<VecSimAllocator> allocator,
								   const char *dbPath) {
	return new SpdbDataStore(allocator, dbPath);
}
