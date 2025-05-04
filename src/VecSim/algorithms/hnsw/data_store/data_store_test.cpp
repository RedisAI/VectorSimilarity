#include "edges_interface.h"

const size_t max_outgoing_links = 32; 
const size_t max_level = 5;
// id % vector_to_level == 0 -> vector exists in this level 
const size_t vector_to_level[max_level] = {
	max_outgoing_links,
	max_outgoing_links << 5,
	max_outgoing_links << 10,
	max_outgoing_links << 15,
	max_outgoing_links << 20
};
const size_t max_vectors = vector_to_level[max_level-1];

int idToLevel(size_t id) {
	for (int level = 0; level < max_level; level++) {
		if (id % vector_to_level[level] != 0)
			return level;
	}
	return max_level;
}

static EdgesInterface  *edges_interface;
void generateLinksOnLevel(size_t id, int level) {
	WriteBatch *wb = edges_interface->newWriteBatch();
	std::vector<idType> edges(max_outgoing_links);
	for (size_t i = 0; i < max_outgoing_links; i++) {
		while (1) {
			int r = rand() % max_vectors;
			if (level > 0) {
				r = (r / vector_to_level[level -1]) * vector_to_level[level -1];
				if (r == 0)
					continue;
				
			}
		    auto iter = std::find(edges.begin(), edges.end(), r);
			if (iter != edges.end())
				continue;
			edges[i] = r;
			
			edges_interface->AddIncomingTarget({r, level}, id, wb);
			break;
		} 
	}
	edges_interface->SetOutgoingAllTargets({id, level}, edges, wb);
	edges_interface->CommitWriteBatch(wb);
}


void verifyLinksOnLevel(size_t id, int level) {
	std::vector<idType> og_edges = edges_interface->GetOutgoingEdges({id, level}, nullptr);
	assert(og_edges.size() > 0);
	
	for (auto og : og_edges) {
		auto inc_edges = edges_interface->GetIncomingEdges({og, level}, nullptr);
		assert(std::find(inc_edges.begin(), inc_edges.end(), id) != inc_edges.end());
	}
	
}
		
	
	

int main() {
	std::shared_ptr<VecSimAllocator> allocator;
	// auto ds = NewRamDataStore(allocator, 0x1000, max_outgoing_links, max_vectors);
	auto ds = NewSpeedbDataStore(allocator, "/tmp/vectordb");
	edges_interface =  new EdgesInterface(allocator, ds);		

	// build
	for (size_t id = 1; id < max_vectors; id++) {
		int level = idToLevel(id);
		for (int l = level; l >= 0; l--) {
			generateLinksOnLevel(id, l);
		}
	}
	edges_interface->Flush();
	for (int i = 0; i < 1000000; i++) {
		size_t id = (rand() % (max_vectors -1)) + 1; // no 0 id;
		int level = idToLevel(id);
		for (int l = level; l >= 0; l--) {
			verifyLinksOnLevel(id, l);
		}
	}
}
		
					 
		
	
	
	
