#pragma once

// Serializing and tests functions.
public:
HNSWIndex(std::ifstream &input, const HNSWParams *params,
          std::shared_ptr<VecSimAllocator> allocator, EncodingVersion version);

// Validates the connections between vectors
HNSWIndexMetaData checkIntegrity() const;

// Index memory size might be changed during index saving.
virtual void saveIndexIMP(std::ofstream &output) override;

// used by index factory to load nodes connections
void restoreGraph(std::ifstream &input);

// used to fix V1 index to V2 (current version)
void restoreGraph_V1_fixes();

private:
// Functions for index saving.
void saveIndexFields(std::ofstream &output) const;
void saveIndexFields_v2(std::ofstream &output) const;

void saveGraph(std::ofstream &output) const;

// Functions for index loading.
void restoreIndexFields(std::ifstream &input);
void HandleLevelGenerator(std::ifstream &input);
void fieldsValidation() const;
