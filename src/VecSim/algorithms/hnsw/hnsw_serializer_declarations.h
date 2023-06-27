#pragma once

// Serializing and tests functions.
public:
HNSWIndex(std::ifstream &input, const HNSWParams *params,
          const AbstractIndexInitParams &abstractInitParams, EncodingVersion version);

// Validates the connections between vectors
HNSWIndexMetaData checkIntegrity() const;

// Index memory size might be changed during index saving.
virtual void saveIndexIMP(std::ofstream &output) override;

// used by index factory to load nodes connections
void restoreGraph(std::ifstream &input);

private:
// Functions for index saving.
void saveIndexFields(std::ofstream &output) const;

void saveGraph(std::ofstream &output) const;

void saveLevel(std::ofstream &output, LevelData &data) const;
void restoreLevel(std::ifstream &input, LevelData &data);

// Functions for index loading.
void restoreIndexFields(std::ifstream &input);
void fieldsValidation() const;
