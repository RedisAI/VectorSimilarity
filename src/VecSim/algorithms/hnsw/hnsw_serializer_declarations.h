#pragma once

// Serializing functions.
public:
HNSWIndex(std::ifstream &input, const HNSWParams *params,
          std::shared_ptr<VecSimAllocator> allocator);

HNSWIndexMetaData checkIntegrity() const;
virtual void saveIndexIMP(std::ofstream &output, EncodingVersion version) const override;
virtual inline bool serializingIsValid() const override {
    return this->checkIntegrity().valid_state;
}

// used by index factory to load nodes connections
void restoreGraph(std::ifstream &input);

private:
virtual void AddToLabelLookup(labelType label, idType id) = 0;
// Functions for index saving.
void saveIndexFields(std::ofstream &output) const;
void saveIndexFields_v2(std::ofstream &output) const;
void saveGraph(std::ofstream &output) const;

// Functions for index loading.
void restoreIndexFields(std::ifstream &input);
void fieldsValidation() const;
