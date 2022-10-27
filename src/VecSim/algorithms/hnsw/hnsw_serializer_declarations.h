#pragma once

// Serializing functions.
public:
// This Ctor can be used with v2 and up.
HNSWIndex(std::string location, std::shared_ptr<VecSimAllocator> allocator);

HNSWIndexMetaData checkIntegrity() const;
virtual void saveIndexIMP(std::ofstream &output, EncodingVersion version) const override;
virtual void loadIndexIMP(std::ifstream &input, EncodingVersion version) override;
virtual inline bool serializingIsValid() const override {
    return this->checkIntegrity().valid_state;
}

private:
virtual void clearLabelLookup() = 0;
virtual void AddToLabelLookup(labelType label, idType id) = 0;

private:
// Functions for index saving.
void saveIndexFields(std::ofstream &output) const;
void saveIndexFields_v2(std::ofstream &output) const;
void saveGraph(std::ofstream &output) const;

// Functions for index loading.
void restoreIndexFields(std::ifstream &input);
void restoreIndexFields_v2(std::ifstream &input);
void restoreGraph(std::ifstream &input);
void fieldsValidation() const;
