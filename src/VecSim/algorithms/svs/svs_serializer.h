/*
* Copyright (c) 2006-Present, Redis Ltd.
* All rights reserved.
*
* Licensed under your choice of the Redis Source Available License 2.0
* (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
* GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <fstream>
#include <string>
#include "VecSim/utils/serializer.h"
#include <filesystem>

class SVSserializer : public Serializer {
public:
    enum class EncodingVersion {
        V0,
        INVALID
    };

    explicit SVSserializer(EncodingVersion version = EncodingVersion::V0);

    static EncodingVersion ReadVersion(std::ifstream &input);

    void saveIndex(const std::string &location) override;


    EncodingVersion getVersion() const;

    virtual void loadIndex(const std::string &location) = 0;

protected:
    EncodingVersion m_version;

    virtual void impl_save(const std::string &location) = 0;

    virtual void saveIndexFields(std::ofstream &output) const = 0;
};
