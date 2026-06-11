#include "base_modification_extractor.h"
#include <stdexcept>
#include <cstring>
#include <cmath>

extern "C" {
#include "htslib/sam.h"
}

namespace pacbio_methylation {

static const uint8_t BASE_TABLE[16] = {
    0, 'A', 'C', 0, 'G', 0, 0, 0, 'T', 0, 0, 0, 0, 0, 0, 'N'
};

static std::vector<float> decode_sam_tag_array(
    const uint8_t* data,
    uint32_t len,
    uint8_t type
) {
    std::vector<float> result;

    if (type == 'c') {
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(reinterpret_cast<const int8_t*>(data)[i]));
        }
    } else if (type == 'C') {
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(data[i]));
        }
    } else if (type == 's') {
        const int16_t* arr = reinterpret_cast<const int16_t*>(data);
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(arr[i]));
        }
    } else if (type == 'S') {
        const uint16_t* arr = reinterpret_cast<const uint16_t*>(data);
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(arr[i]));
        }
    } else if (type == 'i') {
        const int32_t* arr = reinterpret_cast<const int32_t*>(data);
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(arr[i]));
        }
    } else if (type == 'I') {
        const uint32_t* arr = reinterpret_cast<const uint32_t*>(data);
        result.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            result.push_back(static_cast<float>(arr[i]));
        }
    } else if (type == 'f') {
        const float* arr = reinterpret_cast<const float*>(data);
        result.assign(arr, arr + len);
    } else {
        throw std::runtime_error("Unsupported SAM tag array type: " + std::string(1, type));
    }

    return result;
}

std::vector<float> BaseModificationExtractor::extract_ipd(
    const uint8_t* aux_data,
    uint32_t data_length,
    uint8_t code,
    int64_t read_length
) {
    if (!aux_data || data_length < 4) {
        return std::vector<float>(read_length, 0.0f);
    }
    return decode_sam_tag_array(aux_data + 4, data_length - 4, code);
}

std::vector<float> BaseModificationExtractor::extract_pulse_width(
    const uint8_t* aux_data,
    uint32_t data_length,
    uint8_t code,
    int64_t read_length
) {
    if (!aux_data || data_length < 4) {
        return std::vector<float>(read_length, 0.0f);
    }
    return decode_sam_tag_array(aux_data + 4, data_length - 4, code);
}

std::vector<float> BaseModificationExtractor::extract_tag_by_name(
    const void* bam_record,
    const std::string& tag_name,
    int64_t expected_length
) {
    const bam1_t* b = static_cast<const bam1_t*>(bam_record);
    if (!b || tag_name.size() != 2) {
        return std::vector<float>(expected_length, 0.0f);
    }

    uint8_t* aux = bam_aux_get(b, tag_name.c_str());
    if (!aux) {
        return std::vector<float>(expected_length, 0.0f);
    }

    uint8_t type = *aux;
    if (type == 'B') {
        uint8_t subtype = aux[1];
        int32_t count = *reinterpret_cast<int32_t*>(aux + 2);
        return decode_sam_tag_array(aux + 6, count, subtype);
    } else if (type == 'c' || type == 'C' || type == 's' || type == 'S' ||
               type == 'i' || type == 'I' || type == 'f') {
        float val = 0.0f;
        if (type == 'c') val = static_cast<float>(*reinterpret_cast<int8_t*>(aux + 1));
        else if (type == 'C') val = static_cast<float>(aux[1]);
        else if (type == 's') val = static_cast<float>(*reinterpret_cast<int16_t*>(aux + 1));
        else if (type == 'S') val = static_cast<float>(*reinterpret_cast<uint16_t*>(aux + 1));
        else if (type == 'i') val = static_cast<float>(*reinterpret_cast<int32_t*>(aux + 1));
        else if (type == 'I') val = static_cast<float>(*reinterpret_cast<uint32_t*>(aux + 1));
        else if (type == 'f') val = *reinterpret_cast<float*>(aux + 1);
        return std::vector<float>(1, val);
    }

    return std::vector<float>(expected_length, 0.0f);
}

std::vector<uint8_t> BaseModificationExtractor::extract_quality(
    const uint8_t* qual_data,
    int64_t read_length
) {
    std::vector<uint8_t> result(read_length);
    if (qual_data) {
        std::memcpy(result.data(), qual_data, read_length);
    }
    return result;
}

std::vector<uint8_t> BaseModificationExtractor::extract_sequence(
    const uint8_t* seq_data,
    int64_t read_length
) {
    std::vector<uint8_t> result(read_length);
    if (seq_data) {
        for (int64_t i = 0; i < read_length; ++i) {
            result[i] = bam_seqi(seq_data, i);
        }
    }
    return result;
}

std::string BaseModificationExtractor::base_to_string(uint8_t base) {
    char c = static_cast<char>(BASE_TABLE[base & 0xf]);
    return std::string(1, c ? c : 'N');
}

}
