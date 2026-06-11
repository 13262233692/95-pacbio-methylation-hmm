#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace pacbio_methylation {

class BaseModificationExtractor {
public:
    static std::vector<float> extract_ipd(
        const uint8_t* aux_data,
        uint32_t data_length,
        uint8_t code,
        int64_t read_length
    );

    static std::vector<float> extract_pulse_width(
        const uint8_t* aux_data,
        uint32_t data_length,
        uint8_t code,
        int64_t read_length
    );

    static std::vector<float> extract_tag_by_name(
        const void* bam_record,
        const std::string& tag_name,
        int64_t expected_length
    );

    static std::vector<uint8_t> extract_quality(
        const uint8_t* qual_data,
        int64_t read_length
    );

    static std::vector<uint8_t> extract_sequence(
        const uint8_t* seq_data,
        int64_t read_length
    );

    static std::string base_to_string(uint8_t base);
};

}
