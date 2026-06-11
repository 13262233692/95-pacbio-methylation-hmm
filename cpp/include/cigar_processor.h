#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace pacbio_methylation {

struct CigarOperation {
    uint8_t op;
    uint32_t length;
};

struct AlignedPair {
    int64_t ref_pos;
    int64_t read_pos;
    bool is_match;
};

class CigarProcessor {
public:
    static std::vector<CigarOperation> parse_cigar(
        const uint32_t* cigar_data,
        uint32_t n_cigar
    );

    static std::vector<AlignedPair> generate_aligned_pairs(
        const std::vector<CigarOperation>& cigar,
        int64_t ref_start
    );

    static std::vector<int64_t> get_reference_positions(
        const std::vector<CigarOperation>& cigar,
        int64_t ref_start,
        int64_t read_length
    );

    static std::vector<bool> get_clip_mask(
        const std::vector<CigarOperation>& cigar,
        int64_t read_length
    );

    static std::string cigar_op_to_char(uint8_t op);
};

}
