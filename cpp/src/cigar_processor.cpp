#include "cigar_processor.h"
#include <stdexcept>
#include <algorithm>

namespace pacbio_methylation {

static const uint8_t BAM_CMATCH = 0;
static const uint8_t BAM_CINS = 1;
static const uint8_t BAM_CDEL = 2;
static const uint8_t BAM_CREF_SKIP = 3;
static const uint8_t BAM_CSOFT_CLIP = 4;
static const uint8_t BAM_CHARD_CLIP = 5;
static const uint8_t BAM_CPAD = 6;
static const uint8_t BAM_CEQUAL = 7;
static const uint8_t BAM_CDIFF = 8;
static const uint8_t BAM_CBACK = 9;

static const uint32_t BAM_CIGAR_SHIFT = 4;
static const uint32_t BAM_CIGAR_MASK = 0xf;

std::vector<CigarOperation> CigarProcessor::parse_cigar(
    const uint32_t* cigar_data,
    uint32_t n_cigar
) {
    std::vector<CigarOperation> result;
    result.reserve(n_cigar);

    for (uint32_t i = 0; i < n_cigar; ++i) {
        uint32_t v = cigar_data[i];
        CigarOperation op;
        op.op = static_cast<uint8_t>(v & BAM_CIGAR_MASK);
        op.length = v >> BAM_CIGAR_SHIFT;
        result.push_back(op);
    }

    return result;
}

std::vector<AlignedPair> CigarProcessor::generate_aligned_pairs(
    const std::vector<CigarOperation>& cigar,
    int64_t ref_start
) {
    std::vector<AlignedPair> pairs;
    int64_t ref_pos = ref_start;
    int64_t read_pos = 0;

    for (const auto& op : cigar) {
        switch (op.op) {
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                for (uint32_t i = 0; i < op.length; ++i) {
                    pairs.push_back({ref_pos++, read_pos++, true});
                }
                break;
            case BAM_CINS:
                for (uint32_t i = 0; i < op.length; ++i) {
                    pairs.push_back({-1, read_pos++, false});
                }
                break;
            case BAM_CDEL:
            case BAM_CREF_SKIP:
                for (uint32_t i = 0; i < op.length; ++i) {
                    pairs.push_back({ref_pos++, -1, false});
                }
                break;
            case BAM_CSOFT_CLIP:
                read_pos += op.length;
                break;
            case BAM_CHARD_CLIP:
            case BAM_CPAD:
            default:
                break;
        }
    }

    return pairs;
}

std::vector<int64_t> CigarProcessor::get_reference_positions(
    const std::vector<CigarOperation>& cigar,
    int64_t ref_start,
    int64_t read_length
) {
    std::vector<int64_t> ref_positions(read_length, -1);
    int64_t ref_pos = ref_start;
    int64_t read_pos = 0;

    for (const auto& op : cigar) {
        switch (op.op) {
            case BAM_CMATCH:
            case BAM_CEQUAL:
            case BAM_CDIFF:
                for (uint32_t i = 0; i < op.length; ++i) {
                    if (read_pos < read_length) {
                        ref_positions[read_pos] = ref_pos;
                    }
                    ref_pos++;
                    read_pos++;
                }
                break;
            case BAM_CINS:
                read_pos += op.length;
                break;
            case BAM_CDEL:
            case BAM_CREF_SKIP:
                ref_pos += op.length;
                break;
            case BAM_CSOFT_CLIP:
                read_pos += op.length;
                break;
            case BAM_CHARD_CLIP:
            case BAM_CPAD:
            default:
                break;
        }
    }

    return ref_positions;
}

std::vector<bool> CigarProcessor::get_clip_mask(
    const std::vector<CigarOperation>& cigar,
    int64_t read_length
) {
    std::vector<bool> mask(read_length, true);
    int64_t read_pos = 0;

    for (const auto& op : cigar) {
        if (op.op == BAM_CSOFT_CLIP || op.op == BAM_CHARD_CLIP) {
            for (uint32_t i = 0; i < op.length && read_pos + i < read_length; ++i) {
                mask[read_pos + i] = false;
            }
        }
        if (op.op != BAM_CDEL && op.op != BAM_CREF_SKIP) {
            read_pos += op.length;
        }
    }

    return mask;
}

std::string CigarProcessor::cigar_op_to_char(uint8_t op) {
    switch (op) {
        case BAM_CMATCH: return "M";
        case BAM_CINS: return "I";
        case BAM_CDEL: return "D";
        case BAM_CREF_SKIP: return "N";
        case BAM_CSOFT_CLIP: return "S";
        case BAM_CHARD_CLIP: return "H";
        case BAM_CPAD: return "P";
        case BAM_CEQUAL: return "=";
        case BAM_CDIFF: return "X";
        default: return "?";
    }
}

}
