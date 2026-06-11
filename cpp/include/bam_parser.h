#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace pacbio_methylation {

struct BaseModificationData {
    std::string read_id;
    std::string chrom;
    int64_t ref_start;
    int64_t ref_end;
    int64_t read_length;
    int mapq;
    bool is_reverse;

    std::vector<int64_t> ref_positions;
    std::vector<uint8_t> ref_bases;
    std::vector<uint8_t> read_bases;
    std::vector<uint8_t> base_qualities;

    std::vector<float> ipd_values;
    std::vector<float> pulse_width_values;

    std::vector<uint32_t> cigar_ops;
    std::vector<uint32_t> cigar_lens;
};

struct Region {
    std::string chrom;
    int64_t start;
    int64_t end;

    Region() : chrom(""), start(-1), end(-1) {}
    Region(const std::string& c, int64_t s, int64_t e)
        : chrom(c), start(s), end(e) {}
};

class BamParser {
public:
    explicit BamParser(const std::string& bam_path);
    ~BamParser();

    BamParser(const BamParser&) = delete;
    BamParser& operator=(const BamParser&) = delete;

    void set_region(const Region& region);
    void set_min_mapq(int mapq);
    void set_min_baseq(int baseq);

    std::vector<BaseModificationData> parse_all();
    std::vector<BaseModificationData> parse_next_batch(size_t batch_size);

    size_t get_total_reads() const;
    bool has_more() const;
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
