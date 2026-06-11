#include "bam_parser.h"
#include "cigar_processor.h"
#include "base_modification_extractor.h"

#include <htslib/sam.h>
#include <htslib/hts.h>

#include <stdexcept>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace pacbio_methylation {

struct BamParser::Impl {
    std::string bam_path;
    samFile* fp = nullptr;
    bam_hdr_t* header = nullptr;
    hts_idx_t* idx = nullptr;
    bam1_t* record = nullptr;

    Region current_region;
    int min_mapq = 0;
    int min_baseq = 0;

    bool has_index = false;
    bool region_set = false;
    hts_itr_t* iterator = nullptr;

    size_t total_reads_processed = 0;

    Impl() : record(bam_init1()) {}

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (iterator) {
            sam_itr_destroy(iterator);
            iterator = nullptr;
        }
        if (record) {
            bam_destroy1(record);
            record = nullptr;
        }
        if (idx) {
            hts_idx_destroy(idx);
            idx = nullptr;
        }
        if (header) {
            bam_hdr_destroy(header);
            header = nullptr;
        }
        if (fp) {
            sam_close(fp);
            fp = nullptr;
        }
    }

    void open() {
        cleanup();

        fp = sam_open(bam_path.c_str(), "rb");
        if (!fp) {
            throw std::runtime_error("Failed to open BAM file: " + bam_path);
        }

        header = sam_hdr_read(fp);
        if (!header) {
            throw std::runtime_error("Failed to read BAM header: " + bam_path);
        }

        idx = sam_index_load(fp, bam_path.c_str());
        has_index = (idx != nullptr);

        total_reads_processed = 0;
        region_set = false;
    }

    int get_tid(const std::string& chrom) const {
        if (!header) return -1;
        return bam_name2id(header, chrom.c_str());
    }

    void set_iterator_for_region() {
        if (iterator) {
            sam_itr_destroy(iterator);
            iterator = nullptr;
        }

        if (has_index && region_set && !current_region.chrom.empty()) {
            int tid = get_tid(current_region.chrom);
            if (tid >= 0) {
                hts_pos_t start = current_region.start < 0 ? 0 : current_region.start;
                hts_pos_t end = current_region.end < 0 ? header->target_len[tid] : current_region.end;
                iterator = sam_itr_queryi(idx, tid, start, end);
            }
        }
    }

    bool read_next_record() {
        if (!fp || !header || !record) return false;

        int ret;
        if (iterator) {
            ret = sam_itr_next(fp, iterator, record);
        } else {
            ret = sam_read1(fp, header, record);
        }

        if (ret < 0) return false;

        total_reads_processed++;
        return true;
    }

    bool passes_filters() const {
        if (!record) return false;

        if ((record->core.flag & BAM_FUNMAP) != 0) return false;
        if ((record->core.flag & BAM_FSECONDARY) != 0) return false;
        if ((record->core.flag & BAM_FQCFAIL) != 0) return false;
        if ((record->core.flag & BAM_FDUP) != 0) return false;
        if ((record->core.flag & BAM_FSUPPLEMENTARY) != 0) return false;

        if (record->core.qual < min_mapq) return false;

        return true;
    }

    BaseModificationData extract_current_read() {
        BaseModificationData data;

        char* qname = bam_get_qname(record);
        data.read_id = std::string(qname);

        int tid = record->core.tid;
        if (tid >= 0 && header && tid < header->n_targets) {
            data.chrom = std::string(header->target_name[tid]);
        } else {
            data.chrom = "*";
        }

        data.ref_start = record->core.pos;
        data.ref_end = bam_endpos(record);
        data.read_length = record->core.l_qseq;
        data.mapq = record->core.qual;
        data.is_reverse = ((record->core.flag & BAM_FREVERSE) != 0);

        uint32_t* cigar = bam_get_cigar(record);
        auto cigar_ops = CigarProcessor::parse_cigar(cigar, record->core.n_cigar);

        data.cigar_ops.resize(cigar_ops.size());
        data.cigar_lens.resize(cigar_ops.size());
        for (size_t i = 0; i < cigar_ops.size(); ++i) {
            data.cigar_ops[i] = cigar_ops[i].op;
            data.cigar_lens[i] = cigar_ops[i].length;
        }

        data.ref_positions = CigarProcessor::get_reference_positions(
            cigar_ops, data.ref_start, data.read_length
        );

        data.read_bases = BaseModificationExtractor::extract_sequence(
            bam_get_seq(record), data.read_length
        );

        data.base_qualities = BaseModificationExtractor::extract_quality(
            bam_get_qual(record), data.read_length
        );

        data.ref_bases.resize(data.read_length, 0);

        data.ipd_values = BaseModificationExtractor::extract_tag_by_name(
            record, "ip", data.read_length
        );
        if (static_cast<int64_t>(data.ipd_values.size()) != data.read_length) {
            data.ipd_values.resize(data.read_length, 0.0f);
        }

        data.pulse_width_values = BaseModificationExtractor::extract_tag_by_name(
            record, "pw", data.read_length
        );
        if (static_cast<int64_t>(data.pulse_width_values.size()) != data.read_length) {
            data.pulse_width_values.resize(data.read_length, 0.0f);
        }

        return data;
    }
};

BamParser::BamParser(const std::string& bam_path)
    : impl_(std::make_unique<Impl>())
{
    impl_->bam_path = bam_path;
    impl_->open();
}

BamParser::~BamParser() = default;

void BamParser::set_region(const Region& region) {
    impl_->current_region = region;
    impl_->region_set = !region.chrom.empty();
    impl_->set_iterator_for_region();
}

void BamParser::set_min_mapq(int mapq) {
    impl_->min_mapq = mapq;
}

void BamParser::set_min_baseq(int baseq) {
    impl_->min_baseq = baseq;
}

std::vector<BaseModificationData> BamParser::parse_all() {
    std::vector<BaseModificationData> results;

    reset();

    while (impl_->read_next_record()) {
        if (!impl_->passes_filters()) continue;
        results.push_back(impl_->extract_current_read());
    }

    return results;
}

std::vector<BaseModificationData> BamParser::parse_next_batch(size_t batch_size) {
    std::vector<BaseModificationData> results;
    results.reserve(batch_size);

    while (results.size() < batch_size && impl_->read_next_record()) {
        if (!impl_->passes_filters()) continue;
        results.push_back(impl_->extract_current_read());
    }

    return results;
}

size_t BamParser::get_total_reads() const {
    return impl_->total_reads_processed;
}

bool BamParser::has_more() const {
    if (!impl_->fp || !impl_->record) return false;
    return impl_->fp->is_bin ? !impl_->fp->is_bin->eof : true;
}

void BamParser::reset() {
    impl_->open();
    if (impl_->region_set) {
        impl_->set_iterator_for_region();
    }
}

}
