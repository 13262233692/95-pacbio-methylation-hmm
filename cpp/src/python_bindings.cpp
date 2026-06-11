#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bam_parser.h"
#include "cigar_processor.h"
#include "base_modification_extractor.h"

namespace py = pybind11;
using namespace pacbio_methylation;

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    py::array_t<T> arr(vec.size());
    py::buffer_info buf = arr.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(T));
    return arr;
}

PYBIND11_MODULE(_cpp_bindings, m) {
    m.doc() = "C++ bindings for PacBio BAM parsing and methylation signal extraction";

    py::class_<BaseModificationData>(m, "BaseModificationData")
        .def(py::init<>())
        .def_readonly("read_id", &BaseModificationData::read_id)
        .def_readonly("chrom", &BaseModificationData::chrom)
        .def_readonly("ref_start", &BaseModificationData::ref_start)
        .def_readonly("ref_end", &BaseModificationData::ref_end)
        .def_readonly("read_length", &BaseModificationData::read_length)
        .def_readonly("mapq", &BaseModificationData::mapq)
        .def_readonly("is_reverse", &BaseModificationData::is_reverse)
        .def_property_readonly("ref_positions", [](const BaseModificationData& d) {
            return vector_to_numpy(d.ref_positions);
        })
        .def_property_readonly("read_bases", [](const BaseModificationData& d) {
            return vector_to_numpy(d.read_bases);
        })
        .def_property_readonly("base_qualities", [](const BaseModificationData& d) {
            return vector_to_numpy(d.base_qualities);
        })
        .def_property_readonly("ipd_values", [](const BaseModificationData& d) {
            return vector_to_numpy(d.ipd_values);
        })
        .def_property_readonly("pulse_width_values", [](const BaseModificationData& d) {
            return vector_to_numpy(d.pulse_width_values);
        })
        .def_property_readonly("cigar_ops", [](const BaseModificationData& d) {
            return vector_to_numpy(d.cigar_ops);
        })
        .def_property_readonly("cigar_lens", [](const BaseModificationData& d) {
            return vector_to_numpy(d.cigar_lens);
        });

    py::class_<Region>(m, "Region")
        .def(py::init<>())
        .def(py::init<const std::string&, int64_t, int64_t>(),
             py::arg("chrom"), py::arg("start"), py::arg("end"))
        .def_readwrite("chrom", &Region::chrom)
        .def_readwrite("start", &Region::start)
        .def_readwrite("end", &Region::end);

    py::class_<BamParser>(m, "BamParser")
        .def(py::init<const std::string&>(), py::arg("bam_path"))
        .def("set_region", &BamParser::set_region, py::arg("region"))
        .def("set_min_mapq", &BamParser::set_min_mapq, py::arg("mapq"))
        .def("set_min_baseq", &BamParser::set_min_baseq, py::arg("baseq"))
        .def("parse_all", &BamParser::parse_all)
        .def("parse_next_batch", &BamParser::parse_next_batch, py::arg("batch_size"))
        .def("get_total_reads", &BamParser::get_total_reads)
        .def("has_more", &BamParser::has_more)
        .def("reset", &BamParser::reset);

    py::class_<CigarOperation>(m, "CigarOperation")
        .def(py::init<>())
        .def_readwrite("op", &CigarOperation::op)
        .def_readwrite("length", &CigarOperation::length);

    py::class_<AlignedPair>(m, "AlignedPair")
        .def(py::init<>())
        .def_readwrite("ref_pos", &AlignedPair::ref_pos)
        .def_readwrite("read_pos", &AlignedPair::read_pos)
        .def_readwrite("is_match", &AlignedPair::is_match);

    py::class_<CigarProcessor>(m, "CigarProcessor")
        .def_static("cigar_op_to_char", &CigarProcessor::cigar_op_to_char, py::arg("op"));

    m.def("base_to_string", &BaseModificationExtractor::base_to_string, py::arg("base"));
}
