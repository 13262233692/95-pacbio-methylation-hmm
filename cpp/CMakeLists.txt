add_library(bam_parser_core STATIC
    src/bam_parser.cpp
    src/cigar_processor.cpp
    src/base_modification_extractor.cpp
)

target_include_directories(bam_parser_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HTSLIB_INCLUDE_DIRS}
)

target_link_libraries(bam_parser_core PUBLIC
    ${HTSLIB_LIBRARIES}
    pthread
    z
)

pybind11_add_module(_cpp_bindings
    src/python_bindings.cpp
)

target_link_libraries(_cpp_bindings PRIVATE
    bam_parser_core
)

set_target_properties(_cpp_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pacbio_methylation_hmm"
)
