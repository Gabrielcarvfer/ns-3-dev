set(source_files
    helper/adhoc-aloha-noack-ideal-phy-helper.cc
    helper/spectrum-analyzer-helper.cc
    helper/spectrum-helper.cc
    helper/tv-spectrum-transmitter-helper.cc
    helper/waveform-generator-helper.cc
    model/aloha-noack-mac-header.cc
    model/aloha-noack-net-device.cc
    model/constant-spectrum-propagation-loss.cc
    model/friis-spectrum-propagation-loss.cc
    model/half-duplex-ideal-phy-signal-parameters.cc
    model/half-duplex-ideal-phy.cc
    model/hexagonal-wraparound-model.cc
    model/ism-spectrum-value-helper.cc
    model/matrix-based-channel-model.cc
    model/microwave-oven-spectrum-value-helper.cc
    model/multi-model-spectrum-channel.cc
    model/non-communicating-net-device.cc
    model/phased-array-spectrum-propagation-loss-model.cc
    model/single-model-spectrum-channel.cc
    model/spectrum-analyzer.cc
    model/spectrum-channel.cc
    model/spectrum-converter.cc
    model/spectrum-error-model.cc
    model/spectrum-interference.cc
    model/spectrum-model-300kHz-300GHz-log.cc
    model/spectrum-model-ism2400MHz-res1MHz.cc
    model/spectrum-model.cc
    model/spectrum-phy.cc
    model/spectrum-propagation-loss-model.cc
    model/spectrum-signal-parameters.cc
    model/spectrum-transmit-filter.cc
    model/spectrum-value.cc
    model/three-gpp-channel-model.cc
    model/three-gpp-spectrum-propagation-loss-model.cc
    model/trace-fading-loss-model.cc
    model/tv-spectrum-transmitter.cc
    model/two-ray-spectrum-propagation-loss-model.cc
    model/waveform-generator.cc
    model/wraparound-model.cc
)

set(header_files
    helper/adhoc-aloha-noack-ideal-phy-helper.h
    helper/spectrum-analyzer-helper.h
    helper/spectrum-helper.h
    helper/tv-spectrum-transmitter-helper.h
    helper/waveform-generator-helper.h
    model/aloha-noack-mac-header.h
    model/aloha-noack-net-device.h
    model/constant-spectrum-propagation-loss.h
    model/friis-spectrum-propagation-loss.h
    model/half-duplex-ideal-phy-signal-parameters.h
    model/half-duplex-ideal-phy.h
    model/hexagonal-wraparound-model.h
    model/ism-spectrum-value-helper.h
    model/matrix-based-channel-model.h
    model/microwave-oven-spectrum-value-helper.h
    model/multi-model-spectrum-channel.h
    model/non-communicating-net-device.h
    model/phased-array-spectrum-propagation-loss-model.h
    model/single-model-spectrum-channel.h
    model/spectrum-analyzer.h
    model/spectrum-channel.h
    model/spectrum-converter.h
    model/spectrum-error-model.h
    model/spectrum-interference.h
    model/spectrum-model-300kHz-300GHz-log.h
    model/spectrum-model-ism2400MHz-res1MHz.h
    model/spectrum-model.h
    model/spectrum-phy.h
    model/spectrum-propagation-loss-model.h
    model/spectrum-signal-parameters.h
    model/spectrum-transmit-filter.h
    model/spectrum-value.h
    model/three-gpp-channel-model.h
    model/three-gpp-spectrum-propagation-loss-model.h
    model/trace-fading-loss-model.h
    model/tv-spectrum-transmitter.h
    model/two-ray-spectrum-propagation-loss-model.h
    model/waveform-generator.h
    model/wraparound-model.h
    utils/spectrum-test.h
)

# ── HDF5 support ──
find_package(HDF5 COMPONENTS C CXX HL)
if(${HDF5_FOUND})
    message(STATUS "HDF5 found")
else()
    message(WARNING "HDF5 not found — sls_chan_validation will build without HDF5 output")
endif()

# ── WebGPU / SlsChanWgpu integration ──
# The SLS-channel GPU back-end lives in model/sls-chan-wgpu.{h,cc} and the
# matching WGSL kernel in model/sls-chan.wgsl. Set NS3_ENABLE_3GPP_GPU=OFF
# (cmake -DNS3_ENABLE_3GPP_GPU=OFF ..) to compile spectrum without the GPU
# back-end at all (no wgpu-native dependency).
option(NS3_ENABLE_3GPP_GPU "Build ThreeGppChannelModel with WebGPU batch back-end" ON)

# Fetch prebuilt wgpu-native binaries (fastest) — must happen BEFORE build_lib
# so the spectrum library target can link against the imported wgpu target.
include(FetchContent)
set(WEBGPU_LINK_TYPE SHARED)
set(WEBGPU_BACKEND_DEFAULT "WGPU")
FetchContent_Declare(webgpu
        GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
        GIT_TAG        main   # pin to a release tag in production
)
FetchContent_MakeAvailable(webgpu)

get_target_property(WEBGPU_INCLUDES  webgpu INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(WEBGPU_LOCATION  webgpu IMPORTED_LOCATION)
get_target_property(WEBGPU_DEFS      webgpu INTERFACE_COMPILE_DEFINITIONS)

if(NS3_ENABLE_3GPP_GPU)
    message(STATUS "ThreeGppChannelModel GPU back-end: ENABLED (NS3_ENABLE_3GPP_GPU=ON)")
    list(APPEND source_files model/sls-chan-wgpu.cc)
    # NOTE: sls-chan-wgpu.h is deliberately NOT in header_files. It's an
    # implementation detail of the GPU back-end; exposing it as a public
    # header would force every downstream consumer of libspectrum to also
    # have wgpu-native on its include path. The header is consumed only
    # by spectrum's own translation units via a relative include.
else()
    message(STATUS "ThreeGppChannelModel GPU back-end: DISABLED (NS3_ENABLE_3GPP_GPU=OFF)")
endif()

# Library link list — wgpu_native is added when the GPU back-end is enabled.
set(spectrum_link_libs
    ${libpropagation}
    ${libantenna}
    ${libbuildings}
)
if(NS3_ENABLE_3GPP_GPU)
    list(APPEND spectrum_link_libs ${WEBGPU_LOCATION})
endif()

build_lib(
  LIBNAME spectrum
  SOURCE_FILES ${source_files}
  HEADER_FILES ${header_files}
  LIBRARIES_TO_LINK
    ${spectrum_link_libs}
  TEST_SOURCES
    test/spectrum-ideal-phy-test.cc
    test/spectrum-interference-test.cc
    test/spectrum-value-test.cc
    test/spectrum-waveform-generator-test.cc
    test/three-gpp-channel-test-suite.cc
    test/tv-helper-distribution-test.cc
    test/tv-spectrum-transmitter-test.cc
    test/two-ray-splm-test-suite.cc
    test/wraparound-model-test.cc
)

if(NS3_ENABLE_3GPP_GPU)
    # wgpu-native include dirs and defines are only needed when compiling
    # spectrum's own .cc files (sls-chan-wgpu.cc, three-gpp-channel-model.cc).
    # Wrap them in BUILD_INTERFACE so they don't leak into the install
    # interface and CMake doesn't complain about exposing build-tree paths.
    set(_wgpu_build_includes "")
    foreach(_inc IN LISTS WEBGPU_INCLUDES)
        list(APPEND _wgpu_build_includes "$<BUILD_INTERFACE:${_inc}>")
    endforeach()
    target_include_directories(${libspectrum} PRIVATE ${_wgpu_build_includes})
    target_compile_definitions(${libspectrum} PRIVATE
        ${WEBGPU_DEFS}
        NS3_ENABLE_3GPP_GPU=1
    )
    # Make sure the wgpu_native runtime ends up next to the spectrum library
    # so executables that link against it can find it at run-time.
    target_copy_webgpu_binaries(${libspectrum})
endif()

# ── Standalone SLS-channel test / validation binaries ──────────────────────
# These compile their own copy of sls-chan-wgpu.cc rather than linking
# against the spectrum library. The reason is `saveSlsChanToHdf5` lives in
# sls-chan-wgpu.cc behind `#ifdef SLS_CHAN_HDF5`: the spectrum library
# does not define that (to avoid forcing HDF5 onto every consumer), so the
# library's compiled object lacks the symbol the validation binary needs.
# Recompiling the .cc locally with SLS_CHAN_HDF5 also pulls the rest of
# SlsChanWgpu in — slightly wasteful at build time, but completely
# decouples the validation harness from the library's compile flags.
if(NS3_ENABLE_3GPP_GPU)
    add_executable(sls_chan_test examples/sls-chan-example.cc model/sls-chan-wgpu.cc)
    target_link_libraries(sls_chan_test PRIVATE ${WEBGPU_LOCATION})
    target_include_directories(sls_chan_test PRIVATE model/ ${WEBGPU_INCLUDES})
    target_compile_definitions(sls_chan_test PRIVATE ${WEBGPU_DEFS})

    if(${HDF5_FOUND})
        add_executable(sls_chan_validation model/sls-chan-validation.cc model/sls-chan-wgpu.cc)
        target_link_libraries(sls_chan_validation PRIVATE ${WEBGPU_LOCATION} HDF5::HDF5)
        target_include_directories(sls_chan_validation PRIVATE model/ ${WEBGPU_INCLUDES})
        target_compile_definitions(sls_chan_validation PRIVATE ${WEBGPU_DEFS} SLS_CHAN_HDF5)
        target_copy_webgpu_binaries(sls_chan_validation)
    endif()

    # Tiny end-to-end smoke test for the channel model's UseGpu attribute.
    # Links directly against the spectrum DLL + its deps (no test-runner
    # dependency, which currently fails to link on Windows-MinGW due to a
    # wifi DLL export-ordinal overflow). Run it after configure/build:
    #
    #     ./build/gpu_batch_smoke.exe
    #
    # Should print `gpu_batch_smoke OK` if the GPU integration is healthy.
    add_executable(gpu_batch_smoke model/gpu-batch-smoke.cc)
    target_link_libraries(gpu_batch_smoke PRIVATE
        ${libspectrum}
        ${libcore}
        ${libnetwork}
        ${libmobility}
        ${libpropagation}
        ${libantenna}
    )
endif()
