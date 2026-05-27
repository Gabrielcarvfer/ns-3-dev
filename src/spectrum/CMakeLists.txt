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
option(NS3_SLS_PROFILE
       "Compile spectrum lib with sls phase-timing instrumentation"
       OFF)

# Fetch prebuilt wgpu-native (or Dawn) binaries — must happen BEFORE
# build_lib so the spectrum library target can link against the
# imported `webgpu` target. Two knobs:
#   - WEBGPU_BACKEND  WGPU (default) or DAWN. Picks the underlying
#                     WebGPU implementation. Dawn is Chrome's; the
#                     `wgpu` here is gfx-rs/wgpu-native.
#   - WGPU_VERSION    Tag name of a gfx-rs/wgpu-native release.
#                     Only consulted when WEBGPU_BACKEND=WGPU.
#                     Warning: eliemichel's webgpu.hpp wrapper is
#                     tuned to a specific wgpu-native version; major
#                     upgrades may need a wrapper regeneration.
include(FetchContent)
set(WEBGPU_LINK_TYPE SHARED)
set(WEBGPU_BACKEND "WGPU" CACHE STRING "WebGPU implementation backend (WGPU or DAWN)" FORCE)
# wgpu-native is pinned to v24.0.3.1 because eliemichel's C++ wrapper
# (webgpu.hpp) is generated against v24's C API; newer wgpu-native
# releases (v25+) removed types the wrapper references (e.g.
# WGPUProgrammableStageDescriptor) and the wrapper itself doesn't get
# regenerated for each version. To bump beyond this, either:
#   - regenerate the wrapper from the matching webgpu.h (out of scope
#     here), or
#   - switch to WEBGPU_BACKEND=DAWN, which ships its own current
#     wrapper.
set(WGPU_VERSION "v24.0.3.1" CACHE STRING "gfx-rs/wgpu-native release tag (when WEBGPU_BACKEND=WGPU)" FORCE)
FetchContent_Declare(webgpu
        GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
        GIT_TAG        main   # pin to a release tag in production
)
FetchContent_MakeAvailable(webgpu)

get_target_property(WEBGPU_INCLUDES  webgpu INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(WEBGPU_DEFS      webgpu INTERFACE_COMPILE_DEFINITIONS)
# WEBGPU_LOCATION is set differently per backend:
#   - wgpu-native: the `webgpu` target IS the imported wgpu_native.dll,
#     so `webgpu`'s IMPORTED_LOCATION is the .dll path.
#   - Dawn: `webgpu` is an INTERFACE target that links to
#     `dawn::webgpu_dawn`; the actual .dll path lives on the inner
#     target as IMPORTED_LOCATION_RELEASE. The eliemichel module
#     publishes that as a CACHE INTERNAL `WEBGPU_RUNTIME_LIB`.
# We grab whichever is non-empty so downstream link rules work for
# both backends.
get_target_property(WEBGPU_LOCATION  webgpu IMPORTED_LOCATION)
if(NOT WEBGPU_LOCATION OR WEBGPU_LOCATION STREQUAL "WEBGPU_LOCATION-NOTFOUND")
    set(WEBGPU_LOCATION "${WEBGPU_RUNTIME_LIB}")
endif()

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
# Note: the webgpu link dependency is added below via
# `target_link_libraries(${libspectrum} PRIVATE webgpu)` rather than
# being threaded through `LIBRARIES_TO_LINK`. ns-3's build_lib emits
# install(EXPORT ...) rules that reject targets which aren't in its
# export set, and the `webgpu` interface target from FetchContent
# isn't (and shouldn't be) installed.

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
    if(NS3_SLS_PROFILE)
        target_compile_definitions(${libspectrum} PRIVATE SLS_PROFILE_INSTRUMENT=1)
    endif()
    # PRIVATE so the link doesn't leak into ns-3's install export set.
    target_link_libraries(${libspectrum} PRIVATE webgpu)
    # Make sure the wgpu_native/webgpu_dawn runtime ends up next to the
    # spectrum library so executables that link against it can find it
    # at run-time.
    target_copy_webgpu_binaries(${libspectrum})

    # Dawn's D3D12 backend additionally needs the DirectX Shader
    # Compiler (`dxcompiler.dll` + `dxil.dll`) at run-time. Windows SDK
    # ships them at `Windows Kits\10\bin\<version>\x64\`. Copy the
    # first version found next to the spectrum DLL so Dawn-D3D12 works
    # out of the box; if missing, Dawn will fall back to Vulkan via
    # SLS_BACKEND or fail with a clear error.
    if(WEBGPU_BACKEND STREQUAL "DAWN" AND WIN32)
        file(GLOB _dxc_candidates
             "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxcompiler.dll")
        list(LENGTH _dxc_candidates _dxc_n)
        if(_dxc_n GREATER 0)
            list(GET _dxc_candidates 0 _dxc_dll)
            get_filename_component(_dxc_dir "${_dxc_dll}" DIRECTORY)
            message(STATUS "Copying DXC from ${_dxc_dir}")
            configure_file("${_dxc_dir}/dxcompiler.dll"
                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib/dxcompiler.dll"
                           COPYONLY)
            configure_file("${_dxc_dir}/dxil.dll"
                           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib/dxil.dll"
                           COPYONLY)
        else()
            message(WARNING "Dawn D3D12 needs dxcompiler.dll + dxil.dll at "
                            "run-time but none were found under Windows Kits. "
                            "D3D12 backend will fail until they're available.")
        endif()
    endif()

    # When HDF5 is available, compile the SLS-channel GPU back-end with
    # SLS_CHAN_HDF5 so its saveSlsChanToHdf5() member is available to
    # ns-3 callers (e.g. for re-running the outdoor calibration on a
    # batch produced by ThreeGppChannelModel). Without this define the
    # member compiles down to an undefined symbol.
    if(${HDF5_FOUND})
        target_compile_definitions(${libspectrum} PRIVATE SLS_CHAN_HDF5)
        target_link_libraries(${libspectrum} PRIVATE HDF5::HDF5)
    endif()
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
    target_link_libraries(sls_chan_test PRIVATE webgpu)
    target_include_directories(sls_chan_test PRIVATE model/ ${WEBGPU_INCLUDES})
    target_compile_definitions(sls_chan_test PRIVATE ${WEBGPU_DEFS})

    if(${HDF5_FOUND})
        add_executable(sls_chan_validation model/sls-chan-validation.cc model/sls-chan-wgpu.cc)
        target_link_libraries(sls_chan_validation PRIVATE webgpu HDF5::HDF5)
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

    # Profiling harness for the 3GPP channel + spectrum propagation
    # loss models. Standalone binary; the timing macros are compiled
    # INTO the spectrum library when `NS3_SLS_PROFILE` is on (see the
    # top-level option below). Default is off, so the macros are
    # no-ops in normal builds.
    add_executable(spectrum_profile_bench model/spectrum-profile-bench.cc)
    target_link_libraries(spectrum_profile_bench PRIVATE
        ${libspectrum}
        ${libcore}
        ${libnetwork}
        ${libmobility}
        ${libpropagation}
        ${libantenna}
        ${liblte}
    )
endif()
