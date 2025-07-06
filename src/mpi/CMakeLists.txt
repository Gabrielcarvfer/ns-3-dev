set(example_as_test_suite)
if(${ENABLE_EXAMPLES})
  set(example_as_test_suite
      test/mpi-test-suite.cc
  )
endif()

set(mtp_sources)
set(mtp_libraries)
if(${ENABLE_MTP})
  set(mtp_sources
      model/hybrid-simulator-impl.cc
  )
  set(mtp_libraries
      ${libmtp}
  )
endif()

build_lib(
  LIBNAME mpi
  SOURCE_FILES
    ${mtp_sources}
    model/distributed-simulator-impl.cc
    model/granted-time-window-mpi-interface.cc
    model/mpi-interface.cc
    model/mpi-receiver.cc
    model/null-message-mpi-interface.cc
    model/null-message-simulator-impl.cc
    model/parallel-communication-interface.h
    model/remote-channel-bundle-manager.cc
    model/remote-channel-bundle.cc
  HEADER_FILES
    model/mpi-interface.h
    model/mpi-receiver.h
    model/parallel-communication-interface.h
  LIBRARIES_TO_LINK
    ${libnetwork}
    ${mtp_libraries}
    MPI::MPI_CXX
  TEST_SOURCES ${example_as_test_suite}
)
