set(dsr_sources)

# cmake-format: off
if((dsr IN_LIST ns3-all-enabled-modules)
    AND (uan IN_LIST ns3-all-enabled-modules)
    AND (olsr IN_LIST ns3-all-enabled-modules)
    AND (lte IN_LIST ns3-all-enabled-modules))
  set(dsr_sources traced/traced-callback-typedef-test-suite.cc)
endif()
# cmake-format: on

set(csma_sources)
if(applications
   IN_LIST
   ns3-all-enabled-modules
)
  if((csma-layout
      IN_LIST
      ns3-all-enabled-modules
     )
     AND (internet-apps
          IN_LIST
          ns3-all-enabled-modules
         )
  )
    set(csma_sources
        csma-system-test-suite.cc
    )
  endif()
endif()

set(network_sources)
if(network
   IN_LIST
   ns3-all-enabled-modules
)
  set(network_sources
      traced/traced-value-callback-typedef-test-suite.cc
  )
endif()

set(traffic-control_sources)
if(traffic-control
   IN_LIST
   ns3-all-enabled-modules
)
  set(traffic-control_sources
      ns3tc/fq-cobalt-queue-disc-test-suite.cc
      ns3tc/fq-codel-queue-disc-test-suite.cc
      ns3tc/fq-pie-queue-disc-test-suite.cc
      ns3tc/pfifo-fast-queue-disc-test-suite.cc
  )
endif()

add_library(
  test
  OBJECT
  ${PROJECT_SOURCE_DIR}/build-support/empty.cc # empty source file if only
  # libcore is enabled
  ${csma_sources}
  ${dsr_sources}
  ${network_sources}
  ${traffic-control_sources}
)

add_dependencies(
  ${libtest}
  copy_all_headers
)
