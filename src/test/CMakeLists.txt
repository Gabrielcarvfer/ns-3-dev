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

add_library(
  test
  OBJECT
  ${PROJECT_SOURCE_DIR}/build-support/empty.cc # empty source file if only
  # libcore is enabled
  ${csma_sources}
)

add_dependencies(
  ${libtest}
  copy_all_headers
)
