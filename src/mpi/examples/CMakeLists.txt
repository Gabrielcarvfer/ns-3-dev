set(base_examples
    simple-distributed
    simple-distributed-mpi-comm
    nms-p2p-nix-distributed
    simple-distributed-empty-node
)

foreach(
  example
  ${base_examples}
)
  build_lib_example(
    NAME ${example}
    SOURCE_FILES ${example}.cc
                 mpi-test-fixtures.cc
    LIBRARIES_TO_LINK
      ${libmpi}
      ${libpoint-to-point}
      ${libinternet}
      ${libnix-vector-routing}
      ${libapplications}
  )
endforeach()

build_lib_example(
  NAME third-distributed
  SOURCE_FILES third-distributed.cc
               mpi-test-fixtures.cc
  LIBRARIES_TO_LINK
    ${libmpi}
    ${libpoint-to-point}
    ${libinternet}
    ${libmobility}
    ${libwifi}
    ${libcsma}
    ${libapplications}
)
