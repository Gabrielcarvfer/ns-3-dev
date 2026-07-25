build_lib_example(
  NAME aodv-example
  SOURCE_FILES aodv-example.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libinternet}
    ${libaodv}
    ${libinternet-apps}
)

build_lib_example(
  NAME aodv-dense-scaling
  SOURCE_FILES aodv-dense-scaling.cc
  LIBRARIES_TO_LINK
    ${libaodv}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)
