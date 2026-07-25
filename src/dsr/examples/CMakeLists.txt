build_lib_example(
  NAME dsr-example
  SOURCE_FILES dsr-example.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libapplications}
    ${libmobility}
    ${libconfig-store}
    ${libwifi}
    ${libdsr}
)

build_lib_example(
  NAME dsr-dense-scaling
  SOURCE_FILES dsr-dense-scaling.cc
  LIBRARIES_TO_LINK
    ${libdsr}
    ${libcsma}
    ${libinternet}
    ${libapplications}
)
