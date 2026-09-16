build_lib_example(
  NAME simple-mtp
  SOURCE_FILES simple-mtp.cc
  LIBRARIES_TO_LINK
    ${libmtp}
    ${libpoint-to-point}
    ${libinternet}
    ${libnix-vector-routing}
    ${libapplications}
)

build_lib_example(
  NAME fat-tree-mtp
  SOURCE_FILES fat-tree-mtp.cc
  LIBRARIES_TO_LINK
    ${libmtp}
    ${libpoint-to-point}
    ${libinternet}
    ${libnix-vector-routing}
    ${libapplications}
    ${libflow-monitor}
)

build_lib_example(
  NAME wired-wireless-mtp
  SOURCE_FILES wired-wireless-mtp.cc
  LIBRARIES_TO_LINK
    ${libmtp}
    ${libpoint-to-point}
    ${libcsma}
    ${libwifi}
    ${libmobility}
    ${libinternet}
    ${libapplications}
    ${libflow-monitor}
)
