add_library(
  test
  OBJECT
  ${PROJECT_SOURCE_DIR}/build-support/empty.cc # empty source file if only
  # libcore is enabled
)

add_dependencies(
  ${libtest}
  copy_all_headers
)
