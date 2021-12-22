.. include:: replace.txt
.. highlight:: bash

.. Section Separators:
   ----
   ****
   ++++
   ====
   ~~~~

.. _Working with CMake:

Working with CMake
------------------

The ns-3 project used Waf build system in the past, but it has moved to CMake in late 2021.

The steps performed on a typical workflow are the following: 

1. Fetch sources
2. Configure the CMake project
3. Modify files
4. Manually refresh the CMake cache (only needed if adding new modules)
5. CMake cache gets automatically refreshed (after changes to CMakeLists.txt, .cmake and .h files)
6. Build and debug targets

Fetch sources
*************

Source code can be fetched by using the clone command from :ref:`git<Directly cloning ns-3-dev>`.

.. sourcecode:: bash

  ~$ git clone https://gitlab.com/nsnam/ns-3-dev

This will write the contents of the ns-3-dev repository into the ns-3-dev folder. 
If you are working on multiple features concurrently and prefer to keep different repositories, you 
can specify a different output directory.

.. sourcecode:: bash

  ~$ git clone https://gitlab.com/nsnam/ns-3-dev another-ns-3-dev

Configure the project
*********************

Navigate to the cloned ns-3-dev folder, create a CMake cache folder, 
navigate to it and run `CMake`_ pointing to the ns-3-dev folder.

.. sourcecode:: bash

  ~$ cd ns-3-dev
  ~/ns-3-dev$ mkdir cmake_cache
  ~/ns-3-dev$ cd cmake_cache
  ~/ns-3-dev/cmake_cache$ cmake ..

You can pass additional arguments to the CMake command, to configure it. To change variable values,
you should use the -D option followed by the variable name.

As an example, the build type is stored in the variable named `CMAKE_BUILD_TYPE`_. Setting it to one 
of the CMake build types shown in the table below will change compiler settings associated with those
build types and output executable and libraries names, which will receive a suffix.

+------------------+-------------------------------------------------------------------------------------------+
| CMAKE_BUILD_TYPE | `Effects (g++)`_                                                                          |
+==================+===========================================================================================+
| DEBUG            | -g                                                                                        |
+------------------+-------------------------------------------------------------------------------------------+
| RELEASE          | -O3 -DNDEBUG                                                                              |
+------------------+-------------------------------------------------------------------------------------------+
| RELWITHDEBINFO   | -O2 -g -DNDEBUG                                                                           |
+------------------+-------------------------------------------------------------------------------------------+
| MINSIZEREL       | -Os -DNDEBUG                                                                              |
+------------------+-------------------------------------------------------------------------------------------+

You can set the build type with the following command, which assumes your terminal is inside the cache folder
created previously.

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ cmake -DCMAKE_BUILD_TYPE=DEBUG ..

Another common option to change is the `generator`_, which is the real underlying build system called by CMake.
There are many generators supported by CMake, including the ones listed in the table below.

+------------------------------------------------+
| Generators                                     |
+================================================+
| MinGW Makefiles                                |
+------------------------------------------------+
| Unix Makefiles                                 |
+------------------------------------------------+
| MSYS Makefiles                                 |
+------------------------------------------------+
| CodeBlocks - *one of the previous Makefiles*   |
+------------------------------------------------+
| Eclipse CDT4 - *one of the previous Makefiles* |
+------------------------------------------------+
| Ninja                                          |
+------------------------------------------------+
| Xcode                                          |
+------------------------------------------------+

To change the generator, you will need to pass one of these generators with the -G option. For example, if we 
prefer Ninja to Makefiles, which are the default, we need to run the following command:

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ cmake -G Ninja ..

This command may fail if there are different generator files in the same CMake cache folder. It is recommended to clean up
the CMake cache folder, then recreate it and reconfigure setting the generator in the first run.

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ rm -R ./*
  ~/ns-3-dev/cmake_cache$ cmake -DCMAKE_BUILD_TYPE=release -G Ninja ..

After the project has been configured, calling CMake will :ref:`refresh the CMake cache<Manually refresh the CMake cache>`. 
The refresh is required to discover newer targets (libraries, executables and/or modules that were created since the last run)

.. _CMake: https://cmake.org/cmake/help/latest/manual/cmake.1.html
.. _CMAKE_BUILD_TYPE: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
.. _Effects (g++): https://github.com/Kitware/CMake/blob/master/Modules/Compiler/GNU.cmake
.. _generator: https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html


Modifying files
***************

This is the main step of development, where something within the project is modified.
We group modifications into three different groups, based on how they affect the CMake project.

Adding a new module
+++++++++++++++++++

Adding a module is the only case where 
:ref:`manually refreshing the CMake cache<Manually refresh the CMake cache>` is required. 

More information on how to create a new module are provided in :ref:`Adding a New Module to ns3`.

CMake (CMakeLists.txt, .cmake) and header files (.h)
++++++++++++++++++++++++++++++++++++++++++++++++++++

All of the following modifications will trigger an automatic refresh of CMake.

This group modifications include:

#. API changes and related header changes: this is unrelated CMake, so we do not detail this case
#. Module name changes
#. File name changes and updates to the CMakeLists.txt source_files and header_files lists
#. Changes in dependencies of modules, either local dependency to another ns-3 module or external dependency to a third-party library
#. Inclusion of option switches for a module
#. Changes in CMake macros and functions, minimum required compiler version, 

The following sections will detail some of this cases assuming a hypothetical module defined below.
A ns-3 module contains a few variables and a call to either the build_lib or build_contrib_lib macros.

.. sourcecode:: cmake

    set(name hypothetical)

    set(source_files
        helper/hypothetical-helper.cc
        model/hypothetical.cc
    )

    set(header_files
        helper/hypothetical-helper.h
        model/hypothetical.h
        model/colliding-header.h
    )

    set(libraries_to_link ${libcore})

    set(test_sources)

    build_lib("${name}" 
              "${source_files}"
              "${header_files}"
              "${libraries_to_link}"
              "${test_sources}"
              )


Module name changes
======================================================

If the module were already scanned, changing the module name just requires changing the value of ${name}.

.. sourcecode:: cmake

    set(name new-hypothetical-name)

If the module was not already scanned, a :ref:`manual refresh<Manually refresh the CMake cache>`
will be required after making the change.


File name changes and subsequent CMakeLists.txt update
======================================================

Assuming the hypothetical module defined previously has a header name that collides
with the header of a different module.

The name of the colliding-header.h can be changed via the filesystem to 
non-colliding-header.h, and the CMakeLists.txt path needs to be updated to match
the new name. Some IDEs can do that automatically.

.. sourcecode:: cmake

    set(header_files
        helper/hypothetical-helper.h
        model/hypothetical.h
        model/non-colliding-header.h
    )


Changes in dependencies of modules
==================================

Adding a dependency to another ns-3 module just requires adding ${lib${modulename}} 
to the libraries_to_link list, where modulename contains the value of the ns-3 
module which will be depended upon.

.. sourcecode:: cmake

    # now ${libhypothetical} will depend on both core and internet modules
    set(libraries_to_link ${libcore} ${libinternet}) 

Depending on a third-party library is a bit more complicated as we have multiple
ways to handle within CMake.

Search for third-party libraries in a given directory passed by the command-line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we depend on a library passed by a path via the command-line configuration,
we need to check if it exists first.

.. sourcecode:: cmake

    set(NS3_WITH_THIRDPARTYLIB "" CACHE PATH 
      "Build with the third-party library support")
    set(NS3_THIRDPARTYLIBRARY "OFF" CACHE INTERNAL 
      "ON if the third-party library is found in NS3_WITH_THIRDPARTYLIB")

    # If a path for the third-party library was not given, we just skip 
    # the entire module and pretend it does not exist by going back to 
    # parent folder (src or contrib)
    if(NOT NS3_WITH_THIRDPARTYLIB)
      return() 
    endif()

    # If a path for the third-party library was given, we search for both 
    # the library and the header in the NS3_WITH_THIRDPARTYLIB folder
    find_library(third_party_dependency_library 
                 known_third_party_library_name
                 PATHS ${NS3_WITH_THIRDPARTYLIB}
                 PATH_SUFFIXES /build /build/lib /lib
                 )
    find_file(third_party_dependency_header 
              known_third_party_header_name
              HINTS ${NS3_WITH_THIRDPARTYLIB}
              PATH_SUFFIXES /build /build/include /include
              )

    # If both are not found, we return a message indicating it is the
    #     case and stop processing this module by returning to the src folder
    if(NOT (third_party_dependency_library AND third_party_dependency_header))
      message(STATUS 
        "Third-party dependency was not found in ${NS3_WITH_THIRDPARTYLIB}")
      return()
    endif()

    # If both were found, we get the directory containing
    #     the third-party header and use it as an include folder
    get_filename_component(third_party_header_include_folder 
      ${third_party_dependency_header} DIRECTORY)

    # scan for headers in the folder containing the dependency header
    include_directories(${third_party_header_include_folder}) 

    # We also set the NS3_THIRDPARTYLIBRARY variable, indicating it was found
    set(NS3_THIRDPARTYLIBRARY "ON" CACHE INTERNAL 
      "ON if the third-party library is found in NS3_WITH_THIRDPARTYLIB")

    # ... define of module name and create source and headers lists...

    # Now we can link to the third_party_dependency 
    # by adding it to the libraries_to_link list
    set(libraries_to_link ${libcore} ${third_party_dependency_library})

    # ... create list of test sources for the module and call build_lib macro


Search for third-party libraries using CMake's find_package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume we have a module with optional features that rely on a third-party library
that provides a FindThirdPartyPackage.cmake. This Find.cmake file can be distributed
by `CMake itself`_, via library/package managers (APT, Pacman, 
`VcPkg`_), or included to the project tree in the buildsupport/3rd_party directory.

.. _CMake itself: https://github.com/Kitware/CMake/tree/master/Modules
.. _Vcpkg: https://github.com/Microsoft/vcpkg#using-vcpkg-with-cmake

.. sourcecode:: cmake

    # First we search for the package
    find_package(ThirdPartyPackage QUIET)
    if(NOT ${ThirdPartyPackage_FOUND})
      message(STATUS "ThirdPartyPackage was not found. Continuing without it.")
    else()
      message(STATUS "ThirdPartyPackage was found.")

      # Set list of source files that depend on the third-party package
      set(ThirdPartyPackage_sources model/optional-feature.cc)

      # Set a list of libraries exported by the third-party package
      set(ThirdPartyPackage_libraries ${ThirdPartyPackage_LIBRARIES})

      # Add include dir for current module and its examples
      include_directories(${ThirdPartyPackage_INCLUDE_DIR}) 
    endif()

    set(name hypothetical)


    # Then add optional source files and libraries to link
    set(source_files
        helper/hypothetical-helper.cc
        model/hypothetical.cc
        # This list will be empty or contain model/optional-feature.cc
        ${ThirdPartyPackage_sources} 
    )

    # ... create headers list...

    # link to libcore and the third-party libraries if they are found
    set(libraries_to_link ${libcore} ${ThirdPartyPackage_libraries}) 

    # ... create list of test sources for the module and call build_lib macro



Search for third-party libraries using PkgConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume we have a module with optional features that rely on a third-party library
that uses PkgConfig.

.. sourcecode:: cmake

    # Include CMake script to use pkg-config
    include(FindPkgConfig)

    # If pkg-config was found, search for library you want
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(THIRD_PARTY libthird-party)
    endif()

    # Set cached variable if both pkg-config and libthird-party are found
    if(PKG_CONFIG_FOUND AND THIRD_PARTY)
      include_directories(${THIRD_PARTY_INCLUDE_DIRS})
      set(third_party_libs ${THIRD_PARTY_LIBRARIES})
      set(third_party_sources model/optional-feature.cc)
    endif()

    set(name hypothetical)

    # Then add optional source files and libraries to link
    set(source_files
        helper/hypothetical-helper.cc
        model/hypothetical.cc
        # This list will be empty or contain model/optional-feature.cc
        ${third_party_sources} 
    )

    # ... create headers list...

    # link to libcore and the third-party libraries if they are found
    set(libraries_to_link ${libcore} ${third_party_libs}) 

    # ... create list of test sources for the module and call build_lib macro

    

Inclusion of options
~~~~~~~~~~~~~~~~~~~~

There are two ways of managing module options: option switches or cached variables.
Both are present in the main CMakeLists.txt in the ns-3-dev directory and the 
buildsupport/macros_and_definitions.cmake file.


.. sourcecode:: cmake

    # Here are examples of ON and OFF switches
    # option(
    #        NS3_SWITCH # option switch prefixed with NS3\_
    #        "followed by the description of what the option does" 
    #        ON # and the default value for that option
    #        )
    option(NS3_EXAMPLES "Enable examples to be built" OFF)
    option(NS3_TESTS "Enable tests to be built" OFF)

    # Now here is how to let the user indicate a path
    # set( # declares a value
    #     NS3_PREFIXED_VALUE # stores the option value
    #     "" # default value is empty in this case
    #     CACHE # stores that NS3_PREFIXED_VALUE in the CMakeCache.txt file
    #     STRING # type of the cached variable
    #     "description of what this value is used for"
    #     )
    set(NS3_OUTPUT_DIRECTORY "" CACHE PATH "Directory to store built artifacts")

    # The last case are options that can only assume predefined values
    # First we cache different values for that variable
    set(NS3_INT64X64 "INT128" CACHE STRING "Int64x64 implementation")
    set(NS3_INT64X64 "CAIRO" CACHE STRING "Int64x64 implementation")
    set(NS3_INT64X64 "DOUBLE" CACHE STRING "Int64x64 implementation")

    # Then set a cache property for the variable indicating it can assume 
    # specific values
    set_property(CACHE NS3_INT64X64 PROPERTY STRINGS INT128 CAIRO DOUBLE)


More details about these commands can be found in the following links:
`option`_, `set`_, `set_property`_.

.. _option: https://cmake.org/cmake/help/latest/command/option.html
.. _set: https://cmake.org/cmake/help/latest/command/set.html
.. _set_property: https://cmake.org/cmake/help/latest/command/set_property.html


Changes in CMake macros and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order for CMake to feel more familiar to Waf users, a few macros and functions
were created. 

Most of the mostly frequently used macros them can be found in 
buildsupport/macros_and_definitions.cmake. This file includes build type checking, 
compiler family and version checking, enabling and disabling features based
on user options, checking for dependencies of enabled features, 
pre-compiling headers, filtering enabled/disabled modules and dependencies,
and more.

Here are examples of how to do the option and header checking, 
followed by a header configuration:

.. sourcecode:: cmake

  # We always set the ENABLE\_ counterpart of NS\_ option to FALSE before checking
  set(ENABLE_MPI FALSE)

  # If the user option switch is set to ON, we check
  if(${NS3_MPI})
    # Use find_package to look for MPI
    find_package(MPI QUIET)

    # If the package is optional, which is the case for MPI,
    # we can proceed if it is not found
    if(NOT ${MPI_FOUND})
      message(STATUS "MPI was not found. Continuing without it.")
    else()
      # If it is false, we add necessary C++ definitions (e.g. NS3_MPI)
      message(STATUS "MPI was found.")
      add_definitions(-DNS3_MPI)

      # Then set ENABLE_MPI to TRUE, which can be used to check 
      # if NS3_MPI is enabled AND MPI was found
      set(ENABLE_MPI TRUE)
    endif()
  endif()

  # ...

  # These two standard CMake modules allow for header and function checking
  include(CheckIncludeFileCXX)
  include(CheckFunctionExists)

  # Check for required headers and functions, 
  # set flags on the right argument if header in the first argument is found
  # if they are not found, a warning is emitted
  check_include_file_cxx("stdint.h" "HAVE_STDINT_H")
  check_include_file_cxx("inttypes.h" "HAVE_INTTYPES_H")
  check_include_file_cxx("sys/types.h" "HAVE_SYS_TYPES_H")
  check_include_file_cxx("stat.h" "HAVE_SYS_STAT_H")
  check_include_file_cxx("dirent.h" "HAVE_DIRENT_H")
  check_include_file_cxx("stdlib.h" "HAVE_STDLIB_H")
  check_include_file_cxx("signal.h" "HAVE_SIGNAL_H")
  check_include_file_cxx("netpacket/packet.h" "HAVE_PACKETH")
  check_function_exists("getenv" "HAVE_GETENV")

  # This is the CMake command to open up a file template (in this case a header
  # passed as the first argument), then fill its fields with values stored in 
  # CMake variables and save the resulting file to the target destination 
  # (in the second argument)
  configure_file(
    buildsupport/core-config-template.h
    ${CMAKE_HEADER_OUTPUT_DIRECTORY}/core-config.h
  )

The configure_file command is not very clear by itself, as you do not know which
values are being used. So we need to check the template.

.. sourcecode:: cpp

    #ifndef NS3_CORE_CONFIG_H
    #define NS3_CORE_CONFIG_H

    #cmakedefine   HAVE_UINT128_T
    #cmakedefine01 HAVE___UINT128_T
    #cmakedefine   INT64X64_USE_128
    #cmakedefine   INT64X64_USE_DOUBLE
    #cmakedefine   INT64X64_USE_CAIRO
    #cmakedefine01 HAVE_STDINT_H
    #cmakedefine01 HAVE_INTTYPES_H
    #cmakedefine   HAVE_SYS_INT_TYPES_H
    #cmakedefine01 HAVE_SYS_TYPES_H
    #cmakedefine01 HAVE_SYS_STAT_H
    #cmakedefine01 HAVE_DIRENT_H
    #cmakedefine01 HAVE_STDLIB_H
    #cmakedefine01 HAVE_GETENV
    #cmakedefine01 HAVE_SIGNAL_H
    #cmakedefine   HAVE_PTHREAD_H
    #cmakedefine   HAVE_RT

    /*
    * #cmakedefine turns into:
    * //#define HAVE_FLAG // if HAVE_FLAG is not defined in CMake (e.g. unset(HAVE_FLAG))
    * #define HAVE_FLAG // if HAVE_FLAG is defined in CMake (e.g. set(HAVE_FLAG))
    *
    * #cmakedefine01 turns into:
    * #define HAVE_FLAG 0 // if HAVE_FLAG is not defined in CMake
    * #define HAVE_FLAG 1 // if HAVE_FLAG is defined in CMake
    */

    #endif //NS3_CORE_CONFIG_H

Another common thing to implement are custom targets to run specific commands and
manage dependencies. Here is an example for Doxygen:

.. sourcecode:: cmake

  # if the user enabled NS3_DOCS
  if(${NS3_DOCS}) 
    # hide variables from ccmake
    mark_as_advanced(DOXYGEN DOT DIA)

    # fail if doxygen is not found, and store path to it in DOXYGEN
    find_program(DOXYGEN doxygen REQUIRED) 
    find_program(DOT dot)
    find_program(DIA dia)
    if((NOT DOT) OR (NOT DIA))
      message(
        FATAL_ERROR
          "Dot and Dia are required by Doxygen docs."
          "They're shipped within the graphviz and dia packages on Ubuntu"
      )
    endif()

    # Run the print-introspected-doxygen program from utils folder and 
    # save the output to introspected-doxygen.h in the ns-3-dev/doc folder
    add_custom_target(
      run-print-introspected-doxygen
      COMMAND
        ${CMAKE_OUTPUT_DIRECTORY}/utils/ns${NS3_VER}-print-introspected-doxygen${build_profile_suffix}
        print-introspected-doxygen >
        ${PROJECT_SOURCE_DIR}/doc/introspected-doxygen.h
      DEPENDS print-introspected-doxygen # notice it depends on print-introspected-doxygen target
    )

    # Add the run-introspected-command-line target, which
    # runs test.py to get the introspected command-line 
    add_custom_target(
      run-introspected-command-line
      COMMAND ${CMAKE_COMMAND} -E env NS_COMMANDLINE_INTROSPECTION=..
              ${Python3_EXECUTABLE} ./test.py --nowaf --constrain=example
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS all-test-targets # notice it depends on all targets, 
                               # which only exists if ENABLE_TESTS is
                               # set to ON
    )
    
    # Assemble the introspected-command-line
    file(
      WRITE ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      "/* This file is automatically generated by
        * CommandLine::PrintDoxygenUsage() from the CommandLine configuration
        * in various example programs.  Do not edit this file!  Edit the
        * CommandLine configuration in those files instead.
        */
        \n"
    )

    # Add the assemble-introspected-command-line target, which
    # appends all command-line outputs from run-introspected-command-line into
    # ns-3-dev/doc/introspected-command-line.h
    add_custom_target(
      assemble-introspected-command-line
      # works on CMake 3.18 or newer > COMMAND ${CMAKE_COMMAND} -E cat
      # ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line >
      # ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      COMMAND ${cat_command} ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line
              > ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h 2> NULL
      DEPENDS run-introspected-command-line
    )

    # Add the doxygen custom target, which depends on both introspected 
    # doxygen and command-line, and then runs doxygen after they are completed
    add_custom_target(
      doxygen
      COMMAND ${DOXYGEN} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS run-print-introspected-doxygen assemble-introspected-command-line
    )

    # Add a custom doxygen target that only runs doxygen
    add_custom_target(
      doxygen-no-build COMMAND ${DOXYGEN} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endif()

Module macros are located in buildsupport/custom_modules/ns3_module_macros.cmake.
This file contains macros defining a library, the associated test library,
examples and more. It also contains the macro that builds the module header 
that includes all headers from the module for user scripts.

The module macros get specialized to src and contrib modules, which are addressed
slightly differently in order to ensure everything during configuration time works.

Source files (.cc)
++++++++++++++++++

Changes to .cc files should not trigger an automatic refresh unless it 
is part of an API change.


.. _Manually refresh the CMake cache:

Manually refresh the CMake cache
********************************

The refresh is done by running the CMake command from the CMake cache folder. 

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ cmake ..

Previous settings stored in the CMakeCache.txt will be preserved, while new modules will be
scanned and targets will be added.

Build and debug targets
***********************

The build process of targets (either libraries, executables or custom tasks) can be done
invoking CMake build. To build all the targets, run:

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ cmake --build . 

Notice the single dot now refers to the cmake_cache directory, where the underlying 
build system files are stored (referred inside CMake as `PROJECT_BINARY_DIR` or 
`CMAKE_BINARY_DIR`, which have slightly different uses if working with sub-projects).

.. _PROJECT_BINARY_DIR: https://cmake.org/cmake/help/latest/variable/PROJECT_BINARY_DIR.html
.. _CMAKE_BINARY_DIR: https://cmake.org/cmake/help/latest/variable/CMAKE_BINARY_DIR.html

To build specific targets, run:

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ cmake --build . --target target_name

Where target_name is a valid target name. Module libraries are prefixed with "lib" (e.g. libcore),
executables from the scratch folder are prefixed with `scratch_` (e.g. scratch_scratch-simulator).
Executables targets have their source file name without the ".cc" prefix 
(e.g. sample-simulator.cc => sample-simulator).

Running the programs and debugging is less straightforward, since CMake cannot help. 
If you are launching it directly from the command line, you will need to export a few
directories to make sure the program can find the ns-3 libraries (at least in platforms
that do not support `RPATH`_).

.. _RPATH: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_RPATH.html

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ export PATH=$PATH:~/ns-3-dev/build/lib 
  ~/ns-3-dev/cmake_cache$ export PYTHONPATH=~/ns-3-dev/build/bindings/python
  ~/ns-3-dev/cmake_cache$ export LD_LIBRARY_PATH=~/ns-3-dev/build/lib 
  ~/ns-3-dev/cmake_cache$ ../build/scratch/ns3-dev-scratch-simulator

Alternatively, the ns3 wrapper script can be used. 
Both lib and `scratch_` prefixes will be added automatically.
The equivalent command to run scratch-simulator is the following:

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ ../ns3 --run-no-build scratch-simulator

The ns3 wrapper script can also show the underlying CMake and command line commands
used by adding the "--dry-run" flag. To build and then run a target:

.. sourcecode:: bash

  ~/ns-3-dev$ ./ns3 --dry-run --run scratch-simulator
  The following commands would be executed:
  cd cmake_cache; cmake --build . -j 15 --target scratch_scratch-simulator ; cd ..
  export PATH=$PATH:~/ns-3-dev/build/lib 
  export PYTHONPATH=~/ns-3-dev/build/bindings/python
  export LD_LIBRARY_PATH=~/ns-3-dev/build/lib 
  ./build/scratch/ns3-dev-scratch-simulator

Debugging can be done with GDB. Again, we have the two ways to run the program:

.. sourcecode:: bash

  ~/ns-3-dev/cmake_cache$ export PATH=$PATH:~/ns-3-dev/build/lib 
  ~/ns-3-dev/cmake_cache$ export PYTHONPATH=~/ns-3-dev/build/bindings/python
  ~/ns-3-dev/cmake_cache$ export LD_LIBRARY_PATH=~/ns-3-dev/build/lib 
  ~/ns-3-dev/cmake_cache$ gdb ../build/scratch/ns3-dev-scratch-simulator

Or with the ns3 wrapper:

.. sourcecode:: bash

  ~/ns-3-dev$ ./ns3 --gdb --run-no-build scratch-simulator
