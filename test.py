#! /usr/bin/env python3
#
# Copyright (c) 2009 University of Washington
#
# SPDX-License-Identifier: GPL-2.0-only
#
import argparse
import fnmatch
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from utils import get_list_from_file

# Global variable
args = None

# imported from waflib Logs
colors_lst = {
    "USE": True,
    "BOLD": "\x1b[01;1m",
    "RED": "\x1b[01;31m",
    "GREEN": "\x1b[32m",
    "YELLOW": "\x1b[33m",
    "PINK": "\x1b[35m",
    "BLUE": "\x1b[01;34m",
    "CYAN": "\x1b[36m",
    "GREY": "\x1b[37m",
    "NORMAL": "\x1b[0m",
    "cursor_on": "\x1b[?25h",
    "cursor_off": "\x1b[?25l",
}


def get_color(cl):
    if colors_lst["USE"]:
        return colors_lst.get(cl, "")
    return ""


class color_dict(object):
    def __getattr__(self, a):
        return get_color(a)

    def __call__(self, a):
        return get_color(a)


colors = color_dict()

#
# XXX This should really be part of a ns3 command to list the configuration
# items relative to optional ns-3 pieces.
#
# A list of interesting configuration items in the ns3 configuration
# cache which we may be interested in when deciding on which examples
# to run and how to run them.  These are set by ns3 during the
# configuration phase and the corresponding assignments are usually
# found in the associated subdirectory CMakeLists.txt files.
#
interesting_config_items = [
    "NS3_ENABLED_MODULES",
    "NS3_ENABLED_CONTRIBUTED_MODULES",
    "NS3_MODULE_PATH",
    "ENABLE_EXAMPLES",
    "ENABLE_TESTS",
    "EXAMPLE_DIRECTORIES",
    "ENABLE_PYTHON_BINDINGS",
    "NSCLICK",
    "ENABLE_BRITE",
    "ENABLE_OPENFLOW",
    "APPNAME",
    "BUILD_PROFILE",
    "VERSION",
    "PYTHON",
    "VALGRIND_FOUND",
]

ENABLE_EXAMPLES = True
ENABLE_TESTS = True
NSCLICK = False
ENABLE_BRITE = False
ENABLE_OPENFLOW = False
ENABLE_PYTHON_BINDINGS = False
EXAMPLE_DIRECTORIES = []
APPNAME = ""
BUILD_PROFILE = ""
BUILD_PROFILE_SUFFIX = ""
VERSION = ""
PYTHON = ""
VALGRIND_FOUND = True

#
# This will be given a prefix and a suffix when the ns3 config file is
# read.
#
test_runner_name = "test-runner"

#
# If the user has constrained us to run certain kinds of tests, we can tell ns3
# to only build
#
core_kinds = ["core", "performance", "system", "unit"]

#
# Exclude tests that are problematic for valgrind.
#
core_valgrind_skip_tests = [
    "routing-click",
    "lte-rr-ff-mac-scheduler",
    "lte-tdmt-ff-mac-scheduler",
    "lte-fdmt-ff-mac-scheduler",
    "lte-pf-ff-mac-scheduler",
    "lte-tta-ff-mac-scheduler",
    "lte-fdbet-ff-mac-scheduler",
    "lte-ttbet-ff-mac-scheduler",
    "lte-fdtbfq-ff-mac-scheduler",
    "lte-tdtbfq-ff-mac-scheduler",
    "lte-pss-ff-mac-scheduler",
]


#
# Parse the examples-to-run file if it exists.
#
# This function adds any C++ examples or Python examples that are to be run
# to the lists in example_tests and python_tests, respectively.
#
def parse_examples_to_run_file(
    examples_to_run_path,
    cpp_executable_dir,
    python_script_dir,
    example_tests,
    example_names_original,
    python_tests,
):
    # Look for the examples-to-run file exists.
    if not os.path.exists(examples_to_run_path):
        # Also tests for contribs OUTSIDE the ns-3-dev directory
        possible_external_contrib_path = examples_to_run_path.replace(
            "contrib", f"{os.path.dirname(os.path.dirname(__file__))}/ns-3-external-contrib"
        )
        if os.path.exists(possible_external_contrib_path):
            examples_to_run_path = possible_external_contrib_path
        else:
            return

    # Each tuple in the C++ list of examples to run contains
    #
    #     (example_name, do_run, do_valgrind_run)
    #
    # where example_name is the executable to be run, do_run is a
    # condition under which to run the example, and do_valgrind_run is
    # a condition under which to run the example under valgrind.  This
    # is needed because NSC causes illegal instruction crashes with
    # some tests when they are run under valgrind.
    #
    # Note that the two conditions are Python statements that
    # can depend on ns3 configuration variables.  For example,
    # when NSC was in the codebase, we could write:
    #
    #     ("tcp-nsc-lfn", "NSC_ENABLED == True", "NSC_ENABLED == False", "QUICK"),
    #
    cpp_examples = get_list_from_file(examples_to_run_path, "cpp_examples")
    for cpp_example in cpp_examples:
        # Old example specification did not include
        # 'fullness', so for compatibility,
        # allow 3 components, & set the 'fullness' to QUICK
        if len(cpp_example) == 3:
            example_name, do_run, do_valgrind_run = cpp_example
            fullness = "QUICK"
        elif len(cpp_example) == 4:
            example_name, do_run, do_valgrind_run, fullness = cpp_example
            fullness: str = fullness.upper()

            if fullness != "QUICK" and fullness != "EXTENSIVE" and fullness != "TAKES_FOREVER":
                raise ValueError(
                    f"Invalid value provided for example '{example_name}' "
                    + f"expected 'QUICK', 'EXTENSIVE', or 'TAKES_FOREVER', got: '{fullness}'"
                )
        else:
            # If we have the name of the example we're error-ing for, provide it
            # Otherwise, just give a generic message
            if len(cpp_example) >= 1:
                raise RuntimeError(
                    f"Incorrect number of fields declaration of example '{cpp_example[0]}', "
                    + f"expected 3, or 4 got: {len(cpp_example)}"
                )
            else:
                raise RuntimeError(
                    f"Incorrect number of fields declaration of example, "
                    + f"expected 3, or 4 got: {len(cpp_example)}"
                )

        # Separate the example name from its arguments.
        example_name_original = example_name
        example_name_parts = example_name.split(" ", 1)
        if len(example_name_parts) == 1:
            example_name = example_name_parts[0]
            example_arguments = ""
        else:
            example_name = example_name_parts[0]
            example_arguments = example_name_parts[1]

        # Add the proper prefix and suffix to the example name to
        # match what is done in the CMakeLists.txt file.
        example_path = "%s%s-%s%s" % (APPNAME, VERSION, example_name, BUILD_PROFILE_SUFFIX)

        # Set the full path for the example.
        example_path = os.path.join(cpp_executable_dir, example_path)
        example_path += ".exe" if sys.platform == "win32" else ""
        example_name = os.path.join(os.path.relpath(cpp_executable_dir, NS3_BUILDDIR), example_name)
        # Add all of the C++ examples that were built, i.e. found
        # in the directory, to the list of C++ examples to run.
        if os.path.exists(example_path):
            # Add any arguments to the path.
            if len(example_name_parts) != 1:
                example_path = "%s %s" % (example_path, example_arguments)
                example_name = "%s %s" % (example_name, example_arguments)

            # Add this example.
            example_tests.append((example_name, example_path, do_run, do_valgrind_run, fullness))
            example_names_original.append(example_name_original)

    # Each tuple in the Python list of examples to run contains
    #
    #     (example_name, do_run, fullness)
    #
    # where example_name is the Python script to be run and
    # do_run is a condition under which to run the example.
    #
    # Note that the condition is a Python statement that can
    # depend on ns3 configuration variables.  For example,
    #
    #     ("brite-generic-example.py", "ENABLE_BRITE == True", "QUICK"),
    #
    python_examples = get_list_from_file(examples_to_run_path, "python_examples")
    # Old example specification did not include
    # 'fullness', so for compatibility,
    # allow 2 components, & set the 'fullness' to QUICK
    for python_example in python_examples:
        if len(python_example) == 2:
            example_name, do_run = python_example
            fullness = "QUICK"
        elif len(python_example) == 3:
            example_name, do_run, fullness = python_example
        else:
            # If we have the name of the example we're error-ing for, provide it
            # Otherwise, just give a generic message
            if len(python_example) >= 1:
                raise RuntimeError(
                    f"Incorrect number of fields declaration of example '{python_example[0]}', "
                    + f"expected 2, or 3 got: {len(python_example)}"
                )
            else:
                raise RuntimeError(
                    f"Incorrect number of fields declaration of example, "
                    + f"expected 2, or 3 got: {len(python_example)}"
                )

        # Separate the example name from its arguments.
        example_name_parts = example_name.split(" ", 1)
        if len(example_name_parts) == 1:
            example_name = example_name_parts[0]
            example_arguments = ""
        else:
            example_name = example_name_parts[0]
            example_arguments = example_name_parts[1]

        # Set the full path for the example.
        example_path = os.path.join(python_script_dir, example_name)

        # Add all of the Python examples that were found to the
        # list of Python examples to run.
        if os.path.exists(example_path):
            # Add any arguments to the path.
            if len(example_name_parts) != 1:
                example_path = "%s %s" % (example_path, example_arguments)

            # Add this example.
            python_tests.append((example_path, do_run, fullness))


#
# The test suites are going to want to output status.  They are running
# concurrently.  This means that unless we are careful, the output of
# the test suites will be interleaved.  Rather than introducing a lock
# file that could unintentionally start serializing execution, we ask
# the tests to write their output to a temporary directory and then
# put together the final output file when we "join" the test tasks back
# to the main thread.  In addition to this issue, the example programs
# often write lots and lots of trace files which we will just ignore.
# We put all of them into the temp directory as well, so they can be
# easily deleted.
#
TMP_OUTPUT_DIR = "testpy-output"


def read_test(test):
    result = test.find("Result").text
    name = test.find("Name").text
    if not test.find("Reason") is None:
        reason = test.find("Reason").text
    else:
        reason = ""
    if not test.find("Time") is None:
        time_real = test.find("Time").get("real")
    else:
        time_real = ""
    return (result, name, reason, time_real)


#
# A simple example of writing a text file with a test result summary.  It is
# expected that this output will be fine for developers looking for problems.
#
def node_to_text(test, f, test_type="Suite"):
    (result, name, reason, time_real) = read_test(test)
    if reason:
        reason = " (%s)" % reason

    output = '%s: Test %s "%s" (%s)%s\n' % (result, test_type, name, time_real, reason)
    f.write(output)
    for details in test.findall("FailureDetails"):
        f.write("    Details:\n")
        f.write("      Message:   %s\n" % details.find("Message").text)
        f.write("      Condition: %s\n" % details.find("Condition").text)
        f.write("      Actual:    %s\n" % details.find("Actual").text)
        f.write("      Limit:     %s\n" % details.find("Limit").text)
        f.write("      File:      %s\n" % details.find("File").text)
        f.write("      Line:      %s\n" % details.find("Line").text)
    for child in test.findall("Test"):
        node_to_text(child, f, "Case")


def translate_to_text(results_file, text_file):
    text_file += ".txt" if ".txt" not in text_file else ""
    print('Writing results to text file "%s"...' % text_file, end="")
    et = ET.parse(results_file)

    with open(text_file, "w", encoding="utf-8") as f:
        for test in et.findall("Test"):
            node_to_text(test, f)

        for example in et.findall("Example"):
            result = example.find("Result").text
            name = example.find("Name").text
            if not example.find("Time") is None:
                time_real = example.find("Time").get("real")
            else:
                time_real = ""
            output = '%s: Example "%s" (%s)\n' % (result, name, time_real)
            f.write(output)

    print("done.")


#
# A simple example of writing an HTML file with a test result summary.  It is
# expected that this will eventually be made prettier as time progresses and
# we have time to tweak it.  This may end up being moved to a separate module
# since it will probably grow over time.
#
def translate_to_html(results_file, html_file):
    html_file += ".html" if ".html" not in html_file else ""
    print("Writing results to html file %s..." % html_file, end="")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write("<html>\n")
        f.write("<body>\n")
        f.write("<center><h1>ns-3 Test Results</h1></center>\n")

        #
        # Read and parse the whole results file.
        #
        et = ET.parse(results_file)

        #
        # Iterate through the test suites
        #
        f.write("<h2>Test Suites</h2>\n")
        for suite in et.findall("Test"):
            #
            # For each test suite, get its name, result and execution time info
            #
            (result, name, reason, time) = read_test(suite)

            #
            # Print a level three header with the result, name and time.  If the
            # test suite passed, the header is printed in green. If the suite was
            # skipped, print it in orange, otherwise assume something bad happened
            # and print in red.
            #
            if result == "PASS":
                f.write('<h3 style="color:green">%s: %s (%s)</h3>\n' % (result, name, time))
            elif result == "SKIP":
                f.write(
                    '<h3 style="color:#ff6600">%s: %s (%s) (%s)</h3>\n'
                    % (result, name, time, reason)
                )
            else:
                f.write('<h3 style="color:red">%s: %s (%s)</h3>\n' % (result, name, time))

            #
            # The test case information goes in a table.
            #
            f.write('<table border="1">\n')

            #
            # The first column of the table has the heading Result
            #
            f.write("<th> Result </th>\n")

            #
            # If the suite crashed or is skipped, there is no further information, so just
            # declare a new table row with the result (CRASH or SKIP) in it.  Looks like:
            #
            #   +--------+
            #   | Result |
            #   +--------+
            #   | CRASH  |
            #   +--------+
            #
            # Then go on to the next test suite.  Valgrind and skipped errors look the same.
            #
            if result in ["CRASH", "SKIP", "VALGR"]:
                f.write("<tr>\n")
                if result == "SKIP":
                    f.write('<td style="color:#ff6600">%s</td>\n' % result)
                else:
                    f.write('<td style="color:red">%s</td>\n' % result)
                f.write("</tr>\n")
                f.write("</table>\n")
                continue

            #
            # If the suite didn't crash, we expect more information, so fill out
            # the table heading row.  Like,
            #
            #   +--------+----------------+------+
            #   | Result | Test Case Name | Time |
            #   +--------+----------------+------+
            #
            f.write("<th>Test Case Name</th>\n")
            f.write("<th> Time </th>\n")

            #
            # If the test case failed, we need to print out some failure details
            # so extend the heading row again.  Like,
            #
            #   +--------+----------------+------+-----------------+
            #   | Result | Test Case Name | Time | Failure Details |
            #   +--------+----------------+------+-----------------+
            #
            if result == "FAIL":
                f.write("<th>Failure Details</th>\n")

            #
            # Now iterate through all the test cases.
            #
            for case in suite.findall("Test"):
                #
                # Get the name, result and timing information from xml to use in
                # printing table below.
                #
                (result, name, reason, time) = read_test(case)

                #
                # If the test case failed, we iterate through possibly multiple
                # failure details
                #
                if result == "FAIL":
                    #
                    # There can be multiple failures for each test case.  The first
                    # row always gets the result, name and timing information along
                    # with the failure details.  Remaining failures don't duplicate
                    # this information but just get blanks for readability.  Like,
                    #
                    #   +--------+----------------+------+-----------------+
                    #   | Result | Test Case Name | Time | Failure Details |
                    #   +--------+----------------+------+-----------------+
                    #   |  FAIL  | The name       | time | It's busted     |
                    #   +--------+----------------+------+-----------------+
                    #   |        |                |      | Really broken   |
                    #   +--------+----------------+------+-----------------+
                    #   |        |                |      | Busted bad      |
                    #   +--------+----------------+------+-----------------+
                    #

                    first_row = True
                    for details in case.findall("FailureDetails"):
                        #
                        # Start a new row in the table for each possible Failure Detail
                        #
                        f.write("<tr>\n")

                        if first_row:
                            first_row = False
                            f.write('<td style="color:red">%s</td>\n' % result)
                            f.write("<td>%s</td>\n" % name)
                            f.write("<td>%s</td>\n" % time)
                        else:
                            f.write("<td></td>\n")
                            f.write("<td></td>\n")
                            f.write("<td></td>\n")

                        f.write("<td>")
                        f.write("<b>Message: </b>%s, " % details.find("Message").text)
                        f.write("<b>Condition: </b>%s, " % details.find("Condition").text)
                        f.write("<b>Actual: </b>%s, " % details.find("Actual").text)
                        f.write("<b>Limit: </b>%s, " % details.find("Limit").text)
                        f.write("<b>File: </b>%s, " % details.find("File").text)
                        f.write("<b>Line: </b>%s" % details.find("Line").text)
                        f.write("</td>\n")

                        #
                        # End the table row
                        #
                        f.write("</td>\n")
                else:
                    #
                    # If this particular test case passed, then we just print the PASS
                    # result in green, followed by the test case name and its execution
                    # time information.  These go off in <td> ... </td> table data.
                    # The details table entry is left blank.
                    #
                    #   +--------+----------------+------+---------+
                    #   | Result | Test Case Name | Time | Details |
                    #   +--------+----------------+------+---------+
                    #   |  PASS  | The name       | time |         |
                    #   +--------+----------------+------+---------+
                    #
                    f.write("<tr>\n")
                    f.write('<td style="color:green">%s</td>\n' % result)
                    f.write("<td>%s</td>\n" % name)
                    f.write("<td>%s</td>\n" % time)
                    f.write("<td>%s</td>\n" % reason)
                    f.write("</tr>\n")
            #
            # All of the rows are written, so we need to end the table.
            #
            f.write("</table>\n")

        #
        # That's it for all of the test suites.  Now we have to do something about
        # our examples.
        #
        f.write("<h2>Examples</h2>\n")

        #
        # Example status is rendered in a table just like the suites.
        #
        f.write('<table border="1">\n')

        #
        # The table headings look like,
        #
        #   +--------+--------------+--------------+---------+
        #   | Result | Example Name | Elapsed Time | Details |
        #   +--------+--------------+--------------+---------+
        #
        f.write("<th> Result </th>\n")
        f.write("<th>Example Name</th>\n")
        f.write("<th>Elapsed Time</th>\n")
        f.write("<th>Details</th>\n")

        #
        # Now iterate through all the examples
        #
        for example in et.findall("Example"):
            #
            # Start a new row for each example
            #
            f.write("<tr>\n")

            #
            # Get the result and name of the example in question
            #
            (result, name, reason, time) = read_test(example)

            #
            # If the example either failed or crashed, print its result status
            # in red; otherwise green.  This goes in a <td> ... </td> table data
            #
            if result == "PASS":
                f.write('<td style="color:green">%s</td>\n' % result)
            elif result == "SKIP":
                f.write('<td style="color:#ff6600">%s</fd>\n' % result)
            else:
                f.write('<td style="color:red">%s</td>\n' % result)

            #
            # Write the example name as a new tag data.
            #
            f.write("<td>%s</td>\n" % name)

            #
            # Write the elapsed time as a new tag data.
            #
            f.write("<td>%s</td>\n" % time)

            #
            # Write the reason, if it exist
            #
            f.write("<td>%s</td>\n" % reason)

            #
            # That's it for the current example, so terminate the row.
            #
            f.write("</tr>\n")

        #
        # That's it for the table of examples, so terminate the table.
        #
        f.write("</table>\n")

        #
        # And that's it for the report, so finish up.
        #
        f.write("</body>\n")
        f.write("</html>\n")

    print("done.")


#
# Python Control-C handling is broken in the presence of multiple threads.
# Signals get delivered to the runnable/running thread by default and if
# it is blocked, the signal is simply ignored.  So we hook sigint and set
# a global variable telling the system to shut down gracefully.
#
thread_exit = False


def sigint_hook(signal, frame):
    global thread_exit
    thread_exit = True
    return 0


#
# In general, the build process itself naturally takes care of figuring out
# which tests are built into the test runner.  For example, if ns3 configure
# determines that ENABLE_EMU is false due to some missing dependency,
# the tests for the emu net device simply will not be built and will
# therefore not be included in the built test runner.
#
# Examples, however, are a different story.  In that case, we are just given
# a list of examples that could be run.  Instead of just failing, for example,
# an example if its library support is not present, we look into the ns3
# saved configuration for relevant configuration items.
#
# XXX This function pokes around in the ns3 internal state file.  To be a
# little less hacky, we should add a command to ns3 to return this info
# and use that result.
#
platform = sys.platform
platform = "bsd" if "bsd" in platform else platform
lock_filename = ".lock-ns3_%s_build" % platform


def read_ns3_config():
    try:
        # sys.platform reports linux2 for python2 and linux for python3
        with open(lock_filename, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("top_dir ="):
                    key, val = line.split("=")
                    top_dir = eval(val.strip())
                if line.startswith("out_dir ="):
                    key, val = line.split("=")
                    out_dir = eval(val.strip())

    except FileNotFoundError:
        print(
            "The .lock-ns3 file was not found.  You must configure before running test.py.",
            file=sys.stderr,
        )
        sys.exit(2)

    global NS3_BASEDIR
    NS3_BASEDIR = top_dir
    global NS3_BUILDDIR
    NS3_BUILDDIR = out_dir

    with open(lock_filename, encoding="utf-8") as f:
        for line in f.readlines():
            for item in interesting_config_items:
                if line.startswith(item):
                    exec(line, globals())

    if args.verbose:
        for item in interesting_config_items:
            print("%s ==" % item, eval(item))


#
# It seems pointless to fork a process to run ns3 to fork a process to run
# the test runner, so we just run the test runner directly.  The main thing
# that ns3 would do for us would be to sort out the shared library path but
# we can deal with that easily and do here.
#
# There can be many different ns-3 repositories on a system, and each has
# its own shared libraries, so ns-3 doesn't hardcode a shared library search
# path -- it is cooked up dynamically, so we do that too.
#
def make_paths():
    have_DYLD_LIBRARY_PATH = False
    have_LD_LIBRARY_PATH = False
    have_PATH = False
    have_PYTHONPATH = False

    keys = list(os.environ.keys())
    for key in keys:
        if key == "DYLD_LIBRARY_PATH":
            have_DYLD_LIBRARY_PATH = True
        if key == "LD_LIBRARY_PATH":
            have_LD_LIBRARY_PATH = True
        if key == "PATH":
            have_PATH = True
        if key == "PYTHONPATH":
            have_PYTHONPATH = True

    pypath = os.environ["PYTHONPATH"] = os.path.join(NS3_BUILDDIR, "bindings", "python")

    if not have_PYTHONPATH:
        os.environ["PYTHONPATH"] = pypath
    else:
        os.environ["PYTHONPATH"] += ":" + pypath

    if args.verbose:
        print('os.environ["PYTHONPATH"] == %s' % os.environ["PYTHONPATH"])

    if sys.platform == "darwin":
        if not have_DYLD_LIBRARY_PATH:
            os.environ["DYLD_LIBRARY_PATH"] = ""
        for path in NS3_MODULE_PATH:
            os.environ["DYLD_LIBRARY_PATH"] += ":" + path
        if args.verbose:
            print('os.environ["DYLD_LIBRARY_PATH"] == %s' % os.environ["DYLD_LIBRARY_PATH"])
    elif sys.platform == "win32":
        if not have_PATH:
            os.environ["PATH"] = ""
        for path in NS3_MODULE_PATH:
            os.environ["PATH"] += ";" + path
        if args.verbose:
            print('os.environ["PATH"] == %s' % os.environ["PATH"])
    elif sys.platform == "cygwin":
        if not have_PATH:
            os.environ["PATH"] = ""
        for path in NS3_MODULE_PATH:
            os.environ["PATH"] += ":" + path
        if args.verbose:
            print('os.environ["PATH"] == %s' % os.environ["PATH"])
    else:
        if not have_LD_LIBRARY_PATH:
            os.environ["LD_LIBRARY_PATH"] = ""
        for path in NS3_MODULE_PATH:
            os.environ["LD_LIBRARY_PATH"] += ":" + str(path)
        if args.verbose:
            print('os.environ["LD_LIBRARY_PATH"] == %s' % os.environ["LD_LIBRARY_PATH"])


#
# Short note on generating suppressions:
#
# See the valgrind documentation for a description of suppressions.  The easiest
# way to generate a suppression expression is by using the valgrind
# --gen-suppressions option.  To do that you have to figure out how to run the
# test in question.
#
# If you do "test.py -v -g -s <suitename> then test.py will output most of what
# you need.  For example, if you are getting a valgrind error in the
# devices-mesh-dot11s-regression test suite, you can run:
#
#   ./test.py -v -g -s devices-mesh-dot11s-regression
#
# You should see in the verbose output something that looks like:
#
#   Synchronously execute valgrind --suppressions=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/testpy.supp
#   --leak-check=full --error-exitcode=2 /home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/build/debug/utils/ns3-dev-test-runner-debug
#   --suite=devices-mesh-dot11s-regression --basedir=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev
#   --tempdir=testpy-output/2010-01-12-22-47-50-CUT
#   --out=testpy-output/2010-01-12-22-47-50-CUT/devices-mesh-dot11s-regression.xml
#
# You need to pull out the useful pieces, and so could run the following to
# reproduce your error:
#
#   valgrind --suppressions=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/testpy.supp
#   --leak-check=full --error-exitcode=2 /home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/build/debug/utils/ns3-dev-test-runner-debug
#   --suite=devices-mesh-dot11s-regression --basedir=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev
#   --tempdir=testpy-output
#
# Hint: Use the first part of the command as is, and point the "tempdir" to
# somewhere real.  You don't need to specify an "out" file.
#
# When you run the above command you should see your valgrind error.  The
# suppression expression(s) can be generated by adding the --gen-suppressions=yes
# option to valgrind.  Use something like:
#
#   valgrind --gen-suppressions=yes --suppressions=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/testpy.supp
#   --leak-check=full --error-exitcode=2 /home/craigdo/repos/ns-3-allinone-dev/ns-3-dev/build/debug/utils/ns3-dev-test-runner-debug
#   --suite=devices-mesh-dot11s-regression --basedir=/home/craigdo/repos/ns-3-allinone-dev/ns-3-dev
#   --tempdir=testpy-output
#
# Now when valgrind detects an error it will ask:
#
#   ==27235== ---- Print suppression ? --- [Return/N/n/Y/y/C/c] ----
#
# to which you just enter 'y'<ret>.
#
# You will be provided with a suppression expression that looks something like
# the following:
#   {
#     <insert_a_suppression_name_here>
#     Memcheck:Addr8
#     fun:_ZN3ns36dot11s15HwmpProtocolMac8SendPreqESt6vectorINS0_6IePreqESaIS3_EE
#     fun:_ZN3ns36dot11s15HwmpProtocolMac10SendMyPreqEv
#     fun:_ZN3ns36dot11s15HwmpProtocolMac18RequestDestinationENS_12Mac48AddressEjj
#     ...
#     the rest of the stack frame
#     ...
#   }
#
# You need to add a suppression name which will only be printed out by valgrind in
# verbose mode (but it needs to be there in any case).  The entire stack frame is
# shown to completely characterize the error, but in most cases you won't need
# all of that info.  For example, if you want to turn off all errors that happen
# when the function (fun:) is called, you can just delete the rest of the stack
# frame.  You can also use wildcards to make the mangled signatures more readable.
#
# I added the following to the testpy.supp file for this particular error:
#
#   {
#     Suppress invalid read size errors in SendPreq() when using HwmpProtocolMac
#     Memcheck:Addr8
#     fun:*HwmpProtocolMac*SendPreq*
#   }
#
# Now, when you run valgrind the error will be suppressed.
#
# Until ns-3.36, we used a suppression in testpy.supp in the top-level
# ns-3 directory.   It was defined below, but commented out once it was
# no longer needed.  If it is needed again in the future, define the
# below variable again, and remove the alternative definition to None
#
VALGRIND_SUPPRESSIONS_FILE = ".ns3.supp"
# VALGRIND_SUPPRESSIONS_FILE = None

# When the TEST_LOGS environment variable is set to 1 or true,
# NS_LOG is set to NS_LOG=*, and stdout/stderr
# from tests are discarded to prevent running out of memory.
TEST_LOGS = bool(os.getenv("TEST_LOGS", False))


def run_job_synchronously(shell_command, directory, valgrind, is_python, build_path=""):
    if VALGRIND_SUPPRESSIONS_FILE is not None:
        suppressions_path = os.path.join(NS3_BASEDIR, VALGRIND_SUPPRESSIONS_FILE)

    if is_python:
        path_cmd = PYTHON[0] + " " + os.path.join(NS3_BASEDIR, shell_command)
    else:
        if len(build_path):
            path_cmd = os.path.join(build_path, shell_command)
        else:
            path_cmd = os.path.join(NS3_BUILDDIR, shell_command)

    if valgrind:
        if VALGRIND_SUPPRESSIONS_FILE:
            cmd = (
                "valgrind --suppressions=%s --leak-check=full --show-reachable=yes --error-exitcode=2 --errors-for-leak-kinds=all %s"
                % (suppressions_path, path_cmd)
            )
        else:
            cmd = (
                "valgrind --leak-check=full --show-reachable=yes --error-exitcode=2 --errors-for-leak-kinds=all %s"
                % (path_cmd)
            )
    else:
        cmd = path_cmd

    if args.verbose:
        print("Synchronously execute %s" % cmd)

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=directory,
        stdout=subprocess.PIPE if not TEST_LOGS else subprocess.DEVNULL,
        stderr=subprocess.PIPE if not TEST_LOGS else subprocess.STDOUT,
    )
    stdout_results, stderr_results = proc.communicate()
    stdout_results = b"" if stdout_results is None else stdout_results
    stderr_results = b"" if stderr_results is None else stderr_results

    elapsed_time = time.time() - start_time

    retval = proc.returncode

    def decode_stream_results(stream_results: bytes, stream_name: str) -> str:
        try:
            stream_results = stream_results.decode()
        except UnicodeDecodeError:

            def decode(byte_array: bytes):
                try:
                    byte_array.decode()
                except UnicodeDecodeError:
                    return byte_array

            # Find lines where the decoding error happened
            non_utf8_lines = list(map(lambda line: decode(line), stream_results.splitlines()))
            non_utf8_lines = list(filter(lambda line: line is not None, non_utf8_lines))
            print(
                f"Non-decodable characters found in {stream_name} output of {cmd}: {non_utf8_lines}"
            )

            # Continue decoding on errors
            stream_results = stream_results.decode(errors="backslashreplace")
        return stream_results

    stdout_results = decode_stream_results(stdout_results, "stdout")
    stderr_results = decode_stream_results(stderr_results, "stderr")

    if args.verbose:
        print("Return code = ", retval)
        print("stderr = ", stderr_results)

    return (retval, stdout_results, stderr_results, elapsed_time)


#
# This class defines a unit of testing work.  It will typically refer to
# a test suite to run using the test-runner, or an example to run directly.
#
class Job:
    def __init__(self):
        self.is_break = False
        self.is_skip = False
        self.skip_reason = ""
        self.is_example = False
        self.is_pyexample = False
        self.shell_command = ""
        self.display_name = ""
        self.basedir = ""
        self.tempdir = ""
        self.cwd = ""
        self.tmp_file_name = ""
        self.returncode = False
        self.elapsed_time = 0
        self.build_path = ""

    #
    # A job is either a standard job or a special job indicating that a worker
    # thread should exist.  This special job is indicated by setting is_break
    # to true.
    #
    def set_is_break(self, is_break):
        self.is_break = is_break

    #
    # If a job is to be skipped, we actually run it through the worker threads
    # to keep the PASS, FAIL, CRASH and SKIP processing all in one place.
    #
    def set_is_skip(self, is_skip):
        self.is_skip = is_skip

    #
    # If a job is to be skipped, log the reason.
    #
    def set_skip_reason(self, skip_reason):
        self.skip_reason = skip_reason

    #
    # Examples are treated differently than standard test suites.  This is
    # mostly because they are completely unaware that they are being run as
    # tests.  So we have to do some special case processing to make them look
    # like tests.
    #
    def set_is_example(self, is_example):
        self.is_example = is_example

    #
    # Examples are treated differently than standard test suites.  This is
    # mostly because they are completely unaware that they are being run as
    # tests.  So we have to do some special case processing to make them look
    # like tests.
    #
    def set_is_pyexample(self, is_pyexample):
        self.is_pyexample = is_pyexample

    #
    # This is the shell command that will be executed in the job.  For example,
    #
    #  "utils/ns3-dev-test-runner-debug --test-name=some-test-suite"
    #
    def set_shell_command(self, shell_command):
        self.shell_command = shell_command

    #
    # This is the build path where ns-3 was built.  For example,
    #
    #  "/home/craigdo/repos/ns-3-allinone-test/ns-3-dev/build/debug"
    #
    def set_build_path(self, build_path):
        self.build_path = build_path

    #
    # This is the display name of the job, typically the test suite or example
    # name.  For example,
    #
    #  "some-test-suite" or "udp-echo"
    #
    def set_display_name(self, display_name):
        self.display_name = display_name

    #
    # This is the base directory of the repository out of which the tests are
    # being run.  It will be used deep down in the testing framework to determine
    # where the source directory of the test was, and therefore where to find
    # provided test vectors.  For example,
    #
    #  "/home/user/repos/ns-3-dev"
    #
    def set_basedir(self, basedir):
        self.basedir = basedir

    #
    # This is the directory to which a running test suite should write any
    # temporary files.
    #
    def set_tempdir(self, tempdir):
        self.tempdir = tempdir

    #
    # This is the current working directory that will be given to an executing
    # test as it is being run.  It will be used for examples to tell them where
    # to write all of the pcap files that we will be carefully ignoring.  For
    # example,
    #
    #  "/tmp/unchecked-traces"
    #
    def set_cwd(self, cwd):
        self.cwd = cwd

    #
    # This is the temporary results file name that will be given to an executing
    # test as it is being run.  We will be running all of our tests in parallel
    # so there must be multiple temporary output files.  These will be collected
    # into a single XML file at the end and then be deleted.
    #
    def set_tmp_file_name(self, tmp_file_name):
        self.tmp_file_name = tmp_file_name

    #
    # The return code received when the job process is executed.
    #
    def set_returncode(self, returncode):
        self.returncode = returncode

    #
    # The elapsed real time for the job execution.
    #
    def set_elapsed_time(self, elapsed_time):
        self.elapsed_time = elapsed_time


#
# The worker thread class that handles the actual running of a given test.
# Once spawned, it receives requests for work through its input_queue and
# ships the results back through the output_queue.
#
class worker_thread(threading.Thread):
    def __init__(self, input_queue, output_queue):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            job = self.input_queue.get()
            #
            # Worker threads continue running until explicitly told to stop with
            # a special job.
            #
            if job.is_break:
                return
            #
            # If the global interrupt handler sets the thread_exit variable,
            # we stop doing real work and just report back a "break" in the
            # normal command processing has happened.
            #
            if thread_exit == True:
                job.set_is_break(True)
                self.output_queue.put(job)
                continue

            #
            # If we are actually supposed to skip this job, do so.  Note that
            # if is_skip is true, returncode is undefined.
            #
            if job.is_skip:
                if args.verbose:
                    print("Skip %s" % job.shell_command)
                self.output_queue.put(job)
                continue

            #
            # Otherwise go about the business of running tests as normal.
            #
            else:
                if args.verbose:
                    print("Launch %s" % job.shell_command)

                if job.is_example or job.is_pyexample:
                    #
                    # If we have an example, the shell command is all we need to
                    # know.  It will be something like "examples/udp/udp-echo" or
                    # "examples/wireless/mixed-wireless.py"
                    #
                    (
                        job.returncode,
                        job.standard_out,
                        job.standard_err,
                        et,
                    ) = run_job_synchronously(
                        job.shell_command, job.cwd, args.valgrind, job.is_pyexample, job.build_path
                    )
                else:
                    #
                    # If we're a test suite, we need to provide a little more info
                    # to the test runner, specifically the base directory and temp
                    # file name
                    #
                    if args.update_data:
                        update_data = "--update-data"
                    else:
                        update_data = ""
                    (
                        job.returncode,
                        job.standard_out,
                        job.standard_err,
                        et,
                    ) = run_job_synchronously(
                        job.shell_command
                        + " --xml --tempdir=%s --out=%s %s"
                        % (job.tempdir, job.tmp_file_name, update_data),
                        job.cwd,
                        args.valgrind,
                        False,
                    )

                job.set_elapsed_time(et)

                if args.verbose:
                    print("returncode = %d" % job.returncode)
                    print("---------- begin standard out ----------")
                    print(job.standard_out)
                    print("---------- begin standard err ----------")
                    print(job.standard_err)
                    print("---------- end standard err ----------")

                self.output_queue.put(job)


#
# This function loads the list of previously successful or skipped examples and test suites.
#
def load_previously_successful_tests():
    import glob

    previously_run_tests_to_skip = {"test": [], "example": []}
    previous_results = glob.glob(f"{TMP_OUTPUT_DIR}/*-results.xml")
    if not previous_results:
        print("No previous runs to rerun")
        exit(-1)
    latest_result_file = list(
        sorted(previous_results, key=lambda x: os.path.basename(x), reverse=True)
    )[0]

    try:
        previous_run_results = ET.parse(latest_result_file)
    except ET.ParseError:
        print(f"Failed to parse XML {latest_result_file}")
        exit(-1)

    for test_type in ["Test", "Example"]:
        if previous_run_results.find(test_type):
            temp = list(
                map(
                    lambda x: (x.find("Name").text, x.find("Result").text),
                    previous_run_results.findall(test_type),
                )
            )
            temp = list(filter(lambda x: x[1] in ["PASS", "SKIP"], temp))
            temp = [x[0] for x in temp]
            previously_run_tests_to_skip[test_type.lower()] = temp
    return previously_run_tests_to_skip


#
# This is the main function that does the work of interacting with the
# test-runner itself.
#
def run_tests():
    #
    # Pull some interesting configuration information out of ns3, primarily
    # so we can know where executables can be found, but also to tell us what
    # pieces of the system have been built.  This will tell us what examples
    # are runnable.
    #
    read_ns3_config()

    #
    # Set the proper suffix.
    #
    global BUILD_PROFILE_SUFFIX
    if BUILD_PROFILE == "release":
        BUILD_PROFILE_SUFFIX = ""
    else:
        BUILD_PROFILE_SUFFIX = "-" + BUILD_PROFILE

    #
    # Add the proper prefix and suffix to the test-runner name to
    # match what is done in the CMakeLists.txt file.
    #
    test_runner_name = "%s%s-%s%s" % (APPNAME, VERSION, "test-runner", BUILD_PROFILE_SUFFIX)
    test_runner_name += ".exe" if sys.platform == "win32" else ""

    #
    # Run ns3 to make sure that everything is built, configured and ready to go
    # unless we are explicitly told not to.  We want to be careful about causing
    # our users pain while waiting for extraneous stuff to compile and link, so
    # we allow users that know what they're doing to not invoke ns3 at all.
    #
    if not args.no_build:
        # If the user only wants to run a single example, then we can just build
        # that example.
        #
        # If there is no constraint, then we have to build everything since the
        # user wants to run everything.
        #
        if len(args.example):
            build_cmd = "./ns3 build %s" % os.path.basename(args.example.replace("*", ""))
        else:
            build_cmd = "./ns3"

        if sys.platform == "win32":
            build_cmd = f'"{sys.executable}" {build_cmd}'

        if args.verbose:
            print("Building: %s" % build_cmd)

        proc = subprocess.run(build_cmd, shell=True)
        if proc.returncode:
            print("ns3 died. Not running tests", file=sys.stderr)
            return proc.returncode

    #
    # Dynamically set up paths.
    #
    make_paths()

    #
    # Get the information from the build status file.
    #
    if os.path.exists(lock_filename):
        ns3_runnable_programs = get_list_from_file(lock_filename, "ns3_runnable_programs")
        ns3_runnable_scripts = get_list_from_file(lock_filename, "ns3_runnable_scripts")
        ns3_runnable_scripts = [os.path.basename(script) for script in ns3_runnable_scripts]
    else:
        print(
            "The build status file was not found.  You must configure before running test.py.",
            file=sys.stderr,
        )
        sys.exit(2)

    #
    # Make a dictionary that maps the name of a program to its path.
    #
    ns3_runnable_programs_dictionary = {}
    for program in ns3_runnable_programs:
        # Remove any directory names from path.
        program_name = os.path.basename(program)
        ns3_runnable_programs_dictionary[program_name] = program

    # Generate the lists of examples to run as smoke tests in order to
    # ensure that they remain buildable and runnable over time.
    #
    example_tests = []
    example_names_original = []
    python_tests = []
    for directory in EXAMPLE_DIRECTORIES:
        # Set the directories and paths for this example.
        example_directory = os.path.join("examples", directory)
        examples_to_run_path = os.path.join(example_directory, "examples-to-run.py")
        cpp_executable_dir = os.path.join(NS3_BUILDDIR, example_directory)
        python_script_dir = os.path.join(example_directory)

        # Parse this example directory's file.
        parse_examples_to_run_file(
            examples_to_run_path,
            cpp_executable_dir,
            python_script_dir,
            example_tests,
            example_names_original,
            python_tests,
        )

    for module in NS3_ENABLED_MODULES:
        # Remove the "ns3-" from the module name.
        module = module[len("ns3-") :]

        # Set the directories and paths for this example.
        module_directory = os.path.join("src", module)
        example_directory = os.path.join(module_directory, "examples")
        examples_to_run_path = os.path.join(module_directory, "test", "examples-to-run.py")
        cpp_executable_dir = os.path.join(NS3_BUILDDIR, example_directory)
        python_script_dir = os.path.join(example_directory)

        # Parse this module's file.
        parse_examples_to_run_file(
            examples_to_run_path,
            cpp_executable_dir,
            python_script_dir,
            example_tests,
            example_names_original,
            python_tests,
        )

    for module in NS3_ENABLED_CONTRIBUTED_MODULES:
        # Remove the "ns3-" from the module name.
        module = module[len("ns3-") :]

        # Set the directories and paths for this example.
        module_directory = os.path.join("contrib", module)
        example_directory = os.path.join(module_directory, "examples")
        examples_to_run_path = os.path.join(module_directory, "test", "examples-to-run.py")
        cpp_executable_dir = os.path.join(NS3_BUILDDIR, example_directory)
        python_script_dir = os.path.join(example_directory)

        # Parse this module's file.
        parse_examples_to_run_file(
            examples_to_run_path,
            cpp_executable_dir,
            python_script_dir,
            example_tests,
            example_names_original,
            python_tests,
        )

    #
    # If lots of logging is enabled, we can crash Python when it tries to
    # save all of the text.  We just don't allow logging to be turned on when
    # test.py runs.  If you want to see logging output from your tests, you
    # have to run them using the test-runner directly.
    #
    os.environ["NS_LOG"] = "*" if TEST_LOGS else ""

    #
    # There are a couple of options that imply we can to exit before starting
    # up a bunch of threads and running tests.  Let's detect these cases and
    # handle them without doing all of the hard work.
    #
    if args.kinds:
        path_cmd = os.path.join("utils", test_runner_name + " --print-test-type-list")
        (rc, standard_out, standard_err, et) = run_job_synchronously(
            path_cmd, os.getcwd(), False, False
        )
        print(standard_out)

    if args.list:
        list_items = []
        if ENABLE_TESTS:
            if len(args.constrain):
                path_cmd = os.path.join(
                    "utils",
                    test_runner_name
                    + " --print-test-name-list --print-test-types --test-type=%s" % args.constrain,
                )
            else:
                path_cmd = os.path.join(
                    "utils", test_runner_name + " --print-test-name-list --print-test-types"
                )
            (rc, standard_out, standard_err, et) = run_job_synchronously(
                path_cmd, os.getcwd(), False, False
            )
            if rc != 0:
                # This is usually a sign that ns-3 crashed or exited uncleanly
                print(("test.py error:  test-runner return code returned {}".format(rc)))
                print(
                    (
                        "To debug, try running {}\n".format(
                            "'./ns3 run \"test-runner --print-test-name-list\"'"
                        )
                    )
                )
                return
            if isinstance(standard_out, bytes):
                standard_out = standard_out.decode()
            list_items = standard_out.split("\n")
            list_items.sort()
        print("Test Type            Test Name")
        print("---------------      ---------")
        for item in list_items:
            if len(item.strip()):
                print(item)
        examples_sorted = []
        if ENABLE_EXAMPLES:
            examples_sorted = example_names_original
            examples_sorted.sort()
        if ENABLE_PYTHON_BINDINGS:
            python_examples_sorted = []
            for x, y in python_tests:
                if y == "True":
                    python_examples_sorted.append(x)
            python_examples_sorted.sort()
            examples_sorted.extend(python_examples_sorted)
        for item in examples_sorted:
            print("example             ", item)
        print()

    if args.kinds or args.list:
        return

    #
    # We communicate results in two ways.  First, a simple message relating
    # PASS, FAIL, CRASH or SKIP is always written to the standard output.  It
    # is expected that this will be one of the main use cases.  A developer can
    # just run test.py with no options and see that all of the tests still
    # pass.
    #
    # The second main use case is when detailed status is requested (with the
    # --text or --html options).  Typically this will be text if a developer
    # finds a problem, or HTML for nightly builds.  In these cases, an
    # XML file is written containing the status messages from the test suites.
    # This file is then read and translated into text or HTML.  It is expected
    # that nobody will really be interested in the XML, so we write it somewhere
    # with a unique name (time) to avoid collisions.  In case an error happens, we
    # provide a runtime option to retain the temporary files.
    #
    # When we run examples as smoke tests, they are going to want to create
    # lots and lots of trace files.  We aren't really interested in the contents
    # of the trace files, so we also just stash them off in the temporary dir.
    # The retain option also causes these unchecked trace files to be kept.
    #
    date_and_time = time.strftime("%Y-%m-%d-%H-%M-%S-CUT", time.gmtime())

    if not os.path.exists(TMP_OUTPUT_DIR):
        os.makedirs(TMP_OUTPUT_DIR)

    testpy_output_dir = os.path.join(TMP_OUTPUT_DIR, date_and_time)

    if not os.path.exists(testpy_output_dir):
        os.makedirs(testpy_output_dir)

    #
    # Load results from the latest results.xml, then use the list of
    # failed tests to filter out (SKIP) successful tests
    #
    previously_run_tests_to_skip = {"test": [], "example": []}
    if args.rerun_failed:
        previously_run_tests_to_skip = load_previously_successful_tests()

    #
    # Create the main output file and start filling it with XML.  We need to
    # do this since the tests will just append individual results to this file.
    # The file is created outside the directory that gets automatically deleted.
    #
    xml_results_file = os.path.join(TMP_OUTPUT_DIR, f"{date_and_time}-results.xml")
    with open(xml_results_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write("<Results>\n")

    #
    # We need to figure out what test suites to execute.  We are either given one
    # suite or example explicitly via the --suite or --example/--pyexample option,
    # or we need to call into the test runner and ask it to list all of the available
    # test suites.  Further, we need to provide the constraint information if it
    # has been given to us.
    #
    # This translates into allowing the following options with respect to the
    # suites
    #
    #  ./test.py:                                           run all of the suites and examples
    #  ./test.py --constrain=core:                          run all of the suites of all kinds
    #  ./test.py --constrain=unit:                          run all unit suites
    #  ./test.py --suite=some-test-suite:                   run a single suite
    #  ./test.py --example=examples/udp/udp-echo:           run single example
    #  ./test.py --pyexample=examples/wireless/mixed-wireless.py:  run python example
    #  ./test.py --suite=some-suite --example=some-example: run the single suite
    #
    # We can also use the --constrain option to provide an ordering of test
    # execution quite easily.
    #

    # Flag indicating a specific suite was explicitly requested
    single_suite = False

    if len(args.suite):
        # See if this is a valid test suite.
        path_cmd = os.path.join("utils", test_runner_name + " --print-test-name-list")
        (rc, suites, standard_err, et) = run_job_synchronously(path_cmd, os.getcwd(), False, False)

        if isinstance(suites, bytes):
            suites = suites.decode()

        suites = suites.replace("\r\n", "\n")
        suites_found = fnmatch.filter(suites.split("\n"), args.suite)

        if not suites_found:
            print(
                "The test suite was not run because an unknown test suite name was requested.",
                file=sys.stderr,
            )
            sys.exit(2)
        elif len(suites_found) == 1:
            single_suite = True

        suites = "\n".join(suites_found)

    elif ENABLE_TESTS and len(args.example) == 0 and len(args.pyexample) == 0:
        if len(args.constrain):
            path_cmd = os.path.join(
                "utils",
                test_runner_name + " --print-test-name-list --test-type=%s" % args.constrain,
            )
            (rc, suites, standard_err, et) = run_job_synchronously(
                path_cmd, os.getcwd(), False, False
            )
        else:
            path_cmd = os.path.join("utils", test_runner_name + " --print-test-name-list")
            (rc, suites, standard_err, et) = run_job_synchronously(
                path_cmd, os.getcwd(), False, False
            )
    else:
        suites = ""

    #
    # suite_list will either a single test suite name that the user has
    # indicated she wants to run or a list of test suites provided by
    # the test-runner possibly according to user provided constraints.
    # We go through the trouble of setting up the parallel execution
    # even in the case of a single suite to avoid having to process the
    # results in two different places.
    #
    if isinstance(suites, bytes):
        suites = suites.decode()
    suite_list = suites.split("\n")

    #
    # Performance tests should only be run when they are requested,
    # i.e. they are not run by default in test.py.
    # If a specific suite was requested we run it, even if
    # it is a performance test.
    if not single_suite and args.constrain != "performance":
        # Get a list of all of the performance tests.
        path_cmd = os.path.join(
            "utils", test_runner_name + " --print-test-name-list --test-type=%s" % "performance"
        )
        (rc, performance_tests, standard_err, et) = run_job_synchronously(
            path_cmd, os.getcwd(), False, False
        )
        if isinstance(performance_tests, bytes):
            performance_tests = performance_tests.decode()
        performance_test_list = performance_tests.split("\n")

        # Remove any performance tests from the suites list.
        for performance_test in performance_test_list:
            if performance_test in suite_list:
                suite_list.remove(performance_test)

    # We now have a possibly large number of test suites to run, so we want to
    # run them in parallel.  We're going to spin up a number of worker threads
    # that will run our test jobs for us.
    #
    input_queue = queue.Queue(0)
    output_queue = queue.Queue(0)

    jobs = 0
    threads = []

    #
    # In Python 2.6 you can just use multiprocessing module, but we don't want
    # to introduce that dependency yet; so we jump through a few hoops.
    #
    processors = 1

    if sys.platform != "win32":
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            processors = os.sysconf("SC_NPROCESSORS_ONLN")
        else:
            proc = subprocess.Popen(
                "sysctl -n hw.ncpu", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout_results, stderr_results = proc.communicate()
            stdout_results = stdout_results.decode()
            stderr_results = stderr_results.decode()
            if len(stderr_results) == 0:
                processors = int(stdout_results)
    else:
        processors = os.cpu_count()

    if args.process_limit:
        if processors < args.process_limit:
            print("Using all %s processors" % processors)
        else:
            processors = args.process_limit
            print("Limiting to %s worker processes" % processors)

    #
    # Now, spin up one thread per processor which will eventually mean one test
    # per processor running concurrently.
    #
    for i in range(processors):
        thread = worker_thread(input_queue, output_queue)
        threads.append(thread)
        thread.start()

    #
    # Keep track of some summary statistics
    #
    total_tests = 0
    skipped_tests = 0
    skipped_testnames = []

    #
    # We now have worker threads spun up, and a list of work to do.  So, run
    # through the list of test suites and dispatch a job to run each one.
    #
    # Dispatching will run with unlimited speed and the worker threads will
    # execute as fast as possible from the queue.
    #
    # Note that we actually dispatch tests to be skipped, so all the
    # PASS, FAIL, CRASH and SKIP processing is done in the same place.
    #
    for test in suite_list:
        test = test.strip()
        if len(test):
            job = Job()
            job.set_is_example(False)
            job.set_is_pyexample(False)
            job.set_display_name(test)
            job.set_tmp_file_name(os.path.join(testpy_output_dir, "%s.xml" % test))
            job.set_cwd(os.getcwd())
            job.set_basedir(os.getcwd())
            job.set_tempdir(testpy_output_dir)
            if args.multiple:
                multiple = ""
            else:
                multiple = " --stop-on-failure"
            if len(args.fullness):
                fullness = args.fullness.upper()
                fullness = " --fullness=%s" % fullness
            else:
                fullness = " --fullness=QUICK"

            path_cmd = os.path.join(
                "utils", test_runner_name + " --test-name=%s%s%s" % (test, multiple, fullness)
            )

            job.set_shell_command(path_cmd)

            if args.valgrind and test in core_valgrind_skip_tests:
                job.set_is_skip(True)
                job.set_skip_reason("crashes valgrind")

            if args.rerun_failed and test in previously_run_tests_to_skip["test"]:
                job.is_skip = True
                job.set_skip_reason("didn't fail in the previous run")

            if args.verbose:
                print("Queue %s" % test)

            input_queue.put(job)
            jobs = jobs + 1
            total_tests = total_tests + 1

    #
    # We've taken care of the discovered or specified test suites.  Now we
    # have to deal with examples run as smoke tests.  We have a list of all of
    # the example programs it makes sense to try and run.  Each example will
    # have a condition associated with it that must evaluate to true for us
    # to try and execute it.  This is used to determine if the example has
    # a dependency that is not satisfied.
    #
    # We don't care at all how the trace files come out, so we just write them
    # to a single temporary directory.
    #
    # XXX As it stands, all of the trace files have unique names, and so file
    # collisions can only happen if two instances of an example are running in
    # two versions of the test.py process concurrently.  We may want to create
    # uniquely named temporary traces directories to avoid this problem.
    #
    # We need to figure out what examples to execute.  We are either given one
    # suite or example explicitly via the --suite or --example option, or we
    # need to walk the list of examples looking for available example
    # conditions.
    #
    # This translates into allowing the following options with respect to the
    # suites
    #
    #  ./test.py:                                           run all of the examples
    #  ./test.py --constrain=unit                           run no examples
    #  ./test.py --constrain=example                        run all of the examples
    #  ./test.py --suite=some-test-suite                    run no examples
    #  ./test.py --example=some-example                     run the single example with no parameters
    #  ./test.py --example="some-example --args=2"          run the single example with custom parameters
    #  ./test.py --example=some-example*                    run the all examples-to-run.py instances with said example
    #  ./test.py --suite=some-suite --example=some-example  run the single example
    #
    #
    if len(args.suite) == 0 and len(args.pyexample) == 0:
        if len(args.constrain) == 0 or args.constrain == "example":
            if ENABLE_EXAMPLES:
                if args.example:
                    if args.example.endswith("*"):
                        # If an example name is passed without arguments, we filter all examples containing said program
                        example_tests = list(
                            filter(lambda x: args.example[:-1] in x[0], example_tests)
                        )
                        args.example_args = []
                    else:
                        example_tests = list(
                            filter(
                                lambda x: " ".join([args.example, *args.example_args])
                                == x[0].split("/")[-1],
                                example_tests,
                            )
                        )
                        args.example_args = []

                    if not example_tests or args.example_args:
                        # If an example name is passed with arguments, we create an example entry for said example
                        example_name = " ".join([args.example, *args.example_args])
                        example_path = "%s%s-%s%s" % (
                            APPNAME,
                            VERSION,
                            args.example,
                            BUILD_PROFILE_SUFFIX,
                        )
                        if example_path in ns3_runnable_programs_dictionary:
                            example_path = ns3_runnable_programs_dictionary[example_path]
                            example_path += ".exe" if sys.platform == "win32" else ""
                            example_path = " ".join([example_path, *args.example_args])
                            example_tests = [(example_name, example_path, "True", "True", "QUICK")]
                        else:
                            print("No example matching the name %s" % example_name)
                            example_tests = []

                for name, test, do_run, do_valgrind_run, fullness in example_tests:
                    # Remove any arguments and directory names from test.
                    test_name = test.split(" ", 1)[0]
                    test_name = os.path.basename(test_name)
                    test_name = test_name[:-4] if sys.platform == "win32" else test_name

                    # Don't try to run this example if it isn't runnable.
                    if test_name in ns3_runnable_programs_dictionary:
                        if eval(do_run):
                            job = Job()
                            job.set_is_example(True)
                            job.set_is_pyexample(False)
                            job.set_display_name(name)
                            job.set_tmp_file_name("")
                            job.set_cwd(testpy_output_dir)
                            job.set_basedir(os.getcwd())
                            job.set_tempdir(testpy_output_dir)
                            job.set_shell_command(test)
                            job.set_build_path(args.buildpath)

                            if args.valgrind and not eval(do_valgrind_run):
                                job.set_is_skip(True)
                                job.set_skip_reason("skip in valgrind runs")

                            if (
                                args.rerun_failed
                                and name in previously_run_tests_to_skip["example"]
                            ):
                                job.is_skip = True
                                job.set_skip_reason("didn't fail in the previous run")

                            if args.verbose:
                                print("Queue %s" % test)

                            if args.fullness == "QUICK" and fullness != "QUICK":
                                job.set_is_skip(True)
                                job.set_skip_reason(
                                    f"skip {fullness} examples when QUICK run selected"
                                )
                            elif (
                                args.fullness == "EXTENSIVE"
                                and fullness != "EXTENSIVE"
                                and fullness != "QUICK"
                            ):
                                job.set_is_skip(True)
                                job.set_skip_reason(
                                    f"skip {fullness} examples when EXTENSIVE run selected"
                                )
                            # TAKES_FOREVER includes everything, so no need to exclude anything

                            input_queue.put(job)
                            jobs = jobs + 1
                            total_tests = total_tests + 1

    #
    # Run some Python examples as smoke tests.  We have a list of all of
    # the example programs it makes sense to try and run.  Each example will
    # have a condition associated with it that must evaluate to true for us
    # to try and execute it.  This is used to determine if the example has
    # a dependency that is not satisfied.
    #
    # We don't care at all how the trace files come out, so we just write them
    # to a single temporary directory.
    #
    # We need to figure out what python examples to execute.  We are either
    # given one pyexample explicitly via the --pyexample option, or we
    # need to walk the list of python examples
    #
    # This translates into allowing the following options with respect to the
    # suites
    #
    #  ./test.py --constrain=pyexample           run all of the python examples
    #  ./test.py --pyexample=some-example.py:    run the single python example
    #
    if len(args.suite) == 0 and len(args.example) == 0 and len(args.pyexample) == 0:
        if len(args.constrain) == 0 or args.constrain == "pyexample":
            for test, do_run, fullness in python_tests:
                # Remove any arguments and directory names from test.
                test_name = test.split(" ", 1)[0]
                test_name = os.path.basename(test_name)

                # Don't try to run this example if it isn't runnable.
                if test_name in ns3_runnable_scripts:
                    if eval(do_run):
                        job = Job()
                        job.set_is_example(False)
                        job.set_is_pyexample(True)
                        job.set_display_name(test)
                        job.set_tmp_file_name("")
                        job.set_cwd(testpy_output_dir)
                        job.set_basedir(os.getcwd())
                        job.set_tempdir(testpy_output_dir)
                        job.set_shell_command(test)
                        job.set_build_path("")

                        #
                        # Python programs and valgrind do not work and play
                        # well together, so we skip them under valgrind.
                        # We go through the trouble of doing all of this
                        # work to report the skipped tests in a consistent
                        # way through the output formatter.
                        #
                        if args.valgrind:
                            job.set_is_skip(True)
                            job.set_skip_reason("skip in valgrind runs")

                        #
                        # The user can disable python bindings, so we need
                        # to pay attention to that and give some feedback
                        # that we're not testing them
                        #
                        if not ENABLE_PYTHON_BINDINGS:
                            job.set_is_skip(True)
                            job.set_skip_reason("requires Python bindings")

                        if args.verbose:
                            print("Queue %s" % test)

                        if args.fullness == "QUICK" and fullness != "QUICK":
                            job.set_is_skip(True)
                            job.set_skip_reason(f"skip {fullness} examples when QUICK run selected")
                        elif (
                            args.fullness == "EXTENSIVE"
                            and fullness != "EXTENSIVE"
                            and fullness != "QUICK"
                        ):
                            job.set_is_skip(True)
                            job.set_skip_reason(
                                f"skip {fullness} examples when EXTENSIVE run selected"
                            )
                        # TAKES_FOREVER includes everything, so no need to exclude anything

                        input_queue.put(job)
                        jobs = jobs + 1
                        total_tests = total_tests + 1

    elif len(args.pyexample):
        # Find the full relative path to file if only a partial path has been given.
        if not os.path.exists(args.pyexample):
            import glob

            files = glob.glob("./**/%s" % args.pyexample, recursive=True)
            if files:
                args.pyexample = files[0]

        # Don't try to run this example if it isn't runnable.
        example_name = os.path.basename(args.pyexample)
        if example_name not in ns3_runnable_scripts:
            print("Example %s is not runnable." % example_name)
        elif not os.path.exists(args.pyexample):
            print("Example %s does not exist." % example_name)
        else:
            #
            # If you tell me to run a python example, I will try and run the example
            # irrespective of any condition.
            #
            job = Job()
            job.set_is_pyexample(True)
            job.set_display_name(args.pyexample)
            job.set_tmp_file_name("")
            job.set_cwd(testpy_output_dir)
            job.set_basedir(os.getcwd())
            job.set_tempdir(testpy_output_dir)
            job.set_shell_command(args.pyexample)
            job.set_build_path("")

            if args.verbose:
                print("Queue %s" % args.pyexample)

            input_queue.put(job)
            jobs = jobs + 1
            total_tests = total_tests + 1

    #
    # Tell the worker threads to pack up and go home for the day.  Each one
    # will exit when they see their is_break task.
    #
    for i in range(processors):
        job = Job()
        job.set_is_break(True)
        input_queue.put(job)

    #
    # Now all of the tests have been dispatched, so all we have to do here
    # in the main thread is to wait for them to complete.  Keyboard interrupt
    # handling is broken as mentioned above.  We use a signal handler to catch
    # sigint and set a global variable.  When the worker threads sense this
    # they stop doing real work and will just start throwing jobs back at us
    # with is_break set to True.  In this case, there are no real results so we
    # ignore them.  If there are real results, we always print PASS or FAIL to
    # standard out as a quick indication of what happened.
    #
    passed_tests = 0
    failed_tests = 0
    failed_testnames = []
    crashed_tests = 0
    crashed_testnames = []
    valgrind_errors = 0
    valgrind_testnames = []
    failed_jobs = []
    for i in range(jobs):
        job = output_queue.get()
        if job.is_break:
            continue

        if job.is_example or job.is_pyexample:
            kind = "Example"
        else:
            kind = "TestSuite"

        if job.is_skip:
            status = "SKIP"
            status_print = colors.GREY + status + colors.NORMAL
            skipped_tests = skipped_tests + 1
            skipped_testnames.append(job.display_name + (" (%s)" % job.skip_reason))
        else:
            failed_jobs.append(job)
            if job.returncode == 0:
                status = "PASS"
                status_print = colors.GREEN + status + colors.NORMAL
                passed_tests = passed_tests + 1
                failed_jobs.pop()
            elif job.returncode == 1:
                failed_tests = failed_tests + 1
                failed_testnames.append(job.display_name)
                status = "FAIL"
                status_print = colors.RED + status + colors.NORMAL
            elif job.returncode == 2:
                valgrind_errors = valgrind_errors + 1
                valgrind_testnames.append(job.display_name)
                status = "VALGR"
                status_print = colors.CYAN + status + colors.NORMAL
            else:
                crashed_tests = crashed_tests + 1
                crashed_testnames.append(job.display_name)
                status = "CRASH"
                status_print = colors.PINK + status + colors.NORMAL

        print("[%d/%d] %s" % (i, total_tests, status_print), end="")

        if args.duration or args.constrain == "performance":
            print(" (%.3f)" % job.elapsed_time, end="")

        print(":", end="")

        if "NS_COMMANDLINE_INTROSPECTION" in os.environ:
            print(" Wrote example usage for", end="")

        print(" %s %s" % (kind, job.display_name))

        if job.is_example or job.is_pyexample:
            #
            # Examples are the odd man out here.  They are written without any
            # knowledge that they are going to be run as a test, so we need to
            # cook up some kind of output for them.  We're writing an xml file,
            # so we do some simple XML that says we ran the example.
            #
            # XXX We could add some timing information to the examples, i.e. run
            # them through time and print the results here.
            #
            with open(xml_results_file, "a", encoding="utf-8") as f:
                f.write("<Example>\n")
                example_name = "  <Name>%s</Name>\n" % job.display_name
                f.write(example_name)

                if status == "PASS":
                    f.write("  <Result>PASS</Result>\n")
                elif status == "FAIL":
                    f.write("  <Result>FAIL</Result>\n")
                elif status == "VALGR":
                    f.write("  <Result>VALGR</Result>\n")
                elif status == "SKIP":
                    f.write("  <Result>SKIP</Result>\n")
                    f.write("  <Reason>%s</Reason>\n" % job.skip_reason)
                else:
                    f.write("  <Result>CRASH</Result>\n")

                f.write('  <Time real="%.3f"/>\n' % job.elapsed_time)
                f.write("</Example>\n")

        else:
            #
            # If we're not running an example, we're running a test suite.
            # These puppies are running concurrently and generating output
            # that was written to a temporary file to avoid collisions.
            #
            # Now that we are executing sequentially in the main thread, we can
            # concatenate the contents of the associated temp file to the main
            # results file and remove that temp file.
            #
            # One thing to consider is that a test suite can crash just as
            # well as any other program, so we need to deal with that
            # possibility as well.  If it ran correctly it will return 0
            # if it passed, or 1 if it failed.  In this case, we can count
            # on the results file it saved being complete.  If it crashed, it
            # will return some other code, and the file should be considered
            # corrupt and useless.  If the suite didn't create any XML, then
            # we're going to have to do it ourselves.
            #
            # Another issue is how to deal with a valgrind error.  If we run
            # a test suite under valgrind and it passes, we will get a return
            # code of 0 and there will be a valid xml results file since the code
            # ran to completion.  If we get a return code of 1 under valgrind,
            # the test case failed, but valgrind did not find any problems so the
            # test case return code was passed through.  We will have a valid xml
            # results file here as well since the test suite ran.  If we see a
            # return code of 2, this means that valgrind found an error (we asked
            # it to return 2 if it found a problem in run_job_synchronously) but
            # the suite ran to completion so there is a valid xml results file.
            # If the suite crashes under valgrind we will see some other error
            # return code (like 139).  If valgrind finds an illegal instruction or
            # some other strange problem, it will die with its own strange return
            # code (like 132).  However, if the test crashes by itself, not under
            # valgrind we will also see some other return code.
            #
            # If the return code is 0, 1, or 2, we have a valid xml file.  If we
            # get another return code, we have no xml and we can't really say what
            # happened -- maybe the TestSuite crashed, maybe valgrind crashed due
            # to an illegal instruction.  If we get something beside 0-2, we assume
            # a crash and fake up an xml entry.  After this is all done, we still
            # need to indicate a valgrind error somehow, so we fake up an xml entry
            # with a VALGR result.  Thus, in the case of a working TestSuite that
            # fails valgrind, we'll see the PASS entry for the working TestSuite
            # followed by a VALGR failing test suite of the same name.
            #
            if job.is_skip:
                with open(xml_results_file, "a", encoding="utf-8") as f:
                    f.write("<Test>\n")
                    f.write("  <Name>%s</Name>\n" % job.display_name)
                    f.write("  <Result>SKIP</Result>\n")
                    f.write("  <Reason>%s</Reason>\n" % job.skip_reason)
                    f.write("</Test>\n")
            else:
                failed_jobs.append(job)
                if job.returncode == 0 or job.returncode == 1 or job.returncode == 2:
                    with open(xml_results_file, "a", encoding="utf-8") as f_to, open(
                        job.tmp_file_name, encoding="utf-8"
                    ) as f_from:
                        contents = f_from.read()
                        if status == "VALGR":
                            pre = contents.find("<Result>") + len("<Result>")
                            post = contents.find("</Result>")
                            contents = contents[:pre] + "VALGR" + contents[post:]
                        f_to.write(contents)
                        # When running with sanitizers, the program may
                        # crash before ever writing the expected xml
                        # output file
                        try:
                            et = ET.parse(job.tmp_file_name)
                            if et.find("Result").text in ["PASS", "SKIP"]:
                                failed_jobs.pop()
                        except:
                            pass
                else:
                    with open(xml_results_file, "a", encoding="utf-8") as f:
                        f.write("<Test>\n")
                        f.write("  <Name>%s</Name>\n" % job.display_name)
                        f.write("  <Result>CRASH</Result>\n")
                        f.write("</Test>\n")

    #
    # We have all of the tests run and the results written out.  One final
    # bit of housekeeping is to wait for all of the threads to close down
    # so we can exit gracefully.
    #
    for thread in threads:
        thread.join()

    #
    # Back at the beginning of time, we started the body of an XML document
    # since the test suites and examples were going to just write their
    # individual pieces.  So, we need to finish off and close out the XML
    # document
    #
    with open(xml_results_file, "a", encoding="utf-8") as f:
        f.write("</Results>\n")

    #
    # Print a quick summary of events
    #
    print(
        "%d of %d tests passed (%d passed, %d skipped, %d failed, %d crashed, %d valgrind errors)"
        % (
            passed_tests,
            total_tests,
            passed_tests,
            skipped_tests,
            failed_tests,
            crashed_tests,
            valgrind_errors,
        )
    )
    #
    # Repeat summary of skipped, failed, crashed, valgrind events
    #
    if skipped_testnames:
        skipped_testnames.sort()
        print("List of SKIPped tests:\n    %s" % "\n    ".join(map(str, skipped_testnames)))
    if failed_testnames:
        failed_testnames.sort()
        print("List of FAILed tests:\n    %s" % "\n    ".join(map(str, failed_testnames)))
    if crashed_testnames:
        crashed_testnames.sort()
        print("List of CRASHed tests:\n    %s" % "\n    ".join(map(str, crashed_testnames)))
    if valgrind_testnames:
        valgrind_testnames.sort()
        print("List of VALGR failures:\n    %s" % "\n    ".join(map(str, valgrind_testnames)))

    if failed_jobs and args.verbose_failed:
        for job in failed_jobs:
            if job.standard_out or job.standard_err:
                job_type = "example" if (job.is_example or job.is_pyexample) else "test suite"
                print(
                    f"===================== Begin of {job_type} '{job.display_name}' stdout ====================="
                )
                print(job.standard_out)
                print(
                    f"===================== Begin of {job_type} '{job.display_name}' stderr ====================="
                )
                print(job.standard_err)
                print(
                    f"===================== End of {job_type} '{job.display_name}' =============================="
                )
    #
    # The last things to do are to translate the XML results file to "human-
    # readable form" if the user asked for it (or make an XML file somewhere)
    #
    if len(args.html) + len(args.text) + len(args.xml):
        print()

    if len(args.html):
        translate_to_html(xml_results_file, args.html)

    if len(args.text):
        translate_to_text(xml_results_file, args.text)

    if len(args.xml):
        xml_file = args.xml + (".xml" if ".xml" not in args.xml else "")
        print("Writing results to xml file %s..." % xml_file, end="")
        shutil.copyfile(xml_results_file, xml_file)
        print("done.")

    #
    # Let the user know if they need to turn on tests or examples.
    #
    if not ENABLE_TESTS or not ENABLE_EXAMPLES:
        print()
        if not ENABLE_TESTS:
            print("***  Note: ns-3 tests are currently disabled. Enable them by adding")
            print('***  "--enable-tests" to ./ns3 configure or modifying your .ns3rc file.')
            print()
        if not ENABLE_EXAMPLES:
            print("***  Note: ns-3 examples are currently disabled. Enable them by adding")
            print('***  "--enable-examples" to ./ns3 configure or modifying your .ns3rc file.')
            print()

    #
    # Let the user know if they tried to use valgrind but it was not
    # present on their machine.
    #
    if args.valgrind and not VALGRIND_FOUND:
        print()
        print("***  Note: you are trying to use valgrind, but valgrind could not be found")
        print("***  on your machine.  All tests and examples will crash or be skipped.")
        print()

    #
    # If we have been asked to retain all of the little temporary files, we
    # don't delete tm.  If we do delete the temporary files, delete only the
    # directory we just created.  We don't want to happily delete any retained
    # directories, which will probably surprise the user.
    #
    if not args.retain:
        shutil.rmtree(testpy_output_dir)

    if passed_tests + skipped_tests == total_tests:
        return 0  # success
    else:
        return 1  # catchall for general errors


def split_program_and_arguments(argv):
    split_argv = re.findall(r'(?:".*[|*]?"|\S)+', argv)
    program = ""
    program_args = []
    if split_argv:
        program = split_argv[0]
    if len(split_argv) > 1:
        program_args = split_argv[1:]
    return program, program_args


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--buildpath",
        action="store",
        type=str,
        default="",
        help="specify the path where ns-3 was built (defaults to the build directory for the current variant)",
    )

    parser.add_argument(
        "-c",
        "--constrain",
        action="store",
        type=str,
        default="",
        help="constrain the test-runner by kind of test",
    )

    parser.add_argument(
        "-d",
        "--duration",
        action="store_true",
        default=False,
        help="print the duration of each test suite and example",
    )

    parser.add_argument(
        "-e",
        "--example",
        action="store",
        type=str,
        default="",
        help="specify a single example to run (no relative path is needed)",
    )

    parser.add_argument(
        "-u",
        "--update-data",
        action="store_true",
        default=False,
        help="If examples use reference data files, get them to re-generate them",
    )

    parser.add_argument(
        "-f",
        "--fullness",
        action="store",
        type=str,
        default="QUICK",
        choices=["QUICK", "EXTENSIVE", "TAKES_FOREVER"],
        help="choose the duration of tests to run: QUICK, EXTENSIVE, or TAKES_FOREVER, where EXTENSIVE includes QUICK and TAKES_FOREVER includes QUICK and EXTENSIVE (only QUICK tests are run by default)",
    )

    parser.add_argument(
        "-g",
        "--grind",
        action="store_true",
        dest="valgrind",
        default=False,
        help="run the test suites and examples using valgrind",
    )

    parser.add_argument(
        "-k",
        "--kinds",
        action="store_true",
        default=False,
        help="print the kinds of tests available",
    )

    parser.add_argument(
        "-l", "--list", action="store_true", default=False, help="print the list of known tests"
    )

    parser.add_argument(
        "-m",
        "--multiple",
        action="store_true",
        default=False,
        help="report multiple failures from test suites and test cases",
    )

    parser.add_argument(
        "-n",
        "--no-build",
        action="store_true",
        default=False,
        help="do not build before starting testing",
    )

    parser.add_argument(
        "-p",
        "--pyexample",
        action="store",
        type=str,
        default="",
        help="specify a single python example to run (with relative path)",
    )

    parser.add_argument(
        "-r",
        "--retain",
        action="store_true",
        default=False,
        help="retain all temporary files (which are normally deleted)",
    )

    parser.add_argument(
        "-s",
        "--suite",
        action="store",
        type=str,
        default="",
        help="specify a single test suite to run",
    )

    parser.add_argument(
        "-t",
        "--text",
        action="store",
        type=str,
        default="",
        metavar="TEXT-FILE",
        help="write detailed test results into TEXT-FILE.txt",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="print progress and informational messages",
    )

    parser.add_argument(
        "--verbose-failed",
        action="store_true",
        default=False,
        help="print progress and informational messages for failed jobs",
    )

    parser.add_argument(
        "-w",
        "--web",
        "--html",
        action="store",
        type=str,
        dest="html",
        default="",
        metavar="HTML-FILE",
        help="write detailed test results into HTML-FILE.html",
    )

    parser.add_argument(
        "-x",
        "--xml",
        action="store",
        type=str,
        default="",
        metavar="XML-FILE",
        help="write detailed test results into XML-FILE.xml",
    )

    parser.add_argument(
        "--nocolor",
        action="store_true",
        default=False,
        help="do not use colors in the standard output",
    )

    parser.add_argument(
        "--jobs",
        action="store",
        type=int,
        dest="process_limit",
        default=0,
        help="limit number of worker threads",
    )

    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        dest="rerun_failed",
        default=False,
        help="rerun failed tests",
    )

    global args
    args = parser.parse_args()
    args.example, exargs = split_program_and_arguments(args.example)
    setattr(args, "example_args", exargs)
    signal.signal(signal.SIGINT, sigint_hook)

    # From waf/waflib/Options.py
    envcolor = os.environ.get("NOCOLOR", "") and "no" or "auto" or "yes"

    if args.nocolor or envcolor == "no":
        colors_lst["USE"] = False

    return run_tests()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
