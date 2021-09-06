#!/usr/bin/env python3

# Copyright (C) 2017-2021 Universidade de Brasília

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>

help = (
    "This script compares the output of different './test.py -d' runs\n"
    "It can be used to identify performance improvements or regressions by following these steps:\n"
    "1. After making sure all your changes are working correctly, run './test.py -d > after.txt'\n"
    "2. Checkout the ns-3 branch without your commits, then run './test.py -d > before.txt'\n"
    "3. Run './utils/speedup.py before.txt after.txt' to get a table comparing results and speedups\n"
    "4. Use the '-f' to show most relevant results, '-td' to sort by time difference and '-s' to sort by speedup\n"
)

import argparse
import re
import sys

SECS_IN_DAY = 86400
SECS_IN_HOUR = 3600
SECS_IN_MINUTE = 60


def euclidian_div(a, b):
    return a // b, a % b


def seconds_to_dhms(seconds):
    days, seconds = euclidian_div(seconds, SECS_IN_DAY)
    hours, seconds = euclidian_div(seconds, SECS_IN_HOUR)
    minutes, seconds = euclidian_div(seconds, SECS_IN_MINUTE)
    return days, hours, minutes, seconds


def get_timestamp(seconds):
    d, h, m, s = seconds_to_dhms(seconds)
    if d > 0:
        return "%.2dd %.2dh" % (d, h)
    if h > 0:
        return "%.2dh %.2dm" % (h, m)
    if m > 0:
        return "%.2dm %.2ds" % (m, s)
    if s > 0:
        return "%.2ds" % s
    return ''


# Measure speedup between to `./test.py -d` outputs
def get_speedup(before_dict, after_dict):
    for key in before_dict.keys():
        timediff = after_dict[key] - before_dict[key]
        speedup = before_dict[key] / after_dict[key]
        before_dict[key] = {"before": before_dict[key],
                            "after": after_dict[key],
                            "speedup": speedup,
                            "timediff": timediff
                            }
    return before_dict


def print_speedup(results_dict, output_file, sorting_type, filter_relevant_results):
    # Get before and after total times, plus tests that improved or worsened
    improved = []
    worsened = []
    stable = []
    before_time = 0
    after_time = 0
    timediffs = []
    for test in results_dict.keys():
        before_time += results_dict[test]["before"]
        after_time += results_dict[test]["after"]
        if results_dict[test]["speedup"] > 1:
            improved.append(test)
        elif results_dict[test]["speedup"] < 1:
            worsened.append(test)
        else:
            stable.append(test)
        timediffs.append(results_dict[test]["timediff"])

    if sorting_type == "name":
        sorted_tests = sorted(results_dict.keys())
    elif sorting_type == "speedup":
        sorted_tests = sorted(results_dict.keys(),
                              key=lambda x: results_dict[x]["speedup"],
                              reverse=True)
    elif sorting_type == "timediff":
        sorted_tests = sorted(results_dict.keys(),
                              key=lambda x: results_dict[x]["timediff"], )

    if filter_relevant_results:
        filtered_tests = []
        import statistics
        threshold = max(map(abs, statistics.quantiles(timediffs, n=10)))  # pick threshold based on the deciles
        # Filter out results outside the largest decile
        for test in sorted_tests:
            absolute_diff = abs(results_dict[test]["timediff"])
            if absolute_diff > threshold:
                filtered_tests.append(test)
        sorted_tests = filtered_tests

    print("%15s %15s %15s %s" % ("Before (s)",
                                 "After (s)",
                                 "Speedup (x)" if sorting_type != "timediff" else "Difference (s)",
                                 "Test name"
                                 ),
          file=output_file)

    for test in sorted_tests:
        print("%15.3f %15.3f %15.3f %s" % (results_dict[test]["before"],
                                           results_dict[test]["after"],
                                           results_dict[test]["speedup" if sorting_type != "timediff" else "timediff"],
                                           test),
              file=output_file)

    print("\n\n%d improved, %d worsened, %d stable" % (len(improved),
                                                       len(worsened),
                                                       len(stable)
                                                       ),
          file=output_file)
    print("Absolute Speedup: %.3fx (Before %s - After %s)" % (before_time / after_time,
                                                              get_timestamp(before_time),
                                                              get_timestamp(after_time)),
          file=output_file)


def load_time_dict(test_py_output_file):
    lines = test_py_output_file.readlines()

    output_dict = {}

    for line in lines:
        groups = re.findall(".*\((.*)\): (.*)", line)
        if not groups:
            continue
        output_dict[groups[0][1]] = max(float(groups[0][0]), 0.001)
    return output_dict


def main():
    parser = argparse.ArgumentParser(description=help, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', help='output file', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('-s', action='store_true', default=False, help='sort by speedup')
    parser.add_argument('-td', action='store_true', default=False, help='sort by largest time differences')
    parser.add_argument('-f', action='store_true', default=False, help='filter to relevant results')
    parser.add_argument('input_file', nargs=2, help='input files a and b', type=argparse.FileType(), default=None)
    args = parser.parse_args()
    if not args.input_file[0] or not args.input_file[1]:
        raise Exception("Missing input files")
    before = load_time_dict(args.input_file[0])
    after = load_time_dict(args.input_file[1])

    sorting_type = "name"
    if args.s:
        sorting_type = "speedup"
    if args.td:
        sorting_type = "timediff"

    print_speedup(results_dict=get_speedup(before, after),
                  output_file=args.o,
                  sorting_type=sorting_type,
                  filter_relevant_results=args.f)


main()
