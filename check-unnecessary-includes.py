#! /usr/bin/env python3
# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-
#
# Copyright (c) 2021 Universidade de Brasília
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>

import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
import subprocess
import sys


def load_translation_units(compile_commands_json_path):
    if not os.path.exists(compile_commands_json_path) or not os.path.isfile(compile_commands_json_path):
        raise Exception("Inexisting compile_commands.json file:", compile_commands_json_path)

    try:
        with open(compile_commands_json_path, "r") as file:
            contents = json.load(file)
    except Exception as e:
        raise e

    return contents


def optimize_includes_translation_unit(unit):
    changed = False

    # Check if file exists
    if not os.path.exists(unit['file']) or not os.path.isfile(unit['file']):
        raise Exception("Source file not found:", unit['file'])

    comm = unit['command']
    comm = comm.replace('\\"', '/"').replace('\\','/').replace('//','/').replace('/"','\\"')
    if sys.platform == "win32":
        comm = comm.replace('/','\\\\')

    ret = subprocess.run(["python3", "../cleanup.py", *comm.split()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="./cmake-cache")
    if "no changes" not in ret.stdout.decode():
        print(unit['file'], "was changed")
    return

def main():
    translation_units = load_translation_units("./cmake-cache/compile_commands.json")
    n_threads = max(1, os.cpu_count()-1)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        pool.map(optimize_includes_translation_unit, translation_units)
    return


main()
