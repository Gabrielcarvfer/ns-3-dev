#! /usr/bin/env python3

import os
import shutil
import subprocess
import sys
from multiprocessing import Pool


def pool_filter(pool, func, candidates):
    return [c for c, keep in zip(candidates, pool.map(func, candidates)) if keep]


def filter_ns3_symbol(mangled_symbol):
    cxxfilt_executable = shutil.which("c++filt")
    if not cxxfilt_executable:
        return -1
    else:
        cxxfilt_executable = os.path.basename(cxxfilt_executable)

    demangled_symbol = subprocess.check_output([cxxfilt_executable, mangled_symbol]).decode()
    if not demangled_symbol.startswith("ns3::"):
        return False
    return True


def main():
    nm_executable = shutil.which("nm")
    if not nm_executable:
        return -1
    else:
        nm_executable = os.path.basename(nm_executable)

    if len(sys.argv) < 4:
        return -1

    if "python" in sys.argv[0]:
        sys.argv.pop(0)

    module_name = sys.argv[1]
    static_library_path = sys.argv[2]
    export_file_path = sys.argv[3]

    if not os.path.exists(static_library_path):
        return -1

    try:
        ret = subprocess.run(
            [nm_executable, "-g", "--defined-only", static_library_path], capture_output=True
        )
    except Exception as e:
        return -1

    if ret.returncode != 0:
        return -1

    # Decode output
    mangled_symbols = ret.stdout.decode()

    # Split into lines
    mangled_symbols = mangled_symbols.splitlines()

    # Remove lines with more or less than 3 columns
    mangled_symbols = list(filter(lambda x: len(x.split()) > 1, mangled_symbols))

    # Remove address and symbol type
    mangled_symbols = list(map(lambda x: x.split()[-1], mangled_symbols))

    # Filter out standard library symbols
    mangled_symbols = list(
        filter(
            lambda x: not (
                "_ZNKSt" in x or "_ZSt" in x or "_ZNSt" in x or "_ZNSa" in x or "Eigen" in x
            ),
            mangled_symbols,
        )
    )

    # Demangle symbols in parallel and filter them (EXTREMELY slow)
    # with Pool(processes=os.cpu_count()-1) as pool:
    #    mangled_symbols = pool_filter(pool, filter_ns3_symbol, mangled_symbols)

    with open(export_file_path, "w") as f:
        f.write(f"LIBRARY {module_name}\nEXPORTS\n")
        f.write("\n".join(mangled_symbols))
    return 0


if __name__ == "__main__":
    exit(main())
