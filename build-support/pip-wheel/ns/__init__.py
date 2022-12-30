# This is a stub module that loads the actual ns-3
# bindings from nsnam.ns
import sys

try:
    import nsnam.ns
    sys.modules['ns'] = nsnam.ns
except ModuleNotFoundError as e:
    print("Install the nsnam package with pip install nsnam.", file=sys.stderr)
    exit(-1)
