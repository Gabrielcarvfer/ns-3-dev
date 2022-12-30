# This is a stub module that loads the actual visualizer
# from nsnam.visualizer
import sys

try:
    import nsnam.visualizer
    sys.modules['visualizer'] = nsnam.visualizer
except ModuleNotFoundError as e:
    print("Install the nsnam package with pip install nsnam.", file=sys.stderr)
    exit(-1)
