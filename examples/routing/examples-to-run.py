#! /usr/bin/env python3

# A list of C++ examples to run in order to ensure that they remain
# buildable and runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run, do_valgrind_run).
#
# See test.py for more information.
cpp_examples = [
    ("dynamic-global-routing", "True", "True"),
    ("global-injection-slash32", "True", "True"),
    ("global-routing-slash32", "True", "True"),
    ("mixed-global-routing", "True", "True"),
    ("simple-alternate-routing", "True", "True"),
    ("simple-global-routing", "True", "True"),
    ("simple-routing-ping6", "True", "True"),
    ("static-routing-slash32", "True", "True"),
    (
        "dense-routing-scaling --protocol=aodv --nodes=300 --flows=300 --simTime=10s",
        "NS_PERFORMANCE == True",
        "False",
        "TAKES_FOREVER",
    ),
    (
        "dense-routing-scaling --protocol=dsr --nodes=80 --flows=80 --simTime=10s",
        "NS_PERFORMANCE == True",
        "False",
        "TAKES_FOREVER",
    ),
    (
        "dense-routing-scaling --protocol=olsr --nodes=150 --flows=0 --simTime=20s",
        "NS_PERFORMANCE == True",
        "False",
        "TAKES_FOREVER",
    ),
]

# A list of Python examples to run in order to ensure that they remain
# runnable over time.  Each tuple in the list contains
#
#     (example_name, do_run).
#
# See test.py for more information.
python_examples = [
    ("simple-routing-ping6.py", "True"),
]
