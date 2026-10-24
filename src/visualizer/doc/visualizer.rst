.. include:: replace.txt
.. highlight:: python

.. heading hierarchy:
   ============= Chapter
   ------------- Section (#.#)
   ~~~~~~~~~~~~~ Subsection (#.#.#)


PyViz visualizer
================

PyViz is a live visualizer for |ns3| simulations, originally created by
Gustavo Carneiro [`1`_] and rewritten for web browsers by Gabriel Ferreira.
It shows the topology, the packets moving through it and the state of every
object while the simulation runs, without trace files. It runs in a web
browser and only needs the |ns3| Python bindings; the browser page is served
by the Python process that runs the simulation, from ``127.0.0.1``, with no
external resources, so it works offline.

The source code lives in ``src/visualizer``. The visualizer has three layers:


* ``model/pyviz.{h,cc}`` (C++): the ``ns3::visualizer::PyViz`` helper. It
  steps the simulator, pairs each packet transmission with its receptions,
  and keeps transmission, drop, packet and interface statistics samples. It
  only links against the ``core`` and ``network`` modules: its trace
  handlers are generic (a device sent a packet, a device received a packet,
  a node dropped a packet).
* ``visualizer/*.py`` (Python): everything else. ``traces.py`` decides which
  trace sources of which device types feed the generic handlers, skipping
  modules that are not built, and compiles small C++ adapters at run time
  (with Cppyy) for trace sources with other signatures (IPv4 drops, NR device
  transmissions, the LTE eNB and NR gNB applications). ``web.py`` paces the simulation, computes the topology,
  layout and rates, serves the page and implements every user action.
  ``web_plugins/`` holds the panels and overlays.
* ``visualizer/web`` (HTML, CSS, JavaScript): presentation only. The page
  renders the JSON snapshots it receives and posts user actions back.

The page polls the Python process about 30 times per second. Each poll runs
the simulator for the wall-clock time elapsed since the previous poll
multiplied by the chosen speed, so the speed is in simulated seconds per
wall-clock second regardless of the poll rate. One poll never spends more
than about 40 ms running the simulator: when a simulation cannot keep up
with the requested speed it runs slower, the page stays responsive, and the
status line shows the speed actually achieved.

Scope and Limitations
---------------------

* The visualizer runs Python-driven simulations (or C++ programs through
  ``VisualSimulatorImpl``, which starts it from embedded Python), so the
  Python bindings must be enabled.
* Wired links are drawn from point-to-point and CSMA channels; channels with
  more than two devices are drawn as a hub. Wireless devices produce no
  links; their traffic appears as transmission arrows, Wi-Fi stations
  are joined to their access point by a dashed line, and LTE and NR UEs to
  their serving cell. Nodes without a
  mobility model are placed by an automatic layout.
* Transmissions are displayed for Wi-Fi, LR-WPAN, CSMA, point-to-point,
  LTE and NR (with EPC) devices; other devices can be added in ``traces.py``.
* Packet arrows are slowed down so they can be followed: each hop is shown
  for at least 0.7 s, starting at its real offset in time. At high speeds
  only a sample of the packets is drawn.
* Objects created with a Python constructor (for example
  ``ns.CsmaNetDevice()`` instead of ``ns.CreateObject[ns.CsmaNetDevice]()``)
  report their base TypeId in the object tree.

Visualizer General GUI options
------------------------------

.. _fig-pyviz-start:

.. figure:: figures/pyviz-start.*

   The page right after start. The simulation runs at the slowest speed
   (0.01x) so the first events can be seen.

The toolbar is a single row:

* **Pause** / **Play**: one button that pauses and resumes the simulation.
* **Restart**: destroys the simulation and reruns the scenario script from
  the beginning. A C++ program is executed again instead (Linux only); it
  serves on the same port, so the page reconnects to it.
* **Speed**: from 0.01x to 100x simulated seconds per wall-clock second. The
  page starts at 0.01x. When the simulation cannot keep up, the status line
  shows the speed actually reached, for example ``running at 1.32x``.
* **Smooth**: how many steps the link rates are averaged over.
* **Traffic**: draw the transmissions of all nodes, of the selected nodes
  only, or none.
* **Size** (node size), **Trails**, **Fit** (fit the view to the scenario)
  and **Screenshot** (save the topology as SVG).
* **Panels...**: windows that are not tied to one node, such as the flow
  monitor and the energy overview.
* The status line: simulation state, simulated time and number of nodes.

On narrow windows the control labels give way to their tooltips, so the
toolbar stays on one row.

.. _fig-pyviz-overview:

.. figure:: figures/pyviz-overview.*

   Running at 1x. Green arrows are transmissions, labelled with their rate
   and drawn on the right-hand side of the link for each direction. Yellow
   arrows are packets. The red arrow below a node shows the rate of packets
   it drops. Nodes with an energy source show a battery.

In the topology:

* Scroll to zoom, drag the background to pan, drag a node to move it (nodes
  with a mobility model are moved in the simulation too, kept inside the
  model's ``Bounds`` when it has them). While dragging, the position under
  the pointer is shown as Cartesian coordinates ``(x, y, z)`` in metres; z
  is kept.
* Selecting a node whose mobility model has ``Bounds`` (random walk, random
  direction, Gauss-Markov, ...) draws that area as a dashed rectangle.
* Hover a node for a summary: name, mobility model, position, devices,
  addresses and battery.
* Nodes with mobility are drawn in red, the others in green. A white dot is
  a shared channel (CSMA bus). Dashed red lines join Wi-Fi stations to their
  access point, and dashed blue lines join LTE and NR UEs to their serving
  eNB or gNB.
* Nodes with an energy source (energy module) show a battery: its fill and
  colour follow the remaining charge, the percentage is written next to it,
  and hovering it shows the remaining and total charge in mAh and J.
  Clicking it opens the node's **Battery** panel.
* **Trails** draws the recent path of every moving node, brighter towards
  its current position.

.. _fig-pyviz-mobility-bounds:

.. figure:: figures/pyviz-mobility-bounds.*

   A selected Wi-Fi station and the bounds of its random walk (dashed). The
   other station walks in the same area.

.. _fig-pyviz-drag-coordinates:

.. figure:: figures/pyviz-drag-coordinates.*

   Dragging the station: the label shows the position under the pointer.
   Here the pointer is outside the station's bounds (x starts at 110 m), so
   when released the station is placed at the nearest point inside them.

.. _fig-pyviz-trails:

.. figure:: figures/pyviz-trails-battery.*

   Trails of the two Wi-Fi stations, zoomed in. Their batteries are at 28%.

Inspecting objects
~~~~~~~~~~~~~~~~~~

Click a node to select it. Shift-click or Ctrl-click adds nodes to the
selection or removes them; panels such as the flow monitor use the whole
selection.

The **Node** tab shows the selected node as a tree of objects: its devices
and their channels and queues, its applications, and the objects aggregated
to it (IP stack, mobility model, routing, ...). Expanding an entry loads
it. Each object lists its identifying values (IDs, addresses, MTU, IP
addresses) and its attributes; writable attributes can be changed with
**Apply**, and ns-3 validates the new value.

.. _fig-pyviz-object-tree:

.. figure:: figures/pyviz-object-tree.*

   The object tree of the Wi-Fi access point, with its device expanded.

Panels
~~~~~~

Right-click a node for its panels, or use the buttons in the **Node** tab:

* **Node summary**: devices, MAC and IP addresses, mobility and battery.
* **Interface statistics**: packet and byte counters and rates per device.
* **IPv4 routing table**: static and global routes.
* **OLSR routing table**: when the node runs OLSR.
* **Last packets**: the last packets sent, received and dropped, with their
  decoded headers. The checklist selects which headers to capture, combined
  with OR or AND.
* **Follow node**: keeps the node centred while it moves.

.. _fig-pyviz-node-menu:

.. figure:: figures/pyviz-node-menu.*

   The node menu.

Panels open as windows over the topology. Drag them by the title bar, click
one to bring it to the front, and use **Minimize** to send it to the taskbar
in the top left corner of the topology (where panels, which open on the
right, do not cover it); clicking its button restores it. Minimized windows
stop refreshing.

.. _fig-pyviz-panels:

.. figure:: figures/pyviz-panels.*

   Two panel windows side by side, a third one minimized to the taskbar in
   the top left corner.

Types and console
~~~~~~~~~~~~~~~~~

The **Types** tab lists every registered TypeId with its attributes, trace
sources and the methods Cppyy exposes. Attributes that can be set at
construction time can be given a new default for objects created afterwards
(like ``Config::SetDefault()``).

.. _fig-pyviz-types:

.. figure:: figures/pyviz-types.*

   The Types tab.

The **Console** tab runs Python in the simulation process. ``ns``, ``viz``
(the visualizer), ``node`` (the primary selected node) and ``nodes`` (all
selected nodes) are defined. Shift+Enter runs the input.

.. _fig-pyviz-console:

.. figure:: figures/pyviz-console.*

   The console, with the selected node available as ``node``.

The sidebar's **Minimize** button gives the topology the whole width; the
sidebar then waits in the taskbar like a minimized window.

.. _fig-pyviz-sidebar-hidden:

.. figure:: figures/pyviz-sidebar-hidden.*

   The sidebar minimized to the taskbar.

Energy GUI options
------------------

The **Battery** panel of a node (click its battery, or use the node menu)
plots its charge and remaining energy over time and lists its energy
sources, its drainers (the device energy models attached to them, with their
current draw and the energy they consumed) and its charge sources (energy
harvesters, with their power). History is sampled at most every 0.1 s of
simulated time; at high speeds, once per simulation step.

.. _fig-pyviz-battery-panel:

.. figure:: figures/pyviz-battery-panel.*

   The Battery panel of a Wi-Fi station. The radio's energy model switched
   the radio off at 28% here, so the charge stays flat afterwards.

**Panels... > Energy** summarizes the network: the energy at the start, now,
and consumed, the charge of each node over time, the total remaining energy
over time, and every source, drainer and harvester.

.. _fig-pyviz-energy-panel:

.. figure:: figures/pyviz-energy-panel.*

   The Energy panel.

Flow Monitor
------------

The flow monitor panel shows the statistics collected by the
:ref:`flow-monitor` module.

The flow monitor panel plots, per flow, the throughput, the latency and the
jitter (means over each 0.1 s of simulated time), and the total bytes sent
and received, from the ``flow-monitor`` module. It is optional: enable it
from **Panels... > Flow monitor > Enable flow monitor**, or from the start
in the scenario::

  visualizer.enable_flow_monitor()
  # or, with a FlowMonitorHelper the scenario already installed:
  visualizer.enable_flow_monitor(helper)

When the scenario, in Python or C++, already installs a FlowMonitor with a
``FlowMonitorHelper``, the panel shows that monitor from the start, without
any change to the scenario: all ``FlowMonitorHelper`` objects of a simulation
share one FlowMonitor, which the panel retrieves.

With nodes selected, only the flows to or from them are shown, so selecting
the two ends of a connection shows just that traffic. Up to eight flows are
plotted; all of them are listed in the table.

.. _fig-pyviz-flow-monitor:

.. figure:: figures/pyviz-flow-monitor.*

   The flows between two selected nodes.

Hover a chart to read the values at that time. Drag across a chart to zoom
into that time range; **Reset zoom** or a double-click zooms back out.

.. _fig-pyviz-flow-monitor-zoom:

.. figure:: figures/pyviz-flow-monitor-zoom.*

   A zoomed throughput chart.

Usage
-----

This section describes how to build the visualizer, how to run a
simulation with it, and how scenarios can customize the view or extend it
with plugins.

Building
~~~~~~~~

Configure |ns3| with the Python bindings:

.. sourcecode:: bash

  $ ./ns3 configure --enable-python-bindings
  $ ./ns3 build

The visualizer module is built whenever the bindings are enabled. It needs
no other package: GTK, GooCanvas, PyGObject, GraphViz and IPython, which
earlier versions required, are no longer used.

Running
~~~~~~~

In a Python scenario, build the topology and applications as usual and end
with ``visualizer.start()`` instead of ``ns.Simulator.Run()``::

  from ns import ns
  import visualizer

  # ... create nodes, devices, stacks, applications ...
  ns.Simulator.Stop(ns.Seconds(10))
  visualizer.start()

Then run the script, for example with ``./ns3 run scratch/my-scenario.py``.
It prints the address of the page (``PyViz is available at
http://127.0.0.1:...``) and opens it in the default browser. The simulation
advances while the page is open and stops at ``Simulator::Stop()``.

C++ programs, and Python programs that call ``ns.Simulator.Run()``, can be
visualized by selecting ``VisualSimulatorImpl``, which starts the visualizer
when the simulation runs:

.. sourcecode:: bash

  $ ./ns3 run 'csma-bridge.py --SimulatorImplementationType=ns3::VisualSimulatorImpl'
  $ ./ns3 run wifi-simple-adhoc --visualize

Customizing the view
~~~~~~~~~~~~~~~~~~~~

Scenarios can register code that runs when the visualizer starts::

  import visualizer

  def hook(viz):
      viz.set_node_color(0, "#ff00ff")
      viz.set_node_label(1, "router")
      viz.set_node_size(2, 2.5)
      viz.set_node_icon(3, "tower.svg")

  visualizer.add_initialization_hook(hook)
  visualizer.set_bounds(-10, -40, 160, 40)  # initial view, in metres

Plugins
~~~~~~~

A plugin is a function that receives the visualizer object; built-in ones
are in ``visualizer/web_plugins``. Plugins return plain data, which the page
renders::

  def my_plugin(viz):
      viz.add_node_panel(
          "queue", "Queue length",
          lambda viz, node: {"sections": [{"title": "Queue", "items": [["length", "3"]]}]},
          applies=lambda viz, node: node == 0,
      )

  visualizer.register_plugin(my_plugin)

The visualizer object provides:

* ``add_node_panel(id, title, build, action=None, applies=None)`` and
  ``add_panel(id, title, build, action=None)``: windows built from sections
  (``{"title", "items": [[name, value], ...]}``, ``{"title", "columns",
  "rows"}`` tables and ``{"title", "plot": {"unit", "series"}}`` charts) and
  controls (``checklist``, ``radio``, ``select`` and ``button``), whose
  values are passed to ``action``.
* ``add_overlay(callback)``: extra links drawn over the topology.
* ``add_node_info(callback)``: values merged into each node's snapshot, as
  the battery indicator does.
* ``add_tooltip_info(callback)``: lines added to the node summary.
* ``add_periodic_update(callback)``: called after each simulation step.
* ``add_step_guard(callback)``: holds the simulation while it returns True.
* ``selected_nodes``, ``set_node_color()`` and the other appearance setters.

Examples and Tests
------------------

``src/visualizer/test/pyviz-web-test.py`` tests every feature: the HTTP
API, simulation control and pacing, topology, transmissions of each
wireless technology, drops, the object tree and attributes, all panels,
the flow monitor, restart, ``VisualSimulatorImpl``, and, with
Selenium and Chrome or Chromium, the page itself:

.. sourcecode:: bash

  $ pip install selenium
  $ ./ns3 run "src/visualizer/test/pyviz-web-test.py -v"

The browser tests are skipped when Selenium or a browser is missing, unless
``--require-selenium`` is given. ``--chrome-binary``, ``--chromedriver`` and
``--webdriver-url`` select the browser and driver (``--help-options`` lists
them); other arguments, such as test names, go to ``unittest``. The weekly
``weekly-pyviz`` CI job runs the whole suite.

The test scenario, ``src/visualizer/test/pyviz-test-scenario.py``, also runs
on its own and is a compact tour of the visualizer (wired and wireless links,
a shared channel, drops, batteries and mobility):

.. sourcecode:: bash

  $ ./ns3 run "src/visualizer/test/pyviz-test-scenario.py --stopTime=60 --flowMonitor"

Validation
----------

No formal validation has been done. The test suite described in the
previous section checks that every feature works.

References
----------

.. _`1`:

[1] G. Carneiro, "Transparent metropolitan vehicular network: design and fast
prototyping methodology", PhD thesis, Faculdade de Engenharia da Universidade
do Porto, 2012.
