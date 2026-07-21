#!/usr/bin/env python3
# Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
"""Compare the ns-3 TCP loss recovery behavior with a real Linux TCP stack.

The `linux` subcommand replays a tcp-recovery-fsm-trace scenario against the
Linux TCP stack without requiring root: it re-executes itself inside an
unprivileged user and network namespace (unshare -r -n), where a TUN
"reflector" process acts as the wire. The reflector swaps the source and
destination addresses of every packet (checksum neutral), delays every packet
by 50 ms (100 ms RTT), drops the configured relative sequence numbers (each
listed occurrence once, so a repeated number also drops the retransmission)
and logs every segment. The sender socket is configured to match the ns-3
scenario (MSS 512, reno congestion control, initial cwnd 10, per-scenario
minimum RTO and SACK setting) and its congestion state machine is sampled
through getsockopt(TCP_INFO), whose tcpi_ca_state uses the same
OPEN/DISORDER/CWR/RECOVERY/LOSS states as the ns-3 CongState trace.

The `plot` subcommand renders a time-sequence diagram of both stacks, with
the congestion states as background bands, from a tcp-recovery-fsm-trace log
and a `linux` subcommand JSON output.

Typical workflow:

  ./ns3 run 'tcp-recovery-fsm-trace --sack=1 --drops=20001:22001' > ns3.log
  ./examples/tcp/tcp-recovery-fsm-compare.py linux 1 20001:22001 1000 linux.json
  ./examples/tcp/tcp-recovery-fsm-compare.py plot ns3.log linux.json out.png
"""

import fcntl
import heapq
import json
import os
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time

TUNSETIFF = 0x400454CA
IFF_TUN = 0x0001
IFF_NO_PI = 0x1000
CA_NAMES = ["CA_OPEN", "CA_DISORDER", "CA_CWR", "CA_RECOVERY", "CA_LOSS"]
PORT = 5000
TOTAL = 40000
DELAY = 0.05

# ---------------------------------------------------------------------------
# linux subcommand: replay a scenario against the Linux TCP stack
# ---------------------------------------------------------------------------

t0 = None
events = []  # (t, dir, relseq, paylen, flags, relack, dropped, retx)
transitions = []  # (t, from, to)
cwnd_samples = []  # (t, cwnd, ca_state)
retx_counts = {}
lock = threading.Lock()


def now():
    return time.monotonic()


def rel_t(t):
    return t - t0 if t0 is not None else 0.0


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def open_tun():
    fd = os.open("/dev/net/tun", os.O_RDWR)
    ifr = struct.pack("16sH", b"tun0", IFF_TUN | IFF_NO_PI)
    fcntl.ioctl(fd, TUNSETIFF, ifr)
    return fd


class Reflector(threading.Thread):
    """The wire: swap addresses, delay, drop by relative sequence, log."""

    def __init__(self, fd, drops):
        super().__init__(daemon=True)
        self.fd = fd
        self.drops = list(drops)
        self.isn_c = None
        self.isn_s = None
        self.heap = []
        self.seen_data = set()
        self.stop = False

    def parse(self, pkt):
        if len(pkt) < 40 or pkt[0] >> 4 != 4:
            return None
        ihl = (pkt[0] & 0xF) * 4
        if pkt[9] != 6:
            return None
        src = pkt[12:16]
        dst = pkt[16:20]
        tcp = pkt[ihl:]
        sport, dport, seq, ack = struct.unpack("!HHII", tcp[0:12])
        doff = (tcp[12] >> 4) * 4
        flags = tcp[13]
        pay = len(pkt) - ihl - doff
        return src, dst, sport, dport, seq, ack, flags, pay

    def run(self):
        global t0
        poller = select.poll()
        poller.register(self.fd, select.POLLIN)
        while not self.stop:
            timeout = 10
            if self.heap:
                timeout = max(0, (self.heap[0][0] - now()) * 1000)
            for _fd, _ev in poller.poll(timeout):
                pkt = os.read(self.fd, 4096)
                info = self.parse(pkt)
                if info is None:
                    continue
                src, dst, sport, dport, seq, ack, flags, pay = info
                to_server = dst == socket.inet_aton("10.0.1.1") and dport == PORT
                if to_server and flags & 0x02:
                    self.isn_c = seq
                if not to_server and flags & 0x02:
                    self.isn_s = seq
                dropped = False
                retx = False
                if to_server and pay > 0 and self.isn_c is not None:
                    rel = (seq - self.isn_c) & 0xFFFFFFFF
                    retx = rel in self.seen_data
                    self.seen_data.add(rel)
                    if rel in self.drops:
                        self.drops.remove(rel)
                        dropped = True
                    if retx:
                        with lock:
                            retx_counts[rel] = retx_counts.get(rel, 0) + 1
                    if t0 is None and rel == 1:
                        t0 = now()
                    with lock:
                        events.append((rel_t(now()), "data", rel, pay, flags, 0, dropped, retx))
                elif not to_server and self.isn_s is not None and self.isn_c is not None:
                    relack = (ack - self.isn_c) & 0xFFFFFFFF
                    with lock:
                        events.append((rel_t(now()), "ack", 0, pay, flags, relack, False, False))
                if dropped:
                    continue
                # reflect: swap source and destination address (checksum neutral)
                out = pkt[:12] + dst + src + pkt[20:]
                heapq.heappush(self.heap, (now() + DELAY, len(self.heap), out))
            while self.heap and self.heap[0][0] <= now():
                _t, _n, out = heapq.heappop(self.heap)
                os.write(self.fd, out)


def poll_tcp_info(sock, done):
    last_ca = None
    while not done.is_set():
        try:
            ti = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 192)
        except OSError:
            break
        ca = ti[1]
        cwnd = struct.unpack_from("I", ti, 8 + 4 * 18)[0]
        t = rel_t(now())
        with lock:
            cwnd_samples.append((t, cwnd, ca))
        if last_ca is not None and ca != last_ca:
            with lock:
                transitions.append((t, CA_NAMES[last_ca], CA_NAMES[ca]))
        last_ca = ca
        time.sleep(0.001)


def linux_main(argv):
    if os.environ.get("TCP_FSM_INNER") != "1":
        # re-execute inside an unprivileged user + network namespace
        env = dict(os.environ, TCP_FSM_INNER="1")
        os.execvpe(
            "unshare",
            ["unshare", "-r", "-n", sys.executable, os.path.abspath(__file__), "linux"] + argv,
            env,
        )

    sack = int(argv[0])
    drops = [int(x) for x in argv[1].split(":")]
    rto_min_ms = int(argv[2])
    out_file = argv[3]

    sh("ip link set lo up")
    tun_fd = open_tun()
    sh("ip addr add 10.0.0.1/24 dev tun0")
    sh("ip link set tun0 up")
    sh(f"ip route add 10.0.1.0/24 dev tun0 rto_min {rto_min_ms}ms initcwnd 10")
    sh(f"sysctl -qw net.ipv4.tcp_sack={sack}")
    sh("sysctl -qw net.ipv4.tcp_no_metrics_save=1")

    refl = Reflector(tun_fd, drops)
    refl.start()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("10.0.0.1", PORT))
    srv.listen(1)
    received = []

    def server():
        conn, _ = srv.accept()
        n = 0
        while n < TOTAL:
            d = conn.recv(65536)
            if not d:
                break
            n += len(d)
        received.append(n)
        conn.close()

    st = threading.Thread(target=server, daemon=True)
    st.start()

    cli = socket.socket()
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 512)
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, b"reno")
    cli.connect(("10.0.1.1", PORT))

    done = threading.Event()
    pt = threading.Thread(target=poll_tcp_info, args=(cli, done), daemon=True)
    pt.start()

    cli.sendall(b"x" * TOTAL)
    st.join(timeout=15)
    time.sleep(0.3)
    done.set()
    pt.join(timeout=1)
    cli.close()
    refl.stop = True

    with lock:
        json.dump(
            {
                "received": received[0] if received else 0,
                "events": events,
                "transitions": transitions,
                "cwnd": cwnd_samples[::5],
                "retx": retx_counts,
            },
            open(out_file, "w"),
        )
    print(
        f"received={received[0] if received else 0} "
        f"transitions={[(round(t, 4), a, b) for t, a, b in transitions]} "
        f"retx={retx_counts}"
    )


# ---------------------------------------------------------------------------
# plot subcommand: time-sequence diagram of both stacks with state bands
# ---------------------------------------------------------------------------

C_DATA = "#2a78d6"  # data segments
C_RETX = "#ec835a"  # retransmissions
C_DROP = "#d03b3b"  # segments dropped on the wire
C_INK = "#39424e"
BAND = {
    "CA_OPEN": "#eef5ee",
    "CA_DISORDER": "#fdf3dc",
    "CA_RECOVERY": "#fdeadf",
    "CA_LOSS": "#f9e2e2",
}
SURFACE = "#fcfcfb"


def parse_ns3(path):
    tx = []
    drops = []
    trans = []
    first = None
    for line in open(path):
        m = re.match(r"([\d.]+) (\S+)\s+(.*)", line)
        if not m:
            continue
        t, kind, rest = float(m.group(1)), m.group(2), m.group(3)
        if kind == "TX":
            seq = int(re.search(r"seq=(\d+)", rest).group(1))
            if first is None:
                first = t
            tx.append((t, seq, "[RETX]" in rest))
        elif kind == "DROP":
            drops.append((t, int(re.search(r"seq=(\d+)", rest).group(1))))
        elif kind == "STATE":
            a, b = rest.split(" -> ")
            trans.append((t, a.strip(), b.strip()))
    return {
        "tx": [(t - first, s, r) for t, s, r in tx],
        "drops": [(t - first, s) for t, s in drops],
        "trans": [(t - first, a, b) for t, a, b in trans],
    }


def parse_linux(path):
    d = json.load(open(path))
    tx = []
    drops = []
    for t, direction, rel, pay, flags, relack, dropped, retx in d["events"]:
        if direction != "data":
            continue
        tx.append((t, rel, retx))
        if dropped:
            drops.append((t, rel))
    return {"tx": tx, "drops": drops, "trans": [(t, a, b) for t, a, b in d["transitions"]]}


def draw(ax, data, title, tmax):
    state = "CA_OPEN"
    start = 0.0
    spans = []
    for t, _a, b in data["trans"]:
        spans.append((start, t, state))
        state = b
        start = t
    spans.append((start, tmax, state))
    for s, e, st in spans:
        if e - s <= 0:
            continue
        ax.axvspan(s, e, color=BAND.get(st, "#f0f0f0"), zorder=0)
        if e - s > 0.04 * tmax:
            ax.text(
                (s + e) / 2,
                0.985,
                st.replace("CA_", ""),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                color=C_INK,
                alpha=0.85,
            )
    for t, _a, _b in data["trans"]:
        ax.axvline(t, color=C_INK, lw=0.6, ls="--", alpha=0.35, zorder=1)

    new = [(t, s) for t, s, r in data["tx"] if not r]
    retx = [(t, s) for t, s, r in data["tx"] if r]
    ax.plot([t for t, s in new], [s / 1000 for t, s in new], ".", ms=3.5, color=C_DATA, zorder=3)
    ax.plot(
        [t for t, s in retx],
        [s / 1000 for t, s in retx],
        "^",
        ms=8,
        mfc="none",
        mew=1.6,
        color=C_RETX,
        zorder=4,
    )
    ax.plot(
        [t for t, s in data["drops"]],
        [s / 1000 for t, s in data["drops"]],
        "x",
        ms=8,
        mew=1.8,
        color=C_DROP,
        zorder=5,
    )
    ax.set_title(title, fontsize=10, color=C_INK, loc="left")
    ax.set_ylabel("sequence (KB)", fontsize=8.5, color=C_INK)
    ax.tick_params(labelsize=8, colors=C_INK)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.02 * tmax, tmax)


def plot_main(argv):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ns3_log, linux_json, out_png = argv[0], argv[1], argv[2]
    title = argv[3] if len(argv) > 3 else ""
    ns3 = parse_ns3(ns3_log)
    lnx = parse_linux(linux_json)
    tmax = max([t for t, *_ in ns3["tx"]] + [t for t, *_ in lnx["tx"]]) * 1.05
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, dpi=140)
    fig.patch.set_facecolor(SURFACE)
    draw(axes[0], ns3, f"ns-3 {title}".strip(), tmax)
    draw(axes[1], lnx, "Linux (netns + tun reflector, same drops/delay/MSS)", tmax)
    axes[1].set_xlabel("time since first data segment (s)", fontsize=8.5, color=C_INK)
    legend = [
        Line2D([], [], marker=".", ls="", ms=6, color=C_DATA, label="data segment"),
        Line2D(
            [],
            [],
            marker="^",
            ls="",
            ms=8,
            mfc="none",
            mew=1.6,
            color=C_RETX,
            label="retransmission",
        ),
        Line2D([], [], marker="x", ls="", ms=8, mew=1.8, color=C_DROP, label="dropped on wire"),
    ]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    print("wrote", out_png)


# ---------------------------------------------------------------------------
# verify subcommand: check both stacks against the RFC mandated behavior
# ---------------------------------------------------------------------------


def strip_ca(name):
    return name.replace("CA_", "")


def check(ok, what):
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    return ok


def verify_main(argv):
    ns3_log, linux_json, sack, drops_arg = argv[0], argv[1], int(argv[2]), argv[3]
    drops = [int(x) for x in drops_arg.split(":")]
    expected_retx = {}
    for seq in drops:
        expected_retx[seq] = expected_retx.get(seq, 0) + 1

    ns3 = parse_ns3(ns3_log)
    lnx = parse_linux(linux_json)
    lnx_raw = json.load(open(linux_json))

    ok = True
    print(f"scenario: sack={sack} drops={drops}")

    for name, data in (("ns-3", ns3), ("linux", lnx)):
        print(f"{name}:")
        seq = [(strip_ca(a), strip_ca(b)) for _t, a, b in data["trans"]]

        # RFC 5681, Section 3.2 / RFC 6675, Section 5: loss recovery is
        # entered once via the duplicate ACK threshold and left on the ACK
        # covering the recovery point; a retransmission timeout (CA_LOSS)
        # must not occur, since every drop is recoverable from feedback
        ok &= check(
            sum(1 for _a, b in seq if b == "RECOVERY") == 1,
            "loss recovery entered exactly once (RFC 5681/6675/6582)",
        )
        ok &= check(
            all(b != "LOSS" for _a, b in seq),
            "no retransmission timeout: CA_LOSS never entered "
            "(RFC 6582 Section 3.2 / RFC 6675 Section 6 timer restart)",
        )
        ok &= check(
            seq[-1][1] == "OPEN" if seq else False, "recovery left to CA_OPEN on the full ACK"
        )
        # The DISORDER state may last less than the Linux poll interval, so
        # only require it on the ns-3 (event driven) trace
        if name == "ns-3":
            ok &= check(
                ("OPEN", "DISORDER") == seq[0], "first duplicate ACK moved CA_OPEN -> CA_DISORDER"
            )

        # Each dropped occurrence must be repaired by exactly one
        # retransmission: no missing repair and no spurious retransmission
        # (RFC 6582 Section 3.2 for NewReno, RFC 6675 Section 5 with SACK)
        if name == "ns-3":
            got = {}
            for _t, s2, r in data["tx"]:
                if r:
                    got[s2] = got.get(s2, 0) + 1
        else:
            got = {int(k): v for k, v in lnx_raw["retx"].items()}
        ok &= check(got == expected_retx, f"retransmissions match the drops exactly: {got}")

    ok &= check(lnx_raw.get("received", 0) == TOTAL, "linux delivered all data")

    def state_path(trans):
        # states visited, with DISORDER collapsed (it may be shorter than the
        # Linux TCP_INFO poll interval and therefore not sampled)
        path = ["OPEN"] + [strip_ca(b) for _t, _a, b in trans]
        return [st for st in path if st != "DISORDER"]

    ok &= check(
        state_path(ns3["trans"]) == state_path(lnx["trans"]),
        f"ns-3 and Linux visit the same recovery states: "
        f"{' -> '.join(state_path(ns3['trans']))}",
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def usage():
    print(__doc__)
    print("usage:")
    print("  tcp-recovery-fsm-compare.py linux <sack 0|1> <drops a:b:...> <rto_min_ms> <out.json>")
    print("  tcp-recovery-fsm-compare.py verify <ns3.log> <linux.json> <sack 0|1> <drops>")
    print("  tcp-recovery-fsm-compare.py plot <ns3.log> <linux.json> <out.png> [title]")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
    if sys.argv[1] == "linux" and len(sys.argv) >= 6:
        linux_main(sys.argv[2:])
    elif sys.argv[1] == "verify" and len(sys.argv) >= 6:
        sys.exit(verify_main(sys.argv[2:]))
    elif sys.argv[1] == "plot" and len(sys.argv) >= 5:
        plot_main(sys.argv[2:])
    else:
        usage()
