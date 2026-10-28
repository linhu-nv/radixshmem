#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pressure migration across three radix-server processes on one host.

node0 is a prefill node with --enable-pressure-migration and no background
eviction; it is filled past its trigger and must push its coldest prefixes --
FULL path, SWA window and Mamba checkpoint -- to a peer, then drop the local
chain. Prefixes come in pairs sharing a 3-block stem. node1 / node2 are set up
per scenario:

  basic     two empty decode donors; prefix 1 is touched before the pressure,
            so prefix 0 moves without its sibling (chain deletion stops at the
            stem) while later pairs move whole (the stem goes with them).
  prefix    node1 (decode) already holds every stem, node2 (decode) is larger
            and emptier: the longest existing prefix wins, every push goes to node1.
  fallback  node1 (decode) is 95% full, node2 (prefill) is empty: no donor
            qualifies, so pushes go to the prefill peer.
  none      node1 (decode) is 95% full, node2 (prefill) is 80% full: nobody
            qualifies, nothing moves and node0 keeps every prefix.

    python3 test_pressure_migration.py [--index-dev mlx5_0] [--transfer-dev mlx5_0] [--gid-idx 3]
                                       [--scenario basic --scenario prefix ...]

Needs etcd, an RDMA device and the Mooncake backend; exits 77 when missing.
Clients are re-exec'd, never forked (etcd's Go runtime and ibverbs do not
survive fork()).
"""

import argparse
import glob
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

NAME = "/py_mig"
PAIRS = int(os.environ.get("MIG_PAIRS", "10"))  # 2 * PAIRS prefixes; the orchestrator sizes it
STEM, TAIL = 3, 5          # blocks shared within a pair / private to a prefix
N = STEM + TAIL
WINDOW = 4                 # SWA window blocks (<= TAIL)
FULL_BYTES, SWA_BYTES, MAMBA_BYTES = 65536, 16384, 32768
SWA_RATIO, MAMBA_RATIO = "0.2", "0.1"
TRIGGER, TARGET = 0.15, 0.25
SALT = {"full": 0x46554C4C, "swa": 0x535741, "mamba": 0x4D414D42}

# scenario -> [(role, data_bytes, seed)] for node1, node2; seed: "", "stems", "fill:<used share>"
SCENARIOS = {
    "basic": [("decode", "64M", ""), ("decode", "64M", "")],
    "prefix": [("decode", "64M", "stems"), ("decode", "128M", "")],
    "fallback": [("decode", "16M", "fill:0.95"), ("prefill", "64M", "")],
    "none": [("decode", "16M", "fill:0.95"), ("prefill", "16M", "fill:0.80")],
}


def prefix_tokens(k: int):
    import numpy as np
    stem = np.arange(10_000 * (k // 2 + 1), 10_000 * (k // 2 + 1) + STEM, dtype=np.int64)
    tail = np.arange(1_000_000 + 100 * k, 1_000_000 + 100 * k + TAIL, dtype=np.int64)
    return np.concatenate([stem, tail])


def pattern(h: int, salt: int, size: int) -> bytes:
    return (struct.pack("<QQ", int(h), salt) * (size // 16))[:size]


def client_main(role: str, endpoint: str) -> int:
    import numpy as np
    import radixshmem
    from radixshmem._core import COMPONENT_MASK_FULL, COMPONENT_MASK_MAMBA, COMPONENT_MASK_SWA, ComponentType

    FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
    ALL = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA
    c = radixshmem.RadixClient(NAME, endpoint=endpoint)
    hs = [radixshmem.compute_hashes(prefix_tokens(k), 1) for k in range(2 * PAIRS)]
    out = {"role": role, "rank": c.rank()}

    def view(slot, kind):
        return c.slot_view(int(slot), kind)

    def put_full(h):
        m = int(c.query(h, local_only=True, update_meta=False).common_hit)
        if m == len(h):
            return "InsertError.OK"
        slots = c.allocate_slots(len(h) - m)
        for i, s in enumerate(slots):
            view(s, FULL)[:] = pattern(h[m + i], SALT["full"], FULL_BYTES)
        return str(c.insert(h, slots, start=m).error)

    def verify(h, q):
        """Bad slot count of a FULL|SWA|MAMBA hit on this node."""
        bad = 0
        full = np.concatenate([np.asarray(s) for _, _, s in q.full_fragments]) if q.full_fragments else []
        for i, s in enumerate(full):
            bad += bytes(view(s, FULL)) != pattern(h[i], SALT["full"], FULL_BYTES)
        start = int(q.swa_start)
        for j, s in enumerate(q.swa_slots):
            bad += bytes(view(s, SWA)) != pattern(h[start + j], SALT["swa"], SWA_BYTES)
        if q.mamba is not None:
            bad += bytes(view(q.mamba[1], MAMBA)) != pattern(h[-1], SALT["mamba"], MAMBA_BYTES)
        return bad

    if role.startswith("writer"):
        # "writer:<k>@<j>": touch prefix k right after writing prefix j, so it is
        # hot when the pressure starts (migration begins while the writer runs)
        touch = [int(x) for x in role.split(":", 1)[1].split("@")] if ":" in role else None
        errors = set()
        for j, h in enumerate(hs):
            n = len(h)
            errors.add(put_full(h))
            sw = c.allocate_slots(WINDOW, SWA)
            for i, s in enumerate(sw):
                view(s, SWA)[:] = pattern(h[n - WINDOW + i], SALT["swa"], SWA_BYTES)
            errors.add(str(c.insert(h, sw, component=SWA).error))
            mb = c.allocate_slots(1, MAMBA)
            view(mb[0], MAMBA)[:] = pattern(h[-1], SALT["mamba"], MAMBA_BYTES)
            errors.add(str(c.insert(h, mb, component=MAMBA).error))
            time.sleep(0.01)    # distinct LRU stamps: prefix k is older than k + 1
            if touch and j == touch[1]:
                out["touched"] = int(c.query(hs[touch[0]], local_only=True, update_meta=True).common_hit)
        c.flush()
        out["insert_errors"] = sorted(errors)
    elif role == "stems":
        out["insert_errors"] = sorted({put_full(h[:STEM]) for h in hs})
        c.flush()
    elif role.startswith("fill:"):
        share = float(role.split(":", 1)[1])
        target = int(share * c.mempool_total())
        errors, k = set(), 0
        while c.mempool_total() - c.mempool_free() < target:
            want = min(8, target - (c.mempool_total() - c.mempool_free()))
            toks = np.arange(50_000_000 + 1000 * k, 50_000_000 + 1000 * k + want, dtype=np.int64)
            errors.add(put_full(radixshmem.compute_hashes(toks, 1)))
            k += 1
        out["insert_errors"] = sorted(errors)
        out["used"] = int(c.mempool_total() - c.mempool_free())
    elif role.startswith("touch:"):
        for k in [int(x) for x in role.split(":", 1)[1].split(",") if x]:
            q = c.query(hs[k], local_only=True, update_meta=True)
            out[f"touch{k}"] = int(q.common_hit)
    elif role == "census":
        rows = []   # per prefix: FULL hit, joint FULL|SWA|MAMBA hit, stem hit, bad slots
        for h in hs:
            qf = c.query(h, local_only=True, update_meta=False)
            qa = c.query(h, mask=ALL, local_only=True, lock=True, update_meta=False)
            bad = verify(h, qa) if int(qa.common_hit) == len(h) else 0
            row = [int(qf.common_hit), int(qa.common_hit),
                   int(c.query(h[:STEM], local_only=True, update_meta=False).common_hit), bad]
            if qa.finalize is not None:
                qa.finalize()
            rows.append(row)
        out["rows"] = rows
    elif role.startswith("puller:"):
        ks = [int(x) for x in role.split(":", 1)[1].split(",") if x]
        jobs = [c.pull_async(hs[k], ALL) for k in ks]
        res = [j.wait(120) for j in jobs]
        out["hits"] = [int(r.common_hit) for r in res]
        out["remote_blocks"] = sum(int(r.remote_blocks) for r in res)
        for r in res:
            r.finalize()
        bad = 0
        for k in ks:
            q = c.query(hs[k], mask=ALL, local_only=True, lock=True, update_meta=False)
            bad += verify(hs[k], q) if int(q.common_hit) == len(hs[k]) else 1000
            if q.finalize is not None:
                q.finalize()
        out["bad"] = bad
    elif role == "stats":
        out["stats"] = c.server_stats()
    elif role == "health":
        out["health"] = int(c.check_health())
        out["report"] = c.health_report().describe().strip()
    print("RESULT " + json.dumps(out), flush=True)
    c.close()
    return 0


# ------------------------------------------------------------ orchestrator

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_etcd(workdir):
    cport, pport = free_port(), free_port()
    proc = subprocess.Popen(
        ["etcd", "--name", "mig", "--data-dir", os.path.join(workdir, "etcd-data"),
         "--listen-client-urls", f"http://127.0.0.1:{cport}",
         "--advertise-client-urls", f"http://127.0.0.1:{cport}",
         "--listen-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-cluster", f"mig=http://127.0.0.1:{pport}"],
        stdout=open(os.path.join(workdir, "etcd.log"), "wb"), stderr=subprocess.STDOUT)
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("etcd died during startup")
        try:
            socket.create_connection(("127.0.0.1", cport), timeout=1).close()
            return proc, cport
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("etcd did not come up")


def read(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def wait_log(logs, procs, needle, timeout):
    deadline = time.time() + timeout
    pending = set(range(len(logs)))
    while pending and time.time() < deadline:
        for i in list(pending):
            if procs[i].poll() is not None:
                raise RuntimeError(f"server {i} exited (rc={procs[i].returncode}):\n{read(logs[i])[-3000:]}")
            if needle in read(logs[i]):
                pending.discard(i)
        time.sleep(0.2)
    if pending:
        raise RuntimeError(f"timeout waiting for {needle!r} on {sorted(pending)}:\n"
                           + read(logs[sorted(pending)[0]])[-3000:])


def run_client(role, endpoint, timeout=240):
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--client", role, endpoint],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
    if not lines:
        raise RuntimeError(f"client {role} rc={p.returncode}\nstdout:\n{out}\nstderr:\n{err[-3000:]}")
    return json.loads(lines[0][len("RESULT "):])


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def wait_migration(ep, timeout=60):
    """Stats of node0 once its migration is quiet: free >= TARGET, or no progress for 5 s."""
    deadline, last, still = time.time() + timeout, None, 0
    while time.time() < deadline:
        st = run_client("stats", ep)["stats"]
        free = 1 - st["mempool_used"] / st["mempool_total"]
        mig = st.get("migration", {})
        key = (mig.get("migrated_paths", 0), mig.get("no_eligible_peer", 0) > 0)
        if free >= TARGET:
            return st, free
        still = still + 1 if key == last else 0
        last = key
        if still >= 5 and (mig.get("rounds", 0) > 0 or mig.get("no_eligible_peer", 0) > 0):
            return st, free
        time.sleep(1)
    return st, free


def scenario(a, name, workdir, registry) -> None:
    global PAIRS
    peers = SCENARIOS[name]
    cluster_id = f"mig{os.getpid()}{name}"
    eps = [f"unix://{workdir}/{name}{i}.sock" for i in range(3)]
    servers, logs = [], []
    nodes = [("prefill", "16M")] + [(r, b) for r, b, _ in peers]
    try:
        for i, (role, size) in enumerate(nodes):
            logs.append(os.path.join(workdir, f"{name}{i}.log"))
            cmd = [sys.executable, "-m", "radixshmem.cli", "--name", NAME, "--node-name", f"{name}{i}",
                   "--data-name", f"{NAME}_data{i}", "--data-bytes", size, "--no-prefault",
                   "--swa-ratio", SWA_RATIO, "--mamba-ratio", MAMBA_RATIO,
                   "--endpoint", eps[i], "--registry", registry, "--cluster-id", cluster_id,
                   "--expected-min-nodes", "3", "--rpc-address", "0.0.0.0", "--transfer-ip", "127.0.0.1",
                   "--gid-idx", str(a.gid_idx), "--bootstrap-timeout", str(a.bootstrap_timeout),
                   "--background-evict-ratio", "0", "--interval", "1",
                   "--enable-remote-insert", "--enable-remote-query", "--node-role", role]
            if i == 0:
                cmd += ["--enable-pressure-migration", "--migrate-trigger-free-ratio", str(TRIGGER),
                        "--migrate-target-free-ratio", str(TARGET)]
            cmd += ["--index-dev", a.index_dev] if a.index_dev else []
            cmd += [f"--transfer-dev={d}" for d in a.transfer_dev]
            servers.append(subprocess.Popen(cmd, stdout=open(logs[i], "wb"), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONUNBUFFERED="1")))
        wait_log(logs, servers, "Waiting for a client's geometry", 60)
        import radixshmem
        geo = radixshmem.Geometry(block_size=1, full_slot_bytes=FULL_BYTES, swa_slot_bytes=SWA_BYTES,
                                  swa_window_blocks=WINDOW, mamba_slot_bytes=MAMBA_BYTES)
        cl0 = radixshmem.RadixClient(NAME, geo, endpoint=eps[0])
        wait_log(logs, servers, "Cluster ready", a.bootstrap_timeout + 30)
        for ep in eps:
            radixshmem.RadixClient(NAME, endpoint=ep).wait_ready(30)
        pools = cl0.wait_ready(10).geometry["pools"]
        cl0.close()
        ranks = [run_client("stats", ep)["stats"]["rank"] for ep in eps]
        full_slots = pools["full"]["num_slots"]
        PAIRS = int(0.9 * full_slots) // (STEM + 2 * TAIL)
        os.environ["MIG_PAIRS"] = str(PAIRS)      # the re-exec'd clients read it
        print(f"  ranks {ranks}; node0 pools full {full_slots}, swa {pools['swa']['num_slots']}, "
              f"mamba {pools['mamba']['num_slots']}; {2 * PAIRS} prefixes")

        for i, (_, _, seed) in enumerate(peers, start=1):
            if seed:
                r = run_client(seed, eps[i])
                check(r["insert_errors"] == ["InsertError.OK"], f"node{i} seeded with {seed} ({r})")
        time.sleep(2.5)       # capacity records refresh every second

        # basic: touch prefix 1 once 85% of the pairs are written, still below the trigger
        w = run_client(f"writer:1@{int(1.7 * PAIRS) - 1}" if name == "basic" else "writer", eps[0])
        check(w["insert_errors"] == ["InsertError.OK"], f"node0 writer inserted every prefix ({w})")

        st, free = wait_migration(eps[0])
        mig = st.get("migration", {})
        print(f"  node0 migration stats: {mig}")
        fails = {k: v for k, v in mig.items() if k.startswith(("push_fail", "round_errors", "probe_errors"))}
        check(not fails and "last_error" not in mig, f"no migration failures {fails}")
        src = run_client("census", eps[0])["rows"]
        moved = [k for k, r in enumerate(src) if r[0] < N]
        kept = [k for k, r in enumerate(src) if r[0] == N]
        print(f"  moved {moved}")
        if os.environ.get("MIG_DEBUG"):
            for i in (1, 2):
                rows = run_client("census", eps[i])["rows"]
                print(f"  node{i} full hits: {[r[0] for r in rows]}")
            print(f"  node0 full hits: {[r[0] for r in src]}")
        check(all(src[k][1] == N and src[k][3] == 0 for k in kept), "prefixes left on node0 are intact")

        if name == "none":
            check(not moved and mig.get("migrated_paths", 0) == 0 and mig.get("no_eligible_peer", 0) > 0
                  and mig.get("peeked", 0) == 0,
                  f"no peer qualifies: nothing pinned or moved ({mig.get('no_eligible_peer', 0)} backoffs)")
            return
        check(free >= TARGET, f"node0 FULL free share {free:.2f} >= {TARGET}")
        expect_target = {"prefix": 1, "fallback": 2}.get(name)
        dsts = {i: run_client("census", eps[i])["rows"] for i in (1, 2)}
        for k in moved:
            holders = [i for i in (1, 2) if dsts[i][k][1] == N and dsts[i][k][3] == 0]
            check(holders, f"prefix {k} is on node{holders} with FULL + SWA + Mamba, bytes intact")
        if expect_target is not None:
            tr = ranks[expect_target]
            check(mig.get(f"to_rank_{tr}", 0) == mig.get("migrated_paths", -1) > 0,
                  f"every push went to node{expect_target} (rank {tr})")
        if name == "fallback":
            check(mig.get("to_prefill_peer", 0) > 0, "pushes fell back to the prefill peer")
        if name == "prefix":
            check(mig.get("dedup_blocks", 0) >= STEM * len(moved), "node1's stems were deduplicated")
        if name == "basic":
            order = [0] + list(range(2, 2 * PAIRS))   # prefix 1 was touched
            check(moved == order[:len(moved)] and 1 not in moved,
                  "coldest first; the touched prefix 1 stays")
            check(src[0][0] == STEM and src[0][2] == STEM,
                  "prefix 0 moved alone: chain deletion stopped at the stem shared with prefix 1")
            whole = [p for p in range(1, PAIRS) if 2 * p in moved and 2 * p + 1 in moved]
            check(whole and all(src[2 * p][2] == 0 for p in whole),
                  f"pairs {whole} moved whole and took their stem along")
            r = run_client("puller:" + ",".join(map(str, moved)), eps[0])
            check(r["hits"] == [N] * len(moved) and r["bad"] == 0, f"node0 pulls them back intact ({r})")
        for i, ep in enumerate(eps):
            h = run_client("health", ep)
            check(h["health"] == 0, f"node{i} check_health == 0 ({h['report']})")
    finally:
        for p in servers:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in servers:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        for stale in glob.glob(f"/dev/shm{NAME}*"):
            try:
                os.unlink(stale)
            except OSError:
                pass


def orchestrate(a) -> int:
    import radixshmem

    if (not shutil.which("etcd") or not os.path.isdir(f"/sys/class/infiniband/{a.index_dev}")
            or not hasattr(radixshmem, "_data") or "mooncake" not in radixshmem._data.TRANSFER_BACKENDS
            or not hasattr(radixshmem._core, "EtcdKv")):
        print(f"SKIP: needs etcd, RDMA device {a.index_dev!r} and radixshmem built with etcd + Mooncake")
        return 77
    workdir = tempfile.mkdtemp(prefix="radixshmem-mig-")
    etcd_proc = None
    try:
        etcd_proc, cport = start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        print(f"etcd on {registry}   logs in {workdir}")
        for name in a.scenario:
            print(f"\n[{name}] {SCENARIOS[name]}")
            scenario(a, name, workdir, registry)
        print("\nPASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if etcd_proc is not None:
            etcd_proc.terminate()
            etcd_proc.wait(timeout=10)
        if os.environ.get("RADIXSHMEM_KEEP_LOGS"):
            print(f"logs kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        return client_main(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dev", default="mlx5_bond_0")
    ap.add_argument("--transfer-dev", action="append", default=None)
    ap.add_argument("--gid-idx", type=int, default=3)
    ap.add_argument("--bootstrap-timeout", type=int, default=60)
    ap.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), default=None)
    a = ap.parse_args()
    a.transfer_dev = a.transfer_dev or [a.index_dev]
    a.scenario = a.scenario or ["basic", "prefix", "fallback", "none"]
    return orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
