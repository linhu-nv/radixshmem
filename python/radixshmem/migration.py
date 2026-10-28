# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pressure migration: when a node's slot pools run low, push its coldest
prefixes to a peer with room and drop the local copy, so a prefill node can
use the memory of a decode node (or of a lightly loaded prefill node).

Two threads per data-mode cluster server:

* ``CapacityRegistry`` publishes this node's role and per-pool free/total slot
  counts under ``radix/<cluster_id>/capacity/<node>`` on a lease, so the key
  goes away with the process. Every node publishes, migrating or not; it is
  how a source learns which peers can take data.
* ``PressureMigrator`` (``MigrationConfig.enabled``) runs rounds. A round
  starts when some pool's free share is below ``trigger_free_ratio`` and stops
  once every pool is back above ``target_free_ratio``. It peeks the coldest
  unpinned leaves as full root->leaf hash paths and, for each one:

  1. pins the path locally: FULL ``[0, n)``, plus the SWA window and the Mamba
     checkpoint at ``n`` when those pools exist and the END carries them. The
     pins keep the slots from being recycled while RDMA reads them.
  2. picks a target among peers whose published free share is at least
     ``min_peer_free_ratio`` in every pool the push needs. Donors (role
     ``decode``) come first; prefill peers only when no donor qualifies and
     they have ``prefill_peer_free_ratio``. The longest prefix the target
     already holds wins, then the larger free share.
  3. pushes: probes the target's ``[0, m)``, reserves ``n - m`` FULL slots there,
     RDMA-writes ``[m, n)`` through this server's transfer engine and inserts
     remotely (the target dedups, and rejects the insert if it lost part of
     ``[0, m)`` meanwhile); then the SWA window and the Mamba checkpoint the
     same way.
  4. checks that the target now matches ``[0, n)`` and drops the local chain
     (``evict_chain``: the leaf and every ancestor left childless and unpinned).

A push that fails anywhere, or lands only partially, keeps the local copy;
ordinary LRU eviction remains the backstop. Paths deeper than ``max_blocks``
(the remote-op cap) are left to it as well.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from radixshmem import _core
from radixshmem._core import ComponentType
from radixshmem.client import XFER_WRITE

if TYPE_CHECKING:  # pragma: no cover
    from radixshmem.server import RadixServer

FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
_POOL_NAMES = {FULL: "full", SWA: "swa", MAMBA: "mamba"}
_MASK = {FULL: int(_core.COMPONENT_MASK_FULL), SWA: int(_core.COMPONENT_MASK_SWA),
         MAMBA: int(_core.COMPONENT_MASK_MAMBA)}

MAX_REMOTE_OP_BLOCKS = 512   # include/cluster_protocol.h; one remote op carries at most this many hashes
ROLE_PREFILL, ROLE_DECODE = "prefill", "decode"

_LEASE_TTL_S = 10


@dataclass
class MigrationConfig:
    enabled: bool = False
    trigger_free_ratio: float = 0.15       # start when some pool's free share drops below this
    target_free_ratio: float = 0.20        # stop once every pool is back at or above this
    min_peer_free_ratio: float = 0.10      # a target needs this free share in every pool it receives
    prefill_peer_free_ratio: float = 0.30  # ... a prefill-role target this much (donors come first)
    batch: int = 8                         # leaves peeked per round
    max_blocks: int = MAX_REMOTE_OP_BLOCKS # deeper paths are left to ordinary eviction
    scan_factor: int = 64                  # leaves visited per round = batch x scan_factor
    timeout_ms: int = 10000                # one RDMA write job
    idle_s: float = 0.05                   # sleep between checks when not under pressure
    no_target_backoff_s: float = 1.0       # sleep when under pressure but no peer qualifies
    publish_interval_s: float = 1.0        # capacity record refresh
    peer_stale_s: float = 10.0             # a capacity record older than this is ignored

    def __post_init__(self) -> None:
        for k in ("trigger_free_ratio", "target_free_ratio", "min_peer_free_ratio",
                  "prefill_peer_free_ratio"):
            if not 0.0 <= float(getattr(self, k)) < 1.0:
                raise ValueError(f"migration.{k} must be in [0, 1)")
        if self.target_free_ratio < self.trigger_free_ratio:
            raise ValueError("migration.target_free_ratio must be >= trigger_free_ratio")
        if not 0 < self.max_blocks <= MAX_REMOTE_OP_BLOCKS:
            raise ValueError(f"migration.max_blocks must be in (0, {MAX_REMOTE_OP_BLOCKS}]")
        if self.batch <= 0:
            raise ValueError("migration.batch must be > 0")


def pool_counts(index) -> Dict[ComponentType, Tuple[int, int]]:
    """{pool: (free, total)} of a ``_core.RadixServer``, enabled pools only."""
    out = {FULL: (int(index.mempool_free()), int(index.mempool_total()))}
    swa_total, mamba_total = int(index.swa_mempool_total()), int(index.mamba_mempool_total())
    if swa_total > 0:
        out[SWA] = (int(index.swa_mempool_free()), swa_total)
    if mamba_total > 0:
        out[MAMBA] = (int(index.mamba_mempool_free()), mamba_total)
    return {k: v for k, v in out.items() if v[1] > 0}


def free_ratio(free: int, total: int) -> float:
    return free / total if total > 0 else 0.0


class CapacityRegistry:
    """``radix/<cluster_id>/capacity/<node>``: a node's role, rank and per-pool
    ``[free, total]`` slot counts as JSON, on a lease this registry keeps alive.

    Stand-in for the cluster capacity / identity service; the record is what
    ``PressureMigrator`` needs from it."""

    def __init__(self, registry: str, cluster_id: str):
        self._kv = _core.EtcdKv(registry)
        self._prefix = f"radix/{cluster_id}/capacity/"
        self.lease = self._kv.grant_lease(_LEASE_TTL_S)
        self._kv.keep_alive(self.lease)

    def publish(self, node: str, record: Dict[str, Any]) -> None:
        self._kv.put(self._prefix + node, json.dumps(record, sort_keys=True), self.lease)

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for key, value in self._kv.get_prefix(self._prefix):
            try:
                rec = json.loads(value)
            except ValueError:
                continue
            rec.setdefault("node", key[len(self._prefix):])
            out.append(rec)
        return out

    def close(self) -> None:
        try:
            self._kv.revoke(self.lease)
        except Exception:  # noqa: BLE001 - the lease expires on its own
            pass


@dataclass
class _Peer:
    node: str
    rank: int
    role: str
    pools: Dict[ComponentType, Tuple[int, int]]

    def free(self, kind: ComponentType) -> float:
        return free_ratio(*self.pools.get(kind, (0, 0)))


@dataclass
class _Victim:
    """One locally pinned path and the slots a push needs."""
    hashes: np.ndarray
    full: np.ndarray                       # FULL slots of [0, n)
    swa: Optional[np.ndarray] = None       # SWA window ending at n
    mamba: Optional[int] = None            # Mamba checkpoint at n
    finalizers: Tuple = ()

    @property
    def n(self) -> int:
        return len(self.hashes)

    def kinds(self) -> List[ComponentType]:
        return [FULL] + ([SWA] if self.swa is not None else []) + ([MAMBA] if self.mamba is not None else [])

    def release(self) -> None:
        for f in self.finalizers:
            if f is not None:
                f()
        self.finalizers = ()


class PressureMigrator:
    """Capacity publisher plus (when enabled) the migration loop of one server."""

    def __init__(self, server: "RadixServer"):
        self._srv = server
        self.cfg: MigrationConfig = server.cfg.migration
        self.role: str = server.cfg.node_role
        c = server.cfg.cluster
        self._registry = CapacityRegistry(c.registry, c.cluster_id)
        self._node = server.index.node_name()
        self._rank = int(server.index.rank())
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._peers: List[_Peer] = []
        self._peers_at = -1e9
        self.stats: Counter = Counter()
        self.last_error = ""
        self._last_publish = 0.0

    # ----------------------------------------------------------- lifecycle

    def start(self) -> "PressureMigrator":
        self._publish()
        self._spawn(self._publish_loop, "radixshmem-capacity")
        if self.cfg.enabled:
            self._spawn(self._migrate_loop, "radixshmem-migrate")
        return self

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(self.cfg.timeout_ms / 1000 + 5)
        self._threads = []
        self._registry.close()

    # ------------------------------------------------------------ capacity

    def _publish(self) -> None:
        pools = pool_counts(self._srv.index)
        self._registry.publish(self._node, {
            "node": self._node, "rank": self._rank, "role": self.role, "t": time.time(),
            "pools": {_POOL_NAMES[k]: [f, t] for k, (f, t) in pools.items()}})
        self._last_publish = time.monotonic()

    def _publish_loop(self) -> None:
        while not self._stop.wait(self.cfg.publish_interval_s):
            try:
                self._publish()
            except Exception:  # noqa: BLE001 - etcd hiccup: the lease keeps the last record
                self.stats["publish_errors"] += 1

    def peers(self) -> List[_Peer]:
        """Other nodes' fresh capacity records."""
        now, out = time.time(), []
        by_name = {v: k for k, v in _POOL_NAMES.items()}
        for rec in self._registry.list():
            if rec.get("node") == self._node or int(rec.get("rank", -1)) in (-1, self._rank):
                continue
            if now - float(rec.get("t", 0)) > self.cfg.peer_stale_s:
                continue
            pools = {by_name[k]: (int(v[0]), int(v[1])) for k, v in rec.get("pools", {}).items()
                     if k in by_name}
            out.append(_Peer(rec["node"], int(rec["rank"]), rec.get("role", ROLE_PREFILL), pools))
        return out

    # ----------------------------------------------------------- migration

    def _local_free(self) -> float:
        return min(free_ratio(f, t) for f, t in pool_counts(self._srv.index).values())

    def _migrate_loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.run_round()
            except Exception as e:  # noqa: BLE001 - keep the thread; the round is retried
                self.stats["round_errors"] += 1
                self.last_error = f"{type(e).__name__}: {e}"
                busy = False
            if busy is None:     # under pressure, nobody can take data
                self._stop.wait(self.cfg.no_target_backoff_s)
            elif not busy:
                self._stop.wait(self.cfg.idle_s)

    def run_round(self) -> Optional[bool]:
        """One round: True when it moved data (the caller loops on), False when
        idle, None when under pressure but no peer can take FULL data (back off)."""
        c = self.cfg
        if self._local_free() >= c.trigger_free_ratio:
            return False
        now = time.monotonic()
        if now - self._peers_at >= c.publish_interval_s:   # records change no faster
            self._peers, self._peers_at = self.peers(), now
        if not any(self._eligible(p, [FULL]) for p in self._peers):
            self.stats["no_eligible_peer"] += 1
            return None
        self.stats["rounds"] += 1
        client = self._srv.iclient
        paths = client.peek_lru_leaf_paths(c.batch, c.max_blocks, c.batch * c.scan_factor)
        self.stats["peeked"] += len(paths)
        if not paths:
            self.stats["no_victims"] += 1
            return False
        moved = False
        for hashes in paths:
            if self._stop.is_set():
                break
            moved |= self.migrate_path(np.ascontiguousarray(hashes, dtype=np.uint64))
            if self._local_free() >= c.target_free_ratio:
                break
        return moved

    def migrate_path(self, hashes: np.ndarray) -> bool:
        """Push one root->leaf path to a peer and drop the local chain; False
        (local copy kept) when no peer qualifies or the push does not land."""
        victim = self._pin(hashes)
        if victim is None:
            self.stats["skipped_unpinnable"] += 1
            return False
        ok = False
        try:
            target = self._choose_target(victim)
            if target is None:
                self.stats["no_target"] += 1
                return False
            ok = self._push(target, victim)
        finally:
            victim.release()
        if not ok:
            return False
        freed = self._srv.iclient.evict_chain(hashes)
        self.stats["migrated_paths"] += 1
        self.stats["freed_blocks"] += int(freed)
        if freed == 0:
            self.stats["kept_after_push"] += 1   # touched or extended meanwhile
        return True

    def _pin(self, hashes: np.ndarray) -> Optional[_Victim]:
        """Pin [0, n) locally and collect its slots; None when the path is no
        longer fully here (evicted or split since the peek)."""
        client, n = self._srv.iclient, len(hashes)
        pools = pool_counts(self._srv.index)
        q = client.query(hashes, mask=_MASK[FULL], local_only=True, lock=True, update_meta=False)
        if int(q.status) != 0 or int(q.common_hit) != n or not q.full_fragments:
            if q.finalize is not None:
                q.finalize()
            return None
        full = np.concatenate([np.asarray(s, dtype=np.int32) for _, _, s in q.full_fragments])[:n]
        victim = _Victim(hashes, full, finalizers=(q.finalize,))
        if SWA in pools:
            qs = client.query(hashes, mask=_MASK[FULL] | _MASK[SWA], local_only=True, lock=True,
                              update_meta=False)
            if int(qs.status) == 0 and int(qs.common_hit) == n and len(qs.swa_slots):
                victim.swa = np.asarray(qs.swa_slots, dtype=np.int32)
            victim.finalizers += (qs.finalize,)
        if MAMBA in pools:
            qm = client.query(hashes, mask=_MASK[FULL] | _MASK[MAMBA], local_only=True, lock=True,
                              update_meta=False)
            if int(qm.status) == 0 and int(qm.common_hit) == n and qm.mamba is not None:
                victim.mamba = int(qm.mamba[1])
            victim.finalizers += (qm.finalize,)
        return victim

    def _floor(self, p: _Peer) -> float:
        c = self.cfg
        return c.min_peer_free_ratio if p.role == ROLE_DECODE else max(c.min_peer_free_ratio,
                                                                       c.prefill_peer_free_ratio)

    def _eligible(self, p: _Peer, kinds: List[ComponentType]) -> bool:
        floor = self._floor(p)
        return all(k in p.pools and p.free(k) >= floor for k in kinds)

    def _choose_target(self, victim: _Victim) -> Optional[_Peer]:
        need = victim.kinds()
        donors = [p for p in self._peers if p.role == ROLE_DECODE and self._eligible(p, need)]
        group = donors or [p for p in self._peers if p.role != ROLE_DECODE and self._eligible(p, need)]
        best, best_key = None, None
        for p in group:
            r = self._srv.iclient.remote_query(p.rank, victim.hashes, lock=False, update_meta=False)
            if int(r.status) != 0:
                self.stats["probe_errors"] += 1
                continue
            key = (int(r.total_hit_length), p.free(FULL))
            if best_key is None or key > best_key:
                best, best_key = p, key
        if best is not None and group is not donors:
            self.stats["to_prefill_peer"] += 1
        return best

    def _push(self, peer: _Peer, victim: _Victim) -> bool:
        # The target's [0, m) is not pinned: a remote QUERY_RELEASE rebuilds the
        # pin from the wire range only, so a pinned terminal that our own insert
        # then splits at m would keep a reference on the split-off suffix. If the
        # target evicts part of [0, m) meanwhile, its insert rejects the broken
        # prefix and the check below keeps the local copy.
        client, rank, n, hashes = self._srv.iclient, peer.rank, victim.n, victim.hashes
        probe = client.remote_query(rank, hashes, lock=False, update_meta=False)
        if int(probe.status) != 0:
            self.stats["push_fail_probe"] += 1
            return False
        m = int(probe.total_hit_length)
        if m < n and not self._push_component(rank, hashes, FULL, victim.full[m:], start=m):
            return False
        if victim.swa is not None and not self._push_component(rank, hashes, SWA, victim.swa):
            return False
        if victim.mamba is not None and not self._push_component(
                rank, hashes, MAMBA, np.asarray([victim.mamba], dtype=np.int32)):
            return False
        check = client.remote_query(rank, hashes, lock=False, update_meta=False)
        if int(check.status) != 0 or int(check.total_hit_length) < n:
            self.stats["push_partial"] += 1
            return False
        self.stats["pushed_blocks"] += n - m
        self.stats["dedup_blocks"] += m
        self.stats[f"to_rank_{rank}"] += 1
        # optimistic accounting until the peer's next capacity record
        landed = {FULL: n - m, SWA: 0 if victim.swa is None else len(victim.swa),
                  MAMBA: 0 if victim.mamba is None else 1}
        for k, used in landed.items():
            if k in peer.pools:
                free, total = peer.pools[k]
                peer.pools[k] = (max(0, free - used), total)
        return True

    def _push_component(self, rank: int, hashes: np.ndarray, kind: ComponentType,
                        local: np.ndarray, start: int = 0) -> bool:
        """Reserve ``len(local)`` slots of ``kind`` on ``rank``, RDMA-write the
        local slots into them and insert. FULL lands at ``hashes[start:]``;
        SWA covers the tail of ``hashes`` and MAMBA its END (start 0)."""
        client, name = self._srv.iclient, _POOL_NAMES[kind]
        k = len(local)
        rep = client.remote_allocate(rank, k, kind)
        remote = np.asarray(rep.slots, dtype=np.int32)
        if int(rep.status) != 0 or len(remote) != k:
            if len(remote):
                client.remote_recycle(rank, remote, kind)
            self.stats[f"push_fail_reserve_{name}"] += 1
            return False
        if not self._write(rank, kind, local, remote):
            client.remote_recycle(rank, remote, kind)
            self.stats[f"push_fail_write_{name}"] += 1
            return False
        res = client.remote_insert(rank, hashes, start, k, remote, True, kind)
        if res.error != _core.InsertError.OK:
            self.stats[f"push_fail_insert_{name}"] += 1
            return False
        self.stats[f"pushed_{name}_slots"] += k - len(res.unused_slots)
        return True

    def _write(self, rank: int, kind: ComponentType, local: np.ndarray, remote: np.ndarray) -> bool:
        srv, timeout = self._srv, self.cfg.timeout_ms
        ticket = srv.submit([(rank, int(kind), remote.tolist(), local.tolist(), XFER_WRITE)], timeout)
        deadline = time.monotonic() + timeout / 1000 + 5
        while time.monotonic() < deadline:
            rep = srv.wait_any([ticket], 1000)
            replies = rep["done"].get(ticket)
            if replies is not None:
                ok = all(r[0] for r in replies)
                if ok:
                    self.stats["write_bytes"] += sum(int(r[1]) for r in replies)
                return ok
            if ticket in rep["unknown"]:
                return False
        return False

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.stats)
        out["role"], out["enabled"] = self.role, self.cfg.enabled
        if self.last_error:
            out["last_error"] = self.last_error
        return out
