"""Direct expert-bank views over byte-identical safetensors tensors.

The index is intentionally narrower than the FTW format. It supports checkpoints whose
source tensor for one bank layer is already the exact contiguous ``[num_experts, ...]``
runtime bank. No fusion, transpose, quantization, padding, or tensor-parallel slicing may
be required. At present that contract is met by the packed BF16 Qwen3.5/Qwen3.6 MoE
layout.

DISK layers map the source shard directly. Safetensors payload offsets need not be page
aligned, so :class:`HostBank` floors the mmap offset and carries the payload's intra-page
offset in its tensor view. Buffered populate reads and the UFFD pager accept exact
unaligned offsets. O_DIRECT is not used by this reader because its offset and length
alignment contract cannot be guaranteed by safetensors.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import math
import mmap
import os
import re
import struct
import warnings

import torch

from freetoken.utils import download_hf_weight, init_logger


logger = init_logger(__name__)

INDEX_NAME = "freetoken_bank_index.json"
FORMAT_TAG = "freetoken_safetensors_bank_index"
FORMAT_VERSION = 1

_QWEN_PACKED_ARCHES = {"Qwen3_5MoeForConditionalGeneration"}
_QWEN_PACKED_RE = re.compile(
    r"^(?:model\.language_model\.|language_model\.|model\.)"
    r"layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<bank>gate_up_proj|down_proj)$"
)
_BANK_NAME = {"gate_up_proj": "gate_up", "down_proj": "down"}
_TORCH_DTYPE = {"BF16": torch.bfloat16}
_DTYPE_BYTES = {"BF16": 2}


class UnsupportedSafetensorsBankIndex(ValueError):
    """The checkpoint needs a runtime repack and therefore cannot be mapped directly."""


def _checkpoint_folder(model_path: str) -> str:
    folder = download_hf_weight(model_path)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"checkpoint directory not found: {folder}")
    return folder


def index_path(model_path: str) -> str:
    return os.path.join(model_path, INDEX_NAME)


def _shard_names(folder: str) -> list[str]:
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(folder, "*.safetensors")))


def _read_header(path: str) -> tuple[bytes, dict, int]:
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors length prefix: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > size - 8:
            raise ValueError(f"safetensors header exceeds shard size: {path}")
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    return raw, header, 8 + header_len


def _config_geometry(folder: str) -> tuple[str, int, int]:
    path = os.path.join(folder, "config.json")
    with open(path, encoding="utf-8") as handle:
        root = json.load(handle)
    text = root.get("text_config") or root
    architectures = list(root.get("architectures") or ()) + list(text.get("architectures") or ())
    architecture = next((arch for arch in architectures if arch in _QWEN_PACKED_ARCHES), None)
    if architecture is None:
        shown = architectures[0] if architectures else "unknown"
        raise UnsupportedSafetensorsBankIndex(
            f"architecture {shown!r} has no byte-identical safetensors bank layout; "
            "use --bank-source ftw"
        )
    try:
        num_layers = int(text["num_hidden_layers"])
        num_experts = int(text["num_experts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedSafetensorsBankIndex(
            "packed Qwen bank indexing requires num_hidden_layers and num_experts"
        ) from exc
    if num_layers <= 0 or num_experts <= 0:
        raise UnsupportedSafetensorsBankIndex("packed Qwen bank geometry must be positive")
    return architecture, num_layers, num_experts


def _manifest_entry(folder: str, shard: str, raw_header: bytes) -> dict:
    stat = os.stat(os.path.join(folder, shard))
    return {
        "file": shard,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
    }


def _fingerprint(manifest: list[dict]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _records_fingerprint(banks: list[dict]) -> str:
    canonical = json.dumps(banks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _validate_expert_records(entry: dict, num_experts: int) -> None:
    row_bytes = entry["length"] // num_experts
    records = entry["experts"]
    if len(records) != num_experts:
        raise ValueError(f"bank {entry['bank']!r} has {len(records)} expert records")
    for expert, record in enumerate(records):
        expected = entry["offset"] + expert * row_bytes
        if record != [entry["file"], expected, row_bytes]:
            raise ValueError(
                f"bank {entry['bank']!r} expert {expert} has a non-contiguous source range"
            )


def build_safetensors_bank_index(model_path: str) -> dict:
    """Build and atomically persist the supported source-bank index."""
    folder = _checkpoint_folder(model_path)
    architecture, num_layers, num_experts = _config_geometry(folder)
    shards = _shard_names(folder)
    if not shards:
        raise UnsupportedSafetensorsBankIndex("checkpoint contains no safetensors shards")

    manifest: list[dict] = []
    banks: list[dict] = []
    for shard in shards:
        path = os.path.join(folder, shard)
        raw_header, header, payload_base = _read_header(path)
        manifest.append(_manifest_entry(folder, shard, raw_header))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            match = _QWEN_PACKED_RE.match(name)
            if match is None:
                continue
            dtype = tensor.get("dtype")
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            if dtype not in _DTYPE_BYTES:
                raise UnsupportedSafetensorsBankIndex(
                    f"packed expert tensor {name!r} uses {dtype!r}; only BF16 is byte-identical"
                )
            if not isinstance(shape, list) or not shape or int(shape[0]) != num_experts:
                raise UnsupportedSafetensorsBankIndex(
                    f"packed expert tensor {name!r} does not start with {num_experts} experts"
                )
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data_offsets for {name!r}")
            begin, end = map(int, offsets)
            length = end - begin
            expected = math.prod(int(dim) for dim in shape) * _DTYPE_BYTES[dtype]
            if begin < 0 or length != expected or payload_base + end > os.path.getsize(path):
                raise ValueError(
                    f"safetensors byte range mismatch for {name!r}: {length} != {expected}"
                )
            offset = payload_base + begin
            row_bytes = length // num_experts
            entry = {
                "layer": int(match.group("layer")),
                "bank": _BANK_NAME[match.group("bank")],
                "dtype": dtype,
                "shape": [int(dim) for dim in shape],
                "file": shard,
                "offset": offset,
                "length": length,
                # Each tuple is [shard file, absolute byte offset, byte length].
                "experts": [
                    [shard, offset + expert * row_bytes, row_bytes]
                    for expert in range(num_experts)
                ],
            }
            _validate_expert_records(entry, num_experts)
            banks.append(entry)

    banks.sort(key=lambda item: (item["layer"], item["bank"]))
    expected_keys = {
        (layer, bank)
        for layer in range(num_layers)
        for bank in ("gate_up", "down")
    }
    actual_keys = {(entry["layer"], entry["bank"]) for entry in banks}
    if actual_keys != expected_keys or len(banks) != len(expected_keys):
        missing = sorted(expected_keys - actual_keys)[:8]
        extra = sorted(actual_keys - expected_keys)[:8]
        raise UnsupportedSafetensorsBankIndex(
            f"packed Qwen expert banks are incomplete or duplicated; missing={missing}, extra={extra}"
        )

    index = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "architecture": architecture,
        "quant_format": "bf16",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "fingerprint_kind": "sha256(shard-name,size,mtime-ns,ctime-ns,header-sha256)",
        "fingerprint": _fingerprint(manifest),
        "records_sha256": _records_fingerprint(banks),
        "shards": manifest,
        "banks": banks,
    }
    target = index_path(folder)
    temporary = f"{target}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(index, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return index


def _manifest_matches(folder: str, index: dict) -> bool:
    manifest = index.get("shards")
    if not isinstance(manifest, list):
        return False
    if _shard_names(folder) != [item.get("file") for item in manifest]:
        return False
    for item in manifest:
        try:
            stat = os.stat(os.path.join(folder, item["file"]))
        except (KeyError, OSError):
            return False
        if (
            stat.st_size != item.get("size")
            or stat.st_mtime_ns != item.get("mtime_ns")
            or stat.st_ctime_ns != item.get("ctime_ns")
        ):
            return False
        try:
            raw_header, _header, _payload_base = _read_header(
                os.path.join(folder, item["file"])
            )
        except (OSError, ValueError):
            return False
        if hashlib.sha256(raw_header).hexdigest() != item.get("header_sha256"):
            return False
    return (
        index.get("fingerprint") == _fingerprint(manifest)
        and index.get("records_sha256") == _records_fingerprint(index.get("banks", []))
    )


def is_safetensors_bank_index_stale(model_path: str, index: dict | None = None) -> bool:
    folder = _checkpoint_folder(model_path)
    if index is None:
        try:
            with open(index_path(folder), encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, ValueError):
            return True
    if index.get("format") != FORMAT_TAG or index.get("version") != FORMAT_VERSION:
        return True
    return not _manifest_matches(folder, index)


def ensure_safetensors_bank_index(model_path: str) -> tuple[str, dict]:
    """Return ``(local_folder, index)``, rebuilding a missing or stale index once."""
    folder = _checkpoint_folder(model_path)
    # config.json is stable for the checkpoint lifetime and avoids leaving a second
    # sidecar merely for locking. Atomic replacement still publishes the JSON index.
    with open(os.path.join(folder, "config.json"), "rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            current = None
            try:
                with open(index_path(folder), encoding="utf-8") as handle:
                    current = json.load(handle)
            except (OSError, ValueError):
                pass
            if current is None or is_safetensors_bank_index_stale(folder, current):
                current = build_safetensors_bank_index(folder)
                logger.info_rank0(f"expert banks: wrote safetensors index {index_path(folder)}")
            return folder, current
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def has_safetensors_bank_index(model_path: str) -> bool:
    """True when the checkpoint can build a byte-identical bank index."""
    try:
        ensure_safetensors_bank_index(model_path)
        return True
    except (FileNotFoundError, UnsupportedSafetensorsBankIndex):
        return False


def _copy_mapped_entry(bank, file_path: str, entry: dict) -> None:
    map_offset = entry["offset"] // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    view_offset = entry["offset"] - map_offset
    fd = os.open(file_path, os.O_RDONLY)
    try:
        buf = mmap.mmap(
            fd,
            view_offset + entry["length"],
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
            offset=map_offset,
        )
    finally:
        os.close(fd)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable", category=UserWarning
            )
            source = torch.frombuffer(
                buf,
                dtype=_TORCH_DTYPE[entry["dtype"]],
                count=entry["length"] // _DTYPE_BYTES[entry["dtype"]],
                offset=view_offset,
            ).view(*entry["shape"])
        bank.tensor.copy_(source)
        del source
    finally:
        buf.close()


def load_indexed_banks(
    model_path: str,
    *,
    num_layers: int,
    dtype: torch.dtype,
    layer_residency: list[str] | None = None,
    disk_pager=None,
    hot_expert_ids: dict[int, tuple[int, ...]] | None = None,
    hot_expert_capacity: dict[int, int] | None = None,
    hugepages_tmpfs: str | None = None,
    hugepages_tmpfs_margin_bytes: int = 1 << 30,
):
    """Load packed banks, directly mapping only layers labelled ``DISK``."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import (
        HostBank, HostResidency, TmpfsMirrorSource, born_pinned_default,
        prepare_tmpfs_bank_mirrors,
    )

    folder, index = ensure_safetensors_bank_index(model_path)
    if index["num_layers"] != num_layers:
        raise ValueError(
            f"bank index has {index['num_layers']} layers, model config has {num_layers}"
        )
    if dtype is not torch.bfloat16:
        raise UnsupportedSafetensorsBankIndex(
            f"indexed packed banks require torch.bfloat16 serving dtype, got {dtype}"
        )
    residency = layer_residency or [HostResidency.PINNED.value] * num_layers
    if len(residency) != num_layers:
        raise ValueError(f"expected {num_layers} residency labels, got {len(residency)}")
    valid = {item.value for item in HostResidency}
    if set(residency) - valid:
        raise ValueError(f"unknown host residency labels: {sorted(set(residency) - valid)}")

    disk_layers = {layer for layer, label in enumerate(residency) if label == "disk"}
    if hugepages_tmpfs and disk_pager is not None:
        raise ValueError(
            "--moe-bank-hugepages-tmpfs cannot be used with --moe-disk-pager uffd"
        )
    tmpfs_paths: dict[str, str] = {}
    tmpfs_huge = None
    if hugepages_tmpfs:
        mirror_sources = [
            TmpfsMirrorSource(
                f"{entry['bank']}#L{entry['layer']:05d}",
                os.path.join(folder, entry["file"]),
                entry["offset"],
                entry["length"],
            )
            for entry in index["banks"]
            if entry["layer"] in disk_layers
        ]
        tmpfs_paths, tmpfs_huge, capacity = prepare_tmpfs_bank_mirrors(
            hugepages_tmpfs,
            mirror_sources,
            margin_bytes=hugepages_tmpfs_margin_bytes,
        )
        bank_bytes, margin, required, free, reusable, available = capacity
        logger.info_rank0(
            f"MoE DISK tmpfs mirrors: huge={tmpfs_huge}; "
            f"capacity required={bank_bytes} bank + {margin} margin = {required} "
            f"bytes; available={free} free + {reusable} reusable = {available} bytes"
        )
    born = born_pinned_default()
    sources: dict[str, list[torch.Tensor | None]] = {
        "gate_up": [None] * num_layers,
        "down": [None] * num_layers,
    }
    owners: dict[str, list[HostBank | None]] = {
        "gate_up": [None] * num_layers,
        "down": [None] * num_layers,
    }
    shard_files = {item["file"] for item in index["shards"]}
    for entry in index["banks"]:
        _validate_expert_records(entry, index["num_experts"])
        layer = entry["layer"]
        if entry["file"] not in shard_files or os.path.basename(entry["file"]) != entry["file"]:
            raise ValueError(f"indexed bank references an unknown shard: {entry['file']!r}")
        if sources.get(entry["bank"]) is None or not 0 <= layer < num_layers:
            raise ValueError(
                f"invalid indexed bank key {(layer, entry.get('bank'))!r}"
            )
        if sources[entry["bank"]][layer] is not None:
            raise ValueError(f"duplicate indexed bank {(layer, entry['bank'])!r}")
        file_path = os.path.join(folder, entry["file"])
        if layer in disk_layers:
            backing = "uffd" if disk_pager is not None else "file"
            mirror_key = f"{entry['bank']}#L{layer:05d}"
            mapped_path = tmpfs_paths.get(mirror_key, file_path)
            bank = HostBank(
                tuple(entry["shape"]),
                _TORCH_DTYPE[entry["dtype"]],
                backing=backing,
                file_path=mapped_path,
                file_offset=0 if mirror_key in tmpfs_paths else entry["offset"],
                disk_pager=disk_pager,
                tmpfs_backed=mirror_key in tmpfs_paths,
                tmpfs_huge=tmpfs_huge if mirror_key in tmpfs_paths else None,
            )
        else:
            backing = "cuda" if born and residency[layer] == "pinned" else "mmap"
            bank = HostBank(
                tuple(entry["shape"]), _TORCH_DTYPE[entry["dtype"]], backing=backing
            )
            _copy_mapped_entry(bank, file_path, entry)
            if residency[layer] == "pinned":
                bank.pin()
            elif residency[layer] == "locked":
                bank.lock()
        sources[entry["bank"]][layer] = bank.tensor
        owners[entry["bank"]][layer] = bank

    applied = list(residency)
    for layer in range(num_layers):
        for bank_name in sources:
            if sources[bank_name][layer] is None:
                raise ValueError(f"indexed bank {bank_name!r} is missing layer {layer}")
            owner = owners[bank_name][layer]
            if applied[layer] == "locked" and owner.residency is not HostResidency.LOCKED:
                applied[layer] = HostResidency.PAGEABLE.value

    seeded = {
        int(layer): tuple(int(expert) for expert in expert_ids)
        for layer, expert_ids in (hot_expert_ids or {}).items()
    }
    capacities = {
        int(layer): int(capacity)
        for layer, capacity in (hot_expert_capacity or {}).items()
    }
    for layer, expert_ids in seeded.items():
        capacities.setdefault(layer, len(expert_ids))
    hot_eligible_layers = {
        layer for layer, label in enumerate(residency)
        if label in (HostResidency.DISK.value, HostResidency.PINNED.value)
    }
    invalid = (set(seeded) | set(capacities)) - hot_eligible_layers
    if invalid:
        raise ValueError(
            f"HOT expert rows require DISK or PINNED layers, got {sorted(invalid)}"
        )

    for layer, capacity in sorted(capacities.items()):
        expert_ids = seeded.get(layer, ())
        if capacity <= 0 or capacity > index["num_experts"] or len(expert_ids) > capacity:
            raise ValueError(f"invalid HOT capacity {capacity} for layer {layer}")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(f"HOT expert ids for layer {layer} contain duplicates")
        if any(expert < 0 or expert >= index["num_experts"] for expert in expert_ids):
            raise ValueError(f"HOT expert id outside layer {layer}'s expert range")

    return ExpertBanks(
        "bf16",
        {name: [tensor for tensor in layers if tensor is not None] for name, layers in sources.items()},
        layer_residency=applied,
        hot_expert_ids=seeded if capacities else {},
        hot_expert_capacity=capacities,
    )


def indexed_bank_byte_breakdown(model_path: str) -> tuple[int, int] | None:
    try:
        _folder, index = ensure_safetensors_bank_index(model_path)
    except (FileNotFoundError, UnsupportedSafetensorsBankIndex):
        return None
    return sum(int(entry["length"]) for entry in index["banks"]), 0


__all__ = [
    "FORMAT_TAG",
    "FORMAT_VERSION",
    "INDEX_NAME",
    "UnsupportedSafetensorsBankIndex",
    "build_safetensors_bank_index",
    "ensure_safetensors_bank_index",
    "has_safetensors_bank_index",
    "index_path",
    "indexed_bank_byte_breakdown",
    "is_safetensors_bank_index_stale",
    "load_indexed_banks",
]
