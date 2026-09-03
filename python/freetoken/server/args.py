from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from freetoken.distributed import DistributedInfo
from freetoken.scheduler import SchedulerConfig
from freetoken.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # The terminal shell is attached to this server (ft shell --model / ft serve --shell-mode).
    # The workers read it to leave the shell's foreground process group, so the ^C that cancels
    # a turn cannot also kill the engine — see server/launch.py:_detach_process_group.
    shell_mode: bool = False
    served_model_name: str | None = None
    tool_call_parser: str = "llama3"
    # Reasoning parser that splits <think> reasoning from content for OpenAI
    # responses. None disables it (default for models without a reasoning protocol).
    reasoning_parser: str | None = None
    # "model": fill unspecified request sampling params from generation_config.json
    # (temperature/top_k/top_p), like sglang. "none": use framework defaults only.
    sampling_defaults: str = "model"
    # Default max output (decode) tokens for a request that omits one. None falls back to the
    # adapter's built-in default (32k).
    max_output_tokens: int | None = None
    # Cancel scheduler work when the HTTP peer goes away. Kept as on/off to match the CLI
    # spelling and leave room for future disconnect policies without adding inverse flags.
    abort_on_disconnect: str = "on"
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens). Mirrors sglang's --enable-cache-report.
    enable_cache_report: bool = False
    # Comma-separated CORS allow-list for browser/webview clients (e.g. the desktop
    # app). Empty string disables CORS headers entirely; "*" allows any origin.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"
    # --gpu entries in TP-rank order, empty = not given
    gpu: tuple[str, ...] = ()
    # full UUIDs resolved from --gpu, entry i = TP rank i; None = NVML unavailable, each worker then resolves its raw entry against CUDA's own enumeration
    gpu_assigned: "tuple[str, ...] | None" = None

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/freetoken_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/freetoken_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(
    args: List[str],
    run_shell: bool = False,
    prog: str | None = None,
) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from freetoken.attention import validate_attn_backend
    from freetoken.kvcache import SUPPORTED_CACHE_MANAGER
    from freetoken.moe import SUPPORTED_MOE_BACKENDS

    def _parse_moe_cache_rate(value: str) -> float:
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
        if not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1]")
        return rate

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if n < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return n

    def _harness_prefix(value: str) -> str:
        kind, separator, prefix = value.partition("=")
        if not separator or not kind.strip() or not prefix.strip():
            raise argparse.ArgumentTypeError("must use non-empty kind=prefix syntax")
        return f"{kind.strip()}={prefix}"

    def _gpu_prefill_layers(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in ("auto", "off"):
            return normalized
        try:
            n = int(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be 'auto', 'off', or a positive integer"
            ) from exc
        if n < 1:
            raise argparse.ArgumentTypeError(
                "must be 'auto', 'off', or a positive integer"
            )
        return str(n)

    def _hot_adapt_interval(value: str) -> str | int:
        normalized = value.strip().lower()
        if normalized == "auto":
            return normalized
        try:
            interval = int(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be 'auto' or a non-negative integer"
            ) from exc
        if interval < 0:
            raise argparse.ArgumentTypeError(
                "must be 'auto' or a non-negative integer"
            )
        return interval

    def _nonnegative_float(value: str) -> float:
        try:
            n = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a non-negative number") from exc
        if not math.isfinite(n) or n < 0:
            raise argparse.ArgumentTypeError("must be >= 0")
        return n

    def _lazy_gpu_arg(value: str) -> tuple[str, ...]:
        from freetoken.gpu_select import gpu_arg

        return gpu_arg(value)

    def _infer_tool_call_parser(model_path: str) -> str:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        # M3 first: its marker also contains the bare "minimax" substring, but the
        # namespaced tool grammar is a different protocol from M2's.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3_coder"
        if (
            "qwen3_5" in marker
            or "qwen3.5" in marker
            or ("qwen3" in marker and "coder" in marker)
        ):
            return "qwen3_coder"
        if "qwen" in marker:
            return "qwen25"
        if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
            return "deepseekv32"
        if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
            return "deepseekv32"
        if "glm" in marker:
            return "glm47"
        if "mistral" in marker:
            return "mistral"
        return "llama3"

    def _infer_reasoning_parser(model_path: str) -> str | None:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        if "deepseek" in marker and any(
            tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")
        ):
            return "deepseekv32"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3"
        if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
            return "qwen3"
        if "glm" in marker:
            return "glm"
        # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
        # not M2's always-on implicit <think>.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        return None

    parser = argparse.ArgumentParser(prog=prog, description="FreeToken Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--gpu",
        type=_lazy_gpu_arg,
        default=ServerArgs.gpu,
        help=(
            "GPU(s) to run on, comma-separated; entry i is TP rank i. Each entry is a GPU "
            "UUID (GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index"
        ),
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=ServerArgs.max_output_tokens,
        help="Default max output tokens for requests that omit one (default 32k).",
    )

    parser.add_argument(
        "--abort-on-disconnect",
        choices=["on", "off"],
        default=ServerArgs.abort_on_disconnect,
        help="Abort queued or running generation when its HTTP client disconnects.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help=(
            "Fraction of total GPU free memory the engine may use for weights + MoE "
            "cache + KV cache combined; the remainder is reserved runtime headroom."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--ple-backend",
        choices=["pinned", "cached", "disk", "uring", "hmm"],
        default=ServerArgs.ple_backend,
        help=(
            "Qwen3.8 Flash-Next PLE table backend: 'pinned' keeps the full table in "
            "pinned host RAM; 'cached' pins a bounded hot-row bank; 'disk' stages rows "
            "from read-only mappings; 'uring' streams rows from the checkpoint with "
            "Linux io_uring; 'hmm' lets the GPU read those mappings directly."
        ),
    )

    parser.add_argument(
        "--ple-prefill-gather",
        choices=["on", "off"],
        default=ServerArgs.ple_prefill_gather,
        help=(
            "Bulk-stage deduplicated PLE rows before each prefill chunk when "
            "--ple-backend hmm is selected. Decode always keeps direct HMM gathers."
        ),
    )

    parser.add_argument(
        "--speculative-mtp",
        choices=["off", "on"],
        default=ServerArgs.speculative_mtp,
        help=(
            "Qwen3.8 Flash-Next native MTP speculative decoding. The initial "
            "implementation is greedy-only and batch-size 1; unsupported batches "
            "fall back to normal decode."
        ),
    )

    parser.add_argument(
        "--mtp-draft-tokens",
        type=int,
        default=ServerArgs.mtp_draft_tokens,
        help="Number of MTP drafts per step. This fused implementation requires 1.",
    )

    parser.add_argument(
        "--ple-cache-gib",
        type=float,
        default=ServerArgs.ple_cache_gib,
        help="Pinned PLE hot-row bank budget in GiB for --ple-backend cached.",
    )

    parser.add_argument(
        "--ple-cache-warm",
        type=str,
        default=ServerArgs.ple_cache_warm,
        help="Optional JSON row-frequency profile used to warm the PLE cache.",
    )

    parser.add_argument(
        "--ple-cache-profile-out",
        type=str,
        default=ServerArgs.ple_cache_profile_out,
        help="Write cumulative cached-PLE row frequencies to this JSON file.",
    )

    parser.add_argument(
        "--ple-uring-staging-mib",
        type=_positive_int,
        default=ServerArgs.ple_uring_staging_mib,
        help=(
            "Per-PLE-layer resident staging budget in MiB for --ple-backend "
            "uring, including data, scale conversion, local IDs, and io_uring "
            "bounce buffers (default: 64)."
        ),
    )

    parser.add_argument(
        "--ple-uring-queue-depth",
        type=_positive_int,
        default=ServerArgs.ple_uring_queue_depth,
        help="io_uring submission queue depth for --ple-backend uring (default: 64).",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=_positive_int,
        default=ServerArgs.decode_log_interval,
        help="Print one decode scheduler status line every N decode forwards.",
    )

    parser.add_argument(
        "--priority-aging-seconds",
        type=_nonnegative_float,
        default=ServerArgs.priority_aging_seconds,
        help=(
            "Seconds a waiting request needs to gain one effective priority point. "
            "For parked KV-ladder requests, this is also the starvation bound: once "
            "the oldest waiter passes it, that waiter is admitted ahead of priority "
            "ordering and new admissions pause (default: 30; 0 disables both)."
        ),
    )

    kv_capacity_group = parser.add_mutually_exclusive_group()
    kv_capacity_group.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    kv_capacity_group.add_argument(
        "--num-tokens",
        dest="num_token_override",
        type=int,
        default=ServerArgs.num_token_override,
        help=(
            "Total KV-cache capacity in tokens; must be a multiple of the resolved page "
            "size (DSV4: 128 window page, TRTLLM backend: 64). Mutually exclusive with "
            "--num-pages."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "bf16", "fp8_e4m3"],
        default=ServerArgs.kv_cache_dtype,
        help=(
            "FULL-attention K/V storage dtype. Auto selects fp8_e4m3 on sm_89+ "
            "when every selected attention backend can consume it, otherwise bf16."
        ),
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache strategy (naive | radix). For hybrid GDN models 'radix' is materialized "
        "as a GDN-aware radix (cross-request GDN-state prefix reuse); pass 'naive' to opt out.",
    )

    parser.add_argument(
        "--kv-disk-cache-dir",
        type=str,
        default=ServerArgs.kv_disk_cache_dir,
        help="Directory for crash-safe whole-prefix QSA/GDN state entries.",
    )

    parser.add_argument(
        "--kv-disk-cache-gib",
        type=float,
        default=ServerArgs.kv_disk_cache_gib,
        help="Disk prefix-cache byte budget in GiB (default: 0, disabled).",
    )

    parser.add_argument(
        "--kv-harness-prefixes",
        action="append",
        type=_harness_prefix,
        default=None,
        metavar="KIND=PREFIX",
        help=(
            "Coding-harness system-prefix signature. Repeat for multiple signatures. "
            "Supplying any entries replaces the built-in OpenCode and Pi signatures."
        ),
    )

    parser.add_argument(
        "--lazy-restore",
        type=str,
        choices=["on", "off"],
        default=ServerArgs.lazy_restore,
        help=(
            "Demand-load page-indexed QSA KV during disk prefix restore. Entries without "
            "a block index fall back to eager restore."
        ),
    )

    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=ServerArgs.enable_cache_report,
        help=(
            "Return the number of prefix-cached prompt tokens in each response's usage block "
            "(OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic "
            "usage.cache_read_input_tokens, Responses usage.input_tokens_details.cached_tokens). "
            "On /v1/messages this also makes input_tokens EXCLUDE the cached prefix, matching "
            "Anthropic billing semantics."
        ),
    )

    parser.add_argument(
        "--sampling-defaults",
        type=str,
        default=ServerArgs.sampling_defaults,
        choices=["model", "none"],
        help=(
            "Source for unspecified request sampling params. 'model' fills "
            "temperature/top_k/top_p from the checkpoint's generation_config.json "
            "(recommended for reasoning models to avoid greedy repetition loops); "
            "'none' uses framework defaults only."
        ),
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Model id returned by /v1/models. Defaults to the basename of --model.",
    )

    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "llama3",
            "qwen",
            "qwen25",
            "qwen3_coder",
            "mistral",
            "deepseekv32",
            "gemma4",
            "glm47",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gpt_oss",
            "gpt-oss",
        ],
        help="Tool-call parser format for OpenAI-compatible tool responses.",
    )

    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="auto",
        choices=[
            "auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm",
            "minimax", "minimax_m3", "muse_glimmer", "gemma4",
        ],
        help=(
            "Reasoning parser that splits chain-of-thought into reasoning_content "
            "for OpenAI responses. 'auto' selects per model family (gpt-oss Harmony, "
            "<think> for qwen3/glm/minimax, <mm:think> for minimax-m3, ATEM to=self "
            "channels for muse-glimmer, gemma thought, dsv4); 'off' disables it."
        ),
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help=(
            "The MoE backend to use. 'auto' resolves a MoE model to the offload family "
            "(offload, or hybrid when a `ft bench bw` profile recommends it); resident "
            "'fused' experts must be requested explicitly."
        ),
    )

    parser.add_argument(
        "--nvfp4-backend",
        default=ServerArgs.nvfp4_backend,
        choices=["auto", "marlin", "flashinfer", "triton"],
        help=(
            "NVFP4 routed-expert GEMM backend. The implicit default is Triton unless "
            "NVFP4 activation auto-selection chooses the SM120 b12x path. auto picks by "
            "GPU; an explicit choice is never rewritten and fails if it cannot run."
        ),
    )

    parser.add_argument(
        "--moe-activation-dtype",
        default=ServerArgs.moe_activation_dtype,
        choices=["auto", "bf16", "nvfp4"],
        help=(
            "Routed-expert activation dtype. auto uses NVFP4 only on sm120 when every "
            "expert GEMM has a ModelOpt input scale; bf16 keeps W4A16; nvfp4 fails at "
            "startup when the architecture, scales, FlashInfer entry, or FTW layout "
            "cannot support it."
        ),
    )

    parser.add_argument(
        "--expert-load",
        default=ServerArgs.expert_load,
        choices=["auto", "serial", "parallel"],
        help=(
            "How MoE expert banks are read into host RAM. 'auto' (default) reads scattered "
            "experts in parallel (fast) but falls back to serial when free RAM can't cover "
            "the banks + the parallel reader's extra whole-shard buffer; 'serial' forces the "
            "low-memory reclaimable read (slower); 'parallel' forces the fast read."
        ),
    )

    parser.add_argument(
        "--bank-source",
        default=ServerArgs.bank_source,
        choices=["auto", "ftw", "index"],
        help=(
            "Expert bank source. 'auto' prefers FTW, then a supported byte-identical "
            "safetensors index; 'ftw' and 'index' force one path."
        ),
    )

    parser.add_argument(
        "--moe-bank-hugepages",
        default=ServerArgs.moe_bank_hugepages,
        choices=["auto", "on", "off"],
        help=(
            "Transparent huge pages for expert-bank mappings. 'auto' advises eligible "
            "Linux mappings and otherwise disables silently; 'on' requires runtime "
            "MADV_HUGEPAGE support; 'off' disables the advice."
        ),
    )
    parser.add_argument(
        "--moe-bank-hugepages-tmpfs",
        default=ServerArgs.moe_bank_hugepages_tmpfs,
        metavar="DIR",
        help=(
            "Mirror file-backed DISK expert banks into a tmpfs mounted with "
            "huge=always or huge=within_size, then map the mirrors. Requires "
            "--moe-bank-hugepages auto or on; off rejects this option."
        ),
    )
    parser.add_argument(
        "--moe-bank-hugepages-tmpfs-margin-gib",
        type=float,
        default=ServerArgs.moe_bank_hugepages_tmpfs_margin_gib,
        help="Free tmpfs capacity retained beyond all DISK bank mirrors (default: 1 GiB).",
    )

    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=int,
        default=ServerArgs.moe_cache_size,
        help="The number of unified MoE expert slots on GPU.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=_parse_moe_cache_rate,
        default=ServerArgs.moe_cache_rate,
        help="The fraction of all MoE experts to keep in GPU cache.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        default=ServerArgs.moe_cache_auto,
        help=(
            "Auto-pick --moe-cache-size from free VRAM and expert size, MoE-priority "
            "(KV gets --kv-reserve-tokens as a floor). Not supported for owned-KV models."
        ),
    )

    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=ServerArgs.kv_reserve_tokens,
        help=(
            "Usable KV-cache token floor reserved before --moe-cache-auto fills experts "
            "(default 8192; the internal dummy page is additional). With --kv-ladder "
            "the effective floor is at least two 32768-token growth steps, or the "
            "configured growth cap when that cap is lower."
        ),
    )

    kv_ladder_unspecified = object()
    parser.add_argument(
        "--kv-ladder",
        choices=["on", "off"],
        default=kv_ladder_unspecified,
        help=(
            "Grow KV at request boundaries by shrinking the auto-sized MoE expert cache "
            "(default: on). It is inactive unless --moe-cache-auto and a supported "
            "offload cache are in use, and unless --max-running-requests is 1. An explicit "
            "'--kv-ladder on' rejects higher concurrency. With the "
            "ladder active, --num-pages or --num-tokens sets the growth cap instead of "
            "the startup pool size. 'off' keeps the startup reservation fixed."
        ),
    )

    parser.add_argument(
        "--linear-state-cache-ratio",
        type=float,
        default=ServerArgs.linear_state_cache_ratio,
        help=(
            "GDN/linear-state slot pool as a multiple of --max-running-requests. "
            "Slots beyond the running requests hold prefix-cache snapshots; raise "
            "it so multi-turn chats survive between messages (costs VRAM per slot)."
        ),
    )

    parser.add_argument(
        "--moe-cache-policy",
        default=ServerArgs.moe_cache_policy,
        choices=["lru"],
        help="The unified MoE cache eviction policy.",
    )

    parser.add_argument(
        "--moe-cpu-threads",
        type=int,
        default=ServerArgs.moe_cpu_threads,
        help=(
            "Number of CPU worker threads for --moe-backend cpu decode experts. "
            "0 = auto (physical cores)."
        ),
    )

    parser.add_argument(
        "--moe-cpu-executor-mode",
        choices=["sleep", "spin", "auto"],
        default=ServerArgs.moe_cpu_executor_mode,
        help=(
            "CPU MoE worker synchronization: 'sleep' uses condition variables "
            "(default), 'spin' busy-polls for lower wake latency with a bounded "
            "wait fallback, and 'auto' enables spin on suitable single-socket x86 "
            "systems."
        ),
    )

    parser.add_argument(
        "--moe-cpu-layers",
        type=str,
        default=ServerArgs.moe_cpu_layers,
        help=(
            "With --moe-backend offload/hybrid: which MoE layers compute on the "
            "CPU executor instead of the GPU offload/PCIe path (where CUDA pinning "
            "is quota-capped, e.g. WSL, their banks are OS-locked instead of pinned). Explicit id list ('3,7,11'), a count ('8' = 8 "
            "layers evenly strided), or a fraction ('0.5'). Unset = automatic where "
            "CUDA pinning is quota-capped, e.g. WSL (locks just enough head+tail "
            "layers when the banks exceed the pin budget, none otherwise); '0' "
            "forces all layers on GPU."
        ),
    )

    parser.add_argument(
        "--moe-disk-layers",
        type=str,
        default=ServerArgs.moe_disk_layers,
        help=(
            "Which MoE layers keep expert banks as read-only file-backed mappings "
            "and use the selected DISK decode mode. Explicit id list ('3,7,11'), a count "
            "('8' = 8 layers evenly strided), or a fraction ('0.5'). Requires FTW or "
            "a supported byte-identical safetensors bank index."
        ),
    )

    parser.add_argument(
        "--moe-disk-layer-profile",
        type=str,
        default=ServerArgs.moe_disk_layer_profile,
        help=(
            "Versioned JSON profile with per-layer traffic scores and per-expert route "
            "counts. Automatic FTW spill selects the lowest-score layers; HOT expert "
            "residency uses the expert_hits section. Legacy flat layer-score objects "
            "remain accepted. Explicit --moe-disk-layers takes precedence."
        ),
    )

    parser.add_argument(
        "--moe-hot-expert-budget-gib",
        type=float,
        default=ServerArgs.moe_hot_expert_budget_gib,
        help=(
            "Pinned-host byte budget for HOT experts inside DISK layers. A versioned "
            "per-expert profile seeds the partition when provided; otherwise online "
            "adaptation starts all-cold. Requires DISK CPU decode. 0 disables "
            "expert-granular residency."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-halflife-steps",
        type=int,
        default=ServerArgs.moe_hot_adapt_halflife_steps,
        help=(
            "Routing-update-call half-life for online per-expert route counts "
            "(default: 2000)."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-interval-steps",
        type=_hot_adapt_interval,
        default=ServerArgs.moe_hot_adapt_interval_steps,
        help=(
            "Routed tokens between HOT partition recomputes. 'auto' derives fill and "
            "steady cadences from the allocation (default); an integer is fixed. "
            "Due fixed-interval ticks accumulate while work is active, and 0 disables "
            "adaptation."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-max-swap-gib",
        type=float,
        default=ServerArgs.moe_hot_adapt_max_swap_gib,
        help=(
            "Maximum HOT row bytes copied per adaptation tick (default: 0.5 GiB). "
            "A request boundary may consume several accumulated ticks, up to "
            "--moe-hot-adapt-boundary-cap-frac of the HOT budget (node-4 example: "
            "0.5 x 3.98 GiB per boundary)."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-boundary-cap-frac",
        type=float,
        default=ServerArgs.moe_hot_adapt_boundary_cap_frac,
        help=(
            "Maximum fraction of the HOT budget staged at one request boundary "
            "(default: 0.5)."
        ),
    )

    parser.add_argument(
        "--moe-hot-plan-persist",
        choices=["auto", "on", "off"],
        default=ServerArgs.moe_hot_plan_persist,
        help=(
            "Persist the adapted HOT expert plan across restarts. 'auto' reads an "
            "existing plan and writes when the plan directory is writable (default); "
            "'on' requests the same behavior with an explicit startup warning if writes "
            "are unavailable; 'off' disables reads and writes."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-idle-ms",
        type=int,
        default=ServerArgs.moe_hot_adapt_idle_ms,
        help=(
            "Idle time before HOT adaptation may use changed routing counters. "
            "Idle adaptation is supported only when TP == 1; it is inert under "
            "tensor parallelism. 0 disables idle adaptation. "
            "Preemption stops host staging at the next row boundary, but the staged "
            "prefix is still installed on the scheduler stream before the next forward. "
            "Up to the staging row count logged at startup can be queued, costing "
            "25 to 50 ms on node-4 at the default 0.5 GiB swap bound "
            "(default: 500 ms)."
        ),
    )

    parser.add_argument(
        "--moe-hot-adapt-idle-min-interval-ms",
        type=int,
        default=ServerArgs.moe_hot_adapt_idle_min_interval_ms,
        help=(
            "Minimum time between repeated HOT adaptation idle ticks "
            "(default: 2000 ms)."
        ),
    )

    parser.add_argument(
        "--moe-hot-plan-dir",
        type=str,
        default=ServerArgs.moe_hot_plan_dir,
        help=(
            "Directory for freetoken_hot_plan.json. Defaults to the model directory."
        ),
    )

    parser.add_argument(
        "--moe-hot-plan-interval-minutes",
        type=float,
        default=ServerArgs.moe_hot_plan_interval_minutes,
        help="Minutes between background HOT plan writes (default: 10).",
    )

    parser.add_argument(
        "--moe-collect-stats",
        action="store_true",
        default=ServerArgs.moe_collect_stats,
        help=(
            "Collect per-layer realized decode traffic for GET /v1/moe-layer-profile "
            "and report protected-slot oracle versus realized route coverage on status lines."
        ),
    )

    parser.add_argument(
        "--moe-disk-prefill",
        choices=["cpu", "copy"],
        default=ServerArgs.moe_disk_prefill,
        help=(
            "How DISK layers run prefill: 'cpu' computes routed experts through the "
            "CPU executor (default); 'copy' restores the whole-layer pageable copy "
            "to the GPU cache for benchmarking."
        ),
    )

    parser.add_argument(
        "--moe-prefill-coalesce",
        choices=["populate", "on", "off"],
        default=ServerArgs.moe_prefill_coalesce,
        help=(
            "Warm each DISK CPU-prefill layer's bounded routed expert union: "
            "'populate' reads backing-file ranges (default), 'on' uses advisory "
            "WILLNEED only, and 'off' disables the lease."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hot-split",
        choices=["on", "off"],
        default=ServerArgs.moe_prefill_hot_split,
        help=(
            "Run protected HOT routes of DISK prefill layers on the GPU and only "
            "the remaining cold routes on the CPU executor (default: on)."
        ),
    )

    parser.add_argument(
        "--moe-prefill-split-kernel",
        choices=["grouped", "decode"],
        default=ServerArgs.moe_prefill_split_kernel,
        help=(
            "GPU kernel for protected HOT routes during DISK prefill: 'grouped' "
            "uses the chunk GEMM (default). After a grouped-path CUDA fault, "
            "restarting with 'decode' is the only recovery; it uses the "
            "route-at-a-time kernel and can also support A/B measurements."
        ),
    )

    parser.add_argument(
        "--moe-cpu-prefill-batch",
        choices=["on", "off"],
        default=ServerArgs.moe_cpu_prefill_batch,
        help=(
            "Group CPU-prefill routes by expert and run row-batched NVFP4 W4A8 "
            "matmuls (default: on). Setup failures fall back to the serial path."
        ),
    )

    parser.add_argument(
        "--moe-disk-decode",
        choices=["cpu", "gpufetch"],
        default=ServerArgs.moe_disk_decode,
        help=(
            "How DISK layers run decode: 'cpu' uses the CPU executor (default); "
            "'gpufetch' stages LRU-missing routed experts from the file mapping into "
            "the existing GPU slot cache, then runs the normal GPU GEMM."
        ),
    )

    parser.add_argument(
        "--moe-cold-fetch-max",
        type=int,
        default=ServerArgs.moe_cold_fetch_max,
        help=(
            "With DISK CPU decode and a HOT expert budget, fetch at most N distinct "
            "COLD experts per layer and step into non-protected GPU slots. 0 "
            "disables the policy (default)."
        ),
    )

    parser.add_argument(
        "--moe-disk-pager",
        choices=["madvise", "uffd"],
        default=ServerArgs.moe_disk_pager,
        help=(
            "DISK expert residency backend: 'madvise' keeps the existing file-mmap "
            "path (default); Linux-only 'uffd' fills complete expert rows into an "
            "anonymous userfaultfd region."
        ),
    )

    parser.add_argument(
        "--moe-disk-lookahead",
        choices=["on", "off"],
        default=ServerArgs.moe_disk_lookahead,
        help=(
            "Prefetch each madvise DISK layer's preceding decode routes before the "
            "next decode step (default: on). No-op with the UFFD pager."
        ),
    )

    parser.add_argument(
        "--session-expert-prefetch",
        choices=["on", "off"],
        default=ServerArgs.session_expert_prefetch,
        help=(
            "Admission-time restore of a session's routed-expert profile into the GPU "
            "slot cache and disk page cache (default: on)."
        ),
    )

    parser.add_argument(
        "--session-protect-experts",
        type=int,
        default=ServerArgs.session_protect_experts,
        help=(
            "Maximum profiled experts receiving an LRU protection boost per live "
            "session (default: 64; 0 disables protection)."
        ),
    )

    parser.add_argument(
        "--moe-step-timing",
        action="store_true",
        default=ServerArgs.moe_step_timing,
        help=(
            "Measure CPU-head, GPU-middle, CPU-tail, and within-layer CPU/GPU "
            "overlap time for decode steps, and append interval averages to the "
            "decode status line. This diagnostic synchronizes each timed step."
        ),
    )

    parser.add_argument(
        "--host-cache-reserve-gib",
        type=_nonnegative_float,
        default=ServerArgs.host_cache_reserve_gib,
        help=(
            "Host RAM reserved for the OS and expert-tier file cache. The default is "
            "max(8 GiB, 15%% of MemTotal)."
        ),
    )

    parser.add_argument(
        "--moe-pager-budget-gib",
        type=float,
        default=ServerArgs.moe_pager_budget_gib,
        help=(
            "Maximum resident UFFD expert-row bytes in GiB, and the basis for the "
            "bounded CPU prefill sweep ceiling. By default the host-memory governor "
            "derives it together with the expert pin budget."
        ),
    )

    parser.add_argument(
        "--moe-hybrid-max-fetch",
        type=int,
        default=ServerArgs.moe_hybrid_max_fetch,
        help=(
            "For --moe-backend hybrid: max experts fetched over PCIe per (layer, decode "
            "step); the rest of that step's misses are computed on the CPU, overlapped. "
            "-1 (default) = auto: fetch the benched pcie/cpu bandwidth fraction of each "
            "step's misses (perfect overlap; needs an `ft bench bw` profile, else 1). "
            "0 = never fetch (all misses on CPU); large = behaves like plain offload."
        ),
    )

    prefill_residency_group = parser.add_mutually_exclusive_group()
    prefill_residency_group.add_argument(
        "--moe-gpu-prefill-layers",
        type=_gpu_prefill_layers,
        metavar="{auto,N,off}",
        default=ServerArgs.moe_gpu_prefill_layers,
        help=(
            "Pinned MoE layers reserved first for GPU prefill overlap during automatic "
            "host-residency planning: 'auto' fits as many as the pin budget permits, "
            "a positive integer forces exactly that many, and 'off' sends every "
            "automatically planned layer through the CPU prefill path."
        ),
    )

    prefill_residency_group.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        default=ServerArgs.moe_prefill_overlap,
        help=(
            "Disable two-buffer overlap for prefill MoE expert copies. "
            "By default, prefill overlap is enabled and requires "
            "--moe-cache-size >= 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--enable-special-token-ckpt",
        action="store_true",
        dest="special_token_ckpt",
        default=ServerArgs.special_token_ckpt,
        help=(
            "Checkpoint decode state at special tokens (currently the tool-call opener). "
            "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
            "scheduler preserves a reuse point just after it (GDN: a state snapshot "
            "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
            "a client that rewrites the echoed tool call only invalidates the call body, "
            "not the turn."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hit-d2d",
        action="store_true",
        dest="moe_prefill_hit_d2d",
        default=ServerArgs.moe_prefill_hit_d2d,
        help=(
            "During prefill prefetch, copy cache-resident experts device-side into "
            "the double buffer and stream only the misses over PCIe "
            "(cudaMemcpyBatchAsync, CUDA >= 13.0). Effective with "
            "--moe-cache-size > 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--cors-origins",
        type=str,
        default=ServerArgs.cors_origins,
        help=(
            "Comma-separated CORS allow-list for browser/webview clients "
            "(default: local Tauri/Vite dev origins). '' disables, '*' allows any."
        ),
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()
    if kwargs["kv_harness_prefixes"] is None:
        kwargs["kv_harness_prefixes"] = ServerArgs.kv_harness_prefixes
    else:
        kwargs["kv_harness_prefixes"] = tuple(kwargs["kv_harness_prefixes"])
    kwargs["kv_ladder_explicit"] = kwargs["kv_ladder"] is not kv_ladder_unspecified
    if not kwargs["kv_ladder_explicit"]:
        kwargs["kv_ladder"] = ServerArgs.kv_ladder

    # reject a too-long list here with a clear reason, not as a dead rank later
    if len(kwargs["gpu"]) not in (0, kwargs["tensor_parallel_size"]):
        if kwargs["tensor_parallel_size"] == 1 and len(kwargs["gpu"]) > 1:
            parser.error("tensor parallelism is not supported yet: --gpu takes one entry")
        parser.error(
            f"--gpu has {len(kwargs['gpu'])} entries but --tensor-parallel-size is "
            f"{kwargs['tensor_parallel_size']}; give one entry per TP rank"
        )

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs["shell_mode"] = run_shell
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["served_model_name"] is None:
        kwargs["served_model_name"] = (
            os.path.basename(os.path.normpath(kwargs["model_path"])) or kwargs["model_path"]
        )

    if kwargs["tool_call_parser"] == "auto":
        kwargs["tool_call_parser"] = _infer_tool_call_parser(kwargs["model_path"])

    if kwargs["reasoning_parser"] == "auto":
        kwargs["reasoning_parser"] = _infer_reasoning_parser(kwargs["model_path"])
    elif kwargs["reasoning_parser"] == "off":
        kwargs["reasoning_parser"] = None

    # Offload-family backends (offload/cpu/hybrid) need a slot cache; if the user gave no
    # sizing flag at all, default to --moe-cache-auto so a bare `ft serve <FTW MoE>` works
    # out of the box (the scheduler resolves the size from free VRAM). Explicit
    # size/rate/auto is preserved.
    from freetoken.moe import is_offload_moe_backend

    _no_cache_flag = (
        kwargs["moe_cache_size"] == 0
        and not kwargs["moe_cache_auto"]
        and (kwargs["moe_cache_rate"] is None or kwargs["moe_cache_rate"] == 0)
    )
    if is_offload_moe_backend(kwargs["moe_backend"]) and _no_cache_flag:
        kwargs["moe_cache_auto"] = True

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # "auto" (or an unspecified dtype) resolves to the checkpoint's dtype. Multimodal /
    # hybrid configs (e.g. Qwen3.5-MoE) keep it under ``text_config`` and use the newer
    # ``dtype`` key rather than top-level ``torch_dtype``, so check both; default bf16.
    if (dtype_str := kwargs["dtype"]) in ("auto", None):
        from freetoken.utils import cached_load_hf_config

        cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()
        text_cfg = cfg.get("text_config") or {}
        dtype_str = (
            cfg.get("torch_dtype") or cfg.get("dtype")
            or text_cfg.get("torch_dtype") or text_cfg.get("dtype") or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
