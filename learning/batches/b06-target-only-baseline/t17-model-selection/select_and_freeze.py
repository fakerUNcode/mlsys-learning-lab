# T17 模型选型与版本冻结脚本。
#
# 职责与全局位置：本文件是 B06/T17 的唯一实验入口。它做三件事：
#   1. 按 T16 的实测设备预算（RTX 5080 Laptop total 15.920 GiB）筛选候选
#      target 模型：权重字节直接取自 HF 仓库 model.safetensors.index.json 的
#      metadata.total_size（即真实需要驻留 GPU 的权重字节数），KV 用 T16 公式
#      2*L*B*Hkv*S*dh*bkv 计算；
#   2. 对选中的模型冻结不可变 40 位 commit revision（禁止用 main/latest 标签
#      作为实验版本），按该 revision 下载 config/tokenizer/LICENSE 等小文件；
#   3. 离线加载 tokenizer 核验特殊 token 与词表，把全部证据写入
#      frozen_manifest.json（冻结配置）与 reports 下的原始 JSON。
#
# 复现（仓库根目录）：
#   .venv/bin/python learning/batches/b06-target-only-baseline/t17-model-selection/select_and_freeze.py
#
# 已知边界：不下载模型权重（T18 才加载前向）；网络不可达时直接失败退出；
# 设备 free 是共享 GPU 上的时点采样，会波动。

import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = Path(__file__).resolve().parent / "frozen_manifest.json"
RAW_JSON_PATH = REPO_ROOT / "reports/stage-02/2026-10-19-t17/selection_measurements.json"

GIB = 1024**3

# 预算约定全部沿用 T16 的容量表：BF16 权重与 KV（每元素 2 字节）、
# context 4096、验证空间 5 token、激活预留 2 GiB（规划输入，T18/T19 实测替换）、
# runtime+safety 预留设备 total 的 10%。
CONTEXT_TOKENS = 4096
VERIFY_TOKENS = 5
ACTIVATION_RESERVE_BYTES = 2 * GIB
SAFETY_RESERVE_RATIO = 0.10
BYTES_PER_ELEM = 2  # BF16

# 候选与角色：target 候选按"能装下的最大模型"择优；draft 仅用于检查
# 阶段 3 配对（T30 正式验收）是否在预算内有立足空间。
CANDIDATES = [
    {"repo_id": "Qwen/Qwen3-0.6B", "role": "draft", "family": "qwen3"},
    {"repo_id": "Qwen/Qwen3-1.7B", "role": "target", "family": "qwen3", "draft_repo": "Qwen/Qwen3-0.6B"},
    {"repo_id": "Qwen/Qwen3-4B", "role": "target", "family": "qwen3", "draft_repo": "Qwen/Qwen3-0.6B"},
    {"repo_id": "Qwen/Qwen3-8B", "role": "target", "family": "qwen3", "draft_repo": "Qwen/Qwen3-0.6B"},
    {"repo_id": "Qwen/Qwen2.5-1.5B", "role": "target", "family": "qwen2.5", "draft_repo": "Qwen/Qwen2.5-0.5B"},
    {"repo_id": "Qwen/Qwen2.5-0.5B", "role": "draft", "family": "qwen2.5"},
    {"repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "role": "target", "family": "tinyllama", "draft_repo": None},
]

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# 冻结时随 revision 一起落地的小文件；权重 safetensors 不在此列表。
FREEZE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "README.md",
]


def fetch_candidate(api: HfApi, cand: dict) -> dict:
    """拉取一个候选的不可变元数据。

    输入：HF 仓库 id。输出：40 位 revision、许可证（model card 元数据）、
    架构字段（层数/KV 头/head dim/dtype）与真实权重字节。
    所有字段都从 pinned revision 的文件读取，不使用本地记忆的模型参数。
    """
    # files_metadata=True 让 siblings 附带每个文件的真实字节数；单分片小模型
    # 没有 index.json，权重字节统一按 "model*.safetensors 文件字节之和" 取，
    # 该口径含 safetensors 头部，是略偏保守（偏大）的驻留估计。
    info = api.model_info(cand["repo_id"], files_metadata=True)
    sha = info.sha
    if not SHA_RE.match(sha):
        # model_info 返回的默认分支头 commit；不是 40 位十六进制说明
        # 解析异常（如 tag 名），此时不能作为冻结版本。
        raise RuntimeError(f"{cand['repo_id']}: 解析到的 revision 不是 40 位 commit: {sha!r}")

    cfg_path = hf_hub_download(cand["repo_id"], "config.json", revision=sha)
    cfg = json.loads(Path(cfg_path).read_text())

    weight_bytes = sum(
        s.size or 0 for s in info.siblings
        if s.rfilename.endswith(".safetensors") and s.rfilename.startswith("model")
    )
    if weight_bytes == 0:
        raise RuntimeError(f"{cand['repo_id']}: 未找到 safetensors 权重文件")

    # 多分片仓库再读取 index.json 的 metadata.total_size 交叉核对：两个口径
    # 只应相差各分片头部（几 KB 量级），差距过大说明口径理解有误，直接报错。
    index_cross_check = None
    try:
        index_path = hf_hub_download(cand["repo_id"], "model.safetensors.index.json", revision=sha)
        index = json.loads(Path(index_path).read_text())
        index_cross_check = index["metadata"]["total_size"]
        if abs(weight_bytes - index_cross_check) > 64 * 1024 * len(index["weight_map"]):
            raise RuntimeError(
                f"{cand['repo_id']}: 文件求和 {weight_bytes} 与 index total_size "
                f"{index_cross_check} 差距超出头部解释范围"
            )
    except Exception:
        # 单分片仓库没有 index.json，属预期情况。
        pass

    # head_dim：Qwen3 在 config 里显式给出；Qwen2.5/TinyLlama 需要按
    # hidden_size / num_attention_heads 推出（这两家 query 头等宽）。
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]

    return {
        **cand,
        "revision": sha,
        "license": (info.cardData.get("license") if info.cardData else None),
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "num_attention_heads": cfg["num_attention_heads"],
        "hidden_size": cfg["hidden_size"],
        "head_dim": head_dim,
        "dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
        "vocab_size": cfg["vocab_size"],
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "weight_bytes": weight_bytes,
        "index_total_size_cross_check": index_cross_check,
        "param_count_approx": weight_bytes // BYTES_PER_ELEM,
    }


def kv_cache_bytes(m: dict, batch: int, seq: int) -> int:
    """T16 的 KV 公式：每 token 每层每 KV 头各存一条 K 和一条 V。"""
    return (
        2
        * m["num_hidden_layers"]
        * batch
        * m["num_key_value_heads"]
        * seq
        * m["head_dim"]
        * BYTES_PER_ELEM
    )


def main() -> int:
    # ---- 设备与依赖快照（选型依据的一部分，随原始 JSON 存档）----
    free_b, total_b = torch.cuda.mem_get_info()
    device = {
        "name": torch.cuda.get_device_name(0),
        "total_bytes": total_b,
        "free_bytes_at_selection": free_b,
    }
    safety_reserve = int(total_b * SAFETY_RESERVE_RATIO)
    deps = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "huggingface_hub": __import__("huggingface_hub").__version__,
    }

    api = HfApi()
    fetched = [fetch_candidate(api, c) for c in CANDIDATES]
    by_repo = {m["repo_id"]: m for m in fetched}

    # ---- 预算评估：target-only 与阶段 3 配对两个口径 ----
    evaluations = []
    for m in fetched:
        if m["role"] != "target":
            continue
        kv_target_only = kv_cache_bytes(m, batch=1, seq=CONTEXT_TOKENS)
        budget_target_only = m["weight_bytes"] + kv_target_only + ACTIVATION_RESERVE_BYTES + safety_reserve

        draft = by_repo.get(m.get("draft_repo") or "")
        if draft is not None:
            kv_pair = kv_cache_bytes(m, 1, CONTEXT_TOKENS + VERIFY_TOKENS) + kv_cache_bytes(draft, 1, CONTEXT_TOKENS)
            budget_pair = m["weight_bytes"] + draft["weight_bytes"] + kv_pair + ACTIVATION_RESERVE_BYTES + safety_reserve
        else:
            kv_pair, budget_pair = None, None

        evaluations.append({
            "repo_id": m["repo_id"],
            "weight_bytes": m["weight_bytes"],
            "target_only": {
                "kv_bytes": kv_target_only,
                "budget_bytes": budget_target_only,
                "fits_total": budget_target_only <= total_b,
                "fits_measured_free": budget_target_only <= free_b,
            },
            "pair": (
                {
                    "draft_repo": draft["repo_id"],
                    "kv_bytes": kv_pair,
                    "budget_bytes": budget_pair,
                    "fits_total": budget_pair <= total_b,
                    "fits_measured_free": budget_pair <= free_b,
                }
                if draft is not None
                else {"draft_repo": None, "note": "同家族无更小 draft 候选，阶段 3 需跨家族配对"}
            ),
            "license": m["license"],
        })

    # 选择规则：在 target-only 与配对两个口径都装进实测 free 的候选里，
    # 取权重字节最大者——研究主线是推测解码系统行为，target 越大越接近
    # 真实负载；8B 若装不下会作为被排除的证据留在 evaluations 里。
    eligible = [
        e for e in evaluations
        if e["target_only"]["fits_measured_free"]
        and e["pair"].get("fits_measured_free")
    ]
    if not eligible:
        print("没有任何候选同时满足 target-only 与配对预算，终止。", file=sys.stderr)
        return 1
    chosen_eval = max(eligible, key=lambda e: e["weight_bytes"])
    chosen = by_repo[chosen_eval["repo_id"]]

    # ---- 冻结：按 40 位 revision 下载小文件并离线核验 tokenizer ----
    frozen_files = {}
    for fname in FREEZE_FILES:
        try:
            p = hf_hub_download(chosen["repo_id"], fname, revision=chosen["revision"])
        except Exception:
            # 个别候选可能缺 merges.txt/LICENSE 等文件；缺失记录、不中断。
            frozen_files[fname] = None
            continue
        # sha256 作为本地内容指纹：日后核对 cache 是否被改动时直接比对。
        digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        frozen_files[fname] = {"path": p, "bytes": Path(p).stat().st_size, "sha256": digest}

    snapshot_dir = os.path.dirname(frozen_files["config.json"]["path"])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot_dir)

    # Qwen 系词表在 config.vocab_size 之外还有一批注册的控制/保留 token；
    # 逐个查 id，把 T18 采样契约会用到的控制 token 显式列进 manifest。
    control_tokens = {}
    for name in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        ids = tok.convert_tokens_to_ids(name)
        control_tokens[name] = ids

    tokenizer_facts = {
        "vocab_size": tok.vocab_size,
        "len_tokenizer": len(tok),
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "special_tokens_map": tok.special_tokens_map,
        "control_token_ids": control_tokens,
        "has_chat_template": tok.chat_template is not None,
    }

    # generation_config 是 T18 采样契约的一部分：EOS 的权威 id 列表以它为准
    # （Qwen3 同时把 <|im_end|> 和 <|endoftext|> 列为停止符），采样默认值也
    # 记录下来，T18 冻结采样配置时必须显式覆盖而不是隐式继承。
    gen_cfg_path = Path(snapshot_dir) / "generation_config.json"
    generation_facts = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else None

    manifest = {
        "frozen_at": date.today().isoformat(),
        "task": "T17",
        "repo_id": chosen["repo_id"],
        "revision": chosen["revision"],
        "revision_rule": "冻结为 40 位 commit hash；实验与复现禁止用 main/latest 等浮动引用",
        "license": {"card_data": chosen["license"], "license_file_sha_note": "LICENSE 文件已随 revision 下载到本地 HF cache"},
        "config": {
            "num_hidden_layers": chosen["num_hidden_layers"],
            "num_attention_heads": chosen["num_attention_heads"],
            "num_key_value_heads": chosen["num_key_value_heads"],
            "head_dim": chosen["head_dim"],
            "hidden_size": chosen["hidden_size"],
            "vocab_size": chosen["vocab_size"],
            "dtype": chosen["dtype"],
            "tie_word_embeddings": chosen["tie_word_embeddings"],
            "weight_bytes": chosen["weight_bytes"],
            "param_count_approx": chosen["param_count_approx"],
        },
        "tokenizer": tokenizer_facts,
        "generation_config": generation_facts,
        "dependencies": deps,
        "device": device,
        "budget_decision": {
            "convention": "T16: BF16 权重+KV；context 4096；verify 5 token；activation 预留 2 GiB；safety 10% total",
            "chosen_target_only": chosen_eval["target_only"],
            "chosen_pair": chosen_eval["pair"],
        },
        "frozen_files": {k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in frozen_files.items() if v},
        "snapshot_dir": snapshot_dir,
        "weights_download_command": (
            f".venv/bin/python -m huggingface_hub.commands.huggingface_cli download "
            f"{chosen['repo_id']} --revision {chosen['revision']}"
        ),
    }

    raw = {
        "date": date.today().isoformat(),
        "device": device,
        "dependencies": deps,
        "budget_convention": {
            "context_tokens": CONTEXT_TOKENS,
            "verify_tokens": VERIFY_TOKENS,
            "activation_reserve_bytes": ACTIVATION_RESERVE_BYTES,
            "safety_reserve_bytes": safety_reserve,
            "bytes_per_elem": BYTES_PER_ELEM,
        },
        "candidates": {m["repo_id"]: {k: v for k, v in m.items()} for m in fetched},
        "evaluations": evaluations,
        "chosen": chosen_eval["repo_id"],
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2))

    GiB = lambda b: f"{b / GIB:.3f} GiB"
    print(f"设备: {device['name']}  total={GiB(total_b)}  free={GiB(free_b)}")
    for e in evaluations:
        pair = e["pair"]
        pair_note = (
            f"pair={GiB(pair['budget_bytes'])} (fits_free={pair['fits_measured_free']})"
            if pair.get("budget_bytes") is not None else "pair=无同家族 draft"
        )
        print(
            f"{e['repo_id']:<44} license={e['license'] or '?':<14} "
            f"weights={GiB(e['weight_bytes']):<12} target_only={GiB(e['target_only']['budget_bytes'])} "
            f"(fits_free={e['target_only']['fits_measured_free']})  {pair_note}"
        )
    print(f"\n选定 target: {chosen['repo_id']} @ {chosen['revision']}")
    print(f"tokenizer: vocab={tokenizer_facts['vocab_size']} eos={tokenizer_facts['eos_token']!r} "
          f"pad={tokenizer_facts['pad_token']!r} chat_template={tokenizer_facts['has_chat_template']}")
    if generation_facts:
        print(f"generation_config: eos_token_id={generation_facts.get('eos_token_id')} "
              f"do_sample={generation_facts.get('do_sample')} temperature={generation_facts.get('temperature')}")
    print(f"manifest -> {MANIFEST_PATH}")
    print(f"raw json -> {RAW_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
