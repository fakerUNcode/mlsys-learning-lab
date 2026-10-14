"""B04/T12：读取固定 TinyLlama 配置并打印头数/投影形状。

只读本章 sources 目录中的 JSON（JavaScript Object Notation，数据交换格式）；
不联网、不加载权重、不创建真实模型。维度推算供课程对照，非运行性能证据。
"""

import json
from pathlib import Path


def describe_config() -> dict:
    """返回实际配置与推算形状的字典；文件对象由 with 在读取后关闭。

    无参数；路径相对脚本位置，因此不依赖调用者 cwd（当前工作目录）。
    函数不修改源文件；并发调用只能读取同一份固定文件。
    """
    # __file__ 是当前脚本路径；Path.parent 取所属目录，/ 运算符拼接路径。
    folder = Path(__file__).resolve().parent / "sources"
    with (folder / "tinyllama-config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    with (folder / "model-source.json").open(encoding="utf-8") as file:
        source = json.load(file)
    hidden = config["hidden_size"]
    hq, hkv = config["num_attention_heads"], config["num_key_value_heads"]
    # 该 checkpoint 未单列 head_dim；按 hidden_size / 查询头数推算。
    d = config.get("head_dim") or hidden // hq
    return {
        "model_id": source["model_id"], "revision": source["revision"],
        "hidden_size": hidden, "query_heads": hq, "kv_heads": hkv,
        "head_dim": d, "queries_per_kv_head": hq // hkv,
        "q_proj_weight_shape": [hq * d, hidden],
        "k_v_proj_weight_shape_each": [hkv * d, hidden],
        "rope_theta": config["rope_theta"], "rope_scaling": config["rope_scaling"],
        "max_position_embeddings": config["max_position_embeddings"],
        "declared_weight_dtype": config["torch_dtype"],
    }


if __name__ == "__main__":
    print(json.dumps(describe_config(), indent=2))
