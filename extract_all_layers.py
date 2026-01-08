#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, types, argparse, subprocess, numpy as np, torch
from typing import Tuple, List

# ---- NumPy 2.0 兼容别名（旧 fairseq/依赖里可能引用）----
if not hasattr(np, 'float'):   np.float = float
if not hasattr(np, 'int'):     np.int = int
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'bool'):    np.bool = bool
if not hasattr(np, 'object'):  np.object = object
if not hasattr(np, 'long'):    np.long = int
# ---------------------------------------------------------

def decode_mp3_to_mono16k_float32(path: str, target_sr: int = 16000) -> np.ndarray:
    """用 ffmpeg 解码 mp3 -> 单声道 float32 PCM @ target_sr。"""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(target_sr), "pipe:1"
    ]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 解码失败: {path}\n{e.stderr.decode(errors='ignore')}")
    return np.frombuffer(out.stdout, dtype=np.float32)

def pad_or_trim_to_n(x: np.ndarray, n: int) -> np.ndarray:
    """把 1D 波形长度强制为 n（不足补零，超长截断）。"""
    if x.shape[0] == n: return x
    if x.shape[0] < n:
        y = np.zeros(n, dtype=np.float32); y[:x.shape[0]] = x; return y
    return x[:n].astype(np.float32, copy=False)

def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

def _guess_arch_from_state_dict(sd: dict) -> Tuple[int,int,int]:
    """推断 embed_dim, num_layers, num_heads。"""
    embed_dim = None
    for k in ["encoder.layers.0.self_attn.k_proj.weight",
              "encoder.layers.0.self_attn.q_proj.weight",
              "encoder.layers.0.self_attn.v_proj.weight"]:
        t = sd.get(k, None)
        if isinstance(t, torch.Tensor) and t.dim()==2:
            embed_dim = t.shape[0]; break
    if embed_dim is None:
        for k, t in sd.items():
            if isinstance(t, torch.Tensor) and t.dim()==2 and k.endswith("self_attn.k_proj.weight"):
                embed_dim = t.shape[0]; break
    layer_ids = set()
    for k in sd.keys():
        m = re.match(r"encoder\.layers\.(\d+)\.", k)
        if m: layer_ids.add(int(m.group(1)))
    num_layers = (max(layer_ids)+1) if layer_ids else 12
    if embed_dim is None: embed_dim = 768
    num_heads = max(1, embed_dim // 64)  # base:12, large:16
    return embed_dim, num_layers, num_heads

def load_fairseq_w2v2_from_state_dict(ckpt_path: str, device: torch.device):
    """兼容新旧 fairseq API；只用本地 .pth/.pt（可为纯 state_dict）。"""
    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
    try:
        from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config
        has_cfg = True
    except Exception:
        Wav2Vec2Config = None
        has_cfg = False

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("model", None), dict):
        sd = state["model"]
    elif isinstance(state, dict) and isinstance(state.get("state_dict", None), dict):
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        raise RuntimeError("未知 checkpoint 格式：需要 dict/包含 'model' 或 'state_dict'。")

    for pref in ["w2v_encoder.w2v_model.", "w2v_model."]:
        if any(k.startswith(pref) for k in sd.keys()):
            sd = _strip_prefix(sd, pref); break

    embed_dim, num_layers, num_heads = _guess_arch_from_state_dict(sd)
    ffn_dim = embed_dim * 4
    default_conv = "[(512,10,5),(512,3,2),(512,3,2),(512,3,2),(512,3,2),(512,2,2),(512,2,2)]"

    if has_cfg:
        cfg = Wav2Vec2Config()
        if hasattr(cfg, "extractor_mode"):      cfg.extractor_mode = "layer_norm"
        if hasattr(cfg, "conv_feature_layers"): cfg.conv_feature_layers = default_conv
        if hasattr(cfg, "feature_grad_mult"):   cfg.feature_grad_mult = 0.0
        if hasattr(cfg, "conv_bias"):           cfg.conv_bias = False
        if hasattr(cfg, "encoder_layers"):          cfg.encoder_layers = num_layers
        if hasattr(cfg, "encoder_embed_dim"):       cfg.encoder_embed_dim = embed_dim
        if hasattr(cfg, "encoder_ffn_embed_dim"):   cfg.encoder_ffn_embed_dim = ffn_dim
        if hasattr(cfg, "encoder_attention_heads"): cfg.encoder_attention_heads = num_heads
        for k, v in dict(
            dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
            layerdrop=0.0, conv_pos=128, conv_pos_groups=16,
            mask_time_prob=0.0, mask_channel_prob=0.0
        ).items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        model = Wav2Vec2Model(cfg)
    else:
        args = types.SimpleNamespace(
            extractor_mode="layer_norm", conv_feature_layers=default_conv, conv_bias=False,
            feature_grad_mult=0.0,
            encoder_layers=num_layers, encoder_embed_dim=embed_dim,
            encoder_ffn_embed_dim=ffn_dim, encoder_attention_heads=num_heads,
            dropout_input=0.0, dropout_features=0.0, dropout=0.0,
            attention_dropout=0.0, activation_dropout=0.0, layerdrop=0.0,
            layer_norm_first=False, conv_pos=128, conv_pos_groups=16,
            mask_time_prob=0.0, mask_time_length=10, mask_channel_prob=0.0, mask_channel_length=10,
            final_dim=embed_dim,
        )
        try:
            model = Wav2Vec2Model.build_model(args, task=None)
        except TypeError:
            model = Wav2Vec2Model.build_model(args)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[fairseq] embed_dim={embed_dim}, layers={num_layers}, heads={num_heads}")
    if missing:    print(f"[fairseq] 缺失键 {len(missing)}（多为 ASR/量化，提特征可忽略）")
    if unexpected: print(f"[fairseq] 未使用键 {len(unexpected)}（多为任务头等）")
    return model.to(device).eval()

# ---- 布局统一工具：把 (T,B,C) 或 (B,T,C) 统一成 (B,T,C) ----
def _to_BTC(y: torch.Tensor, batch_size: int) -> torch.Tensor:
    if y.dim() != 3:
        raise RuntimeError(f"Unexpected tensor dim: {y.dim()} (expect 3)")
    # 已是 (B,T,C)
    if y.shape[0] == batch_size:
        return y
    # 可能是 (T,B,C)
    if y.shape[1] == batch_size:
        return y.transpose(0, 1)
    # batch_size=1 容错
    if batch_size == 1:
        if y.shape[0] == 1:
            return y
        if y.shape[1] == 1:
            return y.transpose(0, 1)
    raise RuntimeError(f"Cannot infer layout from shape {tuple(y.shape)} for batch_size={batch_size}")

@torch.no_grad()
def forward_all_layers(model: torch.nn.Module, wav_16k_1xT: torch.Tensor, verbose: bool = True) -> List[torch.Tensor]:
    """
    返回每个 Transformer 层的输出（标准化为 B,T,C），列表长度 = num_layers。
    """
    # 取到 encoder layers
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "layers"):
        root = model
        for attr in ["w2v_encoder", "w2v_model"]:
            if hasattr(root, attr):
                root = getattr(root, attr)
        enc = getattr(root, "encoder", None)
    if enc is None or not hasattr(enc, "layers"):
        raise RuntimeError("无法定位 encoder.layers，无法抓取各层隐藏状态。")

    num_layers = len(enc.layers)
    bs = int(wav_16k_1xT.shape[0])
    captured: List[Tuple[int, torch.Tensor]] = []
    handles = []

    def make_hook(layer_idx: int):
        def _hook(module, inp, out):
            y = out[0] if isinstance(out, (list, tuple)) else out
            y = _to_BTC(y, batch_size=bs)  # 统一成 (B,T,C)
            captured.append((layer_idx, y))
            if verbose:
                print(f"[collect] layer {layer_idx+1}/{num_layers} captured, shape=[B={y.shape[0]}, T={y.shape[1]}, C={y.shape[2]}]", flush=True)
        return _hook

    for i, layer in enumerate(enc.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    # 触发一次前向
    if hasattr(model, "extract_features"):
        _ = model.extract_features(wav_16k_1xT, padding_mask=None, mask=False)
    else:
        _ = model(wav_16k_1xT)

    # 清理 hook
    for h in handles:
        try: h.remove()
        except Exception: pass

    # 按层序号排序，并只保留张量（均为 B,T,C）
    captured.sort(key=lambda kv: kv[0])
    feats_per_layer = [t[1] for t in captured]

    if len(feats_per_layer) != num_layers:
        print(f"[WARN] 期望捕获 {num_layers} 层，但得到 {len(feats_per_layer)} 层。")

    return feats_per_layer  # List[(B,T,C)]

def main():
    ap = argparse.ArgumentParser(description="Batch extract 1s mp3 features of ALL transformer layers -> audiofeatures/*.npy (shape [L,T,C]).")
    ap.add_argument("--root", type=str, default=".", help="包含 1..50 目录的根路径")
    ap.add_argument("--start", type=int, default=1, help="起始编号（含）")
    ap.add_argument("--end", type=int, default=50, help="结束编号（含）")
    ap.add_argument("--ckpt", type=str, required=True, help="本地 fairseq wav2vec2 .pth/.pt（可纯 state_dict）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000, help="目标采样率（建议 16000）")
    ap.add_argument("--overwrite", action="store_true", help="已存在同名 .npy 是否覆盖")
    ap.add_argument("--outdir_name", type=str, default="audiofeatures", help="输出文件夹名（默认 audiofeatures）")
    ap.add_argument("--quiet", action="store_true", help="不逐层打印 collecting 提示")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[Device] {device}")
    print(f"[Model] loading from {args.ckpt}")

    # 载入模型（纯本地）
    model = load_fairseq_w2v2_from_state_dict(args.ckpt, device=device)

    # 用 1 秒静音确定统一特征长度 & 维度 & 层数（这里不打印逐层，避免噪音）
    with torch.no_grad():
        dummy = torch.zeros(1, args.sr, dtype=torch.float32, device=device)
        layers_out = forward_all_layers(model, dummy, verbose=False)  # List of (B,T,C)
        if not layers_out:
            raise RuntimeError("未捕获到任何层输出。")
        T_ref = layers_out[0].shape[1]
        C = layers_out[0].shape[2]
        L = len(layers_out)
        # 保险起见确认每层的 C 相同
        for i, y in enumerate(layers_out):
            if y.dim()!=3:
                raise RuntimeError(f"层{i} 输出维度异常：{tuple(y.shape)}，期望 [B,T,C]")
            if y.shape[2] != C:
                raise RuntimeError(f"层{i} 的通道数 {y.shape[2]} 与层0的 {C} 不一致。")
        print(f"[Ref] 各层对齐到形状: (L={L}, T_ref={T_ref}, C={C})")

    total = skipped = processed = 0

    for i in range(args.start, args.end + 1):
        audiochunks = os.path.join(args.root, str(i), "audiochunks")
        outdir = os.path.join(args.root, str(i), args.outdir_name)
        os.makedirs(outdir, exist_ok=True)
        if not os.path.isdir(audiochunks):
            print(f"[WARN] 跳过 {i}: 不存在 {audiochunks}")
            continue

        mp3s = sorted([f for f in os.listdir(audiochunks) if f.lower().endswith(".mp3")])
        if not mp3s:
            print(f"[INFO] {i}: 无 mp3，跳过")
            continue

        print(f"[{i}] 待处理 {len(mp3s)} 段...")
        for fname in mp3s:
            total += 1
            stem = os.path.splitext(fname)[0]
            src = os.path.join(audiochunks, fname)
            dst = os.path.join(outdir, f"{stem}.npy")
            if (not args.overwrite) and os.path.exists(dst):
                skipped += 1
                continue

            try:
                wav = decode_mp3_to_mono16k_float32(src, target_sr=args.sr)  # np.float32 [N]
            except Exception as e:
                print(f"[WARN] 解码失败，跳过: {src}\n  {e}")
                skipped += 1
                continue

            wav = pad_or_trim_to_n(wav, args.sr)  # 精确 1 秒
            x = torch.from_numpy(wav).unsqueeze(0).to(device)  # [1,T]

            # 取所有层（根据 --quiet 决定是否逐层打印）
            feats_layers = forward_all_layers(model, x, verbose=not args.quiet)  # List of (B,T?,C)
            if not feats_layers:
                print(f"[WARN] 无层输出，跳过: {src}")
                skipped += 1
                continue

            # 对齐每层到统一 T_ref，然后堆叠 [L, T_ref, C]
            arr_layers = []
            for y in feats_layers:
                y = y[0].detach().cpu().float().numpy()  # (T,C) -- 取 batch 的第 0 个
                if y.shape[0] < T_ref:
                    pad = np.zeros((T_ref - y.shape[0], y.shape[1]), dtype=np.float32)
                    y = np.concatenate([y, pad], axis=0)
                elif y.shape[0] > T_ref:
                    y = y[:T_ref, :]
                arr_layers.append(y.astype(np.float32, copy=False))

            arr = np.stack(arr_layers, axis=0)  # (L, T_ref, C)
            np.save(dst, arr)
            processed += 1

        print(f"[{i}] 累计成功 {processed}，跳过 {skipped}")

    print(f"\n[Done] 扫描 {total} 段，成功 {processed}，跳过 {skipped}。输出形状为 [L,T,C]，已写入各自的 {args.outdir_name}/*.npy")

if __name__ == "__main__":
    main()
