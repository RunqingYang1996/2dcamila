import os
import hashlib

def get_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- 参数配置 ---
INPUT_DIR = "/Users/runqingyang/Desktop/2026camila/checkpoints/wav_splits"
RECOVERED_PT = "/Users/runqingyang/Desktop/2026camila/checkpoints/epoch_0150_recovered.pt"

# 1. 读取元数据
meta = {}
with open(os.path.join(INPUT_DIR, "manifest.txt"), "r") as f:
    for line in f:
        k, v = line.strip().split(':')
        meta[k] = v

num_chunks = int(meta['count'])
expected_hash = meta['hash']

print(f"[*] 开始从 {num_chunks} 个 WAV 文件中提取数据...")

# 2. 剥离头部并合并
with open(RECOVERED_PT, 'wb') as f_out:
    for i in range(num_chunks):
        wav_path = os.path.join(INPUT_DIR, f"part_{i+1:02d}.wav")
        if os.path.exists(wav_path):
            with open(wav_path, 'rb') as f_in:
                # 跳过前 44 字节的 WAV Header
                f_in.seek(44)
                f_out.write(f_in.read())
            print(f"    - 已处理: {wav_path}")

# 3. 校验一致性
actual_hash = get_file_sha256(RECOVERED_PT)
if actual_hash == expected_hash:
    print(f"\n[SUCCESS] 校验成功！还原后的文件与原始 .pt 100% 一致。")
else:
    print(f"\n[ERROR] 校验失败！文件可能已损坏。")
    print(f"预期 Hash: {expected_hash}")
    print(f"实际 Hash: {actual_hash}")
