import json
import time
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as Kaldi
from tqdm import tqdm


def measure_speed(model, feature_dim, sr, num_iters=1000):
    # 生成随机 10 秒音频信号
    audio = torch.randn(1, sr * 10)  # [1, 10s * sr]
    pbar = tqdm(range(num_iters), desc="Speed Test", leave=False)
    start_time = time.time()
    for i in pbar:
        # 特征计算
        feature = Kaldi.fbank(audio, num_mel_bins=feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        # 模型推理
        with torch.no_grad():
            _ = model(feature.unsqueeze(0))
        # 计算当前 24 小时可处理条数
        elapsed = time.time() - start_time
        count_24h = (i + 1) / elapsed * 86400
        pbar.set_postfix({"24h_count": int(count_24h)})
    # 最终统计
    total_elapsed = time.time() - start_time
    total_count_24h = num_iters / total_elapsed * 86400
    print(f"24h can process: {int(total_count_24h)} utterances")


def main():
    # 加载配置和模型
    config_path = Path(
        "model-bin/sid/campplus/configuration.json"
    )
    config = json.load(config_path.open())
    feature_dim = config["model"]["model_config"]["fbank_dim"]
    sr = config["model"]["model_config"]["sample_rate"]
    model = torch.jit.load(
        "model-bin/sid/campplus/campplus-200k-opt.jit"
    )
    model.eval()

    # 运行速度测试
    measure_speed(model, feature_dim, sr)


if __name__ == "__main__":
    main()
