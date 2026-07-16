"""
从 cfg.txt 读取配置，批量 CNN 推理 + 双边滤波去噪。

用法:
  python run_cfg_test.py                    # 默认 05.02.25/
  python run_cfg_test.py /path/to/testdir   # 自定义测试目录

处理后生成:
  out/frame*.bmp              ← _out.bmp 重命名（双边滤波后的最终结果）
  predictions/out*_cnn.png    ← DN 检测 overlay
  predictions/out*_out.bmp    ← 双边滤波混合结果
  predictions/out*_in.bmp     ← 输入副本
  predictions/out*_pred8x8.bmp ← 预测概率图
"""

import sys, os, time, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
import predict_cnn


def parse_cfg(path):
    """解析 cfg.txt，返回配置字典"""
    cfg = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.split('#')[0].strip()
    for k in ['INPUT_WIDTH', 'INPUT_HEIGHT', 'ROI_WIDTH', 'ROI_HEIGHT',
              'START_NO', 'END_NO', 'OFRAM_NUM', 'INPUT_PRENUM']:
        if k in cfg:
            cfg[k] = int(cfg[k])
    return cfg


def main():
    base = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else \
        os.path.join(os.path.dirname(__file__), "05.02.25")
    cfg_path = os.path.join(base, "cfg.txt")
    cfg = parse_cfg(cfg_path)
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    # 目录
    input_dir = os.path.join(base, cfg['INPUT_DIR']) if 'INPUT_DIR' in cfg else base
    out_dir = os.path.join(base, cfg.get('OUTPUT_PATH', './out/'))
    pred_dir = os.path.join(base, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # 帧参数
    in_pre = cfg.get('INPUT_PREFIX', 'frame')
    out_pre = cfg.get('OUTPUT_PREFIX', 'out')
    prenum = cfg.get('INPUT_PRENUM', 4)
    start = cfg.get('START_NO', 0)
    end = min(cfg.get('END_NO', 299) + 1, start + cfg.get('OFRAM_NUM', 300))
    # 加载模型
    predict_cnn.OUTPUT_DIR = pred_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = predict_cnn.MosquitoDenoiseCNN().to(device)
    model.load_state_dict(torch.load(predict_cnn.MODEL_PATH, map_location=device), strict=False)
    model.eval()

    # 逐帧处理
    t0 = time.time()
    processed = skipped = 0
    for fid in range(start, end):
        name = f"{in_pre}{str(fid).zfill(prenum)}.bmp"
        src = os.path.join(input_dir, name)
        if not os.path.exists(src):
            skipped += 1
            continue
        img = cv2.imread(src)
        if img is None:
            skipped += 1
            continue

        # 推理
        predict_cnn.predict_image(model, device, src, os.path.join(pred_dir, name.replace('.bmp', '_out.bmp')), True)

        # _out.bmp → out/
        out_src = os.path.join(pred_dir, name.replace('.bmp', '_out.bmp'))
        if not os.path.exists(out_src):
            raise FileNotFoundError(f"Missing: {out_src}")
        cv2.imwrite(os.path.join(out_dir, name), cv2.imread(out_src))

        h, w = img.shape[:2]
        print(f"  [{fid}] {name}  {w}x{h}  cfg={cfg.get('INPUT_WIDTH','N/A')}x{cfg.get('INPUT_HEIGHT','N/A')}")
        processed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {time.time()-t0:.0f}s")
    print(f"  out/  {out_dir}")
    print(f"  pred/ {pred_dir}")
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
