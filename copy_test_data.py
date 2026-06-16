import shutil
from pathlib import Path
import os

data_dir = Path(__file__).resolve().parent / "test_data"

jobs = [
    ("hisense_mnr_mis_clarity#out1#mnr_input0002.bmp", "hisense"),
    ("05.02.25#out1#mnr_input0012.bmp",              "dot25"),
    ("001_OnlineNews#out1#mnr_input0007.bmp",          "online_news"),
    ("ring_001#out1#mnr_input0007.bmp",                "ring"),
    ("tcl_4k_img#out1#mnr_input0011.bmp",              "tcl_4k"),
]

copies = 100

for src_name, out_name in jobs:
    src_path = data_dir / src_name
    if not src_path.exists():
        print(f"! 找不到: {src_name}")
        continue

    out_dir = data_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(copies):
        shutil.copy2(src_path, out_dir / f"frame{i:04d}.bmp")

    print(f"OK {src_name} -> {out_dir}/  ({copies} 份)")
