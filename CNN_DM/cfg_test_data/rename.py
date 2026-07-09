import os, re, sys, cv2

base = os.path.dirname(__file__)
targets = sys.argv[1:] if len(sys.argv) > 1 else [
    os.path.join(base, "MNR"),
    os.path.join(base, "MNR_MIS"),
]

pattern = re.compile(r"frame(\d{4})_(in|out|cnn|pred8x8)\.(bmp|png)")
name_map = {"in": "mnr_input", "out": "mnr_output", "cnn": "mnr_cnn", "pred8x8": "mnr_predict"}

total = 0
for t in targets:
    if not os.path.exists(t):
        print(f"[跳过] {t} 不存在")
        continue
    for root, dirs, files in os.walk(t):
        if os.path.basename(root) != "predictions":
            continue
        os.chdir(root)
        renamed = 0
        for fname in files:
            m = pattern.fullmatch(fname)
            if not m:
                continue
            num, tag, suffix = m.groups()
            new_name = f"{name_map[tag]}_{num}.bmp"
            if suffix == "png":
                img = cv2.imread(fname)
                if img is not None:
                    cv2.imwrite(new_name, img)
                    os.remove(fname)
                    print(f"  [转] {fname} → {new_name}")
                else:
                    print(f"  [失败] 无法读取 {fname}")
            else:
                os.rename(fname, new_name)
                print(f"  [改] {fname} → {new_name}")
            renamed += 1
        if renamed:
            print(f"  [{root}] {renamed} 个文件\n")
            total += renamed

print(f"全部完成: 共 {total} 个文件")
