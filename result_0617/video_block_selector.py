import cv2
import numpy as np
import os
import json
import re
from glob import glob
from datetime import datetime


def _natsort_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


# -- 全局状态 --
drawing = False
diff_gain = 10
frame_in = None
frame_out = None
disp = None
video_name = ''
frame_idx = 0
roi_list = []          # 多个框 [(rx,ry,rw,rh), ...]
current_roi = None     # 正在画的框 (x1,y1,x2,y2)
collected_frames = []  # 跨帧对比
video_roi_map = {}     # 视频名 -> {rois: [...], frames: [...]}


def list_videos(data_dir='out_data'):
    files = sorted(glob(os.path.join(data_dir, '*_in.mp4')), key=_natsort_key)
    pairs = []
    for f in files:
        base = f.replace('_in.mp4', '')
        out_f = base + '_out.mp4'
        name = os.path.basename(base)
        pairs.append((name, f, out_f if os.path.exists(out_f) else None))
    return pairs


def mouse_cb(event, x, y, flags, param):
    global drawing, current_roi, disp, frame_in
    h, w = frame_in.shape[:2]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_roi = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        current_roi = (current_roi[0], current_roi[1], x2, y2)
        _refresh_display()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        x1, y1 = current_roi[0], current_roi[1]
        rx, ry = min(x1, x2), min(y1, y2)
        rw, rh = abs(x2 - x1), abs(y2 - y1)
        if rw < 8 or rh < 8:
            current_roi = None
            _refresh_display()
            return
        roi_list.append((int(rx), int(ry), int(rw), int(rh)))
        current_roi = None
        idx = len(roi_list)
        print(f"  [OK] Box #{idx}: ({rx},{ry}) {rw}x{rh}  继续画下一个框，或按 s 保存")
        _refresh_display()
        _show_block_detail(idx - 1)


def _refresh_display():
    global disp
    if frame_in is None:
        return
    disp = frame_in.copy()

    # 画所有已确认的框
    for i, (rx, ry, rw, rh) in enumerate(roi_list):
        color = (0, 255, 0) if i == len(roi_list) - 1 else (0, 200, 255)
        cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(disp, f"#{i+1}", (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 画正在拖的框
    if current_roi:
        x1, y1, x2, y2 = current_roi
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(disp, f"Frame {frame_idx} | Boxes: {len(roi_list)} | drag=new  z:undo  r:clear  s:save  q:quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow('video_block_selector', disp)


def _show_block_detail(box_idx):
    global frame_in, frame_out
    rx, ry, rw, rh = roi_list[box_idx]
    block_in = frame_in[ry:ry + rh, rx:rx + rw]
    block_out = frame_out[ry:ry + rh, rx:rx + rw] if frame_out is not None else None

    max_display = 400
    scale = min(max_display / rw, max_display / rh, 8)
    show_w, show_h = int(rw * scale), int(rh * scale)
    big_in = cv2.resize(block_in, (show_w, show_h), interpolation=cv2.INTER_NEAREST)

    if block_out is not None:
        big_out = cv2.resize(block_out, (show_w, show_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Box Detail - In | Out', np.hstack([big_in, big_out]))
    else:
        cv2.imshow('Box Detail - Input', big_in)


def save_result():
    global frame_in, frame_out, video_name, frame_idx, roi_list, collected_frames, video_roi_map

    if not roi_list:
        print("  no boxes, draw at least one")
        return

    base_name = video_name
    out_dir = os.path.join('selected_blocks', base_name)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"f{frame_idx}_{ts}"

    rois_data = []
    for bi, (rx, ry, rw, rh) in enumerate(roi_list):
        block_in = frame_in[ry:ry + rh, rx:rx + rw]
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_box{bi}_in.png"), block_in)

        box_data = {'box': bi, 'x': int(rx), 'y': int(ry), 'w': int(rw), 'h': int(rh),
                    'files': {'input': f"{prefix}_box{bi}_in.png"}}

        block_out = None
        if frame_out is not None:
            block_out = frame_out[ry:ry + rh, rx:rx + rw]
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_box{bi}_out.png"), block_out)
            diff = cv2.absdiff(block_in, block_out)
            diff_amp = np.clip(diff.astype(np.float32) * diff_gain, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_box{bi}_diff_x{diff_gain}.png"), diff_amp)
            box_data['files']['output'] = f"{prefix}_box{bi}_out.png"
            box_data['files']['diff'] = f"{prefix}_box{bi}_diff_x{diff_gain}.png"

        rois_data.append(box_data)
        print(f"  Box #{bi}: ({rx},{ry}) {rw}x{rh}")

    # 所有框标注在同一张全图上
    full = frame_in.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for bi, (rx, ry, rw, rh) in enumerate(roi_list):
        c = colors[bi % len(colors)]
        cv2.rectangle(full, (rx, ry), (rx + rw, ry + rh), c, 3)
        cv2.putText(full, f"#{bi+1} {rw}x{rh}", (rx, ry - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_boxes_full.png"), full,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    meta = {
        'video': video_name,
        'frame_idx': int(frame_idx),
        'diff_gain': diff_gain,
        'boxes': rois_data,
    }
    meta_path = os.path.join(out_dir, f"{prefix}_roi.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    collected_frames.append((int(frame_idx), roi_list.copy(), frame_in, frame_out))

    # video_roi_map
    vname = base_name
    if vname not in video_roi_map:
        video_roi_map[vname] = {
            'boxes': [{'box': i, 'x': int(rx), 'y': int(ry), 'w': int(rw), 'h': int(rh),
                       'size': f"{int(rw)}x{int(rh)}"}
                      for i, (rx, ry, rw, rh) in enumerate(roi_list)],
            'frames': []
        }
    if int(frame_idx) not in video_roi_map[vname]['frames']:
        video_roi_map[vname]['frames'].append(int(frame_idx))

    print(f"\n  [OK] {len(roi_list)} boxes saved in {out_dir}/")
    print(f"  metadata: {meta_path}")
    print(f"  ============\n")


def _read_frames(in_path, out_path, fidx):
    cap = cv2.VideoCapture(in_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fidx >= total:
        print(f"  [WARN] frame {fidx} out of range (total {total}), skip")
        cap.release()
        return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, fin = cap.read()
    cap.release()
    if not ret:
        return None, None
    fout = None
    if out_path and os.path.exists(out_path):
        cap_out = cv2.VideoCapture(out_path)
        cap_out.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, fout = cap_out.read()
        cap_out.release()
        if not ret:
            fout = None
    return fin, fout


def process_interactive(in_path, out_path, fidx):
    global frame_in, frame_out, disp, roi_list, current_roi, frame_idx, drawing

    roi_list = []
    current_roi = None
    drawing = False
    frame_idx = fidx

    frame_in, frame_out = _read_frames(in_path, out_path, fidx)
    if frame_in is None:
        return None

    h, w = frame_in.shape[:2]
    print(f"\n-- First Frame {fidx} ({w}x{h}) --")
    print("  drag to draw boxes, keep drawing more, press s to finish")

    cv2.namedWindow('video_block_selector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video_block_selector', w // 2, h // 2)
    cv2.setMouseCallback('video_block_selector', mouse_cb)
    _refresh_display()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None
        elif key == ord('s'):
            if roi_list:
                save_result()
                return roi_list.copy()
            else:
                print("  draw at least one box first")
        elif key == ord('z'):
            if roi_list:
                roi_list.pop()
                cv2.destroyWindow('Box Detail - In | Out')
                try:
                    cv2.destroyWindow('Box Detail - Input')
                except:
                    pass
                _refresh_display()
                print(f"  undo Box #{len(roi_list)+1}, {len(roi_list)} box(es) left")
        elif key == ord('r'):
            roi_list = []
            current_roi = None
            cv2.destroyWindow('Box Detail - In | Out')
            try:
                cv2.destroyWindow('Box Detail - Input')
            except:
                pass
            _refresh_display()
            print("  cleared all boxes")


def process_batch(in_path, out_path, fidx, locked_rois):
    global frame_in, frame_out, roi_list, frame_idx

    frame_in, frame_out = _read_frames(in_path, out_path, fidx)
    if frame_in is None:
        return

    roi_list = locked_rois
    frame_idx = fidx

    h, w = frame_in.shape[:2]
    print(f"\n-- Frame {fidx} ({w}x{h}) (reuse {len(roi_list)} boxes) --")
    save_result()


def generate_frame_comparison():
    global collected_frames, video_name

    if len(collected_frames) < 2:
        return

    nb = len(collected_frames[0][1])  # 框数量
    row_h = 200

    for bi in range(nb):
        rows = []
        prev_in = None
        for fidx, rois, fin, fout in collected_frames:
            rx, ry, rw, rh = rois[bi]
            scale = row_h / rh
            row_w = int(rw * scale)

            bin_r = cv2.resize(fin[ry:ry+rh, rx:rx+rw], (row_w, row_h), interpolation=cv2.INTER_CUBIC)
            parts = [bin_r]

            # Output
            if fout is not None:
                bout_r = cv2.resize(fout[ry:ry+rh, rx:rx+rw], (row_w, row_h), interpolation=cv2.INTER_CUBIC)
                parts.append(bout_r)
            else:
                parts.append(np.zeros((row_h, row_w, 3), dtype=np.uint8))

            # dInput
            if prev_in is not None:
                d = cv2.absdiff(fin[ry:ry+rh, rx:rx+rw], prev_in[ry:ry+rh, rx:rx+rw])
                d_a = np.clip(d.astype(np.float32) * diff_gain, 0, 255).astype(np.uint8)
                parts.append(cv2.resize(d_a, (row_w, row_h), interpolation=cv2.INTER_CUBIC))
            else:
                na = np.zeros((row_h, row_w, 3), dtype=np.uint8)
                cv2.putText(na, "1st frame", (6, row_h//2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                parts.append(na)

            row_img = np.hstack(parts)
            cv2.putText(row_img, f"F{fidx}", (8, row_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append(row_img)
            prev_in = fin

        cmp_full = np.vstack(rows)

        header_h = 30
        header = np.full((header_h, cmp_full.shape[1], 3), [30, 30, 40], dtype=np.uint8)
        labels = ["Input", "Output", "dInput"]
        for i, label in enumerate(labels):
            cx = i * row_w + row_w//2 - len(label)*5
            cv2.putText(header, label, (cx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        for i in range(1, len(labels)):
            cv2.line(header, (i*row_w, 0), (i*row_w, header_h-1), (80, 80, 90), 1)

        cmp_full = np.vstack([header, cmp_full])

        out_dir = os.path.join('selected_blocks', video_name)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f"box{bi}_frames_compare_{ts}.png")
        cv2.imwrite(out_path, cmp_full, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        print(f"  -> Box#{bi} compare: {out_path}")


def main():
    global video_name, collected_frames, video_roi_map

    data_dir = os.path.join(os.path.dirname(__file__), 'out_data')
    pairs = list_videos(data_dir)
    if not pairs:
        print("out_data: no videos found")
        return

    print("\nAvailable videos:")
    for i, (name, _, _) in enumerate(pairs):
        print(f"  [{i}] {name}")
    print(f"  [{len(pairs)}] manual path")

    sel = input(f"\nSelect video (0-{len(pairs)}): ").strip()
    if sel.isdigit() and int(sel) < len(pairs):
        video_name, in_path, out_path = pairs[int(sel)]
    else:
        in_path = input("video path: ").strip()
        out_path = input("output video path (optional): ").strip() or None
        video_name = os.path.basename(in_path)

    cap_tmp = cv2.VideoCapture(in_path)
    fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()

    default_str = "30-32"
    frames_input = input(f"\nFrame indices (default {default_str}, e.g. 30-32 or 45,50,60, total {total_frames}): ").strip()

    if frames_input:
        if '-' in frames_input and ',' not in frames_input:
            parts = frames_input.split('-')
            frame_indices = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            frame_indices = [int(f.strip()) for f in frames_input.split(',') if f.strip()]
    else:
        frame_indices = list(range(30, 33))

    print(f"  FPS: {fps:.2f}, total frames: {total_frames}")
    print(f"  processing: {frame_indices}")
    print(f"  first frame: draw boxes, rest: auto-reuse\n")

    collected_frames = []
    cv2.destroyAllWindows()

    locked_rois = process_interactive(in_path, out_path, frame_indices[0])
    if locked_rois is None:
        print("first frame not saved, exit")
        cv2.destroyAllWindows()
        return

    for fidx in frame_indices[1:]:
        process_batch(in_path, out_path, fidx, locked_rois)

    cv2.destroyAllWindows()
    generate_frame_comparison()

    if video_roi_map:
        roi_config_path = os.path.join('selected_blocks', 'roi_config.json')
        os.makedirs('selected_blocks', exist_ok=True)
        existing = {}
        if os.path.exists(roi_config_path):
            try:
                with open(roi_config_path, 'r') as f:
                    existing = json.load(f)
            except:
                pass
        # 合并：保留已有帧号，不覆盖
        for vname, vdata in video_roi_map.items():
            if vname in existing:
                for f in vdata.get('frames', []):
                    if f not in existing[vname].get('frames', []):
                        existing[vname].setdefault('frames', []).append(f)
                existing[vname]['frames'].sort()
            else:
                existing[vname] = vdata
        with open(roi_config_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\n  -> roi_config updated: {roi_config_path}")

    print("all done\n")


if __name__ == '__main__':
    main()
