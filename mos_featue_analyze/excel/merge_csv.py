"""Merge all CSV files under excel/ into two files: merged_dm.csv and merged_not_dm.csv."""
import os
import csv

excel_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = {'merged_dm.csv', 'merged_not_dm.csv'}

dm_rows, not_dm_rows = [], []

for root, dirs, files in os.walk(excel_dir):
    for fname in files:
        if not fname.endswith('.csv') or fname in OUTPUTS:
            continue
        fpath = os.path.join(root, fname)
        relpath = os.path.relpath(fpath, excel_dir)

        if fname == 'dm.csv':
            target = 'dm'
        elif fname == 'not_dm.csv':
            target = 'not_dm'
        else:
            continue

        # utf-8-sig strips BOM so all headers normalize to 'name' not '﻿name'
        with open(fpath, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            rows = list(reader)
            if not rows:
                continue

        print(f"  {target}: {relpath}  ({len(rows)} rows, {len(reader.fieldnames)} cols)")
        if target == 'dm':
            dm_rows.extend(rows)
        else:
            not_dm_rows.extend(rows)


def write_csv(outname, rows):
    if not rows:
        print(f"\nNo rows for {outname}, skipping.")
        return
    all_cols = list(dict.fromkeys(k for r in rows for k in r))
    outpath = os.path.join(excel_dir, outname)
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {outpath}  ({len(rows)} rows, {len(all_cols)} cols)")

write_csv('dm.csv', dm_rows)
write_csv('not_dm.csv', not_dm_rows)
