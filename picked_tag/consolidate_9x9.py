"""Consolidate all subdirectory 9x9_dm.csv / 9x9_not_dm.csv into single files."""
import glob
import pandas as pd

dm_files = glob.glob("*/9x9_dm.csv")
not_dm_files = glob.glob("*/9x9_not_dm.csv")

print(f"Found {len(dm_files)} dm files, {len(not_dm_files)} not_dm files")

for out_name, files in [("9x9_dm.csv", dm_files), ("9x9_not_dm.csv", not_dm_files)]:
    dfs = []
    total_rows = 0
    for f in sorted(files):
        df = pd.read_csv(f)
        dfs.append(df)
        total_rows += len(df)
        print(f"  {f}: {len(df)} rows")
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_name, index=False)
    print(f"Wrote {out_name}: {len(combined)} rows (from {len(files)} files)")

print("Done.")
