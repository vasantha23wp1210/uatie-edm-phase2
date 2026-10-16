"""Print rows start..start+n of a blind CSV for judging: python show.py <csv> <start> <n>."""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1]).sort_values(["query_id", "pair_id"])
s, n = int(sys.argv[2]), int(sys.argv[3])
last_q = None
for r in df.iloc[s:s + n].itertuples():
    if r.query_id != last_q:
        print(f"## {r.query_id}: {r.query_text} || NEED: {r.information_need}")
        last_q = r.query_id
    print(f"{r.pair_id[1:]}: {str(r.narrative)[:330]}")
