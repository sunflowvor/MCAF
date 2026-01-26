import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", default="out/")
parser.add_argument("--out", default="exports/")
args = parser.parse_args()

print("[export] start")
print(f"[export] in={args.inp}")
print(f"[export] out={args.out}")

for i in range(6):
    print(f"[export] exporting... {i}/6")
    time.sleep(0.25)

print("[export] done")

