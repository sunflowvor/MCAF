import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/")
parser.add_argument("--out", default="out/")
args = parser.parse_args()

print("[preprocess] start")
print(f"[preprocess] input={args.input}")
print(f"[preprocess] out={args.out}")

for i in range(10):
    print(f"[preprocess] working... {i}/10")
    time.sleep(0.2)

print("[preprocess] done")

