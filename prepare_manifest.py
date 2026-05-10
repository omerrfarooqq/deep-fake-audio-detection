import os
import csv
import argparse

def build_manifest(root, out_csv):
    rows = []
    for label_name, label in [("real", 0), ("fake", 1)]:
        folder = os.path.join(root, "training", label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                rows.append((os.path.join(folder, fname), label))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} entries to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Root path containing training/fake and training/real")
    p.add_argument("out", help="Output CSV manifest path")
    args = p.parse_args()
    build_manifest(args.root, args.out)
