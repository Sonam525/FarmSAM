#!/usr/bin/env python
"""Generate results/*.csv from the project's results ledger, so table values are copied, not retyped.

Usage: python tools/build_results.py LEDGER.md results/

The ledger is an internal file and is not in this repository. Every value below is copied as text. No value is
computed or re-rounded. Only labels change: record keys become the model names used in the paper.
"""
import csv
import os
import re
import sys

NAMES = [("S-PT-dp0", "PigTrack student without drop-path"), ("S-PT-mined", "PigTrack student (mined)"),
         ("S-PT-rand2", "PigTrack student (random, draw 2)"), ("S-PT-rand", "PigTrack student (random, draw 1)"),
         ("S-Edi-mined", "Edinburgh student (mined)"), ("S-PT", "PigTrack student"), ("S-Edi", "Edinburgh student"),
         ("S-v1", "Calf student (evenly spaced)"), ("S-v2", "Calf student (night-mined)"),
         ("T21", "SAM 2.1-tiny"), ("S21", "SAM 2.1-small"), ("ES-L", "EfficientSAM3 encoder"), ("ET", "EdgeTAM")]


def names(s):
    for k, v in NAMES:
        s = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", v, s)
    s = re.sub(r"^T$", "SAM 3", s)
    s = re.sub(r"(?<=[−-] )T(?=$| )", "SAM 3", s)
    s = re.sub(r"(?<![\w.-])T(?= [−-] )", "SAM 3", s)
    return s


def clean(c):
    c = c.strip().replace("**", "").replace("`", "").replace("\\|", "|")
    c = re.sub(r"\s*\(?\[(?:J\d+\w*|TT|CMI|PTM\w*|DEV2|EDIM|FID\S*|G2)[^\]]*\]\)?", "", c)  # source keys
    c = names(c).replace("−", "-")
    c = c.replace("+ T,", "+ SAM 3,").replace(" (§11)", "").replace(" (Section 1)", "")
    return c.strip()


class Ledger:
    def __init__(self, path):
        self.lines = open(path, encoding="utf-8").read().split("\n")

    def table(self, marker, nth=0, start=None):
        """The nth markdown table after the first line that contains marker."""
        i = start or 0
        while marker not in self.lines[i]:
            i += 1
        n = -1
        while True:
            while not self.lines[i].startswith("|"):
                i += 1
            j = i
            while j < len(self.lines) and self.lines[j].startswith("|"):
                j += 1
            n += 1
            if n == nth:
                rows = [[c for c in re.split(r"(?<!\\)\|", ln)[1:-1]] for ln in self.lines[i:j]]
                return [clean(c) for c in rows[0]], [[clean(c) for c in r] for r in rows[2:]]
            i = j


def write(out, name, head, rows):
    with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def split_cols(head, rows, specs):
    """specs: {column index: [new names]}; cells split on ' / '."""
    h2, r2 = [], []
    for i, c in enumerate(head):
        h2 += specs.get(i, [c])
    for r in rows:
        x = []
        for i, c in enumerate(r):
            if i in specs:
                p = c.split(" / ")
                if len(p) != len(specs[i]):
                    p = (p + [""] * len(specs[i]))[:len(specs[i])]
                x += p
            else:
                x.append(c)
        r2.append(x)
    return h2, r2


def ci(cell):
    m = re.match(r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]$", cell.strip())
    return list(m.groups()) if m else [cell, "", ""]


def main(ledger, out):
    L = Ledger(ledger)
    os.makedirs(out, exist_ok=True)
    # ---------------------------------------------------------------- models
    h, r = L.table("## 1. Arms")
    rows = []
    for x in r:
        p = x[2].split(" / ")
        g = x[3].split(" / ")
        rows.append([x[0], p[0], p[1],
                     g[0] if len(g) == 2 else "same graph as the PigTrack student",
                     g[1] if len(g) == 2 else "same graph as the PigTrack student"])
    write(out, "models.csv", ["model", "params_backbone", "params_total", "gflops_total_4_objects",
                              "gflops_backbone_4_objects"], rows)
    # ---------------------------------------------------------------- Edinburgh
    h, r = L.table("**Pooled, interpolated GT**")
    write(out, "edinburgh_protA_pooled.csv", ["model"] + h[1:], r)
    h, r = L.table("**Per clip, HOTA / MOTA / IDF1**")
    specs = {i: [f"{c} {m}" for m in ("HOTA", "MOTA", "IDF1")] for i, c in enumerate(h) if i}
    write(out, "edinburgh_protA_per_clip.csv", *split_cols(["model"] + h[1:], r, specs))
    h, r = L.table("**Per-clip Δ vs T, HOTA / MOTA / IDF1**")
    specs = {i: [f"{c} ΔHOTA", f"{c} ΔMOTA", f"{c} ΔIDF1"] for i, c in enumerate(h) if i}
    h2, r2 = split_cols(["model minus SAM 3"] + h[1:], r, specs)
    write(out, "edinburgh_protA_delta_vs_sam3.csv", h2, r2)
    h, r = L.table("**Supplement, 9-clip table")
    write(out, "edinburgh_protA_nine_clips_no_student.csv", ["model"] + h[1:], r)
    h, r = L.table("### 2.1 J3 bridge")
    write(out, "edinburgh_old_protocol_bridge.csv", h, r)
    # ---------------------------------------------------------------- fidelity
    h, r = L.table("**Edinburgh, S-Edi**")
    write(out, "fidelity_edinburgh.csv", h, r)
    h, r = L.table("**PigTrack, 374 held-out frames")
    write(out, "fidelity_pigtrack.csv", h, r)
    # ---------------------------------------------------------------- PigTrack
    h, r = L.table("## 4. PigTrack test, Protocol A")
    write(out, "pigtrack_protA.csv", ["model"] + h[1:], r)
    h, r = L.table("**Day / night (`tab:pigtrack_daynight`)**")
    h2, r2 = split_cols(["model"] + h[1:], r, {
        1: ["day HOTA", "day AssA", "day IDF1"], 2: ["night HOTA", "night AssA", "night IDF1"]})
    write(out, "pigtrack_day_night.csv", h2, r2)
    h, r = L.table("**Pre-registered contrasts (`tab:pigtrack`)")
    rows = []
    for x in r:
        v = re.sub(r"\s*\(= .*\)$", "", x[2])
        v = "" if v in ("—", "-") else v
        sec = x[3].split(" / ")
        st = x[4].split(" / ")
        c, kind = x[0], "pre-specified"
        if "(exploratory)" in c:
            c, kind = c.replace(" (exploratory)", ""), "exploratory"
        rows.append([c, kind] + ci(x[1]) + [v] + sec + st)
    write(out, "pigtrack_contrasts.csv", ["contrast", "kind", "delta_hota", "ci95_low", "ci95_high", "verdict",
                                          "delta_deta", "delta_assa", "delta_mota", "delta_idf1", "day", "night"], rows)
    # ---------------------------------------------------------------- Protocol D
    h, r = L.table("**PigTrack test, 25 sequences (`tab:pigtrack_protD`).**")
    h2, r2 = split_cols(["model"] + h[1:], r, {8: ["FP", "FN"], 9: ["HOTA day", "HOTA night"]})
    write(out, "pigtrack_protD.csv", h2, r2)
    h, r = L.table("**Edinburgh test, Protocol D")
    write(out, "edinburgh_protD.csv", ["detector", "model"] + h[2:], [
        [x[0].replace(" (zero-shot sensitivity)", ""), x[1]] + x[2:] for x in r])
    h, r = L.table("**Births** (telemetry")
    write(out, "protD_births.csv", h, r)
    # ---------------------------------------------------------------- calves
    h, r = L.table("**Pooled identity-correct calf-time")
    rows = [[x[0], x[1]] + (ci(x[2]) if x[2] not in ("—", "-") else ["", "", ""]) for x in r]
    write(out, "calf_pooled.csv", ["model", "identity_correct_pct", "delta_vs_sam3_pp", "ci95_low", "ci95_high"], rows)
    h, r = L.table("**By phase, ID-correct % / mask IoU vs T**")
    rows = []
    for x in r:
        for j, m in enumerate(h[1:], 1):
            a, b = x[j].split(" / ")
            rows.append([x[0], names(m) if m != "T" else "SAM 3", a, "" if b in ("—", "-") else b])
    write(out, "calf_by_phase.csv", ["phase", "model", "identity_correct_pct", "mask_iou_vs_sam3"], rows)
    h, r = L.table("**Per window** (`calf_windows.csv`")
    rows = []
    for x in r:
        for j, m in enumerate(h[1:], 1):
            p = x[j].split(" / ")
            rows.append([x[0], names(m) if m != "T" else "SAM 3"] + p[:4] + ["" if p[4] in ("—", "-") else p[4]])
    write(out, "calf_by_window.csv", ["window", "model", "identity_correct_pct", "merge_frame_pct", "fragment_pct",
                                      "coverage_pct", "mask_iou_vs_sam3"], rows)
    # ---------------------------------------------------------------- efficiency
    h, r = L.table("**Per-module latency, ms per frame")
    write(out, "efficiency_latency.csv", ["model", h[1], h[2], h[3], h[4], h[5], "model span", h[7], h[8]], r)
    h, r = L.table("**Memory** (`tab_efficiency.csv`")
    write(out, "efficiency_memory.csv", ["model"] + h[1:], r)
    h, r = L.table("**Concurrent streams per A10")
    write(out, "efficiency_streams.csv", ["model", "level_after_first_pass_mib", "per_stream_mib", "streams"],
          [x[:4] for x in r])
    h, r = L.table("**Pruning factorial (`tab:pruning`)**")
    h2, r2 = split_cols(["model"] + h[1:], r, {2: ["HOTA", "MOTA", "IDF1"], 4: ["peak allocated MiB", "end allocated MiB"]})
    write(out, "pruning_factorial.csv", h2, r2)
    h, r = L.table("**J13, end to end with the detector live")
    write(out, "efficiency_end_to_end.csv", h, r)
    # ---------------------------------------------------------------- mining
    h, r = L.table("**Tracking, PigTrack test, oracle-initialised prompts, 25 sequences")
    disp = ["SAM 3", "SAM 2.1-small (zero-shot)", "PigTrack student (3,000 targets)", "PigTrack student (mined)",
            "PigTrack student (random extra frames, draw 1)", "PigTrack student (random extra frames, draw 2)"]
    rows = [[d] + x[1:] for d, x in zip(disp, r)]
    write(out, "mining_pigtrack_tracking.csv", ["model"] + h[1:], rows)
    h, r = L.table("**The seven registered contrasts")
    write(out, "mining_pigtrack_contrasts.csv", h, [x for x in r])
    h, r = L.table("Secondary statistics of the registered contrasts")
    write(out, "mining_pigtrack_secondary.csv", h, r)
    h, r = L.table("**Held-out distillation fidelity (J16")
    write(out, "mining_pigtrack_fidelity.csv", h, r)
    h, r = L.table("**Training (RECORDED, [PTM] Section 4.3")
    write(out, "mining_pigtrack_training.csv", h, r)
    h, r = L.table("**Diagnostics of the extra sets")
    write(out, "mining_pigtrack_diagnostics.csv", h, r)
    h, r = L.table("### 12.2 Dairy calves")
    write(out, "calf_first_vs_night_mined.csv", ["quantity", "first calf student", "night-mined calf student", "difference"],
          [x[:4] for x in r])
    h, r = L.table("### 12.3 Point-estimate differences")
    for x in r:
        x[4] = x[4].replace("; Section 4)", ")").replace("; Section 4", "")
        x[5] = re.sub(r"exploratory \(the unregistered D4 row.*", "exploratory, not pre-specified", x[5])
        x[5] = x[5].replace(", Section 6", "").replace(" (the SAM 2.1-small calf row was added after the other calf results were known)",
                                                         " (this calf row was added after the other calf results were known)")
    write(out, "point_estimate_differences.csv", h[:-1], [x[:-1] for x in r])
    h, r = L.table("### 12.4 Data for the recipe display")
    for x in r:
        x[6] = re.sub(r"^(0\.25042) Section 3\.3.*$",
                      lambda m: m.group(1) + " (quoted from the development record; not re-read from the history file)",
                      x[6])
    write(out, "recipe.csv", h[:-1], [x[:-1] for x in r])
    h, r = L.table("**(a) Pooled over the four test clips")
    write(out, "mining_edinburgh_pooled.csv", ["metric", "published student", "student with 627 extra frames",
                                               "difference (stored precision)", "difference (rounded)"], r)
    h, r = L.table("**(b) Per clip (RECORDED")
    write(out, "mining_edinburgh_per_clip.csv", ["clip", "published student HOTA", "student with 627 extra frames HOTA",
                                                 "difference (stored precision)", "difference (rounded)"], r)


def check_no_control_bytes(out):
    """Stop if any CSV holds a byte below 0x20 other than tab, newline or carriage return."""
    for name in sorted(os.listdir(out)):
        if name.endswith(".csv"):
            with open(os.path.join(out, name), "rb") as f:
                bad = sorted(set(b for b in f.read() if b < 0x20 and b not in (9, 10, 13)))
            if bad:
                raise SystemExit("control bytes %s in %s" % (bad, name))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    check_no_control_bytes(sys.argv[2])
