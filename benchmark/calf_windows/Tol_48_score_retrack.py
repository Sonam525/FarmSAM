# -*- coding: utf-8 -*-
"""Tol · N48 — score a RE-TRACKED root against the user-verified GT of 04-22 / 04-23 (SPEC_v5 §6.3/§6.4 +
addendum 2026-09-24).  Local CPU script: numpy + scipy + stdlib only (no torch, no cv2, no pycocotools).

WHAT IT DOES
  Two arms are scored by THE SAME CODE:  `cand` = the re-tracked root (e.g. Tracking_v5a10) and `ref` = the
  production Tracking_v4v2 root (re-scored here, never taken from an older report).  Every numeric leaf of the
  per-day metrics is reported for both arms plus the delta cand − ref, and the §6.4 acceptance criteria are
  evaluated into a PASS / FAIL table that is written to a metrics JSON.

  The GT is the frozen, user-verified chain:  Tracking_v4v2 masks + segments + events, identity from
    04-22  Tracking_v4v2_reid/20240422/segment_assignments_reviewed.json            (never relabel_week_*.json)
    04-23  Tracking_v4v2_reid/20240423/segment_assignments.json  mapped through week_ident_map[20240423]
  and, for the frames on which v4v2 itself is dark (metric 5f), the teacher roots
    04-22  Tracking_teacher/20240422 + reid_ground_truth_teacher_0422.json v2 (user-verified rules)
    04-23  Tracking_teacher/20240423_w1800-2400, identity TRANSFERRED from the v4v2 GT by mask association.

  Per aligned row (clip, frame_idx) every cand mask is associated to the GT masks by Hungarian on MIN-AREA IoS
  (>= 0.60; IoU breaks ties) and labelled assigned / MULTI / split / UNK / NONE (definitions at ASSOC_*).
  From that:  gaps (no-mask slot-time, the gaps_* port), segments (breaks, linked vs orphan, blips, churn,
  rebuilds), fragments/oversize, merges and extras (MULTI / NONE inside and outside the GT-dark stratum),
  within-segment identity switches (a different GT animal holds >= 10 s inside one cand segment), and — after
  Tol_30 has been run on a LOCAL SCORING COPY of each arm with its sha pinned (identical snapshot for both
  arms) — every identity-trusted GT animal-frame classed correct / wrong-flagged / wrong-silent / '?' /
  uncovered, with every wrong span, switch and fill attributed to the mechanism that opened its segment.

  Two coded gate evaluators reuse the same scan / metrics / must-hold code and write ONE verdict JSON each
  (gate PASS | FAIL | INCOMPLETE | REVIEW, see GATE_SEMANTICS):
    --g2   the A10 pilot windows (critic finding 5): P0 MH1 MH2 MH7 MH8 AUD1 per window + >= 1 dark_link/dark_open
           (zero = REVIEW, a finding for the user), max detect() per frame <= 1, n_inband > 4 => pen_count_refused
           and nothing opened (detections_<clip>.json.gz x events), timings, precision, clip completeness.
    --g5a  04-24 (critic finding 6): v5 masks -> Tracking_v4v2p2/20240424 (min-area IoS >= 0.6) -> the user's docx
           interval GT (DocxGT, = Tol_29/Tol_30 semantics); P0 MH1 MH2 MH8 AUD1 + ROOT_CONFIG identity + GT coverage;
           the 04-23->24 night gap is REPORTED as agreement with the v4v2 chain (the docx is not independent there).
  MH8 reads the telemetry's NESTED clip['v5'] fields and an EXACT GPU class (critic finding 4); the D6 fallback is
  accepted only with --d6_fallback and a 21 GiB cap, and is then labelled. A delayed free that Tol_47 re-measured
  after gc.collect() (mem_check where='gc_recheck') is judged on the re-measured allocated value, as the tracker does.
  MH4z is D8's hard part: wrong animal-time (flagged + silent) attributable to a step-2 mechanism == 0 (MH3 = silent
  only; MH4 = the +60 min review trigger). Geometry is also reported where v4v2 and the teacher agree (SPEC §6.3).
  Review r1 (2026-09-25): a v4v2 GHOST mask (unassigned segment, area < FRAG_LO x A_REF_EVAL) is not an animal (GT-dark
  stratum, MULTI, the 5f fill trigger); MH7, MH5 and the step-2 / switch evidence use the MAIN stratum only (late
  04-23 23:30-24:00 and the identity exclusions are reported); each step-2 verdict uses trusted identity from the first /
  last 10 s AND the nearest >= 10 s run anywhere in the segment; an unadjudicated action makes MH1 REVIEW, never PASS;
  a duplicate open is caught whichever cand slot the Hungarian gave the GT mask to.
  Review r2 (2026-09-25): SPEC §6.3 'a geometry error counts only where [v4v2 and the teacher] agree' + the ±30 s GT-defect
  geometry exclusions now govern every geometric verdict, not only mask_fidelity. The teacher's view (TAG_*) is taken
  on EVERY row with a MULTI / NONE cand mask or a GT mask two cand masks share. MH7 gates the COUNTED MULTI / NONE time
  (the teacher agrees, or v4v2 is the only reference); time where the teacher contradicts v4v2 or a GT-defect exclusion
  applies is reported with its episodes. The step-2 duplicate / split / non-calf rules decide only on EVALUABLE rows
  (main stratum, GT holding four real animals, outside the exclusions, the teacher not contradicting), else unknown; a
  duplicate on a GT-dark calf is caught by trusted identity (teacher fills); a GT mask over two DISJOINT cand masks
  (a GT merge of two calves) names no identity for either.

USAGE
  python Tol_48_score_retrack.py --selftest                      synthetic cases only (no data)
  python Tol_48_score_retrack.py --selftest --data <v4v2 root>   + reproduce the known v4v2 figures (G0)
  G3 (full 04-22 + 04-23; SPEC §6.4 must-holds MH1-MH8 + MH4z (D8 hard zero) + AUD1 + AUD2):
  python Tol_48_score_retrack.py --cand <root> --ref <v4v2 root> --gt_dir <dir> --days 20240422,20240423 \
         [--cand_day_dir 20240423=20240423_w1152-1259] [--teacher 20240422=<dir> --teacher 20240423=<dir>] \
         [--teacher_rules_0422 <json>] [--run_tol30 --tol30 <Tol_30 path> [--tol30_sha <hex>]] \
         [--week_map_ref <week_ident_map.json>] [--week_map_cand <json> | --nightgap_cand <Tol_16_v2 json>] \
         [--telemetry cand:20240422=<json> ...] [--d6_fallback] [--work <dir>] [--out <metrics.json>] [--jobs 4]
  G2 (A10 pilot windows; the automated subset P0 MH1 MH2 MH7 MH8 AUD1 + the G2-only checks, one verdict JSON):
  python Tol_48_score_retrack.py --g2 --cand <pilot root> --ref <v4v2 root> --gt_dir <dir> \
         --window 20240423=20240423_w1152-1259 --window 20240423=20240423_w2018-2200 \
         [--teacher 20240423=<Tracking_teacher/20240423_w1800-2400>] [--telemetry cand:<window dir>=<json>] \
         [--no_ref] [--out Metrics_v5_G2_verdict.json]
  G5a (04-24, held out for the tracker; v5 masks -> Tracking_v4v2p2/20240424 -> the user's docx interval GT):
  python Tol_48_score_retrack.py --g5a --cand <root> --gt_root <Tracking_v4v2p2 root> \
         --gt24 gt_0424_from_user_notes.json --asg24 <Tracking_v4v2p2_reid/20240424/segment_assignments.json> \
         [--g3_metrics <G3 metrics JSON> --nightgap_cand <Tol_16_v2 04-23->24 JSON>] [--telemetry cand:20240424=<json>] \
         [--no_ref] [--d6_fallback] [--out Metrics_v5_G5a_verdict.json]
  <root> contains one sub-directory per day (20240422/, 20240423/ or a window dir given by --cand_day_dir).
  --gt_dir holds segment_assignments_reviewed.json (04-22), segment_assignments.json (04-23) (either directly
  as <gt_dir>/<day>/<file> or <gt_dir>/<file>_<day>) and week_ident_map.json.
  The local copy of a v5 root must include, per day dir: masks_*.json.gz, segments.json, events.jsonl,
  detections_*.json.gz (G2), ROOT_CONFIG.json (day AND root level; MH8 / G5a), the .done_* markers when available,
  and the telemetry Tol/Metrics_track_v5_<root>_<day dir>.json (found next to the root or via --telemetry).

NEVER writes to the Volume; Tol_30 only ever sees a local copy, and the transferred GT it is handed is named
segment_assignments_transferred.json (never segment_assignments.json — T30:800-801 would pick that up).
"""
import os, sys, re, json, gzip, glob, time, math, hashlib, shutil, argparse, subprocess, bisect
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
try:                                    # optional: the shared v5 core (never required — this file is self-contained)
    if HERE not in sys.path: sys.path.insert(0, HERE)
    import tol_v5_core as CORE          # noqa
except Exception:
    CORE = None

# ============================================================================================================
# 0 · constants (each with its provenance; none is a tracker constant and none was fitted on 04-22..28)
# ============================================================================================================
N_SLOTS = 4
H_FULL, W_FULL = 1296, 2304
DAYS_GT = ("20240422", "20240423")
ASSOC_IOS = 0.60          # SPEC §6.3: Hungarian on min-area IoS >= 0.6 (not Tol_25's IoU >= 0.30)
ASSOC_TIE_IOU = 1e-3      # tie-break weight of IoU inside the Hungarian score (identical masks pair with each other)
MULTI_FRAC = 0.50         # SPEC §6.3: a cand mask containing >= 50 % (inter / GT area) of each of two GT masks
GT_SAME_ANIMAL = 0.50     # two GT masks are ONE animal (a GT merge) if max(IoU, IoS) > 0.5 (= §6.3-4 episode rule)
NONE_IOS_MAX = 0.10       # a cand mask with IoS < 0.10 against every GT mask is NONE (no GT animal); = LINK_CONFLICT_IOS
TRUST_GT_IOS = 0.05       # Tol_44.trust_filter GT_IOS_MAX (T44:179): a geometry-trusted GT mask overlaps no other GT mask
TEACHER_AGREE_IOU = 0.50  # SPEC §6.3 geometry: v4v2 and the teacher 'agree' on a mask at IoU >= 0.5 (= the error bar itself)
# review r2 (SPEC §6.3 'A geometry error counts only where the two agree'): the teacher's view of one v4v2 mask (g_tag) or
# of one cand NONE mask (c_ntag), taken on every row where a geometry label needs it (MULTI / NONE / a shared GT mask).
# -1 = no teacher row (v4v2 is the only reference: counted, conservatively); 0 = the teacher CONTRADICTS v4v2 (it has a
# mask there with another geometry: not a geometry error of the cand); 1 = the teacher AGREES; 2 = the teacher has
# nothing on that v4v2 mask (v4v2 is the only reference: counted, conservatively). No new constant: IoU >= TEACHER_AGREE_IOU
# is mask_fidelity's agreement, IoS >= NONE_IOS_MAX is the NONE boundary of associate().
TAG_NO_ROW, TAG_CONTRA, TAG_AGREE, TAG_TDARK = -1, 0, 1, 2
MERGE_GUARD_S = 5.0       # SPEC §6.3 identity-trust: no GT merge_resolved within ±5 s (= T30 MERGE_GUARD)
SWITCH_MIN_S = 10.0       # SPEC §6.3-5a: a different GT animal holds a continuous >= 10 s run inside one segment
SEED_MAP_S = 600.0        # SPEC §6.3-5b: seed segments map by majority over their first 10 min
FIRST_HOUR_S = 3600.0     # SPEC §6.3-5b fallback: Hungarian over first-hour votes
STEP2_WIN_S = 10.0        # step-2 action verdict: animal of the first / last 10 s around the action
OVERLAP_THR, OVERLAP_MIN_S = 0.50, 1.0     # SPEC §6.3-4: episodes >= 1 s at max(IoU, IoS) > 0.5
ORPHAN_LINK_IOU, ORPHAN_LINK_S = 0.20, 120.0  # SPEC §6.3-2: orphan = no spatial link IoU >= 0.20 within 120 s
SUBMIN_S = 60.0           # SPEC §6.3-2 sub-minute blips
CHURN_N, CHURN_WIN_S = 3, 60.0              # SPEC §6.3-2 churn: >= 3 segments in 60 s
GAP_MERGE_S = 1.0         # gaps_analyze.MERGE_SEC: presence blips < 1 s do not end a no-mask episode
GAP_BINS = ("<2s", "2-10s", "10-60s", "1-5min", ">5min")
GAP_LONG_S = 10.0         # gaps_analyze: the spare-detection analysis is over episodes >= 10 s
SEG_TOL_S = 0.05          # row wall = clip start + t_sec (rounded 0.01 s); segment walls are exact
FRAG_HALF = 0.5           # tune_scout/quality.py: fragment = area < 0.5 x pooled daylight median (1 Hz rows)
D8_DELTA_MIN = 60.0       # SPEC_v5 D8 (settled in the addendum): review trigger on overall wrong time
MH7_MAX_MIN = 1.0         # SPEC §6.4-7: MULTI + NONE outside GT-dark <= 1 min per day
NIGHTGAP_MARGIN = 0.015   # SPEC §6.4-6 (= Tol_30 TAU / Tol_16_v2 hard_min_margin)
A10_ALLOC_GIB, A10_RESERVED_GIB, A10_NPF = 4.0, 10.0, 2   # SPEC §6.4-8 / §6.1 G2
D6_CAP_GIB = 21.0         # addendum D6: A100 with mem_cap_gib=21.0 into Tracking_v5a100cap21 (labelled fallback)
INV7 = dict(acc=0.5, acc_min_n=10, med=0.20, med_min_n=5, cov3=0.5)   # Tol_47 finish_clip_telemetry (= T03 INV-7)
G2_MAX_DET_PER_FRAME = 1  # SPEC §3.4 / RUNBOOK G2: v5 never adds a second detect() per frame
TIMING_KEYS_REQUIRED = ("decode", "sam3", "detector")   # Tol_47 _lap() names; G2 "per-component timings present"
GUARDED_OPEN_REASONS = ("reopen", "displaced", "dark_link", "seed_day", "seed_night")   # the pen-count-guarded opens
STEP2_WINDOW_KEYS = ("dark_link", "dark_open")           # G2: >= 1 per pilot window (zero = a finding for the user)
WALL_TOL_S = 0.02         # detection record t_sec is rounded to 0.01 s; event walls are exact
# v4v2 night-gap reference 04-23 -> 04-24 (HANDOFF 2026-09-17, dense): B-local -> A-local
NIGHTGAP_REF_0423_0424 = {"perm": {0: 0, 1: 2, 2: 3, 3: 1}, "min_margin": 0.0054, "support": 0.54,
                          "source": "HANDOFF.md 2026-09-17 six night gaps (dense), UNRESOLVED (flagged)"}
G2_SUBSET = ("P0", "MH1", "MH2", "MH7", "MH8", "AUD1")   # critic finding 5: the must-holds that apply to a pilot window
G5A_SUBSET = ("P0", "MH1", "MH2", "MH8", "AUD1")         # SPEC §6.1 G5a-3 + RUNBOOK G5a (MH7 reported, not gated)
# train-sealed band constants (train_summary.json `constants`, derived from 04-18/19 only); read at run time when
# --train_summary is given, else these recorded values (verified equal to train_summary_segv1d.json 2026-09-24)
FRAG_LO = {"dayl": 0.44, "dark": 0.43}
OVER_HI = 1.32
# GT-normalising area reference (SPEC §6.3 'A_ref_eval 89,075 / 91,200 is used only to normalise GT'; = this file's
# fragment_metrics a_ref_eval_gt on the v4v2 days, reproduced by the zero-delta run 2026-09-25). Used ONLY to decide
# whether a GT mask is an animal (the ghost rule below), never on a cand mask. Any other day (04-24 / v4v2p2): the
# train-sealed A_REF_TRAIN (tol_v5_core.TRAIN_BAND_REFERENCE, 04-18/19 only).
A_REF_EVAL = {"20240422": 89075.0, "20240423": 91200.0}
A_REF_TRAIN = 87137.0
# review r1 finding 1: a GT (v4v2) mask whose segment is UNASSIGNED (no identity in the user-verified GT) and whose area
# is below FRAG_LO[pool] x A_ref (pool: day rows 'dayl', dusk/night rows 'dark') is a GHOST, not an animal: it does not
# count in n_dist (GT-dark stratum, MH7, the MH1 non-calf rule, the 5f fill trigger) and cannot make a cand mask MULTI.
# E.g. 04-22 21:10-21:27 v4v2 slot 3 = unassigned 8,125 px blob `3_010` while the teacher holds a 93,725 px calf there.
# SPEC §6.3 exclusions (wall clock; "24:00:00" = midnight at the end of the day)
IDENT_EXCL = {"20240422": [("14:00:54", "14:02:56"), ("20:59:51", "21:27:09")], "20240423": []}
LATE_STRATUM = {"20240423": ("23:30:00", "24:00:00")}           # "04-23 23:30-24:00 (its own stratum)"
GEOM_EXCL_SLOT = {"20240422": [(2, "13:47:00", "15:58:59")]}     # "04-22 slot 2 13:47-15:58, geometry only"
GEOM_EXCL_CENTRES = {"20240423": ["11:54:24", "11:54:54", "12:05:24", "13:09:44", "13:11:45", "15:14:29", "17:39:12"]}
GEOM_EXCL_HALF_S = 30.0   # "±30 s geometry exclusions around the known GT defects" (4_failures.md:74,79,91-92)
DAWN = ("05:44:00", "08:00:00")                                  # "plus dawn 05:44-08:00" (reported on every day)
# SPEC §6.3 sentinels: (day, wall, GT slot, first rejecting reason expected under v5 — informational)
SENTINELS = [("20240422", "14:00:55", 3, "cause merge_ambiguous (covered 0.964)"),
             ("20240422", "21:00:08", 0, "cause merge_ambiguous"),
             ("20240423", "12:05:23", 1, "merge_near (forced merge_1_2 pass at 12:05:24.4)"),
             ("20240423", "12:26:02", 0, "not_contained (moved 1,120 px)"),
             ("20240423", "20:35:57", 2, "not_contained (moved 646 px); dusk -> no fast-open"),
             ("20240423", "21:28:02", 2, "cause merge_ambiguous")]
SENTINEL_WIN_S = 60.0
# v4v2 night-gap reference (HANDOFF:1518-1530, dense exemplars): perm B-local -> A-local, min margin, support
NIGHTGAP_REF_0422_0423 = {"perm": {0: 0, 1: 3, 2: 2, 3: 1}, "min_margin": 0.0086, "support": 0.79,
                          "source": "HANDOFF.md 2026-09-17 six night gaps (dense), user-verified 4/4"}
# the known v4v2 figures the SELFTEST must reproduce (SPEC §6.3 'Baseline')
V4V2_KNOWN = {"20240422": {"nomask_h": 0.63, "segments": 27, "merge_resolved": 7, "night_frag_pct": 7.9},
              "20240423": {"nomask_h": 2.01, "segments": 136, "merge_resolved": 74, "night_frag_pct": 41.7}}
# Tol_30 review flags (T30:83-84) — verified against the pinned Tol_30 snapshot at run time
REVIEW_FLAGS_T30 = ("pigeonhole", "pigeonhole_conflict", "soft_inherit", "spatial_inherit", "hop_in",
                    "contested", "reassigned", "provisional", "modal", "bijection_conflict", "pool_bootstrap", "warm_pool")
STEP2_MECH = ("dark_link", "dark_pass_open")
# cand status codes (per cand mask) and GT status codes (per GT mask)
C_ABSENT, C_ASSIGNED, C_MULTI, C_SPLIT, C_UNK, C_NONE, C_NOGT = 0, 1, 2, 3, 4, 5, 6
C_NAMES = {0: "absent", 1: "assigned", 2: "MULTI", 3: "split", 4: "UNK", 5: "NONE", 6: "no_gt_row"}
G_ABSENT, G_ASSIGNED, G_MULTI, G_UNCOVERED = 0, 1, 2, 3
PHASES = ("day", "dusk", "night")
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}

# ============================================================================================================
# 1 · VERBATIM from Tol_03_track_v4.py:280-303 (rle_encode, rle_decode, m_iou, m_ios) — the reference semantics
# ============================================================================================================
def rle_encode(mask):
    h, w = mask.shape
    flat = np.asfortranarray(mask).reshape(-1, order="F")
    idx = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate(([0], idx, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]: counts = [0] + counts
    return {"size": [h, w], "counts": counts}

def rle_decode(r):
    h, w = r["size"]
    c = np.asarray(r["counts"], dtype=np.int64)
    vals = np.zeros(len(c), bool); vals[1::2] = True
    return np.repeat(vals, c).reshape((h, w), order="F")

def m_iou(a, b):
    if a is None or b is None: return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/u) if u else 0.0

def m_ios(a, b):
    if a is None or b is None: return 0.0
    m = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum()/m) if m else 0.0

# ============================================================================================================
# 2 · exact run arithmetic on the stored column-major RLE (no decode).  Integer-exact: S1 in the SELFTEST checks
#     inter/iou/ios against the verbatim m_iou/m_ios above with `==` on adversarial and real masks.
# ============================================================================================================
class Runs:
    __slots__ = ("s", "e", "cum", "area", "lo", "hi", "c0", "c1", "r0", "r1", "h")

def runs_of(rle):
    """foreground runs [s, e) of a column-major RLE + prefix lengths + a (column, row) bounding box"""
    h = int(rle["size"][0])
    c = np.asarray(rle["counts"], dtype=np.int64)
    ends = np.cumsum(c); starts = ends - c
    s = starts[1::2]; e = ends[1::2]
    k = e > s
    if not k.all(): s = s[k]; e = e[k]
    R = Runs(); R.h = h; R.s = s; R.e = e
    R.cum = np.concatenate(([0], np.cumsum(e - s)))
    R.area = int(R.cum[-1])
    if R.area:
        R.lo = int(s[0]); R.hi = int(e[-1]); R.c0 = R.lo // h; R.c1 = (R.hi - 1) // h
        if ((s // h) != ((e - 1) // h)).any(): R.r0, R.r1 = 0, h - 1        # a run wraps a column: no row bound
        else: R.r0 = int((s % h).min()); R.r1 = int(((e - 1) % h).max())
    else:
        R.lo = R.hi = R.c0 = R.c1 = R.r0 = R.r1 = 0
    return R

def rle_area(rle):
    return int(sum(rle["counts"][1::2]))

def _cover(B, x):
    """total foreground of B in [0, x) for every x (vectorised)"""
    j = np.searchsorted(B.s, x, side="right") - 1
    jj = np.maximum(j, 0)
    v = B.cum[jj] + np.minimum(x, B.e[jj]) - B.s[jj]
    return np.where(j >= 0, v, 0)

def inter(A, B):
    if A is None or B is None or A.area == 0 or B.area == 0: return 0
    if A is B: return A.area                      # the same RLE object (a root scored against itself)
    if A.lo >= B.hi or B.lo >= A.hi: return 0
    if A.c1 < B.c0 or B.c1 < A.c0 or A.r1 < B.r0 or B.r1 < A.r0: return 0
    if len(A.s) > len(B.s): A, B = B, A
    return int((_cover(B, A.e) - _cover(B, A.s)).sum())

def iou_r(A, B):
    if A is None or B is None: return 0.0
    i = inter(A, B); u = A.area + B.area - i
    return i / u if u else 0.0

def ios_r(A, B):
    if A is None or B is None: return 0.0
    m = min(A.area, B.area)
    return inter(A, B) / m if m else 0.0

# ============================================================================================================
# 3 · time, clips, phases
# ============================================================================================================
EPOCH = datetime(1970, 1, 1)
def ts_of(w): return (w - EPOCH).total_seconds()
def dt_of(t): return EPOCH + timedelta(seconds=float(t))
RXT = re.compile(r"-(\d{8})-(\d{6})-")
def clip_start(v):
    m = RXT.search(os.path.basename(v).replace("_", "-"))
    if m is None: raise RuntimeError(f"cannot parse a clip start from {v!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
def stem_of_file(p): return os.path.basename(p)[len("masks_"):-len(".json.gz")]
def day_ts(day, hms):
    h, m, s = (int(x) for x in hms.split(":"))
    d = datetime.strptime(day, "%Y%m%d")
    if h >= 24: h -= 24; d += timedelta(days=1)
    return ts_of(d.replace(hour=h, minute=m, second=s))
def phase_of_ts(t):
    h = dt_of(t).hour                    # = gaps_analyze.phase_of / Tol_30.phase_of
    return "night" if (h >= 21 or h < 6) else ("dusk" if h == 20 else "day")
def tod_v4(t):
    h = dt_of(t).hour                    # T03:647-648 tod_of_wall (the v4 cadence / 2-sighting regime)
    return "night" if (h >= 21 or h < 6) else "day"
def dur_bin(s):
    return "<2s" if s < 2 else "2-10s" if s < 10 else "10-60s" if s < 60 else "1-5min" if s < 300 else ">5min"
def iso(t): return None if t is None else dt_of(t).isoformat(timespec="milliseconds")
def hm(t): return dt_of(t).strftime("%H:%M:%S")

def sha256_file(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

TOL48_SHA256 = sha256_file(os.path.abspath(__file__))   # at IMPORT: the code that actually ran (the file may be edited later)
CORE_SHA256 = sha256_file(CORE.__file__) if CORE is not None else None

def gpu_class(name):
    """= tol_v5_core.gpu_class (the SELFTEST asserts equality when the core is importable): an EXACT class, so an
    'NVIDIA A100-SXM4-80GB' is never read as an A10 (critic finding 4: a substring test "A10" in name matched A100)"""
    n = (name or "").upper()
    if "A100" in n: return "A100"
    if re.search(r"A10(?!\d)", n): return "A10"
    return n or "UNKNOWN"

def clip_key(c):
    """one spelling for a clip across masks headers, events, detections records and .done_ markers"""
    b = os.path.basename(str(c or ""))
    if b.endswith(".mp4"): b = b[:-4]
    return b.replace(" ", "_")

class DocxGT:
    """04-24 interval GT from the user's notes (gt_0424_from_user_notes.json, DATA_SPLITS.md:208-213), exactly the
    semantics of Tol_29:91-99 / Tol_30:960-966 / Tol_31b:79-92: a GT (v4v2p2) segment's local assignment `loc`
    (Tracking_v4v2p2_reid/20240424/segment_assignments.json) is shown as video label G[loc]; the calf under that
    label at wall t is intervals[t][label]; None within +-exclude_margin_sec of a change point, inside an excluded
    span, or where the label is not in the interval's map. The calf indices are the docx's own (global only if the
    04-23->24 boundary was right: held out for the TRACKER only, SPEC §6.1 G5a caveat)."""
    def __init__(self, path, day):
        J = json.load(open(path, encoding="utf-8"))
        if J.get("day") and str(J["day"]) != day: raise RuntimeError(f"{path} is the GT of {J['day']}, not {day}")
        self.path, self.sha256, self.day = path, sha256_file(path), day
        self.G = {int(k): int(v) for k, v in J["global_ident_map"].items()}
        self.ivs = [(day_ts(day, f), day_ts(day, t), {int(k): (None if v is None else int(v)) for k, v in m.items()})
                    for f, t, m in J["intervals"]]
        self.exc = [(day_ts(day, e[0]), day_ts(day, e[1])) for e in J.get("excluded", [])]
        self.margin = float(J.get("exclude_margin_sec", 180))
        self.cps = sorted({a for a, _b, _m in self.ivs} | {b for _a, b, _m in self.ivs})
        self.video_root = J.get("video_root")
    def animal(self, loc, t):
        """GT calf (docx index) under local assignment `loc` at wall ts t; -1 where undefined"""
        if loc is None or loc < 0: return -1
        label = self.G.get(int(loc))
        if label is None: return -1
        if any(a <= t < b for a, b in self.exc): return -1
        if any(abs(t - cp) <= self.margin for cp in self.cps): return -1
        for a, b, m in self.ivs:
            if a <= t < b:
                v = m.get(label)
                return -1 if v is None else int(v)
        return -1
    def info(self):
        return dict(path=self.path, sha256=self.sha256, day=self.day, global_ident_map=self.G, n_intervals=len(self.ivs),
                    exclude_margin_sec=self.margin, video_root=self.video_root,
                    note="docx calf indices; global only if the 04-23->24 boundary was right (tracker-only held out)")

# ============================================================================================================
# 4 · inputs: segments, events, GT identity, teacher identity
# ============================================================================================================
class SegIndex:
    """segments.json of one day-dir. Row -> segment: the latest segment of that slot starting at or before the row
    (tolerance SEG_TOL_S) whose end is at or after it — at a close-at-wall/open-at-wall boundary the row belongs to
    the NEW segment, exactly as T03 counts it (n_frames)."""
    def __init__(self, path):
        J = json.load(open(path, encoding="utf-8"))
        self.recs = []
        for i, r in enumerate(J["segments"]):
            t0 = ts_of(datetime.fromisoformat(r["t_start_wall"]))
            t1 = ts_of(datetime.fromisoformat(r["t_end_wall"])) if r.get("t_end_wall") else float("inf")
            self.recs.append(dict(idx=i, sid=r["seg_id"], slot=int(r["slot"]), t0=t0, t1=t1,
                                  open_reason=r.get("open_reason"), close_reason=r.get("close_reason"),
                                  via=r.get("via"), raw=r))
        self.by_sid = {r["sid"]: r for r in self.recs}
        self.per_slot = defaultdict(list)
        for r in sorted(self.recs, key=lambda z: (z["t0"], z["idx"])): self.per_slot[r["slot"]].append(r)
        self.starts = {s: [r["t0"] for r in v] for s, v in self.per_slot.items()}
    def at(self, slot, t):
        st = self.starts.get(slot)
        if not st: return -1
        i = bisect.bisect_right(st, t + SEG_TOL_S) - 1
        if i < 0: return -1
        r = self.per_slot[slot][i]
        return r["idx"] if t <= r["t1"] + SEG_TOL_S else -1

def load_events(path):
    out = []
    if not os.path.exists(path): return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line: continue
        try: e = json.loads(line)
        except Exception: continue
        t = None
        if e.get("wall"):
            try: t = ts_of(datetime.fromisoformat(e["wall"]))
            except Exception: t = None
        if t is None and e.get("clip"):
            try: t = ts_of(clip_start(e["clip"])) + float(e.get("t", 0) or 0)
            except Exception: t = None
        e["_t"] = t
        out.append(e)
    return out

def load_gt_assign(path, day, week_map):
    """{seg_id: GT animal (week-global space) or None}. week_map maps day-local -> global (identity for 04-22)."""
    a = json.load(open(path, encoding="utf-8"))["assignments"]
    gm = {str(k): int(v) for k, v in ((week_map or {}).get(day) or {}).items()}
    if gm and sorted(gm.values()) != list(range(N_SLOTS)): raise RuntimeError(f"week map for {day} is not a permutation: {gm}")
    return {sid: (None if v is None else (gm[str(int(v))] if gm else int(v))) for sid, v in a.items()}

def load_week_map(path):
    if not path: return None
    J = json.load(open(path, encoding="utf-8"))
    J = J.get("global_ident_map", J)
    return {d: {str(k): int(v) for k, v in m.items()} for d, m in J.items()}

def teacher_rules_key(rules, excl, slot, t):
    """04-22 teacher GT v2: teacher slot -> true animal A0..A3 by wall-clock rule; None inside an exclusion"""
    for e in excl:
        if slot in e["slots"] and e["t0"] <= t < e["t1"]: return None
    for a, b, lab in rules.get(slot, []):
        if (a is None or a <= t) and (b is None or t < b): return int(lab)
    return None

def load_teacher_rules(path):
    J = json.load(open(path, encoding="utf-8"))
    rules = {int(k): [(None if a is None else ts_of(datetime.fromisoformat(a)),
                       None if b is None else ts_of(datetime.fromisoformat(b)), int(l)) for a, b, l in v]
             for k, v in J["rules"].items()}
    excl = [dict(t0=ts_of(datetime.fromisoformat(e["from"])), t1=ts_of(datetime.fromisoformat(e["to"])),
                 slots=[int(s) for s in e["slots"]]) for e in J.get("exclusions", [])]
    return rules, excl, dict(version=J.get("version"), status=J.get("status"), sha256=sha256_file(path))

# ============================================================================================================
# 5 · per-row association (pure; unit-tested in the SELFTEST)
# ============================================================================================================
def _components(n_present, overlap_pairs):
    parent = {j: j for j in n_present}
    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in overlap_pairs: parent[f(a)] = f(b)
    return {j: f(j) for j in n_present}

def associate(CR, GR, ghost=None):
    """CR/GR: 4 Runs|None (cand, GT). Returns per-cand (status, gt_slot, ios, iou), per-GT (status, cand_slot,
    max IoS vs other GT masks), number of distinct GT animals (GT masks overlapping > GT_SAME_ANIMAL are one).
    ghost: 4 bools, a GT mask that is not an animal (unassigned + below the band, see A_REF_EVAL): it stays in the
    geometric association (a cand mask on it is 'assigned' with an unknown identity) but a component made only of
    ghosts is not a distinct animal, and a ghost never makes a cand mask MULTI (review r1 finding 1)."""
    gh = [bool(ghost[j]) if ghost is not None else False for j in range(N_SLOTS)]
    ci = [i for i in range(N_SLOTS) if CR[i] is not None and CR[i].area > 0]
    gj = [j for j in range(N_SLOTS) if GR[j] is not None and GR[j].area > 0]
    c_stat = [C_ABSENT] * N_SLOTS; c_g = [-1] * N_SLOTS; c_ios = [0.0] * N_SLOTS; c_iou = [0.0] * N_SLOTS
    g_stat = [G_ABSENT] * N_SLOTS; g_c = [-1] * N_SLOTS; g_ggios = [0.0] * N_SLOTS
    # GT-GT overlaps: geometry trust + animal components
    same = []
    for a in range(len(gj)):
        for b in range(a + 1, len(gj)):
            j1, j2 = gj[a], gj[b]
            it = inter(GR[j1], GR[j2])
            if not it: continue
            u = GR[j1].area + GR[j2].area - it; iou = it / u if u else 0.0
            ios = it / min(GR[j1].area, GR[j2].area)
            g_ggios[j1] = max(g_ggios[j1], ios); g_ggios[j2] = max(g_ggios[j2], ios)
            if max(iou, ios) > GT_SAME_ANIMAL: same.append((j1, j2))
    comp = _components(gj, same)
    n_distinct = len({comp[j] for j in gj if not gh[j]})
    for j in gj: g_stat[j] = G_UNCOVERED
    for i in ci: c_stat[i] = C_NONE
    if not ci or not gj:
        return c_stat, c_g, c_ios, c_iou, g_stat, g_c, g_ggios, n_distinct
    I = np.zeros((N_SLOTS, N_SLOTS), np.int64)
    for i in ci:
        for j in gj: I[i, j] = inter(CR[i], GR[j])
    IOS = np.zeros((N_SLOTS, N_SLOTS)); IOU = np.zeros((N_SLOTS, N_SLOTS))
    for i in ci:
        for j in gj:
            it = int(I[i, j])
            if not it: continue
            m = min(CR[i].area, GR[j].area); u = CR[i].area + GR[j].area - it
            IOS[i, j] = it / m if m else 0.0; IOU[i, j] = it / u if u else 0.0
    multi = set()
    for i in ci:
        hit = {comp[j] for j in gj if not gh[j] and I[i, j] and I[i, j] / GR[j].area >= MULTI_FRAC}
        if len(hit) >= 2: multi.add(i)
    rows = [i for i in ci if i not in multi]
    if rows:
        S = np.array([[IOS[i, j] + ASSOC_TIE_IOU * IOU[i, j] for j in gj] for i in rows])
        ri, cj = linear_sum_assignment(-S)
        for a, b in zip(ri, cj):
            i, j = rows[a], gj[b]
            if IOS[i, j] >= ASSOC_IOS:
                c_stat[i] = C_ASSIGNED; c_g[i] = j; c_ios[i] = float(IOS[i, j]); c_iou[i] = float(IOU[i, j])
                g_stat[j] = G_ASSIGNED; g_c[j] = i
    for i in ci:
        if i in multi:
            c_stat[i] = C_MULTI
            c_ios[i] = float(max(IOS[i, j] for j in gj)); continue
        if c_stat[i] == C_ASSIGNED: continue
        best_j = max(gj, key=lambda j: (IOS[i, j], IOU[i, j]))
        c_ios[i] = float(IOS[i, best_j]); c_iou[i] = float(IOU[i, best_j]); c_g[i] = best_j
        if IOS[i, best_j] >= ASSOC_IOS: c_stat[i] = C_SPLIT          # a second piece of an already-assigned animal
        elif IOS[i, best_j] >= NONE_IOS_MAX: c_stat[i] = C_UNK
        else: c_stat[i] = C_NONE; c_g[i] = -1
    for j in gj:
        if g_stat[j] == G_ASSIGNED: continue
        if any(I[i, j] and I[i, j] / GR[j].area >= MULTI_FRAC for i in multi): g_stat[j] = G_MULTI
    return c_stat, c_g, c_ios, c_iou, g_stat, g_c, g_ggios, n_distinct

def match_best(CR_i, TR):
    """best teacher mask for one cand mask: (slot, ios, iou) with IoS >= ASSOC_IOS, else (-1, best ios, iou)"""
    best = (-1, 0.0, 0.0)
    for k in range(N_SLOTS):
        if TR[k] is None: continue
        o = ios_r(CR_i, TR[k]); u = iou_r(CR_i, TR[k])
        if (o, u) > (best[1], best[2]): best = (k, o, u)
    return best if best[1] >= ASSOC_IOS else (-1, best[1], best[2])

def hungarian_pairs(AR, BR, thr=ASSOC_IOS):
    """one-to-one A->B by IoS (IoU tie-break), accepted at IoS >= thr"""
    ai = [i for i in range(N_SLOTS) if AR[i] is not None]; bj = [j for j in range(N_SLOTS) if BR[j] is not None]
    if not ai or not bj: return {}
    S = np.zeros((len(ai), len(bj))); O = np.zeros_like(S)
    for a, i in enumerate(ai):
        for b, j in enumerate(bj):
            O[a, b] = ios_r(AR[i], BR[j]); S[a, b] = O[a, b] + ASSOC_TIE_IOU * iou_r(AR[i], BR[j])
    ri, cj = linear_sum_assignment(-S)
    return {ai[a]: bj[b] for a, b in zip(ri, cj) if O[a, b] >= thr}

def teacher_gt_tag(GRj, TR, partner):
    """review r2 (SPEC §6.3 'a geometry error counts only where the two agree'): the teacher's view of ONE v4v2 mask.
    AGREE = its Hungarian teacher partner (hungarian_pairs) has IoU >= TEACHER_AGREE_IOU (= mask_fidelity's agreement);
    CONTRA = no agreeing partner but a teacher mask covers it (min-area IoS >= NONE_IOS_MAX: the teacher sees an animal
    there with another geometry, e.g. a v4v2 fragment straddling two calves); TDARK = the teacher has nothing there."""
    if partner is not None and iou_r(TR[partner], GRj) >= TEACHER_AGREE_IOU: return TAG_AGREE
    if any(T is not None and ios_r(T, GRj) >= NONE_IOS_MAX for T in TR): return TAG_CONTRA
    return TAG_TDARK

def teacher_none_tag(CRk, TR):
    """review r2: the teacher's view of a cand NONE mask (IoS < NONE_IOS_MAX against every v4v2 mask). AGREE = the teacher
    would label it NONE too (IoS < NONE_IOS_MAX against every teacher mask); CONTRA = the teacher has an animal under it
    that v4v2 lacks."""
    return TAG_CONTRA if any(T is not None and ios_r(CRk, T) >= NONE_IOS_MAX for T in TR) else TAG_AGREE

# ============================================================================================================
# 6 · the scan: one pass over one arm-day (cand rows aligned to GT rows by (clip, frame_idx))
# ============================================================================================================
def parse_window(day_dir_name):
    m = re.search(r"_w(\d{4})-(\d{4})$", day_dir_name)
    if not m: return None
    return (int(m.group(1)[:2]) * 60 + int(m.group(1)[2:]), int(m.group(2)[:2]) * 60 + int(m.group(2)[2:]))

def in_window(stem, win):
    if win is None: return True
    s = clip_start(stem); mm = s.hour * 60 + s.minute
    return win[0] <= mm < win[1]

class FrameStream:
    """Streaming reader of one masks_<clip>.json.gz: the header fields plus ONE frame dict at a time. A 34-min clip
    is ~230 MB of JSON (~1.5 GB as Python objects); json.load of three of them per worker does not fit in memory."""
    CHUNK = 1 << 20
    TRIM = 2 << 20
    def __init__(self, path):
        self.path = path; self.f = gzip.open(path, "rt", encoding="utf-8")
        self.dec = json.JSONDecoder(); self.buf = ""; self.pos = 0; self.eof = False; self.done = False
        while True:
            k = self.buf.find('"frames"')
            b = self.buf.find("[", k) if k >= 0 else -1
            if b >= 0: break
            if not self._more(): raise RuntimeError(f"{path}: no 'frames' array")
        try: self.header = json.loads(self.buf[:k].rstrip().rstrip(",") + "}")
        except Exception: self.header = {}
        self.pos = b + 1
    def _more(self):
        if self.eof: return False
        s = self.f.read(self.CHUNK)
        if not s: self.eof = True; return False
        if self.pos > self.TRIM: self.buf = self.buf[self.pos:]; self.pos = 0
        self.buf += s
        return True
    def __iter__(self): return self
    def __next__(self):
        if self.done: raise StopIteration
        while True:
            buf, p = self.buf, self.pos; n = len(buf)
            while p < n and buf[p] in " \t\r\n,": p += 1
            self.pos = p
            if p >= n:
                if not self._more(): raise RuntimeError(f"{self.path}: truncated 'frames' array")
                continue
            if buf[p] == "]":
                self.pos = p + 1; self._tail(); self.done = True
                raise StopIteration
            try:
                obj, end = self.dec.raw_decode(buf, p)
            except json.JSONDecodeError:
                if not self._more(): raise
                continue
            self.pos = end
            return obj
    def _tail(self):
        while self._more(): pass
        rest = self.buf[self.pos:].strip().lstrip(",").strip()
        if len(rest) > 1 and rest.endswith("}"):
            try: self.header.update(json.loads("{" + rest))
            except Exception: pass
        self.f.close()

def load_masks(path):
    """whole-file load (SELFTEST / small files only; the scan streams)"""
    with gzip.open(path, "rt") as f: return json.load(f)

ROW_SPEC = dict(ts=((), np.float64, 0), fidx=((), np.int32, 0), pos=((), np.int32, 0), has_gt=((), bool, False),
                c_area=((4,), np.int32, 0), c_seg=((4,), np.int32, -1), c_stat=((4,), np.int8, 0), c_gs=((4,), np.int8, -1),
                c_ios=((4,), np.float32, 0), c_iou=((4,), np.float32, 0), g_area=((4,), np.int32, 0), g_anim=((4,), np.int8, -2),
                g_seg=((4,), np.int32, -1), g_stat=((4,), np.int8, 0), g_c=((4,), np.int8, -1), g_gg=((4,), np.float32, 0),
                g_idt=((4,), bool, False), n_dist=((), np.int8, 0), p_over=((), np.uint8, 0), clip_i=((), np.int16, 0),
                g_tiou=((4,), np.float32, -1.0),   # 1 Hz rows with a teacher row: IoU of the GT mask with its paired
                                                   # teacher mask (0 = unpaired); -1 = no teacher row (SPEC §6.3 geometry)
                g_ghost=((4,), bool, False),       # the GT mask is a ghost (unassigned + below the band: A_REF_EVAL)
                n_dist_raw=((), np.int8, 0),       # distinct GT masks counting ghosts (report: GT-dark added by ghosts)
                c_cov=((4,), np.uint8, 0),         # MULTI cand mask: bit j = it contains >= MULTI_FRAC of (non-ghost) GT mask j
                g_tag=((4,), np.int8, -1),         # the teacher's view of GT mask j on a geometry row (TAG_*; review r2)
                c_ntag=((4,), np.int8, -1))        # the teacher's view of a cand NONE mask (TAG_*; review r2)

class RowBuf:
    def __init__(self, cap=12000):
        self.cap = cap; self.a = {k: np.full((cap,) + shp, fill, dt) for k, (shp, dt, fill) in ROW_SPEC.items()}
    def ensure(self, i):
        if i < self.cap: return
        new = max(i + 1, 2 * self.cap)
        for k, (shp, dt, fill) in ROW_SPEC.items():
            b = np.full((new,) + shp, fill, dt); b[:self.cap] = self.a[k]; self.a[k] = b
        self.cap = new
    def finish(self, n):
        return {k: v[:n].copy() for k, v in self.a.items()}

def scan_day(job):
    """job: dict(arm, day, cand_dir, gt_dir|None, gt_assign|None, week_map|None, teacher_dir|None,
    teacher_rules|None, light). Streams every clip once, cand / GT / teacher merge-joined on frame_idx.
    Returns a dict of numpy arrays + records (pickle-able)."""
    t_start = time.time()
    arm, day, cand_dir = job["arm"], job["day"], job["cand_dir"]
    light = bool(job.get("light"))
    # the scoring window: the cand day dir's _wHHMM-HHMM suffix, or an explicit override (G2 scores the ref arm over
    # the pilot window only)
    win = job.get("window") or parse_window(os.path.basename(os.path.normpath(cand_dir)))
    if win is not None: win = tuple(win)
    cseg = SegIndex(os.path.join(cand_dir, "segments.json"))
    cev = load_events(os.path.join(cand_dir, "events.jsonl"))
    cfiles = {stem_of_file(p): p for p in glob.glob(os.path.join(cand_dir, "masks_*.json.gz"))
              if in_window(stem_of_file(p), win)}
    done_marks = sorted(n[len(".done_"):] for n in os.listdir(cand_dir) if n.startswith(".done_"))
    gt_dir = job.get("gt_dir")
    gseg = gev = gasg = None; gfiles = {}
    merge_ts = []
    docx = DocxGT(job["gt_docx"], day) if (gt_dir and job.get("gt_docx")) else None
    if gt_dir:
        gseg = SegIndex(os.path.join(gt_dir, "segments.json"))
        gev = load_events(os.path.join(gt_dir, "events.jsonl"))
        # docx mode (04-24): the RAW local assignment (the docx maps it to a calf per wall-clock interval, per row)
        gasg = load_gt_assign(job["gt_assign"], day, None if docx else job.get("week_map")) if job.get("gt_assign") else {}
        gfiles = {stem_of_file(p): p for p in glob.glob(os.path.join(gt_dir, "masks_*.json.gz"))}
        merge_ts = sorted(e["_t"] for e in gev if e.get("type") == "merge_resolved" and e.get("_t") is not None)
    gt_anim_of_seg = np.full(max(1, len(gseg.recs) if gseg else 1), -1, np.int16)
    if gseg:
        for r in gseg.recs:
            v = gasg.get(r["sid"]) if gasg else None
            gt_anim_of_seg[r["idx"]] = -1 if v is None else int(v)   # docx mode: the local assignment (per-row map)
    # the ghost rule (review r1 finding 1): only with a loaded GT assignment (otherwise 'unassigned' means nothing).
    # frag_lo travels in the job: ProcessPoolExecutor workers re-import this module and would not see --train_summary
    frag_lo_j = dict(job.get("frag_lo") or FRAG_LO)
    a_ref_g = job.get("a_ref_ghost")
    ghost_src = "job" if a_ref_g else ("A_REF_EVAL" if day in A_REF_EVAL else "A_REF_TRAIN")
    a_ref_g = float(a_ref_g or A_REF_EVAL.get(day) or A_REF_TRAIN)
    ghost_on = bool(gseg is not None and gasg)
    ghost_lo = {"day": frag_lo_j["dayl"] * a_ref_g, "dusk": frag_lo_j["dark"] * a_ref_g, "night": frag_lo_j["dark"] * a_ref_g}
    tdir = job.get("teacher_dir")
    tseg = None; tfiles = {}; trules = texcl = None
    if tdir and gt_dir and not light:
        tseg = SegIndex(os.path.join(tdir, "segments.json"))
        tfiles = {stem_of_file(p): p for p in glob.glob(os.path.join(tdir, "masks_*.json.gz"))}
        if job.get("teacher_rules"): trules, texcl, _ = load_teacher_rules(job["teacher_rules"])
    # scope: cand clips (+ GT clips inside the cand window that the cand lacks, for the alignment precondition)
    stems = sorted(set(cfiles) | {s for s in gfiles if in_window(s, win)}, key=clip_start) if gt_dir else \
        sorted(cfiles, key=clip_start)
    excl = [(day_ts(day, a), day_ts(day, b)) for a, b in IDENT_EXCL.get(day, [])]
    late = LATE_STRATUM.get(day); late = (day_ts(day, late[0]), day_ts(day, late[1])) if late else None
    dawn = (day_ts(day, DAWN[0]), day_ts(day, DAWN[1]))
    geo_slot = [(s, day_ts(day, a), day_ts(day, b)) for s, a, b in GEOM_EXCL_SLOT.get(day, [])]
    geo_c = [(day_ts(day, c) - GEOM_EXCL_HALF_S, day_ts(day, c) + GEOM_EXCL_HALF_S) for c in GEOM_EXCL_CENTRES.get(day, [])]
    def in_ivs(t, ivs): return any(a <= t < b for a, b in ivs)
    def near_merge(t):
        k = bisect.bisect_left(merge_ts, t - MERGE_GUARD_S)
        return k < len(merge_ts) and merge_ts[k] <= t + MERGE_GUARD_S
    chunks = []                                    # per-clip dicts of numpy arrays
    clips_meta, fps_seen, sizes = [], Counter(), Counter()
    first_link, last_mask = {}, [None] * N_SLOTS  # seg idx -> (prev seg idx, gap s, iou); per slot (t, runs, seg)
    seg_nomask = 0
    fills = []                                     # 5f: cand NONE masks on GT-dark rows, with the teacher's view
    tvotes = defaultdict(Counter)                  # teacher key -> GT animal votes (1 Hz, id-trusted)
    g_rows_scope = g_rows_aligned = 0
    row0 = 0
    def stepper(stream, what):
        """next(frame) with a strictly-increasing frame_idx check (the merge-join relies on it)"""
        last = [-1]
        def nxt():
            fr = next(stream, None)
            if fr is not None:
                if fr["frame_idx"] <= last[0]: raise RuntimeError(f"{what}: frame_idx not increasing at {fr['frame_idx']}")
                last[0] = fr["frame_idx"]
            return fr
        return nxt
    for ci, stem in enumerate(stems):
        cp = cfiles.get(stem); gp = gfiles.get(stem) if gt_dir else None
        shared = bool(gp and cp and os.path.realpath(gp) == os.path.realpath(cp))
        tp = tfiles.get(stem) if tseg is not None else None
        CS = FrameStream(cp) if cp else None
        GS = FrameStream(gp) if (gp and not shared) else None
        TS = FrameStream(tp) if tp else None
        g_next = stepper(GS, f"GT {stem}") if GS else None
        t_next = stepper(TS, f"teacher {stem}") if TS else None
        c_next = stepper(CS, f"cand {stem}") if CS else None
        g_cur = g_next() if g_next else None; t_cur = t_next() if t_next else None
        n_g = n_al = 0
        if CS is None:
            while g_cur is not None: n_g += 1; g_cur = g_next()
            g_rows_scope += n_g
            clips_meta.append(dict(stem=stem, cand=False, gt=gp is not None, teacher=TS is not None, rows=0, gt_rows=n_g,
                                   aligned=0, smoke=None))
            continue
        w0 = ts_of(clip_start(CS.header.get("clip", stem)))
        fps_seen[float(CS.header.get("track_fps", 5.0))] += 1
        B = RowBuf(); a = B.a; r_i = 0
        while True:
            rc = c_next()
            if rc is None: break
            f = rc["frame_idx"]
            if shared: rg = rc; n_g += 1; n_al += 1
            elif g_next is not None:
                while g_cur is not None and g_cur["frame_idx"] < f: n_g += 1; g_cur = g_next()
                if g_cur is not None and g_cur["frame_idx"] == f: rg = g_cur; n_g += 1; n_al += 1; g_cur = g_next()
                else: rg = None
            else: rg = None
            rt = None
            if t_next is not None:
                while t_cur is not None and t_cur["frame_idx"] < f: t_cur = t_next()
                if t_cur is not None and t_cur["frame_idx"] == f: rt = t_cur; t_cur = t_next()
            B.ensure(r_i); a = B.a
            t = w0 + float(rc["t_sec"]); a["ts"][r_i] = t; a["fidx"][r_i] = f; a["pos"][r_i] = r_i; a["clip_i"][r_i] = ci
            cm = rc.get("masks") or {}
            gm = (rg.get("masks") or {}) if rg is not None else {}
            cache = {}
            def R_of(rle):
                if not rle: return None
                k_ = id(rle)
                if k_ not in cache: cache[k_] = runs_of(rle)
                return cache[k_]
            for k in range(N_SLOTS):
                rl = cm.get(str(k))
                if rl:
                    sizes[tuple(rl["size"])] += 1
                    a["c_area"][r_i, k] = rle_area(rl)
                    si = cseg.at(k, t); a["c_seg"][r_i, k] = si
                    if si < 0: seg_nomask += 1
            if light: r_i += 1; continue
            CR = [R_of(cm.get(str(k))) for k in range(N_SLOTS)]
            # candidate pairwise overlap (merge episodes, §6.3-4)
            bits = 0
            for pi, (p_, q_) in enumerate(PAIRS):
                if CR[p_] is None or CR[q_] is None: continue
                it = inter(CR[p_], CR[q_])
                if not it: continue
                u = CR[p_].area + CR[q_].area - it
                if max(it / u, it / min(CR[p_].area, CR[q_].area)) > OVERLAP_THR: bits |= (1 << pi)
            a["p_over"][r_i] = bits
            # first-mask spatial link of every new segment to the same slot's previous mask (§6.3-2 orphan rule)
            for k in range(N_SLOTS):
                if CR[k] is None: continue
                si = int(a["c_seg"][r_i, k]); lm = last_mask[k]
                if si >= 0 and si not in first_link and (lm is None or lm[2] != si):
                    first_link[si] = (None, None, None) if lm is None else (lm[2], round(t - lm[0], 3), round(iou_r(CR[k], lm[1]), 4))
                last_mask[k] = (t, CR[k], si)
            if rg is None:
                for k in range(N_SLOTS):
                    if CR[k] is not None: a["c_stat"][r_i, k] = C_NOGT
                r_i += 1; continue
            a["has_gt"][r_i] = True
            GR = [R_of(gm.get(str(j))) for j in range(N_SLOTS)]
            ex = in_ivs(t, excl)            # the late stratum stays trusted -- it is reported as its own stratum
            nm = near_merge(t)
            gh = [False] * N_SLOTS
            lo_g = ghost_lo[phase_of_ts(t)] if ghost_on else 0.0
            for j in range(N_SLOTS):
                if GR[j] is None: continue
                a["g_area"][r_i, j] = GR[j].area
                si = gseg.at(j, t); a["g_seg"][r_i, j] = si
                loc = int(gt_anim_of_seg[si]) if si >= 0 else -1
                an = loc
                if docx is not None: an = docx.animal(an, t)         # 04-24: label -> calf per interval, per row
                a["g_anim"][r_i, j] = an
                a["g_idt"][r_i, j] = an >= 0 and not ex and not nm
                if ghost_on and loc < 0 and GR[j].area < lo_g: gh[j] = True; a["g_ghost"][r_i, j] = True
            cs, cg, cio, ciu, gs, gc_, ggios, nd = associate(CR, GR, gh)
            a["n_dist"][r_i] = nd
            a["n_dist_raw"][r_i] = associate([None] * N_SLOTS, GR)[7] if any(gh) else nd
            for k in range(N_SLOTS):
                if CR[k] is None: continue
                a["c_stat"][r_i, k] = cs[k]; a["c_gs"][r_i, k] = cg[k]; a["c_ios"][r_i, k] = cio[k]; a["c_iou"][r_i, k] = ciu[k]
            for j in range(N_SLOTS):
                a["g_stat"][r_i, j] = gs[j]; a["g_c"][r_i, j] = gc_[j]; a["g_gg"][r_i, j] = ggios[j]
            # ---- fills (5f): cand NONE masks on GT-dark rows; the teacher gives their identity where it has a mask.
            #      Teacher rows also feed (1 Hz) the votes of the teacher -> GT identity map. ----
            need_fill = nd < N_SLOTS and any(cs[k] == C_NONE for k in range(N_SLOTS))
            # ---- review r2: rows whose geometry labels need the teacher's view (SPEC §6.3 agreement rule): a MULTI or
            #      NONE cand mask (MH7, the step-2 non-calf rule) or a GT mask two cand masks point at (duplicate rule) ----
            gt_users = Counter(cg[k] for k in range(N_SLOTS) if CR[k] is not None and cs[k] in (C_ASSIGNED, C_SPLIT) and cg[k] >= 0)
            need_geo = any(cs[k] in (C_MULTI, C_NONE) for k in range(N_SLOTS)) or any(v >= 2 for v in gt_users.values())
            for k in range(N_SLOTS):
                if cs[k] == C_MULTI:
                    a["c_cov"][r_i, k] = sum(1 << j for j in range(N_SLOTS) if GR[j] is not None and GR[j].area and not gh[j]
                                             and inter(CR[k], GR[j]) / GR[j].area >= MULTI_FRAC)
            if rt is None:
                if need_fill:
                    for k in range(N_SLOTS):
                        if cs[k] == C_NONE:
                            fills.append(dict(row=row0 + r_i, slot=k, t=t, tslot=-1, t_ios=None, t_iou=None, tkey=None,
                                              t_on_gt=None, why=("no_teacher_row" if tseg is not None else "no_teacher_root")))
                r_i += 1; continue
            if not need_fill and not need_geo and r_i % 5 != 0: r_i += 1; continue
            tm = rt.get("masks") or {}
            TR = [R_of(tm.get(str(k))) for k in range(N_SLOTS)]
            def tkey(k):
                if trules is not None: return ("A", teacher_rules_key(trules, texcl, k, t))
                si_ = tseg.at(k, t); return ("seg", tseg.recs[si_]["sid"] if si_ >= 0 else None)
            t2g = hungarian_pairs(TR, GR)
            if r_i % 5 == 0:
                # SPEC §6.3 'Geometry reference: v4v2 and the teacher. A geometry error counts only where the two agree'
                for j in range(N_SLOTS):
                    if GR[j] is not None: a["g_tiou"][r_i, j] = 0.0
                for k, j in t2g.items(): a["g_tiou"][r_i, j] = iou_r(TR[k], GR[j])
                for k, j in t2g.items():
                    key = tkey(k)
                    if key[1] is None or not a["g_idt"][r_i, j]: continue
                    tvotes[key][int(a["g_anim"][r_i, j])] += 1
            if need_geo:                     # review r2: every geometry row, not only the 1 Hz ones (g_tiou stays 1 Hz)
                g2t = {j: k for k, j in t2g.items()}
                for j in range(N_SLOTS):
                    if GR[j] is not None and GR[j].area: a["g_tag"][r_i, j] = teacher_gt_tag(GR[j], TR, g2t.get(j))
                for k in range(N_SLOTS):
                    if cs[k] == C_NONE: a["c_ntag"][r_i, k] = teacher_none_tag(CR[k], TR)
            if need_fill:
                def t_span(k, tk):
                    """review r3 (the teacher-side mirror of trusted_animal's r2 rule): the fill's teacher mask spans two
                    cand animals when another PRESENT cand mask that does NOT overlap this one (p_over pair bit clear,
                    = trusted_animal's disjointness) either holds (assigned / split) the GT mask the teacher mask is paired
                    with ('paired_gt_mask'), or is matched by that teacher mask itself (min-area IoS >= ASSOC_IOS,
                    match_best's rule; 'teacher_mask', the GT-dark sibling, where no GT mask is paired). A teacher mask
                    over two calves hands one calf's identity to a cand mask on the other, so it names no identity.
                    Overlapping cand masks (a cand duplicate on one animal) keep it.
                    'paired_gt_mask_unheld' (the same defect when the cand has NO mask of its own holding that GT mask,
                    e.g. a slot that left calf Y and dark-linked onto the adjacent calf Z while the teacher merged Y+Z):
                    a teacher mask paired with GT mask j lies over j AND over this NONE mask, which by definition lies
                    outside every GT mask (IoS < NONE_IOS_MAX) -- two disjoint regions, i.e. possibly two calves. j's
                    animal (t_on_gt) and the teacher-segment identity then name this mask only when a cand mask
                    OVERLAPPING it holds j (one cand animal spans both regions, the kept duplicate case); otherwise they
                    would PASS a wrong link Y -> Z as 'same animal', so no identity. A ghost GT mask j (not an animal,
                    review r1 finding 1) is no second animal: the teacher key decides, as before. Returns the evidence or
                    None."""
                    j = t2g.get(tk, -1)
                    held_over = False
                    for k2 in range(N_SLOTS):
                        if k2 == k or CR[k2] is None or not CR[k2].area: continue
                        holds = j >= 0 and cs[k2] in (C_ASSIGNED, C_SPLIT) and cg[k2] == j
                        if (bits >> PAIR_IDX[(min(k, k2), max(k, k2))]) & 1:
                            held_over = held_over or holds; continue
                        if holds: return dict(slot=k2, via="paired_gt_mask", gt_slot=j)
                        if ios_r(CR[k2], TR[tk]) >= ASSOC_IOS: return dict(slot=k2, via="teacher_mask")
                    if j >= 0 and not gh[j] and not held_over: return dict(slot=None, via="paired_gt_mask_unheld", gt_slot=j)
                    return None
                for k in range(N_SLOTS):
                    if cs[k] != C_NONE: continue
                    tk, tio, tiu = match_best(CR[k], TR)
                    rec = dict(row=row0 + r_i, slot=k, t=t, tslot=tk, t_ios=round(tio, 4), t_iou=round(tiu, 4),
                               tkey=None, t_on_gt=None, why=None if tk >= 0 else "no_teacher_mask")
                    if tk >= 0:
                        sp = t_span(k, tk)
                        if sp is not None:
                            # review r3: neither the GT identity of the paired GT mask nor the teacher-segment identity
                            # (both may name the OTHER calf) -- the fill is teacher_unknown, its action unknown (REVIEW)
                            rec.update(why=("teacher_mask_spans_a_gt_animal_and_this_mask" if sp["via"] == "paired_gt_mask_unheld"
                                            else "teacher_mask_spans_two_cand_animals"), t_span=sp, tkey_dropped=list(tkey(tk)))
                            fills.append(rec); continue
                        rec["tkey"] = tkey(tk)
                        # the teacher mask is a GT-visible animal with a TRUSTED GT identity (not inside an identity
                        # exclusion, not within ±5 s of a GT merge; else the teacher key decides) -- review r1 finding 1.
                        # After t_span (review r3) this is left only where a cand mask OVERLAPPING this one holds that GT
                        # mask (one cand animal over both regions: the duplicate the step-2 identity rule must catch)
                        if tk in t2g and a["g_anim"][r_i, t2g[tk]] >= 0 and a["g_idt"][r_i, t2g[tk]]:
                            rec["t_on_gt"] = int(a["g_anim"][r_i, t2g[tk]])
                    fills.append(rec)
            r_i += 1
        while g_cur is not None: n_g += 1; g_cur = g_next()
        g_rows_scope += n_g; g_rows_aligned += n_al
        clips_meta.append(dict(stem=stem, cand=True, gt=gp is not None, teacher=TS is not None, rows=r_i, gt_rows=n_g,
                               aligned=n_al, smoke=bool(CS.header.get("smoke")) if "smoke" in CS.header else None,
                               header={k: v for k, v in CS.header.items() if k != "frames"}))
        if r_i: chunks.append(B.finish(r_i))
        row0 += r_i
        del B, CS, GS, TS
    A = {k: (np.concatenate([ch[k] for ch in chunks]) if chunks else np.zeros((0,) + shp, dt_))
         for k, (shp, dt_, _) in ROW_SPEC.items()}
    nrow = len(A["ts"])
    phase = np.array([PHASES.index(phase_of_ts(t)) for t in A["ts"]], np.int8) if nrow else np.zeros(0, np.int8)
    strat = np.zeros(nrow, np.int8)                 # 0 main, 1 late stratum, 2 identity-excluded
    if nrow:
        for a, b in excl: strat[(A["ts"] >= a) & (A["ts"] < b)] = 2
        if late: strat[(A["ts"] >= late[0]) & (A["ts"] < late[1])] = 1
    geo_ex = np.zeros((nrow, 4), bool)
    if nrow:
        for a, b in excl + geo_c: geo_ex[(A["ts"] >= a) & (A["ts"] < b), :] = True
        for s, a, b in geo_slot: geo_ex[(A["ts"] >= a) & (A["ts"] < b), s] = True
    fps = max(fps_seen, key=fps_seen.get) if fps_seen else 5.0
    stems_c = sorted(cfiles)
    res = dict(arm=arm, day=day, cand_dir=cand_dir, gt_dir=gt_dir, light=light, window=win, fps=fps,
               fps_seen={str(k): v for k, v in fps_seen.items()}, sizes={str(k): v for k, v in sizes.items()},
               clips=clips_meta, A=A, phase=phase, strat=strat, geo_ex=geo_ex,
               dawn_mask=(A["ts"] >= dawn[0]) & (A["ts"] < dawn[1]) if nrow else np.zeros(0, bool), dawn_range=dawn,
               gt_mode=("docx" if docx is not None else ("segment_assignments" if gt_dir else None)),
               gt_docx=docx.info() if docx is not None else None,
               done=dict(present=bool(done_marks), n=len(done_marks),
                         missing=[s for s in stems_c if clip_key(s) not in set(map(clip_key, done_marks))] if done_marks else None,
                         note=None if done_marks else "no .done_ markers in the local copy (not evaluated)"),
               cseg=[dict(r) for r in cseg.recs],
               gseg=[{k: v for k, v in r.items() if k != "raw"} for r in gseg.recs] if gseg else [],
               gt_anim_of_seg=gt_anim_of_seg.tolist(),
               cev=cev, gev=[e for e in (gev or []) if e.get("type") in ("merge_resolved",)],
               first_link=first_link, rows_without_segment=seg_nomask, fills=fills,
               tvotes={json.dumps(k): dict(v) for k, v in tvotes.items()},
               teacher_mode=("rules" if trules is not None else ("transfer" if tseg is not None else None)),
               ghost_rule=dict(on=ghost_on, a_ref=a_ref_g, a_ref_source=ghost_src, frag_lo=frag_lo_j,
                               lo_px={k: round(v, 1) for k, v in ghost_lo.items()},
                               rule="GT mask of an UNASSIGNED segment with area < frag_lo[pool] x a_ref is not an animal"),
               precondition=dict(gt_rows_in_scope=g_rows_scope, aligned=g_rows_aligned,
                                 frac=(g_rows_aligned / g_rows_scope) if g_rows_scope else None,
                                 cand_clips=len(cfiles), gt_clips_in_scope=len([s for s in gfiles if in_window(s, win)]),
                                 missing_cand_clips=[m["stem"] for m in clips_meta if m["gt"] and not m["cand"]]),
               scan_s=round(time.time() - t_start, 1))
    return res

# ============================================================================================================
# 7 · metrics that need no identity layer
# ============================================================================================================
def seg_mech(rec, ev_by_seg=None):
    """the mechanism that opened a segment (attribution, §6.3-6)"""
    r = rec["open_reason"] or ""
    via = rec.get("via") or ((ev_by_seg or {}).get(rec["sid"]) or {}).get("via")
    if r == "dark_link": return "dark_link"
    if via == "dark_pass" or ((ev_by_seg or {}).get(rec["sid"]) or {}).get("dark_open"): return "dark_pass_open"
    if r in ("reopen",): return "scheduled_reopen"
    if r in ("displaced",): return "scheduled_displacement"
    if r in ("partial_repair", "growth_repair"): return "repair"
    if r == "self_recovery": return "sam3_self_recovery"
    if r in ("seed_day", "seed_night"): return "seeding"
    return f"other:{r}"

def events_by_seg(cev):
    out = defaultdict(dict)
    for e in cev:
        t = e.get("type")
        if t == "segment_open" and e.get("via"): out[e.get("seg_id")]["via"] = e.get("via")
        if t == "dark_open" and e.get("seg_id"): out[e["seg_id"]]["dark_open"] = True
        if t == "dark_link" and e.get("seg_id"):
            out[e["seg_id"]]["dark_link"] = e
    return out

def gap_metrics(S):
    """port of gaps_analyze.py (episodes per slot, merged across < 1 s of presence, pre-seed excluded) +
    the dark time after the first spare detection (scheduled `pass` n_det - matched, v5 `dark_pass` unclaimed)"""
    A = S["A"]; W = A["ts"]; N = len(W)
    if N == 0: return dict(episodes=0, hours=0.0)
    dr = S.get("dawn_range")
    def in_dawn(t): return bool(dr is not None and dr[0] <= t < dr[1])   # dawn 05:44-08:00 overlays night/day
    pres = A["c_area"] > 0
    cev = S["cev"]; evs = events_by_seg(cev)
    passes = sorted([e for e in cev if e.get("type") == "pass" and e.get("_t") is not None], key=lambda e: e["_t"])
    darkp = sorted([e for e in cev if e.get("type") == "dark_pass" and e.get("_t") is not None], key=lambda e: e["_t"])
    pw = [e["_t"] for e in passes]; dw = [e["_t"] for e in darkp]
    collapses = [e for e in cev if e.get("type") == "mask_area_collapse" and e.get("_t") is not None]
    merges = [e for e in cev if e.get("type") == "merge_resolved" and e.get("_t") is not None]
    closes = [e for e in cev if e.get("type") == "segment_close" and e.get("_t") is not None]
    cseg = S["cseg"]
    eps, preseed = [], []
    for s in range(N_SLOTS):
        p = pres[:, s]
        # runs of absent rows
        runs = []
        idx = np.flatnonzero(~p)
        if len(idx):
            br = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([idx[0]], idx[br + 1])); ends = np.concatenate((idx[br], [idx[-1]]))
            runs = [[int(a), int(b)] for a, b in zip(starts, ends)]
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] - 1 > 0 and W[r[0]] - W[merged[-1][1] + 1] < GAP_MERGE_S:
                merged[-1][1] = r[1]
            else: merged.append(r)
        for a, b in merged:
            if a == 0: preseed.append(dict(slot=s, dur=float(W[b] - W[0]) + 1.0 / S["fps"])); continue
            post = b + 1 if b + 1 < N else None
            dur = float(W[post] - W[a]) if post is not None else float(W[b] - W[a] + 1.0 / S["fps"])
            eps.append(dict(slot=s, a=a, b=b, post=post, dur=dur))
    out_eps = []
    for e in eps:
        s, a, b, post = e["slot"], e["a"], e["b"], e["post"]
        wa = float(W[a]); wb = float(W[post]) if post is not None else float(W[b]) + 1.0 / S["fps"]
        cause = None
        if any(c.get("slot") == s and wa - 3.0 <= c["_t"] <= wa + 1.0 for c in collapses): cause = "collapse_guard"
        elif any(s in (m.get("jumpers") or []) and abs(m["_t"] - wa) <= 1.5 for m in merges): cause = "merge_close"
        else:
            cl = [c for c in closes if c.get("slot") == s and abs(c["_t"] - wa) <= 1.5]
            cause = ("segment_close:" + str(cl[0].get("reason"))) if cl else "sam3_no_mask"
        pre_seg = int(A["c_seg"][a - 1, s])
        if post is None: end_how = "day_end"
        else:
            ps = int(A["c_seg"][post, s])
            end_how = "sam3_resume_same_segment" if ps == pre_seg else (seg_mech(cseg[ps], evs) if ps >= 0 else "no_segment")
        # passes inside the episode (gaps_analyze: pw >= wa and <= wb + 0.3)
        j = bisect.bisect_left(pw, wa); inside = [x for x in passes[j:] if x["_t"] <= wb + 0.3]
        jd = bisect.bisect_left(dw, wa); dinside = [x for x in darkp[jd:] if x["_t"] <= wb + 0.3]
        sp = []
        for x in inside:
            spare = int(x.get("n_det", 0)) - len(x.get("matched") or {})
            sp.append((x["_t"], spare, "pass"))
        for x in dinside:
            u = x.get("unclaimed"); spare = len(u) if isinstance(u, (list, dict)) else int(u or 0)
            sp.append((x["_t"], spare, "dark_pass"))
        sp.sort()
        fs = next((z for z in sp if z[1] >= 1), None)
        # counterfactual v4 end (APPROXIMATE): the first scheduled pass with a spare (day), the second (night, the
        # two-sighting rule); if none, the actual end
        sched = [z for z in sp if z[2] == "pass" and z[1] >= 1]
        need = 1 if tod_v4(wa) == "day" else 2
        cf_end = sched[need - 1][0] if len(sched) >= need else wb
        out_eps.append(dict(slot=s, t0=iso(wa), t1=iso(wb), dur=round(e["dur"], 2), bin=dur_bin(e["dur"]),
                            phase=phase_of_ts(wa), dawn=in_dawn(wa), cause=cause,
                            pre_seg=cseg[pre_seg]["sid"] if pre_seg >= 0 else None,
                            end_how=end_how, n_pass_in=len(inside), n_dark_pass_in=len(dinside),
                            first_spare_dt=None if fs is None else round(fs[0] - wa, 2),
                            first_spare_kind=None if fs is None else fs[2],
                            cf_v4_end_dt=round(cf_end - wa, 2), day_end=post is None))
    def agg(sel):
        return dict(n=len(sel), h=round(sum(x["dur"] for x in sel) / 3600, 4))
    fillable = [x for x in out_eps if x["cause"] != "merge_close" and not str(x["cause"]).startswith("segment_close:merge")]
    long_ = [x for x in out_eps if x["dur"] >= GAP_LONG_S]
    after = defaultdict(float); after_n = Counter()
    for x in long_:
        if x["first_spare_dt"] is not None:
            after[x["phase"]] += max(0.0, x["dur"] - x["first_spare_dt"]); after_n[x["phase"]] += 1
    step2_fill = defaultdict(float)
    for x in out_eps:
        if x["end_how"] in STEP2_MECH:
            step2_fill[x["phase"]] += max(0.0, x["cf_v4_end_dt"] - x["dur"])
            if x["dawn"]: step2_fill["dawn"] += max(0.0, x["cf_v4_end_dt"] - x["dur"])
    for x in long_:
        if x["dawn"] and x["first_spare_dt"] is not None:
            after["dawn"] += max(0.0, x["dur"] - x["first_spare_dt"]); after_n["dawn"] += 1
    return dict(episodes=len(out_eps), hours=round(sum(x["dur"] for x in out_eps) / 3600, 4),
                by_bin={b: agg([x for x in out_eps if x["bin"] == b]) for b in GAP_BINS},
                by_phase=dict({p: agg([x for x in out_eps if x["phase"] == p]) for p in PHASES},
                              dawn=agg([x for x in out_eps if x["dawn"]])),
                by_cause={c: agg([x for x in out_eps if x["cause"] == c]) for c in sorted({x["cause"] for x in out_eps})},
                by_end={c: agg([x for x in out_eps if x["end_how"] == c]) for c in sorted({x["end_how"] for x in out_eps})},
                fillable=agg(fillable),
                after_first_spare_h={p: round(after[p] / 3600, 4) for p in PHASES + ("dawn",)},
                after_first_spare_n={p: int(after_n[p]) for p in PHASES + ("dawn",)},
                step2_filled_vs_cf_v4_h={p: round(step2_fill[p] / 3600, 4) for p in PHASES + ("dawn",)},
                preseed=preseed, list=out_eps)

def segment_metrics(S):
    cseg = S["cseg"]; cev = S["cev"]; evs = events_by_seg(cev)
    A = S["A"]; hours = len(A["ts"]) / S["fps"] / 3600 if len(A["ts"]) else 0.0
    breaks = [r for r in cseg if r["open_reason"] not in ("seed_day", "seed_night")]
    cls = Counter(); orphan_list = []
    for r in breaks:
        m = seg_mech(r, evs)
        if m == "dark_link": cls["linked_dark"] += 1; continue
        fl = S["first_link"].get(r["idx"])
        ok = fl is not None and fl[0] is not None and fl[1] is not None and fl[1] <= ORPHAN_LINK_S and fl[2] >= ORPHAN_LINK_IOU
        if ok: cls["linked_spatial"] += 1
        else:
            cls["orphan"] += 1
            orphan_list.append(dict(sid=r["sid"], t=iso(r["t0"]), open_reason=r["open_reason"], mech=m,
                                    link=None if fl is None else dict(prev=None if fl[0] is None else cseg[fl[0]]["sid"], gap_s=fl[1], iou=fl[2])))
    by_reason = Counter((r["open_reason"] or "?") + (f"(via={r['via']})" if r.get("via") else "") for r in cseg)
    by_mech = Counter(seg_mech(r, evs) for r in cseg)
    def seg_dur(r):
        t1 = r["t1"] if math.isfinite(r["t1"]) else (float(A["ts"][-1]) if len(A["ts"]) else r["t0"])
        return t1 - r["t0"]
    submin = [r for r in cseg if seg_dur(r) < SUBMIN_S]
    churn_ep, churn_s = 0, 0.0
    for s in range(N_SLOTS):
        opens = sorted(r["t0"] for r in cseg if r["slot"] == s)
        cur = None
        for k in range(len(opens) - CHURN_N + 1):
            if opens[k + CHURN_N - 1] - opens[k] <= CHURN_WIN_S:
                a, b = opens[k], opens[k + CHURN_N - 1]
                if cur is not None and a <= cur[1]: cur[1] = max(cur[1], b)
                else:
                    if cur is not None: churn_ep += 1; churn_s += cur[1] - cur[0]
                    cur = [a, b]
        if cur is not None: churn_ep += 1; churn_s += cur[1] - cur[0]
    rebuilds = sum(1 for e in cev if e.get("type") in ("pass", "dark_pass") and e.get("rebuilt"))
    by_phase_breaks = Counter(phase_of_ts(r["t0"]) for r in breaks)
    dr = S.get("dawn_range")
    by_phase_breaks["dawn"] = sum(1 for r in breaks if dr is not None and dr[0] <= r["t0"] < dr[1])
    return dict(n=len(cseg), breaks=len(breaks), breaks_by_phase={p: by_phase_breaks.get(p, 0) for p in PHASES + ("dawn",)},
                by_open_reason=dict(sorted(by_reason.items())), by_mechanism=dict(sorted(by_mech.items())),
                linked_dark=cls["linked_dark"], linked_spatial=cls["linked_spatial"], orphan=cls["orphan"],
                sub_minute=len(submin), churn_episodes=churn_ep, churn_s=round(churn_s, 2),
                rebuilds=rebuilds, rebuilds_per_h=round(rebuilds / hours, 3) if hours else None,
                merge_resolved=sum(1 for e in cev if e.get("type") == "merge_resolved"),
                event_types=dict(sorted(Counter(e.get("type") for e in cev).items())),
                rows_without_segment=S["rows_without_segment"], orphan_list=orphan_list)

def fragment_metrics(S):
    """1 Hz rows (every 5th stored row of a clip, as tune_scout/quality.py): fragment / band shares by phase"""
    A = S["A"]
    if not len(A["ts"]): return {}
    sel = (A["pos"] % 5) == 0
    ph = S["phase"][sel]; ca = A["c_area"][sel]; ga = A["g_area"][sel] if "g_area" in A else None
    day = ph == 0
    own = ca[day][ca[day] > 0]
    own_med = float(np.median(own)) if len(own) else None
    gt_med = None; a_ref_eval = None
    if ga is not None and (ga > 0).any():
        gd = ga[day][ga[day] > 0]; gt_med = float(np.median(gd)) if len(gd) else None
        lg = ga[day].max(axis=1); lg = lg[lg > 0]; a_ref_eval = float(np.median(lg)) if len(lg) else None
    slot_med = {}
    for s in range(N_SLOTS):
        v = ca[day, s]; v = v[v > 0]
        slot_med[s] = float(np.median(v)) if len(v) else None
    out = dict(pooled_day_median_own=own_med, pooled_day_median_gt=gt_med, a_ref_eval_gt=a_ref_eval,
               slot_day_median_own=slot_med, rows_1hz=int(sel.sum()), by_phase={})
    # per-row band pool (day rows -> dayl, dusk/night rows -> dark: tol_v5_core.POOL); the dawn stratum 05:44-08:00
    # overlays night (05:44-06:00, dark pool) and day (06:00-08:00, dayl pool)
    dm = S.get("dawn_mask")
    dsel = dm[sel] if dm is not None and len(dm) == len(A["ts"]) else np.zeros(int(sel.sum()), bool)
    lo_row = np.where(ph == 0, FRAG_LO["dayl"], FRAG_LO["dark"])
    for pn, m in [(pn_, ph == pi_) for pi_, pn_ in enumerate(PHASES)] + [("dawn", dsel)]:
        a = ca[m]; pres = a > 0; n = int(pres.sum())
        if n == 0: out["by_phase"][pn] = dict(n_masks=0); continue
        v = a[pres]
        lo = np.broadcast_to(lo_row[m][:, None], a.shape)[pres]
        d = dict(n_masks=n,
                 frag_half_own_pct=round(100 * float((v < FRAG_HALF * own_med).sum()) / n, 3) if own_med else None,
                 frag_half_gt_pct=round(100 * float((v < FRAG_HALF * gt_med).sum()) / n, 3) if gt_med else None,
                 below_frag_lo_pct=round(100 * float((v < lo * a_ref_eval).sum()) / n, 3) if a_ref_eval else None,
                 above_over_hi_pct=round(100 * float((v > OVER_HI * a_ref_eval).sum()) / n, 3) if a_ref_eval else None)
        cnt = 0
        for s in range(N_SLOTS):
            vs = a[:, s]; vs = vs[vs > 0]
            if slot_med[s]: cnt += int((vs < FRAG_HALF * slot_med[s]).sum())
        d["frag_half_slot_own_pct"] = round(100 * cnt / n, 3)
        out["by_phase"][pn] = d
    return out

def _episodes_bool(mask, ts, fps):
    """runs of consecutive True rows (a row gap > 1.5 frame periods ends a run) -> [(i0, i1)]"""
    idx = np.flatnonzero(mask)
    if not len(idx): return []
    gap = np.diff(ts[idx]) > 1.5 / fps
    cut = np.flatnonzero((np.diff(idx) > 1) | gap)
    st = np.concatenate(([idx[0]], idx[cut + 1])); en = np.concatenate((idx[cut], [idx[-1]]))
    return list(zip(st.tolist(), en.tolist()))

def overlap_metrics(S):
    A = S["A"]; ts = A["ts"]; dt = 1.0 / S["fps"]
    out = dict(episodes_ge1s=0, seconds=0.0, by_phase={p: dict(n=0, s=0.0) for p in PHASES + ("dawn",)})
    if not len(ts): return out
    dm = S.get("dawn_mask")
    for pi in range(len(PAIRS)):
        m = (A["p_over"] >> pi) & 1
        for i0, i1 in _episodes_bool(m.astype(bool), ts, S["fps"]):
            d = float(ts[i1] - ts[i0]) + dt
            if d >= OVERLAP_MIN_S:
                out["episodes_ge1s"] += 1; out["seconds"] += d
                p = PHASES[S["phase"][i0]]; out["by_phase"][p]["n"] += 1; out["by_phase"][p]["s"] += d
                if dm is not None and dm[i0]: out["by_phase"]["dawn"]["n"] += 1; out["by_phase"]["dawn"]["s"] += d
    out["seconds"] = round(out["seconds"], 2)
    for p in PHASES + ("dawn",): out["by_phase"][p]["s"] = round(out["by_phase"][p]["s"], 2)
    return out

def label_metrics(S):
    """cand-mask time by association label, inside / outside the GT-dark stratum (a row whose GT holds < 4
    distinct REAL animals: ghost GT masks do not count, see A_REF_EVAL), and GT-animal time by coverage
    (animal-centric uncovered, §6.3-1). Per phase, 'all', 'dawn' and per SPEC §6.3 stratum: 'main' (what MH7 gates),
    'late' (04-23 23:30-24:00) and 'excluded' (the identity exclusions) -- the last two are reported only."""
    A = S["A"]; dt = 1.0 / S["fps"]
    if not len(A["ts"]): return {}
    has = A["has_gt"]; dark = has & (A["n_dist"] < N_SLOTS)
    out = {"cand_s": {}, "gt_animal_s": {}}
    dm = S.get("dawn_mask")
    strat = S.get("strat")
    if strat is None or len(strat) != len(A["ts"]): strat = np.zeros(len(A["ts"]), np.int8)
    for pi, pn in enumerate(PHASES + ("all", "dawn", "main", "late", "excluded")):
        if pn == "dawn":
            if dm is None: continue
            rm = has & dm
        elif pn in ("main", "late", "excluded"): rm = has & (strat == ("main", "late", "excluded").index(pn))
        else: rm = has if pn == "all" else (has & (S["phase"] == pi))
        d = {}
        for code, name in C_NAMES.items():
            if code == C_ABSENT: continue
            m = (A["c_stat"] == code) & rm[:, None]
            d[name] = round(float(m.sum()) * dt, 2)
            d[name + "_in_gt_dark"] = round(float((m & dark[:, None]).sum()) * dt, 2)
            d[name + "_out_gt_dark"] = round(float((m & ~dark[:, None]).sum()) * dt, 2)
        out["cand_s"][pn] = d
        g = {}
        for code, name in ((G_ASSIGNED, "assigned"), (G_MULTI, "multi_covered"), (G_UNCOVERED, "uncovered")):
            g[name] = round(float(((A["g_stat"] == code) & rm[:, None]).sum()) * dt, 2)
        out["gt_animal_s"][pn] = g
    out["gt_dark_rows_s"] = round(float(dark.sum()) * dt, 2)
    out["rows_without_gt_s"] = round(float((~has).sum()) * dt, 2)
    gg = A.get("g_ghost")
    if gg is not None and len(gg) == len(A["ts"]):
        out["gt_ghost_mask_s"] = {sn: round(float((gg & (strat == si)[:, None]).sum()) * dt, 2)
                                  for si, sn in enumerate(("main", "late", "excluded"))}
    nr = A.get("n_dist_raw")
    if nr is not None and len(nr) == len(A["ts"]):     # rows that are GT-dark ONLY because a ghost is not an animal
        out["gt_dark_rows_by_ghost_s"] = round(float((dark & (nr >= N_SLOTS)).sum()) * dt, 2)
    out["mh7_geometry"] = mh7_geometry(S)
    return out

def _row_arr(A, key, shape_tail, fill, dtype):
    """A[key] when the scan wrote it (same length as the rows), else a filled array (synthetic S dicts / old pickles)"""
    v = A.get(key)
    n = len(A["ts"])
    return v if v is not None and len(v) == n else np.full((n,) + shape_tail, fill, dtype)

def _geo_rows(S):
    """SPEC §6.3 geometry exclusions: (per (row, GT slot), rows where EVERY slot is excluded -- the ±30 s GT-defect windows
    and the identity exclusions --, rows where ANY slot is -- adds 04-22 slot 2 13:47-15:58)"""
    n = len(S["A"]["ts"]); g = S.get("geo_ex")
    if g is None or len(g) != n: g = np.zeros((n, N_SLOTS), bool)
    return g, g.all(axis=1), g.any(axis=1)

MH7_CLASSES = {1: "counted_teacher_agrees", 2: "counted_v4v2_only", 3: "excused_teacher_contradicts",
               4: "excused_gt_defect_window"}

def mh7_classes(S):
    """review r2 finding 2 -- per (row, cand slot) class of every MULTI / NONE cand mask on a GT row OUTSIDE GT-dark:
      0 not such a mask; COUNTED (MH7 gates it): 1 the teacher agrees with v4v2 on the geometry that makes the label,
      2 v4v2 is the only reference (no teacher row, or the teacher has nothing on a covered GT mask: conservative);
      EXCUSED (SPEC §6.3 'A geometry error counts only where the two agree' + the ±30 s geometry exclusions around the
      known GT defects): 3 the teacher contradicts v4v2 there, 4 a geometry exclusion covers the GT masks involved.
    MULTI: the GT masks it covers (c_cov) -- any in a geometry exclusion -> 4; any CONTRA -> 3; all AGREE -> 1; else 2.
    NONE: every slot excluded (the ±30 s windows) -> 4; c_ntag CONTRA -> 3; AGREE -> 1; no teacher row -> 4 if a
    per-slot exclusion covers the row (the excluded GT mask may be why v4v2 lacks the animal), else 2."""
    A = S["A"]; n = len(A["ts"])
    cls = np.zeros((n, N_SLOTS), np.int8)
    if not n: return cls
    outd = A["has_gt"] & (A["n_dist"] >= N_SLOTS)
    geo, gx_all, gx_any = _geo_rows(S)
    gbits = np.zeros(n, np.uint8)
    for j in range(N_SLOTS): gbits |= (geo[:, j].astype(np.uint8) << j)
    gtag = _row_arr(A, "g_tag", (N_SLOTS,), TAG_NO_ROW, np.int8)
    ntag = _row_arr(A, "c_ntag", (N_SLOTS,), TAG_NO_ROW, np.int8)
    cov = _row_arr(A, "c_cov", (N_SLOTS,), 0, np.uint8)
    for k in range(N_SLOTS):
        st = A["c_stat"][:, k]
        mu = (st == C_MULTI) & outd; no = (st == C_NONE) & outd
        cv = cov[:, k]
        contra = np.zeros(n, bool); agree_all = np.ones(n, bool)
        for j in range(N_SLOTS):
            bit = ((cv >> j) & 1).astype(bool)
            contra |= bit & (gtag[:, j] == TAG_CONTRA)
            agree_all &= ~bit | (gtag[:, j] == TAG_AGREE)
        c_mu = np.where((cv & gbits) != 0, 4, np.where(contra, 3, np.where(agree_all & (cv != 0), 1, 2)))
        c_no = np.where(gx_all, 4, np.where(ntag[:, k] == TAG_CONTRA, 3, np.where(ntag[:, k] == TAG_AGREE, 1,
                                                                                    np.where(gx_any, 4, 2))))
        cls[:, k] = np.where(mu, c_mu, np.where(no, c_no, 0))
    return cls

def mh7_geometry(S, max_ep=50):
    """review r2 finding 2: the MULTI / NONE seconds OUTSIDE GT-dark per stratum split by mh7_classes (counted_s = what
    MH7 gates), plus the counted and the teacher-contradicted episodes of the main stratum (for the renders, §6.4-7)"""
    A = S["A"]; n = len(A["ts"])
    if not n: return {}
    dt = 1.0 / S["fps"]
    cls = mh7_classes(S)
    strat = S.get("strat")
    if strat is None or len(strat) != n: strat = np.zeros(n, np.int8)
    out = dict(rule="SPEC §6.3: a geometry error counts only where v4v2 and the teacher agree (no teacher row / the teacher "
                    "dark on a covered GT mask = v4v2 only reference: counted); ±30 s GT-defect geometry exclusions excused")
    for si, sn in enumerate(("main", "late", "excluded")):
        rm = (strat == si)[:, None]
        d = {}
        for code, lab in ((C_MULTI, "MULTI"), (C_NONE, "NONE")):
            m = (A["c_stat"] == code) & rm
            cnt = {c: int(((cls == c) & m).sum()) for c in MH7_CLASSES}
            dd = {nm: round(cnt[c] * dt, 2) for c, nm in MH7_CLASSES.items()}
            dd["counted_s"] = round((cnt[1] + cnt[2]) * dt, 2)
            dd["excused_s"] = round((cnt[3] + cnt[4]) * dt, 2)
            d[lab] = dd
        out[sn] = d
    main = strat == 0
    eps = {"counted": [], "excused_teacher_contradicts": []}
    for key, codes in (("counted", (1, 2)), ("excused_teacher_contradicts", (3,))):
        for k in range(N_SLOTS):
            m = np.isin(cls[:, k], codes) & main
            for i0, i1 in _episodes_bool(m, A["ts"], S["fps"]):
                lab = Counter(C_NAMES[int(x)] for x in A["c_stat"][i0:i1 + 1, k][m[i0:i1 + 1]]).most_common(1)[0][0]
                eps[key].append(dict(slot=k, t0=iso(A["ts"][i0]), t1=iso(A["ts"][i1]), s=round(float(A["ts"][i1] - A["ts"][i0]) + dt, 2),
                                     label=lab, classes=sorted({MH7_CLASSES[int(c)] for c in cls[i0:i1 + 1, k][m[i0:i1 + 1]]})))
        eps[key].sort(key=lambda e: -e["s"])
        out["episodes_main_" + key] = dict(n=len(eps[key]), longest=eps[key][:max_ep])
    return out

def mask_fidelity(S, a_ref_eval):
    """assigned cand masks vs GEOMETRY-TRUSTED GT masks (Tol_44.trust_filter band + no GT overlap + exclusions)"""
    A = S["A"]
    if not len(A["ts"]) or not a_ref_eval: return {}
    out = {}
    ratio = A["g_area"] / a_ref_eval
    lo_row = np.where(S["phase"] == 0, FRAG_LO["dayl"], FRAG_LO["dark"])[:, None]   # per-row pool (see fragments)
    dm = S.get("dawn_mask")
    for pi, pn in enumerate(PHASES + ("dawn",)):
        if pn == "dawn":
            if dm is None: continue
            rm = dm & A["has_gt"]
        else: rm = (S["phase"] == pi) & A["has_gt"]
        trusted = (A["g_area"] > 0) & (ratio >= lo_row) & (ratio <= OVER_HI) & (A["g_gg"] < TRUST_GT_IOS) & ~S["geo_ex"] & rm[:, None]
        vals = []
        for j in range(N_SLOTS):
            rows = np.flatnonzero(trusted[:, j] & (A["g_stat"][:, j] == G_ASSIGNED))
            k = A["g_c"][rows, j]
            vals.append(A["c_iou"][rows, k])
        v = np.concatenate(vals) if vals else np.zeros(0)
        out[pn] = dict(n_trusted=int(trusted.sum()), n_assigned=int(len(v)),
                       mean_iou=round(float(v.mean()), 4) if len(v) else None,
                       frac_iou_lt_0_5=round(float((v < 0.5).mean()), 4) if len(v) else None)
        # SPEC §6.3 'Geometry reference: v4v2 and the teacher. A geometry error counts only where the two agree':
        # the same trusted GT masks restricted to the 1 Hz rows where the teacher's paired mask agrees with v4v2
        # (IoU >= TEACHER_AGREE_IOU); a geometry error there = no assigned cand mask, or cand IoU < 0.5
        tg = A.get("g_tiou")
        if tg is not None and len(tg) == len(A["ts"]):
            seen = trusted & (tg >= 0.0)
            agree = seen & (tg >= TEACHER_AGREE_IOU)
            va = []
            for j in range(N_SLOTS):
                rows = np.flatnonzero(agree[:, j] & (A["g_stat"][:, j] == G_ASSIGNED))
                va.append(A["c_iou"][rows, A["g_c"][rows, j]])
            va = np.concatenate(va) if va else np.zeros(0)
            n_ag = int(agree.sum())
            out[pn]["teacher_agree"] = dict(
                n_teacher_seen=int(seen.sum()), n_agree=n_ag, n_assigned=int(len(va)),
                mean_iou=round(float(va.mean()), 4) if len(va) else None,
                geometry_errors=int((n_ag - len(va)) + int((va < 0.5).sum())),
                geometry_error_frac=round(float((n_ag - len(va)) + int((va < 0.5).sum())) / n_ag, 4) if n_ag else None)
    return out

def segment_rows(S):
    """cand segment idx -> sorted row indices (its slot's rows carrying that segment)"""
    A = S["A"]; out = defaultdict(list)
    for k in range(N_SLOTS):
        cs = A["c_seg"][:, k]; pres = A["c_area"][:, k] > 0
        idx = np.flatnonzero(pres & (cs >= 0))
        if not len(idx): continue
        order = np.argsort(cs[idx], kind="stable")
        sv = cs[idx][order]; iv = idx[order]
        cuts = np.flatnonzero(np.diff(sv)) + 1
        for grp_s, grp_i in zip(np.split(sv, cuts), np.split(iv, cuts)):
            out[int(grp_s[0])] = np.sort(grp_i)
    return out

def cand_animal(S, rows, k):
    """GT animal (week-global) under cand mask k at `rows` (-1 unknown): assigned + GT identity known"""
    A = S["A"]
    st = A["c_stat"][rows, k]; gs = A["c_gs"][rows, k].astype(np.int64)
    an = np.full(len(rows), -1, np.int64)
    ok = (st == C_ASSIGNED) & (gs >= 0)
    an[ok] = A["g_anim"][rows[ok], gs[ok]]
    an[an < 0] = -1
    return an

def trusted_animal(S, rows, k, fill_by_row=None):
    """cand_animal restricted to IDENTITY-TRUSTED evidence of the MAIN stratum (review r1 findings 1-2): the GT mask the
    cand mask is assigned to has g_idt (assigned GT segment, outside the identity exclusions, no GT merge_resolved
    within ±5 s -- SPEC §6.3 identity-trust) and the row is not in the late / excluded stratum ('cannot be shown').
    fill_by_row {row: {slot: animal}}: the teacher identity of a cand NONE mask on a GT-dark row (5f) where no GT
    identity exists (none where the teacher mask spans two disjoint cand masks, or a GT-visible animal no overlapping cand
    mask holds plus this mask: scan_day.t_span, review r3). -1 = unknown."""
    A = S["A"]
    an = cand_animal(S, rows, k)
    if not len(rows): return an
    gs = A["c_gs"][rows, k].astype(np.int64)
    ok = (an >= 0) & A["g_idt"][rows, np.maximum(gs, 0)]
    # review r2 (the finding-1 class, GT merged two calves): a GT mask that ANOTHER cand mask also points at (assigned or
    # split) while the two cand masks do NOT overlap (pairwise max(IoU, IoS) <= OVERLAP_THR, p_over bit clear) is one GT
    # mask over two cand animals -- the Hungarian hands its label to whichever mask wins the IoU tie-break, so it names
    # no identity on that row. Overlapping cand masks (a cand duplicate on one animal) keep their evidence.
    pov = _row_arr(A, "p_over", (), 0, np.uint8)[rows]
    car = _row_arr(A, "c_area", (N_SLOTS,), 1, np.int32)[rows]
    for k2 in range(N_SLOTS):
        if k2 == k: continue
        s2 = A["c_stat"][rows, k2]
        pb = PAIR_IDX[(min(k, k2), max(k, k2))]
        ok &= ~((car[:, k2] > 0) & ((s2 == C_ASSIGNED) | (s2 == C_SPLIT)) & (A["c_gs"][rows, k2] == gs)
                & (((pov >> pb) & 1) == 0))
    an = np.where(ok, an, -1)
    if fill_by_row:
        for q, r in enumerate(rows):
            if an[q] < 0:
                v = fill_by_row.get(int(r), {}).get(k)
                if v is not None: an[q] = int(v)
    st = S.get("strat")
    if st is not None and len(st) == len(A["ts"]): an[st[rows] != 0] = -1
    return an

def qualifying_runs(ts, anim, fps, min_s=SWITCH_MIN_S):
    """the qualifying runs of detect_switches: [(animal, t_first, t_last)] of the same known animal (unknown rows
    neutral) lasting >= min_s"""
    runs = []
    for t, a in zip(ts, anim):
        if a < 0: continue
        if runs and runs[-1][0] == a: runs[-1][2] = float(t)
        else: runs.append([int(a), float(t), float(t)])
    return [tuple(r) for r in runs if (r[2] - r[1]) + 1.0 / fps >= min_s]

def detect_switches(ts, anim, fps, min_s=SWITCH_MIN_S):
    """runs of the same known animal (unknown rows neutral); a switch = consecutive qualifying (>= min_s) runs of
    different animals. Returns [(t_prev_end, t_next_start, a_prev, a_next, bracketed)]"""
    runs = []
    for t, a in zip(ts, anim):
        if a < 0: continue
        if runs and runs[-1][0] == a: runs[-1][2] = t; runs[-1][3] += 1
        else: runs.append([int(a), t, t, 1])
    q = [r for r in runs if (r[2] - r[1]) + 1.0 / fps >= min_s]
    out = []
    for p, n in zip(q, q[1:]):
        if p[0] != n[0]:
            # bracketed: the change happened across rows with no known animal (a GT-dark / unassigned span)
            known_between = [r for r in runs if r[1] > p[2] and r[2] < n[1]]
            out.append((p[2], n[1], p[0], n[0], len(known_between) == 0 and (n[1] - p[2]) > 1.5 / fps))
    return out

def switch_metrics(S, segrows):
    """5a on identity-TRUSTED main-stratum evidence (trusted_animal): the identity exclusions and the late stratum are
    GT windows whose identities are known to be unreliable, so a GT animal change there is not a switch; a change
    ACROSS them is still found (bracketed) -- same false-FAIL class as review r1 finding 1, applied to MH2"""
    A = S["A"]; cseg = S["cseg"]; evs = events_by_seg(S["cev"]); out = []
    for si, rows in segrows.items():
        rec = cseg[si]; k = rec["slot"]
        an = trusted_animal(S, rows, k)
        for tp, tn, ap, anx, br in detect_switches(A["ts"][rows], an, S["fps"]):
            mech = seg_mech(rec, evs)
            out.append(dict(sid=rec["sid"], slot=k, t_from=iso(tp), t_to=iso(tn), gt_from=ap, gt_to=anx, bracketed=bool(br),
                            phase=phase_of_ts(tn), seg_open_mech=mech, mechanism="sam3_in_segment_hop",
                            step2=mech in STEP2_MECH, since_open_s=round(tn - rec["t0"], 1)))
    return dict(n=len(out), n_bracketed=sum(1 for x in out if x["bracketed"]),
                n_step2=sum(1 for x in out if x["step2"]), list=out)

def seed_mapping(S, segrows, lab=None):
    """SPEC §6.3-5b: Tol_30 bootstraps label = slot of each seed segment; map label -> GT animal by the seed's
    majority over its first 10 min; fallback Hungarian over first-hour votes (by Tol_30 label when available)"""
    A = S["A"]; cseg = S["cseg"]
    seeds = [r for r in cseg if r["open_reason"] in ("seed_day", "seed_night")]
    info = dict(method=None, seeds=[], map={}, note=None)
    if seeds:
        t_first = min(r["t0"] for r in seeds)
        first = [r for r in seeds if r["t0"] <= t_first + 1.0]
        maj = {}
        for r in first:
            rows = segrows.get(r["idx"], np.zeros(0, int))
            rows = rows[A["ts"][rows] <= r["t0"] + SEED_MAP_S] if len(rows) else rows
            an = cand_animal(S, rows, r["slot"]) if len(rows) else np.zeros(0, int)
            gs = A["c_gs"][rows, r["slot"]].astype(int) if len(rows) else np.zeros(0, int)
            trust = np.array([A["g_idt"][rr, g] if g >= 0 else False for rr, g in zip(rows, gs)], bool) if len(rows) else np.zeros(0, bool)
            v = Counter(int(a) for a, tr in zip(an, trust) if a >= 0 and tr)
            m = v.most_common(2)
            maj[r["slot"]] = m[0][0] if m and (len(m) == 1 or m[0][1] > m[1][1]) else None
            info["seeds"].append(dict(sid=r["sid"], slot=r["slot"], votes=dict(v), majority=maj[r["slot"]]))
        if len(first) == N_SLOTS and None not in maj.values() and len(set(maj.values())) == N_SLOTS:
            info.update(method="seed_majority_10min", map={int(k): int(v) for k, v in maj.items()})
            return info
    # fallback: Hungarian over first-hour votes, by label when available else by slot
    if not len(A["ts"]): return info
    t0 = float(A["ts"][0]); rows = np.flatnonzero(A["ts"] <= t0 + FIRST_HOUR_S)
    V = np.zeros((N_SLOTS, N_SLOTS))
    for k in range(N_SLOTS):
        an = cand_animal(S, rows, k)
        L = lab[rows, k] if lab is not None else np.full(len(rows), k)
        gs = A["c_gs"][rows, k].astype(int)
        for r_, a, l, g in zip(rows, an, L, gs):
            if a >= 0 and l >= 0 and g >= 0 and A["g_idt"][r_, g]: V[int(l), int(a)] += 1
    ri, cj = linear_sum_assignment(-V)
    info.update(method="hungarian_first_hour" + ("_by_label" if lab is not None else "_by_slot"),
                map={int(a): int(b) for a, b in zip(ri, cj) if V[a, b] > 0}, votes=V.astype(int).tolist(),
                note="seed-majority rule not applicable (fewer than 4 first seeds, an unknown or a duplicated majority)")
    return info

# ============================================================================================================
# 8 · Tol_30 on a local scoring copy (sha pinned; the same snapshot for both arms)
# ============================================================================================================
def tol30_snapshot(src, work, want_sha=None):
    os.makedirs(work, exist_ok=True)
    h = sha256_file(src)
    if want_sha and not h.startswith(want_sha.lower()):
        raise RuntimeError(f"Tol_30 sha {h} does not match the pinned {want_sha}")
    dst = os.path.join(work, f"Tol_30_snapshot_{h[:12]}.py")
    if not os.path.exists(dst): shutil.copy2(src, dst)
    if sha256_file(dst) != h: raise RuntimeError("Tol_30 snapshot copy differs from its source")
    txt = open(dst, encoding="utf-8").read()
    m = re.search(r"REVIEW_FLAGS\s*=\s*\(([^)]*)\)", txt, re.S)
    flags = tuple(re.findall(r'"([^"]+)"', m.group(1))) if m else REVIEW_FLAGS_T30
    mt = re.search(r"^TAU\s*=\s*([0-9.]+)", txt, re.M)
    return dict(path=dst, src=src, sha256=h, review_flags=flags, review_flags_match_builtin=(flags == REVIEW_FLAGS_T30),
                tau=float(mt.group(1)) if mt else None)

def transferred_gt(S, segrows, seed_map):
    """cand segment -> GT animal by majority of its identity-trusted assigned rows, expressed in the cand LABEL
    space (inverse of the seed map) so Tol_30's own printout is meaningful. Informational for Tol_30 only."""
    inv = {v: k for k, v in (seed_map or {}).items()}
    A = S["A"]; out = {}
    for r in S["cseg"]:
        rows = segrows.get(r["idx"])
        if rows is None or not len(rows): out[r["sid"]] = None; continue
        an = cand_animal(S, rows, r["slot"]); gs = A["c_gs"][rows, r["slot"]].astype(int)
        v = Counter(int(a) for rr, a, g in zip(rows, an, gs) if a >= 0 and g >= 0 and A["g_idt"][rr, g])
        if not v: out[r["sid"]] = None; continue
        a = v.most_common(1)[0][0]
        out[r["sid"]] = int(inv[a]) if inv and a in inv else (None if inv else int(a))
    return out

def run_tol30(snap, arm_root, day_dir, day, work_arm, gt_asg, T=3.0, bands="crop_bands3_dense.npz", log=print):
    """copy the Tol_30 inputs of one day to work_arm/<day>/ and run the production call there"""
    dst = os.path.join(work_arm, day); os.makedirs(dst, exist_ok=True)
    src = os.path.join(arm_root, day_dir)
    need = ["segments.json", "events.jsonl", "observations_meta.json", bands]
    miss = [f for f in need if not os.path.exists(os.path.join(src, f))]
    if miss: return dict(ran=False, why=f"missing Tol_30 inputs in {src}: {miss}")
    for f in need:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s) and sha256_file(d) == sha256_file(s)):
            shutil.copy2(s, d)
    for bad in ("segment_assignments.json", "segment_assignments_reviewed.json"):
        if os.path.exists(os.path.join(dst, bad)): raise RuntimeError(f"{bad} must never exist in a scoring copy ({dst})")
    with open(os.path.join(dst, "segment_assignments_transferred.json"), "w") as f:
        json.dump({"day": day, "source": "Tol_48 transfer (majority GT animal of each cand segment, cand label space)",
                   "assignments": gt_asg}, f, indent=1)
    cmd = [sys.executable, snap["path"], "--root", work_arm, "--day", day, "--T", str(int(T) if float(T).is_integer() else T),
           "--bands", bands, "--gt", "segment_assignments_transferred.json", "--out_dir", dst]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=dst)
    open(os.path.join(dst, f"tol30_{day}.log"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0: raise RuntimeError(f"Tol_30 failed on {dst}:\n{p.stderr[-2000:]}")
    tag = f"T{int(T) if float(T).is_integer() else T}"
    out = os.path.join(dst, f"periodic_identity_{day}_{tag}.json")
    if sha256_file(snap["path"]) != snap["sha256"]: raise RuntimeError("Tol_30 snapshot changed during the run")
    vs = [l.strip() for l in p.stdout.splitlines() if "vs GT:" in l]
    return dict(ran=True, cmd=" ".join(cmd), runtime_s=round(time.time() - t0, 1), out=out, sha256=snap["sha256"],
                stdout_vs_gt=vs[0] if vs else None, inputs_sha256={f: sha256_file(os.path.join(dst, f)) for f in need})

def run_tol30_week(snap, work_arm, days, week_map_path, T=3.0, bands="crop_bands3_dense.npz"):
    out_dir = os.path.join(work_arm, "week"); os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, snap["path"], "--root", work_arm, "--days", ",".join(days), "--week_map", week_map_path,
           "--T", str(int(T) if float(T).is_integer() else T), "--bands", bands, "--out_dir", out_dir]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
    open(os.path.join(out_dir, "tol30_week.log"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0: raise RuntimeError(f"Tol_30 week mode failed:\n{p.stderr[-2000:]}")
    return dict(ran=True, cmd=" ".join(cmd), runtime_s=round(time.time() - t0, 1),
                out=os.path.join(out_dir, "periodic_identity_WEEK.json"), sha256=snap["sha256"])

def load_t30_runs(path, T=3.0, key_prefix=""):
    """{sid: [run dict(t0, t1, label, flags, contested=[(a, b)])]} from a Tol_30 periodic_identity JSON"""
    J = json.load(open(path, encoding="utf-8"))
    runs = {}
    for sid, rl in J["runs"].items():
        if key_prefix:
            if not sid.startswith(key_prefix): continue
            sid = sid[len(key_prefix):]
        lst = [dict(t0=ts_of(datetime.fromisoformat(r["start"])), t1=ts_of(datetime.fromisoformat(r["end"])),
                    label=r["label"], flags=set(r.get("flags") or []), contested=[]) for r in rl]
        lst.sort(key=lambda z: z["t0"])
        runs[sid] = lst
    W = T * 60.0
    for d in J.get("decisions", []):
        if d.get("status", "ok") == "ok": continue
        sid = d["sid"]
        if key_prefix:
            if not sid.startswith(key_prefix): continue
            sid = sid[len(key_prefix):]
        rl = runs.get(sid)
        if not rl: continue
        t = ts_of(datetime.fromisoformat(d["t"]))
        pick = rl[0]
        for r in rl:
            if r["t0"] <= t < r["t1"]: pick = r; break
            if r["t0"] <= t: pick = r
        pick["contested"].append((t, t + W))
    return runs, J

def label_arrays(S, segrows, runs, review_flags):
    """per row x slot: Tol_30 label (-1 = '?', -2 = no run) and 'flagged' (review flag or contested window)"""
    A = S["A"]; n = len(A["ts"])
    lab = np.full((n, N_SLOTS), -2, np.int8); flg = np.zeros((n, N_SLOTS), bool)
    RF = set(review_flags)
    for si, rows in segrows.items():
        rec = S["cseg"][si]; k = rec["slot"]; rl = runs.get(rec["sid"])
        if not rl: continue
        t = A["ts"][rows]
        st = np.array([r["t0"] for r in rl]); en = np.array([r["t1"] for r in rl])
        L = np.array([-1 if r["label"] is None else int(r["label"]) for r in rl], np.int8)
        F = np.array([bool(r["flags"] & RF) for r in rl])
        j = np.searchsorted(st, t + SEG_TOL_S, side="right") - 1
        j = np.where((j < 0) & (t >= st[0] - SEG_TOL_S), 0, j)          # a row just before the first run start
        ok = j >= 0
        jj = np.maximum(j, 0)
        last = jj == len(rl) - 1
        ok &= (t < en[jj]) | (last & (t <= en[jj] + SEG_TOL_S))
        rr = rows[ok]; jv = jj[ok]; tv = t[ok]
        lab[rr, k] = L[jv]
        f = F[jv].copy()
        for q, r in enumerate(rl):                                         # contested T-min windows of each run
            for a, b in r["contested"]:
                f |= (jv == q) & (tv >= a) & (tv < b)
        flg[rr, k] = f
    return lab, flg

def identity_classes(S, lab, flg, amap, mech_of_seg):
    """every identity-trusted GT animal-frame -> correct / wrong_flagged / wrong_silent / '?' / uncovered, per
    stratum; wrong frames attributed to the opening mechanism of the cand segment that holds them"""
    A = S["A"]; dt = 1.0 / S["fps"]
    n = len(A["ts"])
    if not n: return {}
    mp = np.full(N_SLOTS + 2, -1, np.int64)
    for l, a in (amap or {}).items(): mp[int(l)] = int(a)
    cls = np.full((n, N_SLOTS), -1, np.int8)      # -1 not scored; 0 correct 1 wrong_flagged 2 wrong_silent 3 '?' 4 uncovered
    tr = A["g_idt"] & (A["g_area"] > 0)
    k = A["g_c"].astype(np.int64)
    rows, js = np.nonzero(tr)
    kk = k[rows, js]; has = kk >= 0; kc = np.maximum(kk, 0)
    L = np.where(has, lab[rows, kc].astype(np.int64), -1)
    mapped = np.where(L >= 0, mp[np.maximum(L, 0)], -1)
    ga = A["g_anim"][rows, js].astype(np.int64)
    fl = flg[rows, kc]
    v = np.full(len(rows), 4, np.int8)
    v[has & (mapped < 0)] = 3
    v[has & (mapped >= 0) & (mapped == ga)] = 0
    v[has & (mapped >= 0) & (mapped != ga) & fl] = 1
    v[has & (mapped >= 0) & (mapped != ga) & ~fl] = 2
    cls[rows, js] = v
    names = ("correct", "wrong_flagged", "wrong_silent", "q", "uncovered")
    strata = {}
    dark = A["has_gt"] & (A["n_dist"] < N_SLOTS)
    for sn, rm in (("main", S["strat"] == 0), ("late", S["strat"] == 1), ("excluded", S["strat"] == 2),
                   ("day", (S["strat"] == 0) & (S["phase"] == 0)), ("dusk", (S["strat"] == 0) & (S["phase"] == 1)),
                   ("night", (S["strat"] == 0) & (S["phase"] == 2)), ("dawn", (S["strat"] == 0) & S["dawn_mask"]),
                   ("gt_dark_rows", (S["strat"] == 0) & dark)):
        d = {nm: round(float(((cls == ci) & rm[:, None]).sum()) * dt / 60.0, 3) for ci, nm in enumerate(names)}
        d["trusted_min"] = round(float((tr & rm[:, None]).sum()) * dt / 60.0, 3)
        d["wrong_min"] = round(d["wrong_flagged"] + d["wrong_silent"], 3)
        lab_ = d["correct"] + d["wrong_min"]
        d["acc_labelled"] = round(d["correct"] / lab_, 5) if lab_ else None
        strata[sn] = d
    # attribution + wrong spans (main stratum)
    by_mech = defaultdict(lambda: dict(wrong_flagged=0.0, wrong_silent=0.0))
    spans = []
    main = S["strat"] == 0
    for j in range(N_SLOTS):
        rr = np.flatnonzero(((cls[:, j] == 1) | (cls[:, j] == 2)) & main)
        cur = None
        for r in rr:
            c = int(k[r, j]); si = int(A["c_seg"][r, c]); m = mech_of_seg.get(si, "unknown")
            kind = "wrong_flagged" if cls[r, j] == 1 else "wrong_silent"
            by_mech[m][kind] += dt / 60.0
            key = (c, si, int(lab[r, c]), int(A["g_anim"][r, j]), kind)
            if cur and cur["key"] == key and A["ts"][r] - cur["t1"] <= 2.0: cur["t1"] = float(A["ts"][r]); cur["n"] += 1
            else:
                cur = dict(key=key, t0=float(A["ts"][r]), t1=float(A["ts"][r]), n=1); spans.append(cur)
    span_list = []
    for s_ in spans:
        c, si, L, a, kind = s_["key"]
        span_list.append(dict(slot=c, sid=S["cseg"][si]["sid"] if si >= 0 else None, t0=iso(s_["t0"]), t1=iso(s_["t1"]),
                              minutes=round(s_["n"] * dt / 60.0, 3), label=L, label_maps_to=int(mp[L]) if L >= 0 else None,
                              gt=a, kind=kind, mechanism=mech_of_seg.get(si, "unknown"),
                              step2=mech_of_seg.get(si) in STEP2_MECH))
    span_list.sort(key=lambda z: -z["minutes"])
    return dict(strata=strata, wrong_by_mechanism={m: {kk_: round(v, 3) for kk_, v in d.items()} for m, d in sorted(by_mech.items())},
                step2_wrong_silent_min=round(sum(d["wrong_silent"] for m, d in by_mech.items() if m in STEP2_MECH), 3),
                step2_wrong_min=round(sum(d["wrong_silent"] + d["wrong_flagged"] for m, d in by_mech.items() if m in STEP2_MECH), 3),
                wrong_spans=span_list[:200], n_wrong_spans=len(span_list), _cls=cls)

def slot_time_summary(S, runs, gt_asg_label_space, review_flags):
    """ablation/time_ab.py semantics (segment-level slot-time from the run intervals), for continuity with the
    HANDOFF figures; GT = the transferred assignment in the cand label space"""
    lab = corr = wrong = sil = abst = 0.0
    RF = set(review_flags)
    for r in S["cseg"]:
        g = gt_asg_label_space.get(r["sid"])
        s0, s1 = r["t0"], r["t1"]
        for x in runs.get(r["sid"], []):
            a0 = max(s0, x["t0"]); b0 = min(s1, x["t1"]); m = max(0.0, (b0 - a0) / 60.0)
            if x["label"] is None: abst += m; continue
            lab += m
            if g is None: continue
            if x["label"] == g: corr += m
            else:
                wrong += m
                if not (x["flags"] & RF): sil += m
    return dict(labelled_min=round(lab, 2), correct_min=round(corr, 2), wrong_min=round(wrong, 2),
                silent_min=round(sil, 2), abstain_min=round(abst, 2),
                acc=round(corr / (corr + wrong), 5) if (corr + wrong) else None)

# ============================================================================================================
# 9 · teacher fills (5f), step-2 action verdicts, sentinels
# ============================================================================================================
def teacher_map(S):
    """teacher key -> GT animal. rules mode (04-22): A-label -> GT by Hungarian over votes; transfer mode (04-23):
    each teacher segment -> its majority GT animal"""
    V = {tuple(json.loads(k)): Counter({int(a): n for a, n in v.items()}) for k, v in S["tvotes"].items()}
    if S["teacher_mode"] == "rules":
        keys = sorted({k[1] for k in V})
        M = np.zeros((len(keys), N_SLOTS))
        for i, a in enumerate(keys):
            for g, n in V[("A", a)].items(): M[i, g] = n
        if not len(keys): return {}, {}
        ri, cj = linear_sum_assignment(-M)
        mp = {("A", keys[i]): int(j) for i, j in zip(ri, cj) if M[i, j] > 0}
        purity = {str(keys[i]): round(float(M[i, j] / M[i].sum()), 4) for i, j in zip(ri, cj) if M[i].sum()}
        return mp, dict(mode="rules", map={str(k[1]): v for k, v in mp.items()}, purity=purity)
    mp, pur = {}, {}
    for k, c in V.items():
        a, n = c.most_common(1)[0]; tot = sum(c.values())
        mp[k] = int(a); pur[str(k[1])] = dict(animal=int(a), votes=int(tot), purity=round(n / tot, 4))
    return mp, dict(mode="transfer", segments=pur)

def fill_metrics(S, lab, amap, mech_of_seg):
    tm, tinfo = teacher_map(S)
    A = S["A"]; dt = 1.0 / S["fps"]
    mp = {int(l): int(a) for l, a in (amap or {}).items()}
    agg = defaultdict(lambda: defaultdict(float)); lst = []; whys = defaultdict(float)
    other = defaultdict(lambda: defaultdict(float))       # fills in the late / excluded strata: reported, not gated
    strat = S.get("strat")
    for f in S["fills"]:
        r, k = f["row"], f["slot"]
        si = int(A["c_seg"][r, k]); mech = mech_of_seg.get(si, "unknown")
        in_main = strat is None or int(strat[r]) == 0
        truth = None; why = None
        if f["t_on_gt"] is not None: why = "teacher_mask_is_a_gt_visible_animal"; truth = f["t_on_gt"]
        elif f["tkey"] is not None:
            key = tuple(f["tkey"]) if isinstance(f["tkey"], (list, tuple)) else f["tkey"]
            truth = tm.get(key)
            why = "teacher_identity" if truth is not None else "teacher_identity_unknown"
        else: why = f.get("why") or "no_teacher_mask"
        L = int(lab[r, k]) if lab is not None else -2
        if truth is None: c = "teacher_unknown"
        elif L < 0 or L not in mp: c = "q"
        elif mp[L] == truth: c = "correct"
        else: c = "wrong"
        if not in_main:
            other[mech][c] += dt
            if c == "wrong": lst.append(dict(t=iso(f["t"]), slot=k, sid=S["cseg"][si]["sid"] if si >= 0 else None, label=L,
                                              label_maps_to=mp.get(L), teacher=truth, mechanism=mech, why=why,
                                              stratum=("main", "late", "excluded")[int(strat[r])]))
            continue
        agg[mech][c] += dt
        whys[why] += dt
        if c == "wrong": lst.append(dict(t=iso(f["t"]), slot=k, sid=S["cseg"][si]["sid"] if si >= 0 else None, label=L,
                                          label_maps_to=mp.get(L), teacher=truth, mechanism=mech, why=why, stratum="main"))
    tot = defaultdict(float)
    for m, d in agg.items():
        for c, v in d.items(): tot[c] += v
    return dict(teacher=tinfo, fill_s=round(sum(tot.values()), 2), by_class_s={c: round(v, 2) for c, v in sorted(tot.items())},
                by_source_s={w: round(v, 2) for w, v in sorted(whys.items())},
                by_mechanism_s={m: {c: round(v, 2) for c, v in d.items()} for m, d in sorted(agg.items())},
                step2_wrong_s=round(sum(d.get("wrong", 0.0) for m, d in agg.items() if m in STEP2_MECH), 2),
                other_strata=dict(note="late (04-23 23:30-24:00) + identity-excluded strata: cannot be shown (SPEC §6.3); "
                                       "reported, not gated (MH5 = main stratum)",
                                  by_mechanism_s={m: {c: round(v, 2) for c, v in d.items()} for m, d in sorted(other.items())},
                                  step2_wrong_s=round(sum(d.get("wrong", 0.0) for m, d in other.items() if m in STEP2_MECH), 2)),
                wrong_list=lst[:200])

def _majority(an):
    """(animal | 'tie' | None, votes) over the known (>= 0) entries"""
    v = Counter(int(a) for a in an if a >= 0)
    m = v.most_common(2)
    if not m: return None, dict(v)
    if len(m) > 1 and m[0][1] == m[1][1]: return "tie", dict(v)
    return m[0][0], dict(v)

def _side_evidence(ts, an, fps, first, anchor=None):
    """one side of a step-2 action (review r1 finding 2): the majority animal of the first (last) STEP2_WIN_S s and the
    animal of the FIRST (LAST) qualifying >= SWITCH_MIN_S known run ANYWHERE in the segment (unknown rows neutral, as
    detect_switches). ids = the distinct animals named; 'consistent' = exactly one animal and no tie. anchor: the wall
    the 10 s window hangs on (new side: the segment open; default the first / last row)."""
    if not len(ts): return dict(win=None, win_votes={}, run=None, run_t=None, ids=[], consistent=False)
    if anchor is None: anchor = float(ts[0]) if first else float(ts[-1])
    w = (ts <= anchor + STEP2_WIN_S) if first else (ts >= anchor - STEP2_WIN_S)
    maj, votes = _majority(an[w])
    q = qualifying_runs(ts, an, fps)
    run = (q[0] if first else q[-1]) if q else None
    ids = sorted({x for x in (maj, run[0] if run else None) if x is not None and x != "tie"})
    return dict(win=maj, win_votes=votes, run=None if run is None else run[0],
                run_t=None if run is None else [iso(run[1]), iso(run[2])], ids=ids,
                consistent=(len(ids) == 1 and maj != "tie"))

def step2_actions(S, segrows):
    """every step-2 action (dark_link / dark-pass open) gets an automated verdict (review r1 findings 1-3).
    Evidence = trusted_animal: the cand mask ASSIGNED to an identity-trusted GT mask of the MAIN stratum, or the teacher
    identity of a fill; everything else is unknown. Each side (_side_evidence) is the majority of the first / last
    10 s PLUS the first / last qualifying >= 10 s known run anywhere in the segment, so an action whose first 10 s are
    GT-dark is still judged once the GT sees the animal again.
      dark_link: correct iff both sides name one and the same animal; wrong iff each side names one animal and they
        differ; otherwise unknown (no evidence on a side, or the two pieces of one side disagree).
      dark-pass open: wrong if > half of the first 10 s is NONE on EVALUABLE rows = non-calf, or if > half of the first
        10 s another cand slot points (assigned or split) at the same GT mask as the new slot on EVALUABLE rows =
        duplicate, or if > half of the first 10 s another cand slot carries the SAME trusted animal (GT-assigned or the
        teacher identity of a 5f fill, so a duplicate on a GT-dark calf is caught too; review r2; a teacher mask spanning
        two disjoint cand masks, or a GT-visible animal and the fill mask, names no fill identity, review r3 --
        scan_day.t_span) = duplicate;
        correct iff the new side names one animal; else unknown. EVALUABLE (review r2 finding 1) = main stratum, the GT
        holds four distinct REAL animals (ghost GT masks do not count), outside every SPEC §6.3 geometry exclusion
        (±30 s GT-defect windows), and the teacher does not contradict v4v2 on the mask concerned (g_tag / c_ntag
        CONTRA; SPEC §6.3 agreement rule). Non-calf or duplicate evidence that holds a majority only with the
        non-evaluable rows (a GT merge of two calves, a GT defect) is unknown, never wrong and never correct.
      An action whose open lies in the late (04-23 23:30-24:00) or identity-excluded stratum is unknown ('cannot be
      shown', SPEC §6.3), its evidence recorded. MH1: FAIL on any wrong, REVIEW on any unknown, PASS only when every
      action is adjudicated correct."""
    A = S["A"]; cseg = S["cseg"]; evs = events_by_seg(S["cev"]); out = []
    fps = S["fps"]
    strat = S.get("strat")
    if strat is None or len(strat) != len(A["ts"]): strat = np.zeros(len(A["ts"]), np.int8)
    geo_ex, geo_all, geo_any = _geo_rows(S)
    gtag = _row_arr(A, "g_tag", (N_SLOTS,), TAG_NO_ROW, np.int8)
    ntag = _row_arr(A, "c_ntag", (N_SLOTS,), TAG_NO_ROW, np.int8)
    fill_by_row = defaultdict(dict)
    for f in S["fills"]:
        if f["t_on_gt"] is not None: fill_by_row[f["row"]][f["slot"]] = f["t_on_gt"]
    tm, _ = teacher_map(S) if S.get("tvotes") else ({}, {})
    for f in S["fills"]:
        if f["t_on_gt"] is None and f["tkey"] is not None:
            key = tuple(f["tkey"]); a = tm.get(key)
            if a is not None: fill_by_row[f["row"]][f["slot"]] = a
    sent = [day_ts(d, h) for d, h, _s, _e in SENTINELS if d == S["day"]]
    late = LATE_STRATUM.get(S["day"]); late = (day_ts(S["day"], late[0]), day_ts(S["day"], late[1])) if late else None
    excl = [(day_ts(S["day"], a), day_ts(S["day"], b)) for a, b in IDENT_EXCL.get(S["day"], [])]
    def stratum_at(t):                 # = scan_day's strat (the late stratum takes precedence)
        if late and late[0] <= t < late[1]: return "late"
        return "excluded" if any(a <= t < b for a, b in excl) else "main"
    for r in cseg:
        m = seg_mech(r, evs)
        if m not in STEP2_MECH: continue
        rows = segrows.get(r["idx"], np.zeros(0, int)); k = r["slot"]
        ts = A["ts"][rows] if len(rows) else np.zeros(0)
        an = trusted_animal(S, rows, k, fill_by_row) if len(rows) else np.zeros(0, np.int64)
        new = _side_evidence(ts, an, fps, first=True, anchor=r["t0"])
        first = rows[ts <= r["t0"] + STEP2_WIN_S] if len(rows) else rows
        st = Counter(C_NAMES[int(x)] for x in A["c_stat"][first, k]) if len(first) else Counter()
        near = [hm(x) for x in sent if abs(x - r["t0"]) <= SENTINEL_WIN_S]
        stratum = stratum_at(r["t0"])
        rec = dict(sid=r["sid"], slot=k, t=iso(r["t0"]), mechanism=m, stratum=stratum, near_sentinel=near,
                   first_10s_status=dict(st), new=new, new_animal=new["ids"][0] if new["consistent"] else None,
                   verdict="unknown", why=None)
        if m == "dark_link":
            raw = r["raw"]; evl = (evs.get(r["sid"]) or {}).get("dark_link") or {}
            prev = raw.get("link_from") or evl.get("prev_seg_id")
            rec.update(prev=prev, cause=raw.get("cause", evl.get("cause")), sole_dark=raw.get("sole_dark", evl.get("sole_dark")),
                       n_dark=raw.get("n_dark", evl.get("n_dark")))
            pr = next((x for x in cseg if x["sid"] == prev), None)
            pv = dict(win=None, win_votes={}, run=None, run_t=None, ids=[], consistent=False)
            if pr is not None:
                prow = segrows.get(pr["idx"], np.zeros(0, int))
                if len(prow): pv = _side_evidence(A["ts"][prow], trusted_animal(S, prow, pr["slot"], fill_by_row), fps, first=False)
            rec["prev_side"] = pv
            rec["prev_animal"] = pv["ids"][0] if pv["consistent"] else None
            if stratum != "main":
                rec["why"] = f"link opened in the {stratum} stratum (cannot be shown, SPEC §6.3)"
            elif new["consistent"] and pv["consistent"]:
                same = new["ids"][0] == pv["ids"][0]
                rec["verdict"] = "correct" if same else "wrong"
                rec["why"] = "same GT animal across the link" if same else "link joined two different GT animals"
            elif not new["ids"] or not pv["ids"]:
                rec["why"] = "an end of the link has no trusted GT / teacher identity anywhere in its segment"
            else:
                rec["why"] = "conflicting evidence at an end of the link (first/last 10 s vs the nearest >= 10 s run)"
        else:
            n1 = len(first)
            # review r2 finding 1: a row is EVALUABLE for the geometric rules only where the GT holds four distinct REAL
            # animals, in the main stratum, outside the SPEC §6.3 all-slot geometry exclusions (the ±30 s GT-defect
            # windows); a per-slot exclusion (04-22 slot 2 13:47-15:58) is handled per rule below
            four = (strat[first] == 0) & A["has_gt"][first] & (A["n_dist"][first] >= N_SLOTS)
            gok = four & ~geo_all[first]
            ev_rows = first[gok]
            # non-calf: NONE on an evaluable row where the teacher does not see an animal under the mask either; on a
            # per-slot exclusion row only where the teacher positively agrees (the excluded GT mask may be the animal)
            is_none = A["c_stat"][first, k] == C_NONE
            nt = ntag[first, k]
            none_ev = is_none & gok & (nt != TAG_CONTRA) & (~geo_any[first] | (nt == TAG_AGREE))
            none_out = int(none_ev.sum())
            none_nev = int((is_none & four & ~none_ev).sum())        # GT-defect window / the teacher contradicts v4v2
            # review r1 finding 3 -- duplicate: another cand slot associated (assigned OR split) to the SAME GT mask as
            # the new slot (the Hungarian may give the GT mask to the new slot and make the old holder 'split');
            # review r2 finding 1 -- it decides only on EVALUABLE rows where the teacher does not contradict that GT mask
            # (a GT mask over two calves, as at the 04-23 12:05:24 GT merge, holds two CORRECT cand masks)
            dup = dup_ev = n_assoc = split_ev = id_dup = 0
            an_k = trusted_animal(S, first, k, fill_by_row) if n1 else np.zeros(0, np.int64)
            an_o = {k2: trusted_animal(S, first, k2, fill_by_row) for k2 in range(N_SLOTS) if k2 != k} if n1 else {}
            for q, rr in enumerate(first):
                # review r2 (the GT-dark sibling of finding 3): another cand slot carries the SAME trusted animal
                # (GT-assigned, or the teacher identity of a 5f fill on a GT-dark row) on this row
                if an_k[q] >= 0 and any(A["c_area"][rr, k2] > 0 and an_o[k2][q] == an_k[q] for k2 in an_o): id_dup += 1
                g = int(A["c_gs"][rr, k])
                if g < 0 or int(A["c_stat"][rr, k]) not in (C_ASSIGNED, C_SPLIT): continue
                n_assoc += 1
                ok_g = bool(gok[q]) and not geo_ex[rr, g] and int(gtag[rr, g]) != TAG_CONTRA
                if int(A["c_stat"][rr, k]) == C_SPLIT and ok_g: split_ev += 1
                if any(k2 != k and A["c_area"][rr, k2] > 0 and int(A["c_stat"][rr, k2]) in (C_ASSIGNED, C_SPLIT)
                       and int(A["c_gs"][rr, k2]) == g for k2 in range(N_SLOTS)):
                    dup += 1
                    if ok_g: dup_ev += 1
            split = st.get("split", 0)
            rec.update(first_10s_rows=n1, first_10s_evaluable=int(len(ev_rows)), none_out=none_out,
                       none_not_evaluable=none_nev, dup_rows=dup, dup_rows_evaluable=dup_ev, split_rows_evaluable=split_ev,
                       id_dup_rows=id_dup, assoc_rows=n_assoc)
            if stratum != "main":
                rec["why"] = f"opened in the {stratum} stratum (cannot be shown, SPEC §6.3)"
            elif n1 and none_out > n1 / 2:
                rec.update(verdict="wrong", why="opened on no GT animal while the GT holds four real animals (non-calf)")
            elif n1 and (dup_ev > n1 / 2 or split_ev > n1 / 2):
                rec.update(verdict="wrong", why="opened on an animal another cand slot already holds (duplicate)")
            elif n1 and id_dup > n1 / 2:
                rec.update(verdict="wrong", why="opened on an animal another cand slot already holds (duplicate by trusted "
                                                "identity: GT-assigned or the teacher identity of a 5f fill)")
            elif len(ev_rows) and none_out > len(ev_rows) / 2:
                rec["why"] = "non-calf evidence on the evaluable part of a partly GT-dark first 10 s"
            elif n1 and none_out + none_nev > n1 / 2:
                rec["why"] = ("non-calf evidence only where v4v2 cannot adjudicate it (a SPEC §6.3 GT-defect geometry "
                              "exclusion, or the teacher has an animal under the mask)")
            elif n1 and (dup > n1 / 2 or split > n1 / 2):
                rec["why"] = ("duplicate evidence only where the GT cannot adjudicate it (GT-dark / GT-merged row, a GT-defect "
                              "geometry exclusion, or the teacher contradicts the shared GT mask)")
            elif n_assoc and dup > n_assoc / 2:
                rec["why"] = "duplicate evidence on the GT-associated part of the first 10 s"
            elif new["consistent"]:
                rec.update(verdict="correct", why="opened on a GT animal")
            elif new["ids"]:
                rec["why"] = "conflicting evidence (first 10 s vs the first >= 10 s run)"
            else: rec["why"] = "no trusted GT / teacher identity anywhere in the new segment"
        out.append(rec)
    unk = [x for x in out if x["verdict"] == "unknown"]
    return dict(n=len(out), n_wrong=sum(1 for x in out if x["verdict"] == "wrong"), n_unknown=len(unk),
                n_unknown_near_sentinel=sum(1 for x in unk if x["near_sentinel"]),
                n_unknown_dark_link=sum(1 for x in unk if x["mechanism"] == "dark_link"),
                n_unknown_other_strata=sum(1 for x in unk if x["stratum"] != "main"), list=out)

def sentinel_report(S, segrows):
    A = S["A"]; out = []
    if not len(A["ts"]): return out
    for day, hms_, gslot, expect in SENTINELS:
        if day != S["day"]: continue
        t = day_ts(day, hms_)
        rec = dict(day=day, t=hms_, gt_slot=gslot, expected_first_reason=expect,
                   note="GT verdict to be confirmed on a render before use (SPEC §6.3)")
        if not (float(A["ts"][0]) - SENTINEL_WIN_S <= t <= float(A["ts"][-1]) + SENTINEL_WIN_S):
            rec["in_scope"] = False; out.append(rec); continue            # outside a pilot window
        rec["in_scope"] = True
        pre = np.flatnonzero((A["ts"] >= t - 5.0) & (A["ts"] < t) & (A["g_area"][:, gslot] > 0))
        post = np.flatnonzero((A["ts"] >= t) & (A["ts"] <= t + 1800.0) & (A["g_area"][:, gslot] > 0))
        a_pre = Counter(int(x) for x in A["g_anim"][pre, gslot] if x >= 0).most_common(1)
        a_post = None
        if len(post):
            t_re = float(A["ts"][post[0]]); pr = post[A["ts"][post] <= t_re + 5.0]
            c = Counter(int(x) for x in A["g_anim"][pr, gslot] if x >= 0).most_common(1)
            a_post = c[0][0] if c else None; rec["gt_reappears"] = iso(t_re)
        rec["gt_pre"] = a_pre[0][0] if a_pre else None; rec["gt_post"] = a_post
        ks = Counter(int(x) for x in A["g_c"][pre, gslot] if x >= 0).most_common(1)
        if ks:
            k = ks[0][0]; si = Counter(int(A["c_seg"][r, k]) for r in pre if A["g_c"][r, gslot] == k).most_common(1)[0][0]
            rec["cand_slot"] = k; rec["cand_seg_pre"] = S["cseg"][si]["sid"] if si >= 0 else None
            rows = segrows.get(si, np.zeros(0, int))
            later = rows[A["ts"][rows] >= t] if len(rows) else rows
            an = cand_animal(S, later, k) if len(later) else np.zeros(0, int)
            spans = rec["gt_post"] is not None and rec["gt_pre"] is not None and rec["gt_post"] != rec["gt_pre"] and \
                    int((an == rec["gt_post"]).sum()) * (1.0 / S["fps"]) >= 1.0
            rec["cand_segment_spans_change_without_break"] = bool(spans)
            rec["cand_seg_pre_rows_after_t_on_gt_post_s"] = round(int((an == (rec["gt_post"] if rec["gt_post"] is not None else -9)).sum()) / S["fps"], 2)
        else:
            rec["cand_slot"] = None; rec["cand_segment_spans_change_without_break"] = None
        acts = []; first_rej = None
        ks_ = rec.get("cand_slot")
        for e in sorted((e for e in S["cev"] if e.get("_t") is not None and abs(e["_t"] - t) <= SENTINEL_WIN_S),
                        key=lambda e: e["_t"]):
            if e.get("type") in ("segment_open", "segment_close", "dark_link", "dark_open", "dark_pass", "pen_count_refused",
                                 "merge_resolved", "dark_onset", "dark_suspended", "dark_open_refused"):
                a_ = dict(type=e["type"], dt=round(e["_t"] - t, 2), slot=e.get("slot"), seg_id=e.get("seg_id"),
                          reason=e.get("reason") or e.get("cause") or e.get("path"))
                if e["type"] == "dark_pass":
                    rj = (e.get("rejected") or {}).get(str(ks_)) if ks_ is not None else None
                    a_.update(refused=e.get("refused"), deferred=e.get("deferred"), rejected_cand_slot=rj,
                              links=sorted((e.get("links") or {}).keys()), opens=sorted((e.get("opens") or {}).keys()))
                    # the first rejecting reason for the sentinel's cand slot at or after the sentinel (SPEC §6.3 table)
                    if first_rej is None and e["_t"] >= t - 1.0:
                        if e.get("refused"): first_rej = e["refused"]
                        elif e.get("deferred"): first_rej = e["deferred"]
                        elif rj: first_rej = rj[0][1] if isinstance(rj[0], (list, tuple)) and len(rj[0]) > 1 else str(rj[0])
                acts.append(a_)
        rec["first_rejecting_reason"] = first_rej
        rec["actions_within_60s"] = acts
        out.append(rec)
    return out

# ============================================================================================================
# 10 · telemetry (operations, §6.3-7) and must-hold 8
# ============================================================================================================
def read_root_config(day_dir):
    """ROOT_CONFIG.json of a v5 day dir and of its root (Tol_47:1545-1557; core.root_config_record + written_by,
    created, echo). EXPLICIT keys only: a regex over the dump matched the `gpu_tf` widget before `gpu_class`."""
    out = {}
    for lvl, p in (("day", os.path.join(day_dir, "ROOT_CONFIG.json")),
                   ("root", os.path.join(os.path.dirname(os.path.normpath(day_dir)), "ROOT_CONFIG.json"))):
        if not os.path.exists(p): continue
        try: J = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            out[lvl] = dict(path=p, error=f"unreadable: {e}"); continue
        echo = J.get("echo") or {}; w = J.get("widgets") or {}
        g = echo.get("gpu") or {}
        out[lvl] = dict(path=p, sha256=sha256_file(p), gpu_class=J.get("gpu_class"),
                        gpu_name=g.get("name") if isinstance(g, dict) else None,
                        matmul_precision=echo.get("matmul_precision"),
                        mem_cap_gib=w.get("mem_cap_gib", echo.get("mem_cap_gib")),
                        v5_fixes=w.get("v5_fixes"), dark_fire=w.get("dark_fire"), detector=w.get("detector"),
                        written_by=J.get("written_by"),
                        record={k: J.get(k) for k in ("widgets", "shas", "gpu_class", "band")})
    return out

def ops_metrics(S):
    """operations from the tracker's own events (v5 adds mem_check / dark_* / pen_count_refused / resume). v4 roots
    simply have zeros / None here. The GPU class comes from ROOT_CONFIG.json (read_root_config), never a regex."""
    cev = S["cev"]; c = Counter(e.get("type") for e in cev)
    mem = [e for e in cev if e.get("type") in ("mem_check", "mem_assert_fail")]
    # a DELAYED FREE is not a leak (full A100 profile 2026-09-24: a torn-down session can sit in a reference cycle until
    # the next gc.collect()). Tol_47 _mem_check re-measures an allocated failure once after gc.collect() and logs it as
    # mem_check where='gc_recheck' at the same frame; its verdict (and so MH8's) uses the re-measured ALLOCATED value,
    # while RESERVED stays the first measurement. The first reading of a rechecked frame therefore does not enter the
    # allocated peak (a mem_assert_fail after the recheck still does). The telemetry peak is a high-water mark and is
    # taken as is (Tol_47 gc.collect()s right after every teardown, so no garbage coexists with a new session).
    rech = {(clip_key(e.get("clip")), e.get("si")) for e in mem if e.get("type") == "mem_check" and e.get("where") == "gc_recheck"}
    def mx(key):
        v = [float(e[key]) for e in mem if isinstance(e.get(key), (int, float))
             and not (key == "allocated_gib" and e.get("type") == "mem_check" and e.get("where") != "gc_recheck"
                      and (clip_key(e.get("clip")), e.get("si")) in rech)]
        return max(v) if v else None
    dms = [float(e["det_ms"]) for e in cev if e.get("type") in ("pass", "dark_pass") and isinstance(e.get("det_ms"), (int, float))]
    pen = Counter(str(e.get("path")) for e in cev if e.get("type") == "pen_count_refused")
    rss = [float(e["host_rss_gib"]) for e in mem if isinstance(e.get("host_rss_gib"), (int, float))]
    rc = read_root_config(S["cand_dir"])
    classes = sorted({v.get("gpu_class") for v in rc.values() if v.get("gpu_class")})
    return dict(passes=c.get("pass", 0), pass_no_detections=c.get("pass_no_detections", 0), seeded=c.get("seeded", 0),
                dark_passes=c.get("dark_pass", 0), dark_links=c.get("dark_link", 0), dark_opens=c.get("dark_open", 0),
                dark_open_refused=c.get("dark_open_refused", 0), dark_suspended=c.get("dark_suspended", 0),
                dark_onsets=c.get("dark_onset", 0), pen_count_refused=dict(pen), resumes=c.get("resume", 0),
                resume_clips=[e.get("clip") for e in cev if e.get("type") == "resume"],
                mem_checks=sum(1 for e in mem if e.get("type") == "mem_check" and e.get("where") != "gc_recheck"),
                mem_gc_rechecks=len(rech), mem_assert_fail=c.get("mem_assert_fail", 0),
                peak_alloc_gib=mx("allocated_gib"), peak_reserved_gib=mx("reserved_gib"), max_processed_frames=mx("processed_frames"),
                host_rss_gib_max=max(rss) if rss else None,
                det_ms_median=round(float(np.median(dms)), 2) if dms else None,
                det_ms_p95=round(float(np.percentile(dms, 95)), 2) if dms else None,
                root_config={k: {kk: vv for kk, vv in v.items() if kk != "record"} for k, v in rc.items()},
                gpu_class=classes, gpu=sorted({v.get("gpu_name") for v in rc.values() if v.get("gpu_name")}))

def inv7_of(c):
    """Recompute Tol_47 finish_clip_telemetry's INV-7 abort tests from one telemetry clip entry. The entry is written
    BEFORE the checks raise (Tol_47:1614-1631), so a telemetry entry does not prove the clip passed. matched ==
    len(accepted); n_open_rows is not stored, so the coverage test is applied without its >= 1000-row floor
    (conservative: may flag a short clip the tracker would have let pass)."""
    why = []
    acc, med, cov = c.get("acceptance"), c.get("median_accepted_overlap"), c.get("coverage3_frac")
    if (c.get("assignable") or 0) >= INV7["acc_min_n"] and acc is not None and acc < INV7["acc"]: why.append(f"acceptance {acc}")
    if (c.get("matched") or 0) >= INV7["med_min_n"] and med is not None and med < INV7["med"]: why.append(f"median_accepted_overlap {med}")
    if cov is not None and cov < INV7["cov3"]: why.append(f"coverage3_frac {cov}")
    return why

def telemetry_summary(path, rows_hours=None):
    """Tol_47 telemetry Tol/Metrics_track_v5_<root>_<stage><wsuf>.json. The v5 per-clip fields are NESTED under
    clip['v5'] (Tol_47:1615 `entry["v5"] = _v5_clip_entry(...)`, fields at Tol_47:1709-1725) -- critic finding 4:
    reading clip['gpu'] saw nothing and MH8 failed on a correct A10 run. Explicit keys, no regex."""
    if not path or not os.path.exists(path): return dict(available=False, path=path)
    J = json.load(open(path, encoding="utf-8")); clips = J.get("clips", [])
    P = J.get("params") or {}; echo = P.get("v5") or {}; guard = P.get("v5_guard") or {}
    per = []
    for c in clips:
        v = c.get("v5") or {}
        name = v.get("gpu", c.get("gpu"))
        cls = v.get("gpu_class") or (gpu_class(name) if name else None)
        tm = v.get("ms_per_frame") or {}
        per.append(dict(clip=c.get("clip"), frames=int(c.get("frames", 0) or 0), track_min=c.get("track_min"),
                        stage_min=c.get("stage_min"), fps=v.get("fps"), gpu=name, gpu_class=cls,
                        peak_alloc_gib=v.get("peak_allocated_gib"), peak_reserved_gib=v.get("peak_reserved_gib"),
                        mem_free_gib=v.get("mem_free_gib"), max_processed_frames=v.get("max_processed_frames"),
                        mem_checks=v.get("mem_checks"), host_rss_gib=v.get("host_rss_gib"),
                        max_det_calls_per_frame=v.get("max_det_calls_per_frame"), det_calls=v.get("det_calls"),
                        det_ms_p95=v.get("det_ms_p95"), rebuilds_per_hour=v.get("rebuilds_per_hour"),
                        cpu_ms_per_frame=v.get("cpu_ms_per_frame"),
                        host_rss_trend_gib_per_run_h=v.get("host_rss_trend_gib_per_run_h"),
                        session_tracked_entries=v.get("session_tracked_entries"), mem_rechecks=v.get("mem_rechecks"),
                        sessions=v.get("sessions"), rebuilds=v.get("rebuilds"),
                        timings=sorted(k for k, x in tm.items() if isinstance(x, (int, float))),
                        pen_count_refused=v.get("pen_count_refused"), dark=v.get("dark"),
                        resumed=v.get("resumed"), has_v5=bool(v), inv7=inv7_of(c),
                        acceptance=c.get("acceptance"), coverage3_frac=c.get("coverage3_frac"),
                        median_accepted_overlap=c.get("median_accepted_overlap")))
    def mx(k, agg=max):
        vals = [float(p[k]) for p in per if isinstance(p[k], (int, float)) and not isinstance(p[k], bool)]
        return agg(vals) if vals else None
    frames = sum(p["frames"] for p in per)
    tmin = sum(float(p["track_min"] or 0) for p in per); smin = sum(float(p["stage_min"] or 0) for p in per)
    wall_min = [float(p["track_min"] or 0) + float(p["stage_min"] or 0) for p in per]
    g_echo = echo.get("gpu") or {}
    w_echo = echo.get("widgets") or {}
    out = dict(available=True, path=path, sha256=sha256_file(path), n_clips=len(per),
               n_clips_v5=sum(1 for p in per if p["has_v5"]),
               clip_keys=sorted(clip_key(p["clip"]) for p in per if p["clip"]), frames=frames,
               track_h=round(tmin / 60, 3), stage_h=round(smin / 60, 3),
               fps=round(frames / (tmin * 60), 3) if tmin else None,
               recorded_h=round(rows_hours, 3) if rows_hours else None,
               wall_h_per_24h_recorded=round((tmin + smin) / 60 / rows_hours * 24, 2) if rows_hours else None,
               per_clip_wall_min_median=round(float(np.median(wall_min)), 2) if wall_min else None,
               per_clip_fps_median=mx("fps", lambda v: round(float(np.median(v)), 3)),
               peak_alloc_gib=mx("peak_alloc_gib"), peak_reserved_gib=mx("peak_reserved_gib"),
               max_processed_frames=mx("max_processed_frames"), host_rss_gib=mx("host_rss_gib"),
               max_det_calls_per_frame=mx("max_det_calls_per_frame"), cpu_ms_per_frame_max=mx("cpu_ms_per_frame"),
               det_ms_p95_max=mx("det_ms_p95"), rebuilds_per_hour_max=mx("rebuilds_per_hour"),
               # host RSS grows ~0.46 GB per run-hour (SAM3 never prunes session.frames_tracked_per_obj; full A100
               # profile 2026-09-24): REPORTED for the deployability claim, no gate
               host_rss_trend_gib_per_run_h_max=mx("host_rss_trend_gib_per_run_h"),
               host_rss_trend_gib_per_run_h_median=mx("host_rss_trend_gib_per_run_h", lambda v: round(float(np.median(v)), 3)),
               session_tracked_entries_max=mx("session_tracked_entries"),
               mem_rechecks=int(sum(float(p["mem_rechecks"]) for p in per if isinstance(p["mem_rechecks"], (int, float)))),
               rebuilds=int(sum(float(p["rebuilds"]) for p in per if isinstance(p["rebuilds"], (int, float)))),
               gpu=sorted({str(p["gpu"]) for p in per if p["gpu"]}),
               gpu_class=sorted({str(p["gpu_class"]) for p in per if p["gpu_class"]}),
               guard_gpu_class=guard.get("gpu_class"),
               echo_gpu=g_echo.get("name") if isinstance(g_echo, dict) else None,
               echo_gpu_class=gpu_class(g_echo.get("name")) if isinstance(g_echo, dict) and g_echo.get("name") else None,
               matmul_precision=echo.get("matmul_precision"),
               mem_cap_gib=w_echo.get("mem_cap_gib", echo.get("mem_cap_gib")), v5_fixes=w_echo.get("v5_fixes"),
               timings_missing=[p["clip"] for p in per if p["has_v5"] and not set(TIMING_KEYS_REQUIRED) <= set(p["timings"])],
               timing_components=sorted({k for p in per for k in p["timings"]}),
               inv7_violations={str(p["clip"]): p["inv7"] for p in per if p["inv7"]},
               resumed_clips=[p["clip"] for p in per if p["resumed"]],
               acceptance_min=mx("acceptance", min), coverage3_min=mx("coverage3_frac", min),
               median_accepted_overlap_min=mx("median_accepted_overlap", min),
               clips=per)
    return out

def find_telemetry(root, day_dir_name):
    """Tol/Metrics_track_v5_<root>_<stage><wsuf>.json (Tol_47:1581): next to the root (the local copy of Tol/), in the
    root, or in the day dir"""
    rn = os.path.basename(os.path.normpath(root))
    nm = f"Metrics_track_v5_{rn}_{day_dir_name}.json"
    for d in (os.path.dirname(os.path.normpath(root)), root, os.path.join(root, day_dir_name)):
        p = os.path.join(d, nm)
        if os.path.exists(p): return p
    return None

# ============================================================================================================
# 11 · per arm-day assembly, deltas, must-holds
# ============================================================================================================
def day_metrics(S):
    t0 = time.time()
    segrows = segment_rows(S) if len(S["A"].get("ts", [])) else {}
    M = dict(day=S["day"], arm=S["arm"], cand_dir=S["cand_dir"], fps=S["fps"], window=S["window"],
             rows=int(len(S["A"]["ts"])), hours=round(len(S["A"]["ts"]) / S["fps"] / 3600, 4),
             sizes=S["sizes"], fps_seen=S["fps_seen"], precondition=S["precondition"],
             smoke_clips=[c["stem"] for c in S["clips"] if c.get("smoke")],
             done=S.get("done"), gt_mode=S.get("gt_mode"), gt_docx=S.get("gt_docx"),
             clip_stems=[c["stem"] for c in S["clips"] if c.get("cand")])
    M["gaps"] = gap_metrics(S)
    M["segments"] = segment_metrics(S)
    M["ops"] = ops_metrics(S)
    M["fragments"] = fragment_metrics(S)
    if not S["light"]:
        M["overlap"] = overlap_metrics(S)
        M["labels"] = label_metrics(S)
        M["switches"] = switch_metrics(S, segrows)
        M["mapping_seed"] = seed_mapping(S, segrows)
        M["mask_fidelity"] = mask_fidelity(S, M["fragments"].get("a_ref_eval_gt"))
        M["step2_actions"] = step2_actions(S, segrows)
        M["sentinels"] = sentinel_report(S, segrows)
        M["teacher"] = dict(mode=S["teacher_mode"], n_fill_rows=len(S["fills"]))
        M["ghost_rule"] = S.get("ghost_rule")
    M["metrics_s"] = round(time.time() - t0, 1)
    return M, segrows

def numeric_leaves(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).startswith("_") or k in ("list", "orphan_list", "wrong_spans", "wrong_list", "cand_dir", "path",
                                                   "metrics_s", "scan_s", "runtime_s"): continue
            out.update(numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, bool): out[prefix] = int(d)
    elif isinstance(d, (int, float)) and not (isinstance(d, float) and math.isnan(d)): out[prefix] = d
    return out

def deltas(cand, ref):
    a, b = numeric_leaves(cand), numeric_leaves(ref)
    keys = sorted(set(a) | set(b))
    D = {}; nz = []; only = []
    for k in keys:
        if k in a and k in b:
            v = a[k] - b[k]
            D[k] = round(v, 6) if isinstance(v, float) else v
            if v != 0: nz.append(k)
        else: only.append(k)
    return D, nz, only

def mh(id_, name, status, value=None, threshold=None, severity="must_hold", note=None):
    return dict(id=id_, name=name, status=status, value=value, threshold=threshold, severity=severity, note=note)

def _num_max(*xs):
    v = [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return max(v) if v else None

def mh8_row(R, days, args, fr):
    """SPEC §6.4-8 / §6.1 G2 / RUNBOOK G2+G3-8: runs on the A10 -- complete, peak allocated <= 4 GiB, reserved <=
    10 GiB, n_pf <= 2, no mem_assert_fail, no INV-7 abort; GPU = A10 in the telemetry AND ROOT_CONFIG.json (EXACT
    class: critic finding 4). With --d6_fallback the class must be A100 AND the root's mem_cap_gib must be 21.0
    (addendum D6; the row is then labelled: memory verified under a 21 GiB cap on an A100, A10 time projected)."""
    d6 = bool(getattr(args, "d6_fallback", False))
    want = "A100" if d6 else "A10"
    vals, bad, missing = {}, [], []
    for d in days:
        t = R.get("telemetry", {}).get(f"cand:{d}") or {}
        C = R["cand"].get(d) or {}
        o = C.get("ops") or {}
        rc = o.get("root_config") or {}
        tel = bool(t.get("available"))
        v = dict(peak_alloc_gib=_num_max(t.get("peak_alloc_gib"), o.get("peak_alloc_gib")),
                 peak_reserved_gib=_num_max(t.get("peak_reserved_gib"), o.get("peak_reserved_gib")),
                 max_processed_frames=_num_max(t.get("max_processed_frames"), o.get("max_processed_frames")),
                 mem_assert_fail=o.get("mem_assert_fail"), mem_check_events=o.get("mem_checks"),
                 telemetry=t.get("path") if tel else None,
                 gpu_class_telemetry=t.get("gpu_class") or [], gpu_telemetry=t.get("gpu") or [],
                 gpu_class_root_config={k: x.get("gpu_class") for k, x in rc.items()},
                 mem_cap_gib_root_config={k: x.get("mem_cap_gib") for k, x in rc.items()},
                 inv7_violations=t.get("inv7_violations") if tel else None,
                 done=(C.get("done") or {}), resumes=o.get("resumes"), resume_clips=o.get("resume_clips"),
                 fps=t.get("fps"), wall_h_per_24h_recorded=t.get("wall_h_per_24h_recorded"),
                 host_rss_gib=_num_max(t.get("host_rss_gib"), o.get("host_rss_gib_max")),
                 cpu_ms_per_frame_max=t.get("cpu_ms_per_frame_max"))
        vals[d] = v
        if not tel and not o.get("mem_checks"):
            missing.append(f"{d}: no memory evidence (no telemetry, no mem_check events)"); continue
        if not tel: missing.append(f"{d}: telemetry Metrics_track_v5_<root>_<day>.json not found (--telemetry)")
        if not rc: missing.append(f"{d}: ROOT_CONFIG.json missing in the local copy (day and root level)")
        if v["peak_alloc_gib"] is None or v["peak_reserved_gib"] is None or v["max_processed_frames"] is None:
            missing.append(f"{d}: a memory field is missing"); continue
        if v["peak_alloc_gib"] > A10_ALLOC_GIB: bad.append(f"{d}: peak allocated {v['peak_alloc_gib']} GiB > {A10_ALLOC_GIB}")
        if v["peak_reserved_gib"] > A10_RESERVED_GIB: bad.append(f"{d}: peak reserved {v['peak_reserved_gib']} GiB > {A10_RESERVED_GIB}")
        if v["max_processed_frames"] > A10_NPF: bad.append(f"{d}: n_pf {v['max_processed_frames']} > {A10_NPF}")
        if v["mem_assert_fail"]: bad.append(f"{d}: {v['mem_assert_fail']} mem_assert_fail events")
        if v["inv7_violations"]: bad.append(f"{d}: INV-7 abort signature in telemetry {v['inv7_violations']}")
        if v["done"].get("present") and v["done"].get("missing"): bad.append(f"{d}: clips without .done_ {v['done']['missing']}")
        if tel:                    # Tol_47 writes masks BEFORE the telemetry entry: a mask file without an entry = a crash
            tk = set(t.get("clip_keys") or [])
            no_tel = [x for x in (C.get("clip_stems") or []) if clip_key(x) not in tk]
            v["clips_without_telemetry"] = no_tel
            if no_tel: bad.append(f"{d}: masks without a telemetry entry {no_tel[:5]} (crashed before the INV-7 checks?)")
        classes = set(v["gpu_class_telemetry"]) | {x for x in v["gpu_class_root_config"].values() if x}
        if classes and classes != {want}:
            bad.append(f"{d}: GPU class {sorted(classes)} != {want}" + ("" if d6 else " (a D6 fallback must be scored with --d6_fallback)"))
        if d6:
            caps = [x for x in v["mem_cap_gib_root_config"].values() if x is not None]
            try: capok = bool(caps) and all(abs(float(x) - D6_CAP_GIB) < 1e-9 for x in caps)
            except (TypeError, ValueError): capok = False
            if not capok: bad.append(f"{d}: --d6_fallback but ROOT_CONFIG mem_cap_gib {caps} != {D6_CAP_GIB}")
    if not all(fr.get(d) is not None and fr[d] >= 0.99 for d in days): bad.append("a day is incomplete (P0)")
    name = ("runs on the A10: complete, alloc <= 4 GiB, reserved <= 10 GiB, n_pf <= 2, no mem_assert_fail, no INV-7 "
            "abort, GPU class A10 in telemetry AND ROOT_CONFIG.json")
    label = ("D6 FALLBACK: memory verified under a 21 GiB cap on an A100; A10 time is projected (x1.21), not measured"
             if d6 else None)
    if bad:
        return mh("MH8", name, "FAIL", vals, dict(alloc=A10_ALLOC_GIB, reserved=A10_RESERVED_GIB, n_pf=A10_NPF, gpu=want),
                  note="; ".join(bad + missing) + (f" [{label}]" if label else ""))
    if missing:
        return mh("MH8", name, "NOT_EVALUATED", vals, note="; ".join(missing))
    return mh("MH8", name, "PASS", vals, dict(alloc=A10_ALLOC_GIB, reserved=A10_RESERVED_GIB, n_pf=A10_NPF, gpu=want),
              note=label or "INV-7 recomputed from telemetry; .done_ markers checked when present in the local copy")

def must_holds(R, days, args):
    """SPEC §6.4 must-holds on the CAND arm (ref values given for comparison) -> list of rows"""
    T = []
    C = {d: R["cand"].get(d) for d in days}; F = {d: R["ref"].get(d) for d in days}
    # P0 precondition
    fr = {d: (C[d] or {}).get("precondition", {}).get("frac") for d in days}
    ok = all(v is not None and v >= 0.99 for v in fr.values())
    T.append(mh("P0", ">= 99 % of GT rows have a cand row with the same frame_idx (each day)", "PASS" if ok else "FAIL",
                {d: None if v is None else round(v, 5) for d, v in fr.items()}, 0.99))
    ident = all((C[d] or {}).get("identity") for d in days)
    # MH1
    acts = {d: (C[d] or {}).get("step2_actions", {}) for d in days}
    nw = sum(a.get("n_wrong", 0) for a in acts.values()); nu = sum(a.get("n_unknown", 0) for a in acts.values())
    # review r1 finding 2: an action that cannot be adjudicated is never a PASS (the addendum removed the contact sheet
    # that was to adjudicate it): FAIL on any wrong, REVIEW on any unknown (gate_v5 maps REVIEW -> REVIEW, i.e. an ack
    # after a look at the listed actions), PASS only when every action is adjudicated correct
    unk = [dict(day=d, sid=x["sid"], t=x["t"], mechanism=x["mechanism"], stratum=x.get("stratum"),
                near_sentinel=x.get("near_sentinel"), why=x.get("why"))
           for d, a in acts.items() for x in (a.get("list") or []) if x.get("verdict") == "unknown"]
    T.append(mh("MH1", "no wrong-animal step-2 action (dark_link / dark-pass open); every action adjudicated",
                "FAIL" if nw else ("REVIEW" if nu else "PASS"),
                dict(n_actions=sum(a.get("n", 0) for a in acts.values()), n_wrong=nw, n_unknown=nu,
                     n_unknown_near_sentinel=sum(1 for x in unk if x["near_sentinel"]),
                     n_unknown_dark_link=sum(1 for x in unk if x["mechanism"] == "dark_link"),
                     unknown=unk[:50]), 0,
                note="automated verdict from trusted GT / teacher identity (first/last 10 s + nearest >= 10 s run, main "
                     "stratum); any unknown action -> REVIEW (listed), never PASS"))
    # MH2
    sw = sum((C[d] or {}).get("switches", {}).get("n_step2", 0) for d in days)
    T.append(mh("MH2", "no within-segment switch attributable to a step-2 mechanism (5a, conservative attribution: any "
                       "switch inside a step-2-opened segment)", "PASS" if sw == 0 else "FAIL",
                dict(step2_switches=sw, all_switches=sum((C[d] or {}).get("switches", {}).get("n", 0) for d in days)), 0))
    # MH3
    if ident:
        v = sum(C[d]["identity"]["step2_wrong_silent_min"] for d in days)
        T.append(mh("MH3", "silent-wrong animal-time after Tol_30 attributable to step 2 == 0", "PASS" if v == 0 else "FAIL", v, 0.0))
    else: T.append(mh("MH3", "silent-wrong animal-time after Tol_30 attributable to step 2 == 0", "NOT_EVALUATED", note="Tol_30 not run"))
    # MH4 (D8 review trigger)
    if ident and all((F[d] or {}).get("identity") for d in days):
        wc = sum(C[d]["identity"]["strata"]["main"]["wrong_min"] for d in days)
        wr = sum(F[d]["identity"]["strata"]["main"]["wrong_min"] for d in days)
        T.append(mh("MH4", f"overall wrong animal-time (flagged + silent) <= v4v2 re-scored + {D8_DELTA_MIN:g} min (04-22 + 04-23)",
                    "PASS" if wc <= wr + D8_DELTA_MIN else "REVIEW", dict(cand_min=round(wc, 3), ref_min=round(wr, 3)),
                    round(wr + D8_DELTA_MIN, 3), severity="review_trigger"))
    else: T.append(mh("MH4", "overall wrong time <= v4v2 + delta", "NOT_EVALUATED", severity="review_trigger", note="Tol_30 not run on both arms"))
    # MH4z -- D8's HARD part (SPEC D8 as settled by the addendum: delta "sits on top of a hard zero for anything
    # attributable to step 2"): wrong animal-time after Tol_30, FLAGGED + SILENT, in segments opened by a step-2
    # mechanism (dark_link / dark-pass open) == 0. MH3 covers the silent part only; this row is the flagged part too.
    nm4z = "wrong animal-time (flagged + silent) after Tol_30 attributable to a step-2 mechanism == 0 (D8 hard zero)"
    if ident:
        v = {d: (C[d]["identity"] or {}).get("step2_wrong_min") for d in days}
        miss = [d for d, x in v.items() if x is None]
        wk = {d: round(sum(x.get("wrong_flagged", 0.0) + x.get("wrong_silent", 0.0)
                           for m, x in ((C[d].get("identity_week") or {}).get("wrong_by_mechanism") or {}).items()
                           if m in STEP2_MECH), 3) for d in days if C[d].get("identity_week")}
        tot = sum(x for x in v.values() if x is not None)
        T.append(mh("MH4z", nm4z, "NOT_EVALUATED" if miss else ("PASS" if tot == 0 else "FAIL"),
                    dict(per_day_min=v, week_mode_min_reported=wk or None), 0.0,
                    note=(f"no step2_wrong_min for {miss}" if miss else
                          "per-day production call (as MH3); the week-mode value is reported, not gated (a wrong week "
                          "map would be charged to whatever mechanism opened the segment)")))
    else: T.append(mh("MH4z", nm4z, "NOT_EVALUATED", note="Tol_30 not run"))
    # MH5
    if ident:
        v = sum((C[d].get("fills") or {}).get("step2_wrong_s", 0.0) for d in days)
        have_t = any((C[d].get("teacher") or {}).get("mode") for d in days)
        T.append(mh("MH5", "no wrong fill attributable to a step-2 action on v4v2-dark frames (5f, teacher identity)",
                    ("PASS" if v == 0 else "FAIL") if have_t else "NOT_EVALUATED", v, 0.0,
                    note=None if have_t else "no teacher root given"))
    else: T.append(mh("MH5", "no wrong fill attributable to step 2 (5f)", "NOT_EVALUATED", note="Tol_30 not run"))
    # MH6 night gap
    ng = R.get("nightgap", {}).get("cand")
    if ng and ng.get("evaluable"):
        bad = ng["correct"] is False and (ng.get("min_margin") is not None and ng["min_margin"] >= NIGHTGAP_MARGIN)
        st = "FAIL" if bad else "PASS"
        T.append(mh("MH6", "night gap 04-22->23: no WRONG permutation at margin >= 0.015", st, ng, NIGHTGAP_MARGIN,
                    note="a correct permutation below 0.015 is reported UNRESOLVED (as v4v2)"))
    else: T.append(mh("MH6", "night gap 04-22->23: no wrong permutation at margin >= 0.015", "NOT_EVALUATED",
                      note="needs --nightgap_cand (Tol_16_v2 output on the cand root)"))
    # MH7 (review r1 finding 1): gated on the MAIN stratum; GT-dark is ghost-aware (a v4v2 ghost mask is not an animal,
    # see A_REF_EVAL); the late (04-23 23:30-24:00) and identity-excluded strata 'cannot be shown' and are reported
    # review r2 finding 2: SPEC §6.3 'A geometry error counts only where the two [v4v2 and the teacher] agree' + the ±30 s
    # geometry exclusions around the known GT defects -> MH7 gates the COUNTED part (mh7_classes: the teacher agrees, or
    # v4v2 is the only reference); the time where the teacher contradicts v4v2 or a GT-defect exclusion applies is
    # reported with its episodes (for the renders, §6.4-7), never gated
    vals = {}
    for d in days:
        LB = (C[d] or {}).get("labels", {}) or {}
        L = LB.get("cand_s", {})
        cs = L.get("main", {})
        G = LB.get("mh7_geometry") or {}
        gm = G.get("main")
        vals[d] = dict(multi_out_s=gm["MULTI"]["counted_s"] if gm else cs.get("MULTI_out_gt_dark"),
                       none_out_s=gm["NONE"]["counted_s"] if gm else cs.get("NONE_out_gt_dark"),
                       none_in_s=cs.get("NONE_in_gt_dark"),
                       raw_out_gt_dark=dict(multi_s=cs.get("MULTI_out_gt_dark"), none_s=cs.get("NONE_out_gt_dark")),
                       geometry_main=gm, counted_basis="mh7_geometry" if gm else "raw (no geometry split available)",
                       episodes_counted=G.get("episodes_main_counted"),
                       episodes_excused_teacher=G.get("episodes_main_excused_teacher_contradicts"),
                       reported={sn: dict(multi_out_s=(L.get(sn) or {}).get("MULTI_out_gt_dark"),
                                          none_out_s=(L.get(sn) or {}).get("NONE_out_gt_dark"),
                                          none_in_s=(L.get(sn) or {}).get("NONE_in_gt_dark"),
                                          geometry=G.get(sn)) for sn in ("late", "excluded")},
                       gt_ghost_mask_s=LB.get("gt_ghost_mask_s"))
    ok = all(v["multi_out_s"] is not None and v["none_out_s"] is not None
             and (v["multi_out_s"] + v["none_out_s"]) / 60.0 <= MH7_MAX_MIN for v in vals.values())
    T.append(mh("MH7", "MULTI + NONE time outside GT-dark <= 1 min per day, main stratum, counted where v4v2 and the teacher "
                       "agree or v4v2 is the only reference, outside the ±30 s GT-defect geometry exclusions (SPEC §6.3); "
                       "excused time, NONE inside GT-dark, the late and the excluded strata reported",
                "PASS" if ok else "FAIL", vals, MH7_MAX_MIN))
    # MH8: memory from the telemetry AND the tracker's own mem_check events (the max of both), GPU class from the
    # telemetry (clip['v5']) AND ROOT_CONFIG.json, completeness from P0 + .done_ markers + the INV-7 recompute
    T.append(mh8_row(R, days, args, fr))
    # provenance audit: dark_link with a cause outside {collapse, no_mask} (SPEC §6.4 'audited by provenance'); the
    # cause is read from the segment record, else from its dark_link event; a MISSING cause counts as outside
    n_bad, bad_l, n_links, n_ev = 0, [], 0, 0
    for d in days:
        for a in (C[d] or {}).get("step2_actions", {}).get("list", []):
            if a["mechanism"] != "dark_link": continue
            n_links += 1
            if a.get("cause") not in ("collapse", "no_mask"): n_bad += 1; bad_l.append(dict(day=d, sid=a["sid"], cause=a.get("cause")))
        n_ev += int(((C[d] or {}).get("ops") or {}).get("dark_links", 0) or 0)
    T.append(mh("AUD1", "dark_link events with a cause outside {collapse, no_mask} == 0", "PASS" if n_bad == 0 else "FAIL",
                dict(n_bad=n_bad, n_link_segments=n_links, n_dark_link_events=n_ev, bad=bad_l[:50]), 0,
                note=None if n_links == n_ev else f"{n_ev} dark_link events vs {n_links} dark_link segments (a resumed "
                                                  "clip re-emits the events of its aborted first run; check the resumes)"))
    # AUD2 (critic finding 12, RUNBOOK G3 'confirm no Tol_15_v2 output in the root'): Tol_15_v2's unknown-reason
    # fallback soft-inherits dark_link segments (SPEC §3.11). LOCAL evidence only: a Tol_15_v2 segment_assignments
    # file inside a cand day dir, or a sibling <root>_reid dir in the local copy; the Volume-wide check stays a listing.
    root = getattr(args, "cand", None)
    found = []
    if root:
        for d in days:
            dd = (R["cand"].get(d) or {}).get("cand_dir") or os.path.join(root, d)
            found += [p for p in glob.glob(os.path.join(dd, "segment_assignments*.json"))]
        rr = os.path.normpath(root) + "_reid"
        if os.path.isdir(rr): found.append(rr)
    T.append(mh("AUD2", "no Tol_15_v2 output for the cand root (local copy: no segment_assignments*.json in its day dirs, "
                        "no sibling <root>_reid)", "PASS" if not found else "FAIL", dict(found=found[:20]), 0,
                note="Volume-wide: `databricks fs ls $VT` must show no <root>_reid"))
    # overall gate
    hard = [t for t in T if t["severity"] == "must_hold"]
    gate = "FAIL" if any(t["status"] == "FAIL" for t in hard) else ("INCOMPLETE" if any(t["status"] == "NOT_EVALUATED" for t in hard) else "PASS")
    if gate == "PASS" and any(t["status"] == "REVIEW" for t in T): gate = "PASS_WITH_REVIEW"
    return T, gate

def report_only(R, days):
    """SPEC §6.4 'reported against expectation' rows: cand vs ref, not pass/fail"""
    out = []
    for d in days:
        c, f = R["cand"].get(d) or {}, R["ref"].get(d) or {}
        def g(x, *ks):
            for k in ks:
                x = (x or {}).get(k) if isinstance(x, dict) else None
            return x
        out.append(dict(day=d,
            fillable_nomask_h=dict(cand=g(c, "gaps", "fillable", "h"), ref=g(f, "gaps", "fillable", "h")),
            nomask_h=dict(cand=g(c, "gaps", "hours"), ref=g(f, "gaps", "hours")),
            orphan_breaks=dict(cand=g(c, "segments", "orphan"), ref=g(f, "segments", "orphan")),
            linked_breaks=dict(cand=(g(c, "segments", "linked_dark") or 0) + (g(c, "segments", "linked_spatial") or 0),
                               ref=(g(f, "segments", "linked_dark") or 0) + (g(f, "segments", "linked_spatial") or 0)),
            correct_min=dict(cand=g(c, "identity", "strata", "main", "correct"), ref=g(f, "identity", "strata", "main", "correct")),
            q_min=dict(cand=g(c, "identity", "strata", "main", "q"), ref=g(f, "identity", "strata", "main", "q")),
            night_q_min=dict(cand=g(c, "identity", "strata", "night", "q"), ref=g(f, "identity", "strata", "night", "q")),
            night_frag_gt_pct=dict(cand=g(c, "fragments", "by_phase", "night", "frag_half_gt_pct"), ref=g(f, "fragments", "by_phase", "night", "frag_half_gt_pct"))))
    return out

def nightgap_eval(path, map22, map23, ref=None):
    """Tol_16_v2 JSON (frozen.pooled = day-B local -> day-A local) scored against the GT via the per-day seed maps
    (map22 = day-A label -> animal, map23 = day-B label -> animal)"""
    if not path: return dict(evaluable=False)
    J = json.load(open(path, encoding="utf-8")); fz = J.get("frozen") or {}
    perm = {int(k): int(v) for k, v in (fz.get("pooled") or {}).items()}
    if not perm or not map22 or not map23: return dict(evaluable=False, why="permutation or seed maps missing")
    ok = all(map23.get(b) is not None and map22.get(a) is not None and map23[b] == map22[a] for b, a in perm.items())
    return dict(evaluable=True, perm=perm, correct=bool(ok), min_margin=fz.get("min_margin"), support=fz.get("support"),
                verdict=J.get("verdict"), ref=ref or NIGHTGAP_REF_0422_0423)

# ============================================================================================================
# 12 · driver
# ============================================================================================================
def find_gt_file(gt_dir, day):
    names = {"20240422": "segment_assignments_reviewed.json", "20240423": "segment_assignments.json"}
    nm = names.get(day, "segment_assignments.json")
    for p in (os.path.join(gt_dir, day, nm), os.path.join(gt_dir, f"{nm[:-5]}_{day}.json"), os.path.join(gt_dir, nm)):
        if os.path.exists(p): return p
    raise FileNotFoundError(f"no GT assignment file for {day} under {gt_dir} (looked for {nm})")

def _scan_worker(job):
    return (job["arm"], job.get("key", job["day"]), scan_day(job))

def run_scans(jobs, n_jobs=1):
    """{(arm, key): scan} -- key = the job's 'key' (a window dir for G2) or its day"""
    names = ", ".join("%s:%s" % (j["arm"], j.get("key", j["day"])) for j in jobs)
    print(f"[tol48] {len(jobs)} scans ({names}) with {n_jobs} worker(s)", flush=True)
    SCANS = {}
    def note(arm, key, S): print(f"  scanned {arm}:{key} rows={len(S['A']['ts'])} in {S['scan_s']} s", flush=True)
    if n_jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(n_jobs, len(jobs))) as ex:
            for arm, key, S in ex.map(_scan_worker, jobs): SCANS[(arm, key)] = S; note(arm, key, S)
    else:
        for j in jobs:
            arm, key, S = _scan_worker(j); SCANS[(arm, key)] = S; note(arm, key, S)
    return SCANS

def score(args):
    t_all = time.time()
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    cand_dd = dict(kv.split("=", 1) for kv in (args.cand_day_dir or []))
    ref_dd = dict(kv.split("=", 1) for kv in (args.ref_day_dir or []))
    teach = dict(kv.split("=", 1) for kv in (args.teacher or []))
    wm_path = args.week_map_gt or os.path.join(args.gt_dir, "week_ident_map.json")
    week_map = load_week_map(wm_path)
    work = args.work or os.path.join(os.path.dirname(os.path.abspath(args.out)), "tol48_work")
    os.makedirs(work, exist_ok=True)
    if args.train_summary:
        C_ = json.load(open(args.train_summary))["constants"]
        FRAG_LO["dayl"] = float(C_["frag_lo[dayl]"]["value"]); FRAG_LO["dark"] = float(C_["frag_lo[dark]"]["value"])
        globals()["OVER_HI"] = float(C_["over_hi"]["value"])
    arms = {"cand": (args.cand, cand_dd), "ref": (args.ref, ref_dd)}
    jobs = []
    for arm, (root, dd) in arms.items():
        for d in days:
            gt_dir = os.path.join(args.ref if not args.gt_root else args.gt_root, d)
            jobs.append(dict(arm=arm, day=d, cand_dir=os.path.join(root, dd.get(d, d)), gt_dir=gt_dir,
                             gt_assign=find_gt_file(args.gt_dir, d), week_map=week_map,
                             teacher_dir=teach.get(d), teacher_rules=(args.teacher_rules_0422 if d == "20240422" and teach.get(d) else None),
                             frag_lo=dict(FRAG_LO)))
    tol30_src_sha0 = sha256_file(args.tol30) if (args.run_tol30 and args.tol30 and os.path.exists(args.tol30)) else None
    # Tol_30 is patched concurrently by another workflow: snapshot (and check the pin) BEFORE the hour-long scans, so a
    # wrong pin fails fast and a patch landing mid-run cannot reach either arm (both arms use this one snapshot)
    snap = tol30_snapshot(args.tol30, work, args.tol30_sha) if args.run_tol30 else None
    if snap is not None:
        print(f"[tol48] Tol_30 snapshot sha256 {snap['sha256']} taken at start (pin {args.tol30_sha or 'run-time'})", flush=True)
    SCANS = run_scans(jobs, args.jobs)
    R = {"cand": {}, "ref": {}, "telemetry": {}, "tol30": {}}
    SEGROWS = {}
    for (arm, day), S in SCANS.items():
        M, sr = day_metrics(S); R[arm][day] = M; SEGROWS[(arm, day)] = sr
        print(f"  metrics {arm}:{day} nomask {M['gaps']['hours']:.3f} h, segments {M['segments']['n']}, "
              f"merge_resolved {M['segments']['merge_resolved']}, night frag(own) {M['fragments']['by_phase'].get('night', {}).get('frag_half_own_pct')} %", flush=True)
    # ---- Tol_30 (identical pinned snapshot for both arms) ----
    if args.run_tol30:
        R["tol30"]["snapshot"] = {k: v for k, v in snap.items() if k != "path"}
        print(f"[tol48] Tol_30 snapshot sha256 {snap['sha256']} (review flags match built-in: {snap['review_flags_match_builtin']})", flush=True)
        for arm, (root, dd) in arms.items():
            for d in days:
                S = SCANS[(arm, d)]; sr = SEGROWS[(arm, d)]; M = R[arm][d]
                sm = M["mapping_seed"]
                gt_asg = transferred_gt(S, sr, sm.get("map") if sm.get("method") == "seed_majority_10min" else None)
                info = run_tol30(snap, root, dd.get(d, d), d, os.path.join(work, arm), gt_asg)
                R["tol30"][f"{arm}:{d}"] = {k: v for k, v in info.items()}
                if not info.get("ran"): print(f"  Tol_30 {arm}:{d} NOT run: {info.get('why')}", flush=True); continue
                print(f"  Tol_30 {arm}:{d} {info['runtime_s']} s | {info.get('stdout_vs_gt')}", flush=True)
                runs, J = load_t30_runs(info["out"])
                lab, flg = label_arrays(S, sr, runs, snap["review_flags"])
                mp = seed_mapping(S, sr, lab)
                if sm.get("method") == "seed_majority_10min": mp = sm
                M["mapping"] = {k: v for k, v in mp.items()}
                # best permutation over the whole day (check only)
                V = np.zeros((N_SLOTS, N_SLOTS)); A = S["A"]
                for j in range(N_SLOTS):
                    rows = np.flatnonzero(A["g_idt"][:, j] & (A["g_c"][:, j] >= 0))
                    kk = A["g_c"][rows, j].astype(int); L = lab[rows, kk].astype(int); an = A["g_anim"][rows, j].astype(int)
                    ok = (L >= 0) & (an >= 0)
                    np.add.at(V, (L[ok], an[ok]), 1)
                ri, cj = linear_sum_assignment(-V)
                best = {int(a): int(b) for a, b in zip(ri, cj) if V[a, b] > 0}
                M["mapping"]["best_perm_check"] = best
                M["mapping"]["best_perm_agrees"] = all(mp["map"].get(k) == v for k, v in best.items())
                mech = {r["idx"]: seg_mech(r, events_by_seg(S["cev"])) for r in S["cseg"]}
                IC = identity_classes(S, lab, flg, mp["map"], mech)
                IC.pop("_cls", None)
                M["identity"] = IC
                M["fills"] = fill_metrics(S, lab, mp["map"], mech) if S["teacher_mode"] else dict(teacher=None)
                M["tol30_slot_time"] = slot_time_summary(S, runs, gt_asg, snap["review_flags"])
                M["tol30_run_flags"] = dict(Counter(f for rl in runs.values() for r in rl for f in r["flags"]))
                if arm == "ref":
                    orig = load_gt_assign(find_gt_file(args.gt_dir, d), d, None)
                    same = {k: v for k, v in gt_asg.items() if v is not None}
                    M["transferred_gt_vs_original"] = dict(n=len(same), agree=sum(1 for k, v in same.items() if orig.get(k) == v),
                                                           disagree=[k for k, v in same.items() if orig.get(k) != v])
                # week mode (5e)
        wk = {"cand": args.week_map_cand, "ref": args.week_map_ref}
        for arm in ("cand", "ref"):
            p = wk.get(arm)
            if arm == "cand" and not p and args.nightgap_cand:
                J = json.load(open(args.nightgap_cand)); perm = (J.get("frozen") or {}).get("pooled") or {}
                if perm:
                    p = os.path.join(work, "week_map_cand_from_nightgap.json")
                    json.dump({"global_ident_map": {days[0]: {str(i): i for i in range(N_SLOTS)}, days[1]: {str(k): int(v) for k, v in perm.items()}},
                               "source": f"Tol_48 from {args.nightgap_cand}"}, open(p, "w"), indent=1)
            if not p or len(days) < 2: continue
            root, dd = arms[arm]
            if any(dd.get(d, d) != d for d in days): continue
            info = run_tol30_week(snap, os.path.join(work, arm), days, p)
            R["tol30"][f"{arm}:week"] = info
            for d in days:
                S = SCANS[(arm, d)]; sr = SEGROWS[(arm, d)]; M = R[arm][d]
                runs, _ = load_t30_runs(info["out"], key_prefix=f"{d}/")
                lab, flg = label_arrays(S, sr, runs, snap["review_flags"])
                m22 = R[arm][days[0]].get("mapping", {}).get("map", {})
                mech = {r["idx"]: seg_mech(r, events_by_seg(S["cev"])) for r in S["cseg"]}
                IC = identity_classes(S, lab, flg, m22, mech); IC.pop("_cls", None)
                M["identity_week"] = dict(week_map=p, anchored_on=days[0], strata=IC["strata"],
                                          wrong_by_mechanism=IC["wrong_by_mechanism"], n_wrong_spans=IC["n_wrong_spans"],
                                          wrong_spans=IC["wrong_spans"][:50])
    # ---- night gap (MH6) ----
    R["nightgap"] = {}
    if len(days) >= 2:
        m22 = (R["cand"][days[0]].get("mapping") or R["cand"][days[0]].get("mapping_seed") or {}).get("map")
        m23 = (R["cand"][days[1]].get("mapping") or R["cand"][days[1]].get("mapping_seed") or {}).get("map")
        R["nightgap"]["cand"] = nightgap_eval(args.nightgap_cand, {int(k): v for k, v in (m22 or {}).items()}, {int(k): v for k, v in (m23 or {}).items()})
        r22 = (R["ref"][days[0]].get("mapping") or R["ref"][days[0]].get("mapping_seed") or {}).get("map")
        r23 = (R["ref"][days[1]].get("mapping") or R["ref"][days[1]].get("mapping_seed") or {}).get("map")
        R["nightgap"]["ref"] = nightgap_eval(args.nightgap_ref, {int(k): v for k, v in (r22 or {}).items()}, {int(k): v for k, v in (r23 or {}).items()})
        if r22 and r23:
            perm = NIGHTGAP_REF_0422_0423["perm"]
            R["nightgap"]["ref_documented_perm_correct_under_ref_seed_maps"] = all(int(r23.get(b, r23.get(str(b), -1))) == int(r22.get(a, r22.get(str(a), -2))) for b, a in perm.items())
    # ---- telemetry (explicit --telemetry, else found next to the cand root: Tol_47:1581 naming) ----
    tel = dict(kv.split("=", 1) for kv in (args.telemetry or []))
    for d in days:
        if f"cand:{d}" not in tel:
            p = find_telemetry(args.cand, cand_dd.get(d, d))
            if p: tel[f"cand:{d}"] = p
    for key, p in tel.items():
        arm, d = key.split(":", 1)
        rh = (R.get(arm, {}).get(d) or {}).get("hours")
        R["telemetry"][key] = telemetry_summary(p, rh)
    # ---- deltas + must-holds ----
    DEL = {}
    for d in days:
        D, nz, only = deltas(R["cand"][d], R["ref"][d])
        DEL[d] = dict(n_numeric=len(D), n_nonzero=len(nz), nonzero_keys=nz[:500], only_one_arm=only[:200], deltas=D)
    table, gate = must_holds(R, days, args)
    if snap is not None:                       # Tol_30 is patched concurrently: record its sha before AND after
        R["tol30"]["source_sha256_at_start"] = tol30_src_sha0
        R["tol30"]["source_sha256_at_end"] = sha256_file(args.tol30) if os.path.exists(args.tol30) else None
        R["tol30"]["pinned_by"] = "--tol30_sha" if args.tol30_sha else "run-time snapshot (sha recorded, no CLI pin)"
        if R["tol30"]["source_sha256_at_end"] != snap["sha256"]:
            print(f"  NOTE: {args.tol30} changed during the run; both arms were scored with the snapshot {snap['sha256'][:12]}")
    out = dict(tool="Tol_48_score_retrack", created=datetime.now().isoformat(timespec="seconds"),
               tol48_sha256=TOL48_SHA256, tol48_sha256_on_disk_at_end=sha256_file(os.path.abspath(__file__)),
               tol_v5_core=(dict(path=CORE.__file__, sha256=CORE_SHA256) if CORE is not None else None),
               args=vars(args), constants=dict(ASSOC_IOS=ASSOC_IOS, MULTI_FRAC=MULTI_FRAC, GT_SAME_ANIMAL=GT_SAME_ANIMAL,
                   NONE_IOS_MAX=NONE_IOS_MAX, TRUST_GT_IOS=TRUST_GT_IOS, MERGE_GUARD_S=MERGE_GUARD_S, SWITCH_MIN_S=SWITCH_MIN_S,
                   SEED_MAP_S=SEED_MAP_S, OVERLAP=[OVERLAP_THR, OVERLAP_MIN_S], ORPHAN=[ORPHAN_LINK_IOU, ORPHAN_LINK_S],
                   FRAG_LO=FRAG_LO, OVER_HI=globals()["OVER_HI"], D8_DELTA_MIN=D8_DELTA_MIN, MH7_MAX_MIN=MH7_MAX_MIN,
                   IDENT_EXCL=IDENT_EXCL, LATE_STRATUM=LATE_STRATUM, GEOM_EXCL_SLOT=GEOM_EXCL_SLOT, GEOM_EXCL_CENTRES=GEOM_EXCL_CENTRES),
               gt=dict(week_map=wm_path, week_map_sha256=sha256_file(wm_path),
                       files={d: dict(path=find_gt_file(args.gt_dir, d), sha256=sha256_file(find_gt_file(args.gt_dir, d))) for d in days}),
               must_hold=table, gate=gate, report_only=report_only(R, days), deltas=DEL,
               cand=R["cand"], ref=R["ref"], tol30=R["tol30"], nightgap=R["nightgap"], telemetry=R["telemetry"],
               runtime_s=round(time.time() - t_all, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, indent=1, default=str)
    print("\n=== MUST-HOLD TABLE (cand) ===")
    for t in table:
        print(f"  {t['id']:5s} {t['status']:14s} {t['name']}" + (f"  value={json.dumps(t['value'], default=str)[:160]}" if t["value"] is not None else "") + (f"  [{t['note']}]" if t.get("note") else ""))
    print(f"  GATE: {gate}")
    for d in days:
        print(f"  deltas {d}: {DEL[d]['n_nonzero']} non-zero of {DEL[d]['n_numeric']} numeric leaves")
    print(f"[tol48] wrote {args.out} ({time.time() - t_all:.0f} s)")
    return out

# ============================================================================================================
# 12b · coded gate evaluators: G2 (A10 pilot windows) and G5a (04-24, held out for the tracker)
#       Both write ONE verdict JSON. Gate values: PASS (proceed) | FAIL (stop) | INCOMPLETE (evidence missing: fix
#       the local copy / inputs and re-run; never proceed) | REVIEW (every automated check passed but a SPEC finding
#       needs the user's decision before the next launch, e.g. zero step-2 actions in a pilot window).
# ============================================================================================================
GATE_ORDER = {"PASS": 0, "REVIEW": 1, "INCOMPLETE": 2, "FAIL": 3}
GATE_SEMANTICS = {"PASS": "proceed", "FAIL": "stop (a must-hold failed)",
                  "INCOMPLETE": "stop: evidence missing (a check could not be evaluated) -- fix the inputs and re-run",
                  "REVIEW": "all automated checks passed, but a SPEC finding needs the user's decision before proceeding"}

def gate_of(rows):
    g = "PASS"
    for t in rows:
        st, sev = t["status"], t.get("severity", "must_hold")
        if st == "FAIL" and sev == "must_hold": w = "FAIL"
        elif st == "NOT_EVALUATED" and sev == "must_hold": w = "INCOMPLETE"
        elif st in ("REVIEW", "FAIL") or (st == "NOT_EVALUATED" and sev != "must_hold"): w = "REVIEW"
        else: w = "PASS"
        if GATE_ORDER[w] > GATE_ORDER[g]: g = w
    return g

def load_detections(day_dir):
    """detections_<clip>.json.gz (Tol_47 _write_detections: {clip, detector, det_run_tag, op_threshold, band_px,
    records:[{clip, frame_idx, t_sec, kind, n_raw, det_ms, n_inband, raw, kept_idx}]}); the RLEs are dropped"""
    files, recs = {}, []
    for p in sorted(glob.glob(os.path.join(day_dir, "detections_*.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f: J = json.load(f)
        stem = os.path.basename(p)[len("detections_"):-len(".json.gz")]
        files[clip_key(stem)] = dict(n=len(J.get("records") or []), detector=J.get("detector"), band_px=J.get("band_px"))
        for r in J.get("records") or []:
            recs.append(dict(clip=clip_key(r.get("clip") or J.get("clip") or stem), frame_idx=r.get("frame_idx"),
                             t_sec=r.get("t_sec"), kind=r.get("kind"), n_inband=r.get("n_inband"), n_raw=r.get("n_raw"),
                             n_kept=len(r.get("kept_idx") or [])))
    return files, recs

def pen_count_audit(recs, cev, v5_fixes=True):
    """critic finding 5 / RUNBOOK G2: every detect() call with n_inband > N_CALVES logged pen_count_refused and opened
    nothing. Per call kind (Tol_47): seed -> pen_count_refused(seed) and no `seeded` on that frame (§3.8-4, v5_fixes);
    pass -> the `pass` event has pen_refused true and opened [] (§3.8-3, v5_fixes) and >= 1 pen_count_refused; dark
    -> the `dark_pass` event has refused 'pen_count', no links / opens, and pen_count_refused(dark) (§3.6 B). In every
    kind no segment_open with a guarded reason (reopen / displaced / dark_link / seed_*) at that frame's wall.
    Repairs are not pen-count guarded (SPEC §3.8 leaves them unchanged) and are listed, not failed."""
    def key(clip, t): return (clip_key(clip), None if t is None else round(float(t), 2))
    pen, passes, dpasses, seeded = defaultdict(list), {}, {}, {}
    for e in cev:
        ty = e.get("type")
        if ty == "pen_count_refused": pen[key(e.get("clip"), e.get("t"))].append(e.get("path"))
        elif ty == "pass": passes[key(e.get("clip"), e.get("t"))] = e
        elif ty == "dark_pass": dpasses[key(e.get("clip"), e.get("t"))] = e
        elif ty == "seeded": seeded[key(e.get("clip"), e.get("t"))] = e
    opens = sorted((e["_t"], e.get("reason"), e.get("seg_id")) for e in cev
                   if e.get("type") == "segment_open" and e.get("_t") is not None)
    ow = [o[0] for o in opens]
    rows, n_over, n_na = [], 0, 0
    for r in recs:
        if r["n_inband"] is None or r["n_inband"] <= N_SLOTS: continue
        n_over += 1
        k = key(r["clip"], r["t_sec"]); why = []; paths = pen.get(k, [])
        try: wall = ts_of(clip_start(r["clip"])) + float(r["t_sec"])
        except Exception: wall = None
        near = [] if wall is None else opens[bisect.bisect_left(ow, wall - WALL_TOL_S):bisect.bisect_right(ow, wall + WALL_TOL_S)]
        guarded = [o for o in near if o[1] in GUARDED_OPEN_REASONS]
        repairs = [o for o in near if o[1] not in GUARDED_OPEN_REASONS]
        na = False
        if r["kind"] == "seed":
            if not v5_fixes: na = True
            else:
                if "seed" not in paths: why.append("no pen_count_refused(path=seed)")
                if k in seeded: why.append("seeded on this call")
        elif r["kind"] == "pass":
            pe = passes.get(k)
            if pe is None: why.append("no pass event for this call")
            elif not v5_fixes: na = not (pe.get("dark") or {}).get("links") and not (pe.get("dark") or {}).get("opens")
            else:
                # Tol_47 detector_pass: on EVERY scheduled call with detections, v5_fixes runs core.reopen_considered,
                # which refuses whenever n_inband > N_CALVES and then logs pen_count_refused(path='scheduled')
                if pe.get("opened"): why.append(f"pass opened slots {pe['opened']}")
                if not pe.get("pen_refused"): why.append("pass.pen_refused is false")
                if "scheduled" not in paths: why.append(f"no pen_count_refused(path=scheduled) (paths {paths})")
            if pe is not None and (pe.get("dark") or {}).get("links"): why.append(f"dark step linked {pe['dark']['links']}")
            if pe is not None and (pe.get("dark") or {}).get("opens"): why.append(f"dark step opened {pe['dark']['opens']}")
        elif r["kind"] == "dark":
            de = dpasses.get(k)
            if de is None: why.append("no dark_pass event for this call")
            else:
                if de.get("links") or de.get("opens"): why.append("dark_pass linked / opened")
                # RUNBOOK G2: the refusal is recorded by dark_pass.refused == 'pen_count' (the dark step's own
                # decision) and/or pen_count_refused(path=dark); a dark-only call whose dark step found nothing
                # actionable logs the dark_pass alone. Either record suffices, nothing opened is required.
                if de.get("refused") != "pen_count" and "dark" not in paths:
                    why.append(f"dark_pass.refused = {de.get('refused')!r} and no pen_count_refused(path=dark)")
        else: why.append(f"unknown detect kind {r['kind']!r}")
        if guarded and not na: why.append(f"guarded segment_open at this frame: {[(o[1], o[2]) for o in guarded]}")
        n_na += int(na)
        rows.append(dict(clip=r["clip"], frame_idx=r["frame_idx"], t_sec=r["t_sec"], kind=r["kind"], n_inband=r["n_inband"],
                         pen_paths=paths, ok=(not why) or na, not_applicable=na, why=why,
                         repairs_same_frame=[(o[1], o[2]) for o in repairs]))
    bad = [x for x in rows if not x["ok"]]
    return dict(n_calls=len(recs), n_over_pen=n_over, n_not_applicable=n_na, n_violations=len(bad),
                violations=bad[:100], calls_over_pen=rows[:200])

def _det_per_frame(recs):
    c = Counter((r["clip"], r["frame_idx"]) for r in recs)
    return max(c.values()) if c else 0

def _v5_fixes_of(M, tel):
    for x in ((M.get("ops") or {}).get("root_config") or {}).values():
        if x.get("v5_fixes") is not None: return str(x["v5_fixes"]).lower() in ("on", "true", "1")
    if tel and tel.get("v5_fixes") is not None: return str(tel["v5_fixes"]).lower() in ("on", "true", "1")
    return True                                   # unknown: audit strictly (the pilot runs the v5 defaults)

def g2_extra_checks(M, S, tel):
    """the G2-only rows (critic finding 5; SPEC §6.1 G2; RUNBOOK G2)"""
    rows = []
    cand_dir = S["cand_dir"]
    files, recs = load_detections(cand_dir)
    stems = sorted({clip_key(c["stem"]) for c in S["clips"] if c.get("cand")})
    # G2.STEP2 -- >= 1 dark_link or dark_open in the window; zero is a FINDING for the user (SPEC §6.1 G2)
    o = M.get("ops") or {}
    n2 = int(o.get("dark_links", 0) or 0) + int(o.get("dark_opens", 0) or 0)
    rows.append(mh("G2.STEP2", ">= 1 dark_link or dark_open in the window (zero = a finding: report to the user, wait)",
                   "PASS" if n2 >= 1 else "REVIEW", dict(dark_links=o.get("dark_links"), dark_opens=o.get("dark_opens"),
                   dark_passes=o.get("dark_passes"), dark_open_refused=o.get("dark_open_refused")), 1, severity="finding"))
    # G2.DET1 -- max detect() calls per frame <= 1, from the telemetry AND the detections log
    t_max = tel.get("max_det_calls_per_frame") if tel.get("available") else None
    d_max = _det_per_frame(recs) if recs else None
    v = dict(telemetry_max=t_max, detections_max=d_max, n_detect_calls=len(recs))
    if t_max is None and d_max is None: st = "NOT_EVALUATED"
    else: st = "PASS" if max(x for x in (t_max, d_max) if x is not None) <= G2_MAX_DET_PER_FRAME else "FAIL"
    rows.append(mh("G2.DET1", "max_det_calls_per_frame <= 1 (telemetry and detections_<clip>.json.gz)", st, v, G2_MAX_DET_PER_FRAME))
    # G2.PEN -- n_inband > 4 => pen_count_refused and nothing opened
    if not files:
        rows.append(mh("G2.PEN", "every detect() call with n_inband > 4 logged pen_count_refused and opened nothing",
                       "NOT_EVALUATED", note=f"no detections_*.json.gz in {cand_dir} (copy them with the masks)"))
    else:
        au = pen_count_audit(recs, S["cev"], _v5_fixes_of(M, tel))
        rows.append(mh("G2.PEN", "every detect() call with n_inband > 4 logged pen_count_refused and opened nothing",
                       "PASS" if au["n_violations"] == 0 else "FAIL", au, 0))
    # G2.CLIPS -- every tracked clip has its detections log and its telemetry entry (Tol_47 writes both before .done_)
    miss_det = [s for s in stems if s not in files]
    tk = set(tel.get("clip_keys") or []) if tel.get("available") else None
    miss_tel = [s for s in stems if tk is not None and s not in tk]
    st = "NOT_EVALUATED" if tk is None else ("PASS" if not miss_det and not miss_tel else "FAIL")
    rows.append(mh("G2.CLIPS", "every tracked clip has detections_<clip>.json.gz and a telemetry entry", st,
                   dict(n_clips=len(stems), missing_detections=miss_det, missing_telemetry=miss_tel),
                   note=None if tk is not None else "telemetry not found (--telemetry cand:<window dir>=<json>)"))
    # G2.TIM -- per-component timings present (Tol_47 laps, synchronised)
    if not tel.get("available"): st, v = "NOT_EVALUATED", None
    else:
        v = dict(missing=tel.get("timings_missing"), components=tel.get("timing_components"), required=list(TIMING_KEYS_REQUIRED),
                 clips_without_v5=tel.get("n_clips", 0) - tel.get("n_clips_v5", 0))
        st = "PASS" if not tel.get("timings_missing") and v["clips_without_v5"] == 0 and tel.get("n_clips") else "FAIL"
    rows.append(mh("G2.TIM", "per-component timings present in every clip's telemetry", st, v))
    # G2.PREC -- the precision assert held: matmul precision 'highest' in the telemetry echo and ROOT_CONFIG
    precs = {"telemetry": tel.get("matmul_precision") if tel.get("available") else None}
    for k, x in ((M.get("ops") or {}).get("root_config") or {}).items(): precs[f"root_config:{k}"] = x.get("matmul_precision")
    have = {k: x for k, x in precs.items() if x is not None}
    st = "NOT_EVALUATED" if not have else ("PASS" if all(x == "highest" for x in have.values()) else "FAIL")
    rows.append(mh("G2.PREC", "float32 matmul precision 'highest' (telemetry echo + ROOT_CONFIG echo; detect() raises otherwise)",
                   st, precs, "highest"))
    return rows

def sizing_of(tel):
    """RUNBOOK G2 'Sizing for L4': per-clip wall minutes and fps -> projected hours per recorded day, monitor thresholds"""
    if not tel.get("available"): return dict(available=False)
    med = tel.get("per_clip_wall_min_median"); fps = tel.get("per_clip_fps_median")
    return dict(available=True, per_clip_wall_min_median=med, per_clip_fps_median=fps,
                wall_h_per_24h_recorded=tel.get("wall_h_per_24h_recorded"), recorded_h=tel.get("recorded_h"),
                projected_h_18h_day=round(tel["wall_h_per_24h_recorded"] * 18 / 24, 2) if tel.get("wall_h_per_24h_recorded") else None,
                monitor_stall_min=round(2.5 * med, 1) if med else None, monitor_fps_alarm=round(0.5 * fps, 3) if fps else None,
                host_rss_gib=tel.get("host_rss_gib"), cpu_ms_per_frame_max=tel.get("cpu_ms_per_frame_max"))

def _slim(M):
    """the per-window / per-day evidence the verdict JSON carries (the full metrics stay in the scoring JSON)"""
    keep = ("rows", "hours", "precondition", "done", "gt_mode", "gt_docx", "switches", "step2_actions", "sentinels",
            "mapping_seed", "segments", "gaps", "labels", "overlap", "fragments", "ops", "teacher")
    out = {k: M.get(k) for k in keep if k in M}
    if out.get("gaps"): out["gaps"] = {k: v for k, v in out["gaps"].items() if k != "list"}
    if out.get("segments"): out["segments"] = {k: v for k, v in out["segments"].items() if k != "orphan_list"}
    return out

def _verdict_header(tool_mode, args):
    return dict(tool="Tol_48_score_retrack", mode=tool_mode, created=datetime.now().isoformat(timespec="seconds"),
                tol48_sha256=TOL48_SHA256, tol48_sha256_on_disk_at_end=sha256_file(os.path.abspath(__file__)),
                tol_v5_core=(dict(path=CORE.__file__, sha256=CORE_SHA256) if CORE is not None else None),
                args={k: v for k, v in vars(args).items()}, gate_semantics=GATE_SEMANTICS)

def g2_evaluate(args):
    """critic finding 5: the coded G2 pilot evaluator. Per window (--window DAY=SUBDIR, e.g. 20240423=
    20240423_w1152-1259): scan the cand window against the v4v2 GT (identity via week_ident_map), and the v4v2 ref
    restricted to the same window (re-scored by the same code, reported, not gated); must-holds subset G2_SUBSET plus
    G2.STEP2 / DET1 / PEN / CLIPS / TIM / PREC. MH3-MH6 need Tol_30 and the night gap and do not apply to a pilot."""
    t_all = time.time()
    wins = [kv.split("=", 1) for kv in (args.window or [])]
    if not wins: raise SystemExit("--g2 needs at least one --window DAY=SUBDIR")
    teach = dict(kv.split("=", 1) for kv in (args.teacher or []))
    wm_path = args.week_map_gt or os.path.join(args.gt_dir, "week_ident_map.json")
    week_map = load_week_map(wm_path)
    tel_arg = dict(kv.split("=", 1) for kv in (args.telemetry or []))
    jobs = []
    for day, sub in wins:
        gt_dir = os.path.join(args.gt_root or args.ref, day)
        base = dict(day=day, gt_dir=gt_dir, gt_assign=find_gt_file(args.gt_dir, day), week_map=week_map,
                    teacher_dir=teach.get(day), frag_lo=dict(FRAG_LO),
                    teacher_rules=(args.teacher_rules_0422 if day == "20240422" and teach.get(day) else None))
        jobs.append(dict(base, arm="cand", key=sub, cand_dir=os.path.join(args.cand, sub)))
        if not args.no_ref:
            jobs.append(dict(base, arm="ref", key=sub, cand_dir=os.path.join(args.ref, day), window=parse_window(sub)))
    SCANS = run_scans(jobs, args.jobs)
    out = _verdict_header("G2", args); out["windows"] = {}
    gates = []
    for day, sub in wins:
        S = SCANS[("cand", sub)]
        Mc, _sr = day_metrics(S)
        Mr = day_metrics(SCANS[("ref", sub)])[0] if ("ref", sub) in SCANS else None
        # a per-DAY key is accepted only when that day has one window (two pilot windows of 04-23 have two telemetry
        # files: Metrics_track_v5_<root>_<day>_w<from>-<to>.json)
        one = sum(1 for d_, _s in wins if d_ == day) == 1
        tp = tel_arg.get(f"cand:{sub}") or (tel_arg.get(f"cand:{day}") if one else None) or find_telemetry(args.cand, sub)
        tel = telemetry_summary(tp, Mc.get("hours"))
        R = {"cand": {day: Mc}, "ref": {day: Mr} if Mr else {}, "telemetry": {f"cand:{day}": tel}, "nightgap": {}}
        rows_all, _g = must_holds(R, [day], args)
        rows = [t for t in rows_all if t["id"] in G2_SUBSET] + g2_extra_checks(Mc, S, tel)
        g = gate_of(rows); gates.append(g)
        D = deltas(Mc, Mr)[0:2] if Mr else None
        out["windows"][sub] = dict(day=day, gate=g, must_hold=rows, telemetry=({k: v for k, v in tel.items() if k != "clips"}),
                                   sizing=sizing_of(tel), cand=_slim(Mc), ref=_slim(Mr) if Mr else None,
                                   ref_note="v4v2 restricted to the same window, re-scored by the same code (report only)",
                                   n_nonzero_deltas=(len(D[1]) if D else None),
                                   step2_unverifiable=[a for a in (Mc.get("step2_actions") or {}).get("list", [])
                                                       if a["verdict"] == "unknown"])
        print(f"\n=== G2 window {sub} ({day}) : {g} ===")
        for t in rows:
            print(f"  {t['id']:9s} {t['status']:14s} {t['name']}" + (f"  [{str(t['note'])[:200]}]" if t.get("note") else ""))
    gate = max(gates, key=lambda x: GATE_ORDER[x]) if gates else "INCOMPLETE"
    out.update(gate=gate, proceed=(gate == "PASS"), runtime_s=round(time.time() - t_all, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, indent=1, default=str)
    print(f"\n  G2 GATE: {gate}  ({GATE_SEMANTICS[gate]})\n[tol48] wrote {args.out}")
    return out

def root_config_identity(cand_root, days):
    """G5a: ROOT_CONFIG.json identical across the days of a root (core.root_config_record keys: widgets without
    stage/from/to/max_*, the four shas, the GPU class, the band)"""
    recs = {}
    for d in days:
        rc = read_root_config(os.path.join(cand_root, d))
        for lvl, x in rc.items():
            if "record" in x: recs[f"{d}:{lvl}"] = x["record"]
    keys = sorted(recs)
    if len(keys) < 2 or not any(k.startswith(days[-1] + ":") for k in keys):
        return mh("G5a.ROOTCFG", "ROOT_CONFIG.json of 04-24 identical to 04-22 / 04-23 (one config per root)", "NOT_EVALUATED",
                  dict(found=keys), note="copy ROOT_CONFIG.json of every day dir and of the root")
    ref = recs[keys[0]]
    diff = {k: sorted(kk for kk in set(ref) | set(recs[k]) if ref.get(kk) != recs[k].get(kk)) for k in keys[1:]}
    diff = {k: v for k, v in diff.items() if v}
    return mh("G5a.ROOTCFG", "ROOT_CONFIG.json of 04-24 identical to 04-22 / 04-23 (one config per root)",
              "PASS" if not diff else "FAIL", dict(compared=keys, differing=diff), note=None)

def g5a_evaluate(args):
    """critic finding 6: the automated part of G5a (SPEC §6.1 G5a-2..4, RUNBOOK G5a). v5 masks of 04-24 are
    associated to Tracking_v4v2p2/20240424 (min-area IoS >= 0.6, the §6.3 rule) and each v4v2p2 mask gets the calf of
    the user's docx interval GT under its video label (DocxGT). Must-holds G5A_SUBSET + ROOT_CONFIG identity; the
    04-23->24 night gap and MH7 are REPORTED. The v4v2p2 ref arm (scored against the same docx GT by the same code)
    is reported for comparison, e.g. its own known in-segment hops (08:07 drift, 08:36-09:19 feeding exchange)."""
    t_all = time.time()
    day = "20240424"
    if not (args.gt24 and args.asg24 and args.gt_root): raise SystemExit("--g5a needs --gt_root, --gt24 and --asg24")
    cand_sub = dict(kv.split("=", 1) for kv in (args.cand_day_dir or [])).get(day, day)
    gt_dir = os.path.join(args.gt_root, day)
    base = dict(day=day, gt_dir=gt_dir, gt_assign=args.asg24, gt_docx=args.gt24, frag_lo=dict(FRAG_LO))
    jobs = [dict(base, arm="cand", cand_dir=os.path.join(args.cand, cand_sub))]
    if not args.no_ref: jobs.append(dict(base, arm="ref", cand_dir=gt_dir))
    SCANS = run_scans(jobs, args.jobs)
    S = SCANS[("cand", day)]
    Mc, _sr = day_metrics(S)
    Mr = day_metrics(SCANS[("ref", day)])[0] if ("ref", day) in SCANS else None
    tp = dict(kv.split("=", 1) for kv in (args.telemetry or [])).get(f"cand:{day}") or find_telemetry(args.cand, cand_sub)
    tel = telemetry_summary(tp, Mc.get("hours"))
    R = {"cand": {day: Mc}, "ref": {day: Mr} if Mr else {}, "telemetry": {f"cand:{day}": tel}, "nightgap": {}}
    rows_all, _g = must_holds(R, [day], args)
    rows = [t for t in rows_all if t["id"] in G5A_SUBSET]
    rows.append(root_config_identity(args.cand, ["20240422", "20240423", cand_sub]))
    # GT coverage sanity: the docx GT must give identities to a real share of the aligned rows, else every
    # identity must-hold above passes vacuously
    A = S["A"]; idt = int((A["g_idt"] & (A["g_area"] > 0)).sum()); tot = int((A["g_area"] > 0).sum())
    frac = idt / tot if tot else 0.0
    rows.append(mh("G5a.GTCOV", "the docx GT identifies >= 50 % of the aligned v4v2p2 animal-frames (non-vacuous checks)",
                   "PASS" if frac >= 0.5 else "FAIL", dict(identity_trusted_frac=round(frac, 4), n=tot), 0.5,
                   note="docx: 16.6 h scored, +-3 min around change points and 22:49-24:00 excluded (DATA_SPLITS.md:213)"))
    # 04-23 -> 04-24 night gap: REPORTED (the docx calf index is global only under v4v2's chain, so 'agrees' means
    # 'agrees with the v4v2 chain', not 'correct')
    ng = dict(evaluable=False, why="needs --nightgap_cand (Tol_16_v2 04-23->24 on the cand root) and --g3_metrics")
    if args.nightgap_cand and args.g3_metrics:
        G3 = json.load(open(args.g3_metrics, encoding="utf-8"))
        c23 = (G3.get("cand") or {}).get("20240423") or {}
        m23 = (c23.get("mapping") or c23.get("mapping_seed") or {}).get("map") or {}
        m24 = (Mc.get("mapping_seed") or {}).get("map") or {}
        ng = nightgap_eval(args.nightgap_cand, {int(k): int(v) for k, v in m23.items()}, {int(k): int(v) for k, v in m24.items()},
                           ref=NIGHTGAP_REF_0423_0424)
        if ng.get("evaluable"):
            ng["agrees_with_v4v2_chain"] = ng.pop("correct")
            ng["note"] = "docx calf indices = global only under v4v2's 04-23->24 permutation: this is agreement, not correctness"
    rep = dict(nightgap_0423_0424=ng,
               mh7_values=[t for t in rows_all if t["id"] == "MH7"][0]["value"],
               switches_cand=(Mc.get("switches") or {}), switches_ref=(Mr or {}).get("switches"),
               step2_actions=(Mc.get("step2_actions") or {}),
               step2_unverifiable=[a for a in (Mc.get("step2_actions") or {}).get("list", []) if a["verdict"] == "unknown"])
    g = gate_of(rows)
    out = _verdict_header("G5a", args)
    out.update(day=day, gate=g, proceed=(g == "PASS"), must_hold=rows, report=rep,
               telemetry={k: v for k, v in tel.items() if k != "clips"}, cand=_slim(Mc), ref=_slim(Mr) if Mr else None,
               user_watch_2_required=True, runtime_s=round(time.time() - t_all, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f: json.dump(out, f, indent=1, default=str)
    print(f"\n=== G5a {day} : {g} ===")
    for t in rows:
        print(f"  {t['id']:11s} {t['status']:14s} {t['name']}" + (f"  [{str(t['note'])[:200]}]" if t.get("note") else ""))
    print(f"  night gap 04-23->24: {json.dumps(ng, default=str)[:300]}")
    print(f"  switches cand {rep['switches_cand'].get('n')} (step-2 {rep['switches_cand'].get('n_step2')}) | ref "
          f"{(rep['switches_ref'] or {}).get('n')}")
    print(f"\n  G5a GATE: {g}  ({GATE_SEMANTICS[g]}); user watch #2 still required\n[tol48] wrote {args.out}")
    return out

# ============================================================================================================
# 13 · SELFTEST (synthetic, no data) + reproduction of the known v4v2 figures (with --data)
# ============================================================================================================
def _mk(h, w, boxes):
    m = np.zeros((h, w), bool)
    for (y0, y1, x0, x1) in boxes: m[y0:y1, x0:x1] = True
    return m

def _st_rle_ops(fails):
    rng = np.random.default_rng(48)
    H, W = 37, 53
    masks = [np.zeros((H, W), bool), np.ones((H, W), bool)]
    for (y, x) in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        m = np.zeros((H, W), bool); m[y, x] = True; masks.append(m)
    m = np.zeros((H, W), bool); m[:, 0] = True; masks.append(m)
    m = np.zeros((H, W), bool); m[:, -1] = True; masks.append(m)
    m = np.zeros((H, W), bool); m[5, :] = True; masks.append(m)                     # one-row stripe (runs of length 1)
    masks.append((np.indices((H, W)).sum(0) % 2).astype(bool))                        # checkerboard
    m = np.zeros((H, W), bool); m[H - 3:, 4:9] = True; m[:3, 5:10] = True; masks.append(m)   # runs wrapping columns
    for _ in range(40): masks.append(rng.random((H, W)) < rng.uniform(0.02, 0.6))
    for _ in range(40):
        y0, x0 = rng.integers(0, H - 5), rng.integers(0, W - 5)
        masks.append(_mk(H, W, [(y0, y0 + rng.integers(1, H - y0), x0, x0 + rng.integers(1, W - x0))]))
    R = [runs_of(rle_encode(m)) for m in masks]
    n = 0
    for a in range(len(masks)):
        if not np.array_equal(rle_decode(rle_encode(masks[a])), masks[a]): fails.append(f"S1 round-trip {a}")
        if R[a].area != int(masks[a].sum()): fails.append(f"S1 area {a}")
        for b in range(len(masks)):
            i1 = int(np.logical_and(masks[a], masks[b]).sum()); i2 = inter(R[a], R[b])
            if i1 != i2: fails.append(f"S1 inter {a},{b}: {i1} vs {i2}")
            if iou_r(R[a], R[b]) != m_iou(masks[a], masks[b]): fails.append(f"S2 iou {a},{b}")
            if ios_r(R[a], R[b]) != m_ios(masks[a], masks[b]): fails.append(f"S2 ios {a},{b}")
            n += 1
    if ios_r(None, R[2]) != m_ios(None, masks[2]) or iou_r(R[2], None) != m_iou(masks[2], None): fails.append("S2 None")
    if CORE is not None:
        for nm in ("rle_decode", "m_iou", "m_ios"):
            f = getattr(CORE, nm, None)
            if f is None: continue
            for a in range(0, len(masks), 7):
                for b in range(0, len(masks), 5):
                    if nm == "rle_decode":
                        if not np.array_equal(f(rle_encode(masks[a])), masks[a]): fails.append(f"core {nm} {a}")
                    elif f(masks[a], masks[b]) != globals()[nm](masks[a], masks[b]): fails.append(f"core {nm} {a},{b}")
    return f"S1/S2 run arithmetic == verbatim T03 on {n} pairs of {len(masks)} adversarial masks" + (" (+ tol_v5_core verbatim copies)" if CORE is not None else "")

def _st_associate(fails):
    H, W = 60, 100
    g = [_mk(H, W, [(5, 25, 5, 25)]), _mk(H, W, [(5, 25, 40, 60)]), _mk(H, W, [(35, 55, 5, 25)]), _mk(H, W, [(35, 55, 60, 90)])]
    GR = [runs_of(rle_encode(m)) for m in g]
    # identical
    cs, cg, *_ , gs, gc, gg, nd = associate(GR, GR)
    if cs != [C_ASSIGNED] * 4 or cg != [0, 1, 2, 3] or nd != 4: fails.append(f"A1 identical {cs} {cg} {nd}")
    # permuted cand slots, a fragment inside GT 2, a merged blob over GT 0+1, a non-calf, a split piece
    c = [None] * 4
    c[0] = runs_of(rle_encode(_mk(H, W, [(5, 25, 5, 60)])))         # MULTI (covers GT 0 and GT 1)
    c[1] = runs_of(rle_encode(_mk(H, W, [(40, 50, 10, 20)])))       # fragment inside GT 2 -> assigned (IoS 1)
    c[2] = runs_of(rle_encode(_mk(H, W, [(0, 4, 90, 99)])))         # non-calf -> NONE
    c[3] = runs_of(rle_encode(_mk(H, W, [(35, 55, 60, 90)])))       # GT 3 exactly
    cs, cg, cio, ciu, gs, gc, gg, nd = associate(c, GR)
    if cs != [C_MULTI, C_ASSIGNED, C_NONE, C_ASSIGNED] or cg[1] != 2 or cg[3] != 3:
        fails.append(f"A2 labels {cs} {cg}")
    if gs != [G_MULTI, G_MULTI, G_ASSIGNED, G_ASSIGNED]: fails.append(f"A2 gt status {gs}")
    # split: two pieces of GT 0
    c2 = [runs_of(rle_encode(_mk(H, W, [(5, 25, 5, 15)]))), runs_of(rle_encode(_mk(H, W, [(5, 25, 15, 25)]))), None, None]
    cs, cg, *_ , gs, gc, gg, nd = associate(c2, GR)
    if sorted(cs[:2]) != sorted([C_ASSIGNED, C_SPLIT]): fails.append(f"A3 split {cs}")
    # UNK: a mask half on GT 3 and half on straw (IoS 0.5)
    c3 = [None, None, None, runs_of(rle_encode(_mk(H, W, [(35, 55, 45, 75)])))]
    cs, *_ = associate(c3, GR)
    if cs[3] != C_UNK: fails.append(f"A4 UNK {cs}")
    # GT merge (two GT slots on one animal) is ONE animal: a cand mask covering it is not MULTI, n_distinct = 3
    gm = [g[0], _mk(H, W, [(6, 24, 6, 24)]), g[2], g[3]]
    GRm = [runs_of(rle_encode(m)) for m in gm]
    cs, cg, *_ , gs, gc, gg, nd = associate([GR[0], None, None, None], GRm)
    if cs[0] != C_ASSIGNED or nd != 3: fails.append(f"A5 GT merge {cs} nd={nd}")
    # identical masks nested (A inside B): the IoU tie-break pairs each mask with itself
    nest = [GRm[0], GRm[1], None, None]
    cs, cg, *_ = associate(nest, [GRm[0], GRm[1], None, None])
    if cg[:2] != [0, 1]: fails.append(f"A6 tie-break {cg}")
    return "associate: assigned / MULTI / split / UNK / NONE, GT-merge = one animal, IoU tie-break"

def _st_switches_gaps(fails):
    fps = 5.0
    ts = np.arange(0, 60, 0.2)
    an = np.array([1] * 150 + [2] * 150)                                  # 30 s of 1 then 30 s of 2
    if len(detect_switches(ts, an, fps)) != 1: fails.append("W1 one switch")
    an = np.array([1] * 100 + [2] * 25 + [1] * 175)                       # 5 s excursion -> no switch
    if detect_switches(ts, an, fps): fails.append("W2 short excursion must not switch")
    an = np.array([1] * 100 + [-1] * 100 + [2] * 100)                     # change across an unknown span -> bracketed
    sw = detect_switches(ts, an, fps)
    if len(sw) != 1 or not sw[0][4]: fails.append(f"W3 bracketed {sw}")
    an = np.array([1] * 60 + [2] * 60 + [1] * 60 + [3] * 60 + [1] * 60)  # 12 s runs of 2 and 3 -> 4 transitions
    if len(detect_switches(ts, an, fps)) != 4: fails.append("W4 multi")
    an = np.array([1] * 49 + [2] * 251)                                  # a 9.8 s first run does not qualify
    if detect_switches(ts, an, fps): fails.append("W5 9.8 s")
    # gap episodes (gaps_analyze semantics): presence blip < 1 s merges; pre-seed run excluded
    n = 300; area = np.full((n, 4), 50000, np.int32)
    area[:10, :] = 0                                                      # pre-seed
    area[50:60, 0] = 0; area[62:70, 0] = 0                                # 0.4 s blip -> one episode 50..69 (4.0 s)
    area[100:200, 1] = 0; area[205:210, 1] = 0                            # 1.0 s presence -> two episodes
    S = dict(A=dict(ts=np.arange(n) * 0.2 + ts_of(datetime(2024, 4, 22, 12, 0, 0)), c_area=area,
                    c_seg=np.zeros((n, 4), np.int32)), fps=fps, cev=[], cseg=[dict(idx=0, sid="x_001", slot=0, t0=0, t1=1e12,
                    open_reason="seed_day", via=None, raw={})])
    G = gap_metrics(S)
    durs = sorted(round(x["dur"], 2) for x in G["list"])
    if durs != [1.0, 4.0, 20.0] or len(G["preseed"]) != 4: fails.append(f"G1 gap episodes {durs} preseed={len(G['preseed'])}")
    return "switch detection (>=10 s runs, neutral unknowns, bracketed) + gap episodes (1 s merge, pre-seed excluded)"

def _st_stream(fails, tmp):
    """FrameStream == json.load on a file whose frames straddle many tiny chunks (boundary handling)"""
    rng = np.random.default_rng(7)
    frames = []
    for i in range(60):
        m = rng.random((23, 31)) < 0.3
        frames.append({"frame_idx": 5 * i, "t_sec": round(0.2 * i, 2), "masks": {"0": rle_encode(m), "1": None, "2": rle_encode(~m), "3": None}})
    J = {"clip": "Tol_ch01-20240423-120000-120100.mp4", "track_fps": 5.0, "mask_format": "rle_column_major", "frames": frames, "smoke": True}
    p = os.path.join(tmp, "masks_Tol_ch01-20240423-120000-120100.json.gz")
    with gzip.open(p, "wt") as f: json.dump(J, f)
    for chunk in (37, 97, 1 << 20):
        old = FrameStream.CHUNK, FrameStream.TRIM
        FrameStream.CHUNK, FrameStream.TRIM = chunk, 3 * chunk
        try:
            fs = FrameStream(p); got = list(fs)
        finally:
            FrameStream.CHUNK, FrameStream.TRIM = old
        if got != frames: fails.append(f"ST1 frames differ at chunk {chunk}")
        if fs.header.get("clip") != J["clip"] or fs.header.get("smoke") is not True or fs.header.get("track_fps") != 5.0:
            fails.append(f"ST1 header {fs.header} at chunk {chunk}")
    return "FrameStream == json.load across chunk boundaries (37 B / 97 B / 1 MiB), header + trailing fields"

def _write_root(root, day, stems, frames_by_stem, segs, events):
    d = os.path.join(root, day); os.makedirs(d, exist_ok=True)
    for stem in stems:
        with gzip.open(os.path.join(d, f"masks_{stem}.json.gz"), "wt") as f:
            json.dump({"clip": stem + ".mp4", "track_fps": 5.0, "mask_format": "rle_column_major", "frames": frames_by_stem[stem], "smoke": False}, f)
    json.dump({"day": day, "track_fps": 5.0, "segments": segs}, open(os.path.join(d, "segments.json"), "w"), indent=1)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        for e in events: f.write(json.dumps(e) + "\n")

def _st_end_to_end(fails, tmp):
    """a tiny synthetic day (60x100 px masks, 3 clips x 400 rows = 4 min): cand == GT gives an all-zero delta and
    the right absolute counts; a perturbed cand triggers exactly the injected defects"""
    day = "20240423"; H, W = 60, 100
    stems = ["Tol_ch01-20240423-120000-120120", "Tol_ch01-20240423-120120-120240", "Tol_ch01-20240423-120240-120400"]
    boxes = {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}
    def fr(stem, rows, fn):
        out = []
        for i in range(rows):
            ms = fn(stem, i)
            out.append({"frame_idx": i * 5, "t_sec": round(i * 0.2, 2), "masks": {str(k): (rle_encode(m) if m is not None else None) for k, m in ms.items()}})
        return out
    base = {k: _mk(H, W, [v]) for k, v in boxes.items()}
    def gt_fn(stem, i):
        ms = dict(base)
        if stem == stems[1] and 100 <= i < 150: ms[3] = None                   # GT slot 3 dark 10 s (clip 2)
        return ms
    w0 = datetime(2024, 4, 23, 12, 0, 0)
    t_ = lambda ci, i: (w0 + timedelta(seconds=80 * ci + i * 0.2)).isoformat()
    gt_frames = {s: fr(s, 400, gt_fn) for s in stems}
    gsegs = [dict(seg_id=f"{k}_001", slot=k, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                  t_end_wall=(w0 + timedelta(seconds=240)).isoformat(), close_reason="day_end") for k in range(3)]
    gsegs += [dict(seg_id="3_001", slot=3, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                   t_end_wall=t_(1, 99), close_reason="absence_break"),
              dict(seg_id="3_002", slot=3, t_start_wall=t_(1, 150), open_reason="self_recovery", n_frames=0,
                   t_end_wall=(w0 + timedelta(seconds=240)).isoformat(), close_reason="day_end")]
    gev = [dict(type="seeded", clip=stems[0] + ".mp4", t=0.0, slots=[0, 1, 2, 3], day=True)]
    gev += [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"]) for s in gsegs]
    root_gt = os.path.join(tmp, "gt"); _write_root(root_gt, day, stems, gt_frames, gsegs, gev)
    gtd = os.path.join(tmp, "gtfiles"); os.makedirs(os.path.join(gtd, day), exist_ok=True)
    json.dump({"assignments": {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3, "3_002": 3}},
              open(os.path.join(gtd, day, "segment_assignments.json"), "w"))
    wm = {"global_ident_map": {day: {"0": 0, "1": 3, "2": 2, "3": 1}}}
    json.dump(wm, open(os.path.join(gtd, "week_ident_map.json"), "w"))
    # perturbed cand: (1) slots 1/2 exchange animals inside their segments from clip 3 row 100 (40 s later),
    # (2) a worker blob as slot 0's mask for 70 s while GT holds all four (NONE outside GT-dark),
    # (3) slot 3 goes dark with GT and comes back as a dark_link segment on the RIGHT animal
    worker = _mk(H, W, [(22, 36, 40, 60)])
    def cand_fn(stem, i):
        ms = gt_fn(stem, i)
        ci = stems.index(stem)
        if ci == 2 and i >= 100: ms[1], ms[2] = base[2], base[1]
        if ci == 0 and 50 <= i < 400: ms[0] = worker
        return ms
    c_frames = {s: fr(s, 400, cand_fn) for s in stems}
    csegs = [dict(g) for g in gsegs]
    csegs[-1] = dict(csegs[-1], open_reason="dark_link", link_from="3_001", cause="no_mask", sole_dark=True, n_dark=1)
    cev = gev[:1] + [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"]) for s in csegs]
    cev.append(dict(type="dark_link", slot=3, seg_id="3_002", prev_seg_id="3_001", wall=t_(1, 150), cause="no_mask"))
    root_c = os.path.join(tmp, "cand"); _write_root(root_c, day, stems, c_frames, csegs, cev)
    out = os.path.join(tmp, "m.json")
    ns = argparse.Namespace(cand=root_c, ref=root_gt, gt_root=None, gt_dir=gtd, days=day, cand_day_dir=None, ref_day_dir=None,
                            teacher=None, teacher_rules_0422=None, run_tol30=False, tol30=None, tol30_sha=None,
                            week_map_gt=None, week_map_ref=None, week_map_cand=None, nightgap_cand=None, nightgap_ref=None,
                            telemetry=None, work=os.path.join(tmp, "work"), out=out, jobs=1, train_summary=None)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): res = score(ns)
    c, f = res["cand"][day], res["ref"][day]
    # ref (GT scored against itself): nothing wrong anywhere
    if f["switches"]["n"] != 0: fails.append(f"E1 ref switches {f['switches']['n']}")
    if f["labels"]["cand_s"]["all"]["NONE_out_gt_dark"] != 0: fails.append("E1 ref NONE")
    if abs(f["gaps"]["hours"] * 3600 - 10.0) > 0.21: fails.append(f"E1 ref gap {f['gaps']['hours'] * 3600}")
    if f["mapping_seed"]["map"] != {0: 0, 1: 3, 2: 2, 3: 1}: fails.append(f"E1 seed map {f['mapping_seed']}")
    if f["precondition"]["frac"] != 1.0: fails.append("E1 precondition")
    # cand: two switches (slots 1 and 2), NONE outside GT-dark = 70 s, one correct dark_link, MH7 fails, MH1/MH2 pass
    if c["switches"]["n"] != 2: fails.append(f"E2 cand switches {c['switches']['n']}")
    none_s = c["labels"]["cand_s"]["all"]["NONE_out_gt_dark"]
    if abs(none_s - 70.0) > 0.01: fails.append(f"E2 NONE outside GT-dark {none_s}")
    sa = c["step2_actions"]
    if sa["n"] != 1 or sa["list"][0]["verdict"] != "correct": fails.append(f"E2 step-2 verdict {sa}")
    mhs = {t["id"]: t["status"] for t in res["must_hold"]}
    if mhs.get("MH7") != "FAIL" or mhs.get("MH1") != "PASS" or mhs.get("MH2") != "PASS" or mhs.get("P0") != "PASS":
        fails.append(f"E2 must-holds {mhs}")
    # cand == GT: an all-zero delta
    ns2 = argparse.Namespace(**{**vars(ns), "cand": root_gt, "out": os.path.join(tmp, "m2.json")})
    with contextlib.redirect_stdout(io.StringIO()): res2 = score(ns2)
    if res2["deltas"][day]["n_nonzero"] != 0: fails.append(f"E3 identical arms delta {res2['deltas'][day]['nonzero_keys'][:5]}")
    # wrong dark_link: relink 3_002 from slot 0's segment (a different animal) -> MH1 FAIL
    csegs2 = [dict(s) for s in csegs]; csegs2[-1]["link_from"] = "0_001"
    root_c3 = os.path.join(tmp, "cand3"); _write_root(root_c3, day, stems, gt_frames, csegs2, cev)
    ns3 = argparse.Namespace(**{**vars(ns), "cand": root_c3, "out": os.path.join(tmp, "m3.json")})
    with contextlib.redirect_stdout(io.StringIO()): res3 = score(ns3)
    mh3 = {t["id"]: t["status"] for t in res3["must_hold"]}
    if mh3.get("MH1") != "FAIL": fails.append(f"E4 wrong link must fail MH1: {res3['cand'][day]['step2_actions']}")
    # identity classes with synthetic Tol_30 runs: slot labels = slots; cand arm has 2 x 40 s x ... of wrong time
    S = scan_day(dict(arm="cand", day=day, cand_dir=os.path.join(root_c, day), gt_dir=os.path.join(root_gt, day),
                      gt_assign=os.path.join(gtd, day, "segment_assignments.json"), week_map=load_week_map(os.path.join(gtd, "week_ident_map.json"))))
    sr = segment_rows(S)
    runs = {r["sid"]: [dict(t0=r["t0"], t1=r["t1"], label=r["slot"], flags=({"provisional"} if r["slot"] == 2 else set()), contested=[])] for r in S["cseg"]}
    lab, flg = label_arrays(S, sr, runs, REVIEW_FLAGS_T30)
    mp = seed_mapping(S, sr, lab)
    mech = {r["idx"]: seg_mech(r, events_by_seg(S["cev"])) for r in S["cseg"]}
    IC = identity_classes(S, lab, flg, mp["map"], mech)
    st = IC["strata"]["main"]
    if abs(st["wrong_silent"] - 1.0) > 0.01 or abs(st["wrong_flagged"] - 1.0) > 0.01:
        fails.append(f"E5 identity classes {st}")
    if IC["step2_wrong_silent_min"] != 0: fails.append("E5 step-2 attribution")
    return "end-to-end synthetic day: ref==GT clean, injected switch/NONE/links detected, identical arms -> zero delta, identity classes"

def _fx_frames(stems, fn, rows=400):
    """masks rows at 5 fps (frame_idx stride 5 at 25 fps), t_sec = 0.2 * i"""
    out = {}
    for s in stems:
        out[s] = [{"frame_idx": i * 5, "t_sec": round(i * 0.2, 2),
                   "masks": {str(k): (rle_encode(m) if m is not None else None) for k, m in fn(s, i).items()}}
                  for i in range(rows)]
    return out

def _fx_v5_extras(root, day_dir, stems, gpu="NVIDIA A10-24Q", cls="A10", prec="highest", mem_cap="0", acc=1.0,
                  timings=None, widgets_extra=None, rc=True, done=True, alloc=3.1, max_det=1, v5_class=True):
    """Tol_47-shaped side files of a v5 day dir: ROOT_CONFIG.json (day + root; widgets FIRST, with gpu_tf, exactly the
    trap of critic finding 4), the telemetry next to the root (per-clip fields NESTED under 'v5'), .done_ markers"""
    dd = os.path.join(root, day_dir)
    widgets = {"gpu_tf": "off", "v5_fixes": "on", "dark_fire": "on", "mem_cap_gib": mem_cap, "detector": "rfdetr"}
    widgets.update(widgets_extra or {})
    rec = {"widgets": widgets, "shas": {"detector": "d", "student": "s", "core": "c", "source": "x"}, "gpu_class": cls,
           "band": {"A_REF_TRAIN": 87137.0, "FRAG_LO": {"dayl": 0.44, "dark": 0.43}, "OVER_HI": 1.32}}
    echo = {"gpu": {"name": gpu, "class": cls}, "matmul_precision": prec, "mem_cap_gib": float(mem_cap), "widgets": widgets}
    if rc:
        for p in (os.path.join(dd, "ROOT_CONFIG.json"), os.path.join(root, "ROOT_CONFIG.json")):
            if not os.path.exists(p):
                json.dump(dict(rec, written_by=day_dir, created="2026-09-24T00:00:00", echo=echo), open(p, "w"), indent=1)
    tm = timings if timings is not None else {"decode": 3.0, "decode_wait": 1.0, "sam3": 100.0, "detector": 5.0, "student": 20.0}
    clips = [dict(clip=s, frames=400, passes=1, assignable=20, matched=20, acceptance=acc, median_accepted_overlap=0.9,
                  strikes=0, opens=0, closes=0, merge_events=0, coverage3_frac=1.0, track_min=1.0, stage_min=0.1,
                  v5=dict(gpu=gpu, **({"gpu_class": cls} if v5_class else {}), fps=6.5, peak_allocated_gib=alloc, peak_reserved_gib=6.0, mem_free_gib=10.0,
                          max_processed_frames=2, mem_checks=2, sessions=1, rebuilds=1, rebuilds_per_hour=45.0,
                          det_calls={"pass": 2, "dark": 1, "seed": 1}, max_det_calls_per_frame=max_det, det_ms_median=40.0,
                          det_ms_p95=60.0, dark={}, pen_count_refused={"dark": 0, "scheduled": 1, "seed": 0},
                          host_rss_gib=3.2, cpu_ms_per_frame=30.0, ms_per_frame=tm, lap_sync=True, decode="seek_thread",
                          host_rss_trend_gib_per_run_h=0.46, session_tracked_entries=1234, mem_rechecks=0,
                          resumed=False, restored_clips=[])) for s in stems]
    J = dict(stage=day_dir.split("_")[0], track_fps=5.0, params={"v5": echo, "v5_guard": {"gpu_class": cls}}, clips=clips)
    rn = os.path.basename(os.path.normpath(root))
    json.dump(J, open(os.path.join(os.path.dirname(os.path.normpath(root)), f"Metrics_track_v5_{rn}_{day_dir}.json"), "w"))
    if done:
        for s in stems: open(os.path.join(dd, f".done_{s}"), "w").close()

def _st_gpu_telemetry(fails, tmp):
    """critic finding 4: GPU class is exact (A100 is never an A10), telemetry fields are read from clip['v5'],
    ROOT_CONFIG is read by key (its widgets hold gpu_tf before gpu_class), MH8 labels the D6 fallback"""
    names = {"NVIDIA A10": "A10", "NVIDIA A10-24Q": "A10", "NVIDIA A100-SXM4-80GB": "A100", "NVIDIA A100 80GB PCIe": "A100",
             "NVIDIA A100-SXM4-40GB": "A100", "Tesla T4": "TESLA T4", "": "UNKNOWN", None: "UNKNOWN"}
    for n, want in names.items():
        if gpu_class(n) != want: fails.append(f"T1 gpu_class({n!r}) = {gpu_class(n)!r} != {want!r}")
        if CORE is not None and hasattr(CORE, "gpu_class") and CORE.gpu_class(n) != gpu_class(n):
            fails.append(f"T1 gpu_class({n!r}) differs from tol_v5_core.gpu_class")
    stems = ["Tol_ch01-20240423-120000-120120"]
    def run(tag, **kw):
        root = os.path.join(tmp, "tel_" + tag, "Tracking_v5a10"); dd = os.path.join(root, "20240423"); os.makedirs(dd)
        d6 = kw.pop("d6", False); rm_rc = kw.pop("rm_rc", False); extra = kw.pop("extra_stems", [])
        _fx_v5_extras(root, "20240423", stems, **kw)
        if rm_rc:
            for p in (os.path.join(dd, "ROOT_CONFIG.json"), os.path.join(root, "ROOT_CONFIG.json")): os.remove(p)
        tel = telemetry_summary(find_telemetry(root, "20240423"), 0.11)
        o = ops_metrics(dict(cev=[dict(type="mem_check", si=250, allocated_gib=2.0, reserved_gib=5.0, processed_frames=2,
                                       host_rss_gib=3.0)], cand_dir=dd))
        R = {"cand": {"20240423": dict(ops=o, done=dict(present=True, n=1, missing=[]), clip_stems=stems + extra)},
             "telemetry": {"cand:20240423": tel}}
        return tel, o, mh8_row(R, ["20240423"], argparse.Namespace(d6_fallback=d6), {"20240423": 1.0})
    tel, o, r = run("a10")
    if tel.get("gpu_class") != ["A10"] or tel.get("gpu") != ["NVIDIA A10-24Q"] or tel.get("peak_alloc_gib") != 3.1             or tel.get("max_det_calls_per_frame") != 1 or tel.get("peak_reserved_gib") != 6.0 or tel.get("host_rss_gib") != 3.2:
        fails.append(f"T2 nested telemetry fields not read: {tel.get('gpu')} {tel.get('gpu_class')} {tel.get('peak_alloc_gib')}")
    if o.get("gpu_class") != ["A10"]: fails.append(f"T2 ROOT_CONFIG gpu class read as {o.get('gpu_class')} (gpu_tf trap)")
    if tel.get("host_rss_trend_gib_per_run_h_max") != 0.46 or tel.get("session_tracked_entries_max") != 1234:
        fails.append(f"T2b host-RSS trend not reported: {tel.get('host_rss_trend_gib_per_run_h_max')} {tel.get('session_tracked_entries_max')}")
    if r["status"] != "PASS": fails.append(f"T3 MH8 A10 run: {r['status']} {r.get('note')}")
    _t, _o, r = run("a100", gpu="NVIDIA A100-SXM4-80GB", cls="A100")
    if r["status"] != "FAIL" or "A10" not in str(r.get("note")): fails.append(f"T4 MH8 must FAIL on an unlabelled A100: {r['status']}")
    _t, _o, r = run("d6", gpu="NVIDIA A100-SXM4-80GB", cls="A100", mem_cap="21.0", d6=True)
    if r["status"] != "PASS" or "D6 FALLBACK" not in str(r.get("note")): fails.append(f"T5 MH8 labelled D6: {r['status']} {r.get('note')}")
    _t, _o, r = run("d6cap", gpu="NVIDIA A100-SXM4-80GB", cls="A100", mem_cap="0", d6=True)
    if r["status"] != "FAIL": fails.append(f"T6 MH8 D6 without the 21 GiB cap must FAIL: {r['status']}")
    _t, _o, r = run("norc", rm_rc=True)
    if r["status"] != "NOT_EVALUATED": fails.append(f"T7 MH8 without ROOT_CONFIG must be NOT_EVALUATED: {r['status']}")
    _t, _o, r = run("inv7", acc=0.3)
    if r["status"] != "FAIL" or "INV-7" not in str(r.get("note")): fails.append(f"T8 MH8 INV-7 signature: {r['status']}")
    _t, _o, r = run("alloc", alloc=4.5)
    if r["status"] != "FAIL": fails.append(f"T9 MH8 peak allocated 4.5 GiB must FAIL: {r['status']}")
    _t, _o, r = run("notel", extra_stems=["Tol_ch01-20240423-120120-120240"])
    if r["status"] != "FAIL" or "telemetry entry" not in str(r.get("note")): fails.append(f"T11 masks without telemetry: {r['status']}")
    tel, _o, _r = run("noclass", gpu="NVIDIA A100-SXM4-80GB", cls="A100", v5_class=False)   # class derived from the name
    if tel.get("gpu_class") != ["A100"]: fails.append(f"T10 class from the nested gpu name: {tel.get('gpu_class')}")
    return "GPU class exact (A100 != A10, = core), telemetry clip['v5'] fields, ROOT_CONFIG by key, MH8 D6 label / INV-7 / tripwires"

def _g2_fixture(tmp, tag, variant=()):
    """a synthetic 3-clip pilot window 20240423_w1204-1208 (4 min): cand = the GT masks, slot 3 dark 10 s in clip 2 and
    back as a CORRECT dark_link; one scheduled call with 5 in-band detections that was refused (pen count); the SPEC
    sentinel 12:05:23 (GT slot 1) lies inside the window"""
    day, H, W = "20240423", 60, 100
    stems = [f"Tol_ch01-{day}-120400-120520", f"Tol_ch01-{day}-120520-120640", f"Tol_ch01-{day}-120640-120800"]
    boxes = {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}
    base = {k: _mk(H, W, [v]) for k, v in boxes.items()}
    def gt_fn(stem, i):
        ms = dict(base)
        if stem == stems[1] and 100 <= i < 150: ms[3] = None
        return ms
    w0 = datetime(2024, 4, 23, 12, 4, 0)
    t_ = lambda ci, i: (w0 + timedelta(seconds=80 * ci + i * 0.2)).isoformat()
    tend = (w0 + timedelta(seconds=240)).isoformat()
    frames = _fx_frames(stems, gt_fn)
    gt_root = os.path.join(tmp, "g2_gt")
    if not os.path.isdir(os.path.join(gt_root, day)):
        gsegs = [dict(seg_id=f"{k}_001", slot=k, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                      t_end_wall=tend, close_reason="day_end") for k in range(3)]
        gsegs += [dict(seg_id="3_001", slot=3, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                       t_end_wall=t_(1, 99), close_reason="absence_break"),
                  dict(seg_id="3_002", slot=3, t_start_wall=t_(1, 150), open_reason="self_recovery", n_frames=0,
                       t_end_wall=tend, close_reason="day_end")]
        gev = [dict(type="seeded", clip=stems[0] + ".mp4", t=0.0, slots=[0, 1, 2, 3], day=True)]
        gev += [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"]) for s in gsegs]
        _write_root(gt_root, day, stems, frames, gsegs, gev)
        gtd = os.path.join(tmp, "g2_gtfiles"); os.makedirs(os.path.join(gtd, day), exist_ok=True)
        json.dump({"assignments": {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3, "3_002": 3}},
                  open(os.path.join(gtd, day, "segment_assignments.json"), "w"))
        json.dump({"global_ident_map": {day: {"0": 0, "1": 3, "2": 2, "3": 1}}}, open(os.path.join(gtd, "week_ident_map.json"), "w"))
    dark = "nodark" not in variant
    csegs = [dict(seg_id=f"{k}_001", slot=k, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                  t_end_wall=tend, close_reason="day_end") for k in range(3)]
    csegs += [dict(seg_id="3_001", slot=3, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                   t_end_wall=t_(1, 150), close_reason="dark_gap" if dark else "absence_break"),
              dict(seg_id="3_002", slot=3, t_start_wall=t_(1, 150), n_frames=0, t_end_wall=tend, close_reason="day_end",
                   **(dict(open_reason="dark_link", link_from="3_001", cause="no_mask", sole_dark=True, n_dark=1)
                      if dark else dict(open_reason="self_recovery")))]
    c0, c1, c2 = (s + ".mp4" for s in stems)
    def pass_ev(clip, t, n_in, **kw):
        e = dict(type="pass", clip=clip, t=t, forced=None, n_det=n_in, tracked=4, matched={"0": 0.9, "1": 0.9, "2": 0.9},
                 struck=[], opened=[], refreshed=[], rebuilt=False, dark=None, n_raw=n_in, n_inband=n_in, det_ms=40.0,
                 detector="rfdetr", pen_refused=False)
        e.update(kw); return e
    cev = [dict(type="seeded", clip=c0, t=0.0, slots=[0, 1, 2, 3], day=True)]
    cev += [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"]) for s in csegs]
    cev += [pass_ev(c0, 15.0, 4), pass_ev(c0, 60.0, 5, pen_refused=True), pass_ev(c2, 15.0, 4),
            dict(type="mem_check", si=250, allocated_gib=2.5, reserved_gib=5.0, processed_frames=2, clip=c0, host_rss_gib=3.0)]
    if "pen_noevent" not in variant:
        cev.append(dict(type="pen_count_refused", path="dark" if "pen_path_dark" in variant else "scheduled", n_inband=5,
                        areas=[1] * 5, scores=[0.9] * 5, clip=c0, t=60.0))
    if "dark_pen" in variant or "dark_pen_link" in variant:   # a dark-only call over the pen count (no pen event logged)
        cev.append(dict(type="dark_pass", clip=c1, t=40.0, wall=t_(1, 200), pool="dayl", kind="dark_only", n_raw=5, n_det=5,
                        n_inband=5, det_ms=41.0, dark_slots={}, unclaimed=[], candidates=[], opens={},
                        links=({"2": {"det": 0}} if "dark_pen_link" in variant else {}), rejected={},
                        refused="pen_count", deferred=None, rebuilt=False, areas=[300] * 5, scores=[0.9] * 5))
    if "pen_open" in variant:
        cev.append(dict(type="segment_open", seg_id="0_099", slot=0, wall=t_(0, 300), reason="reopen"))
    if dark:
        cev.append(dict(type="dark_pass", clip=c1, t=30.0, wall=t_(1, 150), pool="dayl", kind="dark_only", n_raw=1, n_det=1,
                        n_inband=1, det_ms=41.0, dark_slots={}, unclaimed=[0], candidates=[0], links={"3": {"det": 0}},
                        opens={}, rejected={"1": [[None, "cause"]]}, refused=None, deferred=None, rebuilt=True, areas=[300], scores=[0.9]))
        cev.append(dict(type="dark_link", slot=3, seg_id="3_002", prev_seg_id="3_001", closed_occ=True, gap_s=20.2, ios=1.0,
                        iou=1.0, det_area=300, conflict=0.0, cause="no_mask", n_dark=1, sole_dark=True, clip=c1, t=30.0))
    root = os.path.join(tmp, "g2_" + tag, "Tracking_v5a10_pilot"); sub = f"{day}_w1204-1208"
    _write_root(root, sub, stems, frames, csegs, cev)
    recs = {stems[0]: [dict(frame_idx=75, t_sec=15.0, kind="pass", n_inband=4), dict(frame_idx=300, t_sec=60.0, kind="pass", n_inband=5)],
            stems[1]: [dict(frame_idx=150, t_sec=30.0, kind="dark", n_inband=1)] if dark else [],
            stems[2]: [dict(frame_idx=75, t_sec=15.0, kind="pass", n_inband=4)]}
    if "det2" in variant: recs[stems[2]].append(dict(frame_idx=75, t_sec=15.0, kind="dark", n_inband=4))
    if "dark_pen" in variant or "dark_pen_link" in variant:
        recs[stems[1]].append(dict(frame_idx=200, t_sec=40.0, kind="dark", n_inband=5))
    for s, rl in recs.items():
        with gzip.open(os.path.join(root, sub, f"detections_{s}.json.gz"), "wt") as f:
            json.dump({"clip": s, "detector": "rfdetr", "records": [dict(r, clip=s + ".mp4", n_raw=r["n_inband"], det_ms=40.0,
                                                                             raw=[], kept_idx=[]) for r in rl]}, f)
    kw = {}
    if "a100" in variant: kw.update(gpu="NVIDIA A100-SXM4-80GB", cls="A100")
    if "d6" in variant: kw.update(mem_cap="21.0")
    if "norc" in variant: kw.update(rc=False)
    if "inv7" in variant: kw.update(acc=0.3)
    if "prec" in variant: kw.update(prec="high")
    if "timing" in variant: kw.update(timings={"decode": 3.0, "sam3": 100.0})
    if "det_tel" in variant: kw.update(max_det=2)
    _fx_v5_extras(root, sub, stems, **kw)
    ns = argparse.Namespace(cand=root, ref=gt_root, gt_root=None, gt_dir=os.path.join(tmp, "g2_gtfiles"), window=[f"{day}={sub}"],
                            teacher=None, teacher_rules_0422=None, week_map_gt=None, telemetry=None, no_ref=("ref" not in variant),
                            jobs=1, out=os.path.join(tmp, "g2_" + tag, "verdict.json"), d6_fallback=("d6" in variant))
    return ns, sub

def _st_g2(fails, tmp):
    """critic finding 5: the coded G2 evaluator on synthetic pilot windows, one injected defect per variant"""
    import io, contextlib
    def run(tag, variant=()):
        ns, sub = _g2_fixture(tmp, tag, variant)
        with contextlib.redirect_stdout(io.StringIO()): r = g2_evaluate(ns)
        return r, {t["id"]: t["status"] for t in r["windows"][sub]["must_hold"]}, r["windows"][sub]
    r, st, w = run("base", ("ref",))
    want = {"P0": "PASS", "MH1": "PASS", "MH2": "PASS", "MH7": "PASS", "MH8": "PASS", "AUD1": "PASS", "G2.STEP2": "PASS",
            "G2.DET1": "PASS", "G2.PEN": "PASS", "G2.CLIPS": "PASS", "G2.TIM": "PASS", "G2.PREC": "PASS"}
    if st != want or r["gate"] != "PASS": fails.append(f"G2-0 baseline: gate {r['gate']} {st}")
    pen = [t for t in w["must_hold"] if t["id"] == "G2.PEN"][0]["value"]
    if pen["n_over_pen"] != 1 or pen["n_violations"] != 0: fails.append(f"G2-0 pen audit {pen}")
    if (w["cand"]["step2_actions"] or {}).get("list", [{}])[0].get("verdict") != "correct": fails.append("G2-0 link verdict")
    sn = {x["t"]: x for x in w["cand"]["sentinels"]}
    if not sn.get("12:05:23", {}).get("in_scope") or sn["12:05:23"].get("first_rejecting_reason") != "cause"             or sn["12:05:23"].get("cand_slot") != 1 or sn.get("20:35:57", {}).get("in_scope") is not False:
        fails.append(f"G2-0 sentinels {[(k, v.get('in_scope'), v.get('first_rejecting_reason')) for k, v in sn.items()]}")
    for tag, variant, rid, want_st, want_gate in (
            ("pen_open", ("pen_open",), "G2.PEN", "FAIL", "FAIL"),
            ("pen_noev", ("pen_noevent",), "G2.PEN", "FAIL", "FAIL"),
            ("pen_pathdark", ("pen_path_dark",), "G2.PEN", "FAIL", "FAIL"),   # a pass refusal must be path=scheduled
            ("dark_pen", ("dark_pen",), "G2.PEN", "PASS", "PASS"),            # dark_pass.refused == pen_count suffices
            ("dark_pen_link", ("dark_pen_link",), "G2.PEN", "FAIL", "FAIL"),  # ... but it must link / open nothing
            ("det2", ("det2",), "G2.DET1", "FAIL", "FAIL"),
            ("det_tel", ("det_tel",), "G2.DET1", "FAIL", "FAIL"),
            ("nodark", ("nodark",), "G2.STEP2", "REVIEW", "REVIEW"),
            ("a100", ("a100",), "MH8", "FAIL", "FAIL"),
            ("a100d6", ("a100", "d6"), "MH8", "PASS", "PASS"),
            ("norc", ("norc",), "MH8", "NOT_EVALUATED", "INCOMPLETE"),
            ("inv7", ("inv7",), "MH8", "FAIL", "FAIL"),
            ("prec", ("prec",), "G2.PREC", "FAIL", "FAIL"),
            ("timing", ("timing",), "G2.TIM", "FAIL", "FAIL")):
        r, st, _w = run(tag, variant)
        if st.get(rid) != want_st or r["gate"] != want_gate:
            fails.append(f"G2-{tag}: {rid}={st.get(rid)} (want {want_st}), gate {r['gate']} (want {want_gate}); {st}")
    return "G2 evaluator: baseline PASS; pen-count open / missing refusal, 2 detects per frame, A100, INV-7, precision, " \
           "timings -> FAIL; zero step-2 -> REVIEW; no ROOT_CONFIG -> INCOMPLETE; labelled D6 -> PASS"

def _truth_tol29(gt, day, label, w):
    """Tol_29_periodic_reid_test.py:91-99 verbatim semantics on datetimes (the reference for DocxGT)"""
    def hh(s):
        h, m, sec = (int(x) for x in s.split(":")); d = datetime.strptime(day, "%Y%m%d")
        if h >= 24: h -= 24; d += timedelta(days=1)
        return d.replace(hour=h, minute=m, second=sec)
    ivs = [(hh(f), hh(t), {int(k): v for k, v in m.items()}) for f, t, m in gt["intervals"]]
    exc = [(hh(f), hh(t)) for f, t, *_ in gt.get("excluded", [])]
    margin = timedelta(seconds=gt.get("exclude_margin_sec", 180))
    cps = sorted({iv[0] for iv in ivs} | {iv[1] for iv in ivs})
    for f, t in exc:
        if f <= w < t: return None
    for cp in cps:
        if abs((w - cp).total_seconds()) <= margin.total_seconds(): return None
    for f, t, m in ivs:
        if f <= w < t: return m.get(label)
    return None

def _st_g5a(fails, tmp):
    """critic finding 6: DocxGT == Tol_29's truth(); G5a on a synthetic 04-24 whose docx GT exchanges labels 1/2 at
    12:03 inside the GT segments: a seed segment carrying it = a (non-step-2) switch; a dark_link segment carrying it
    = MH2 FAIL; a ROOT_CONFIG that differs from 04-22/23 = FAIL; the 04-23->24 night-gap report"""
    import io, contextlib
    day, H, W = "20240424", 60, 100
    gt = {"day": day, "global_ident_map": {"0": 0, "1": 2, "2": 1, "3": 3},
          "intervals": [["11:00:00", "12:03:00", {"0": 0, "1": 1, "2": 2, "3": 3}],
                        ["12:03:00", "13:00:00", {"0": 0, "1": 2, "2": 1, "3": None}]],
          "excluded": [["12:03:30", "12:03:40", "test"]], "exclude_margin_sec": 10}
    gp = os.path.join(tmp, "gt_0424_test.json"); json.dump(gt, open(gp, "w"))
    D = DocxGT(gp, day)
    w0 = datetime(2024, 4, 24, 12, 0, 0)
    for sec in list(range(-5, 300, 1)) + [180 - 10, 180 + 10, 180 - 10.5, 210, 219.9, 220]:
        w = w0 + timedelta(seconds=sec)
        for loc in range(4):
            ref = _truth_tol29(gt, day, D.G[loc], w)
            got = D.animal(loc, ts_of(w))
            if (ref if ref is not None else -1) != got: fails.append(f"D1 DocxGT loc {loc} at {w.time()}: {got} vs Tol_29 {ref}")
    if D.animal(None, ts_of(w0)) != -1: fails.append("D1 unassigned segment must be -1")
    # the day: 3 clips 12:00-12:04, all four GT masks present throughout
    stems = [f"Tol_ch01-{day}-120000-120120", f"Tol_ch01-{day}-120120-120240", f"Tol_ch01-{day}-120240-120400"]
    boxes = {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}
    base = {k: _mk(H, W, [v]) for k, v in boxes.items()}
    frames = _fx_frames(stems, lambda s, i: dict(base))
    tend = (w0 + timedelta(seconds=240)).isoformat()
    gsegs = [dict(seg_id=f"{k}_001", slot=k, t_start_wall=w0.isoformat(), open_reason="seed_day", n_frames=0,
                  t_end_wall=tend, close_reason="day_end") for k in range(4)]
    gev = [dict(type="seeded", clip=stems[0] + ".mp4", t=0.0, slots=[0, 1, 2, 3], day=True)]
    gev += [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"]) for s in gsegs]
    gt_root = os.path.join(tmp, "g5_gt"); _write_root(gt_root, day, stems, frames, gsegs, gev)
    ap_ = os.path.join(tmp, "asg24.json")
    json.dump({"assignments": {"0_001": 0, "1_001": 2, "2_001": 1, "3_001": 3}}, open(ap_, "w"))   # shown label = G[loc]
    def run(tag, link=False, rc_widget=None, ng=None):
        root = os.path.join(tmp, "g5_" + tag, "Tracking_v5a10")
        segs = [dict(s) for s in gsegs]; ev = [dict(e) for e in gev]
        if link:
            tl = (w0 + timedelta(seconds=150)).isoformat()
            segs[1] = dict(segs[1], t_end_wall=tl, close_reason="dark_gap")
            segs.append(dict(seg_id="1_002", slot=1, t_start_wall=tl, open_reason="dark_link", link_from="1_001", cause="no_mask",
                             sole_dark=True, n_dark=1, n_frames=0, t_end_wall=tend, close_reason="day_end"))
            ev += [dict(type="segment_close", seg_id="1_001", slot=1, wall=tl, reason="dark_gap"),
                   dict(type="segment_open", seg_id="1_002", slot=1, wall=tl, reason="dark_link"),
                   dict(type="dark_link", slot=1, seg_id="1_002", prev_seg_id="1_001", cause="no_mask", clip=stems[1] + ".mp4", t=70.0)]
        _write_root(root, day, stems, frames, segs, ev)
        for d in ("20240422", "20240423"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
            _fx_v5_extras(root, d, [], done=False)
        _fx_v5_extras(root, day, stems, widgets_extra=({"detect_night_sec": rc_widget} if rc_widget else None))
        ns = argparse.Namespace(cand=root, gt_root=gt_root, gt24=gp, asg24=ap_, cand_day_dir=None, telemetry=None,
                                no_ref=False, jobs=1, out=os.path.join(tmp, "g5_" + tag, "verdict.json"),
                                nightgap_cand=None, g3_metrics=None, d6_fallback=False)
        if ng is not None:
            ngp = os.path.join(tmp, "g5_" + tag, "ng.json"); json.dump({"frozen": {"pooled": ng, "min_margin": 0.005, "support": 0.5}}, open(ngp, "w"))
            g3p = os.path.join(tmp, "g5_" + tag, "g3.json")
            json.dump({"cand": {"20240423": {"mapping": {"map": {"0": 0, "1": 3, "2": 2, "3": 1}}}}}, open(g3p, "w"))
            ns.nightgap_cand, ns.g3_metrics = ngp, g3p
        with contextlib.redirect_stdout(io.StringIO()): r = g5a_evaluate(ns)
        return r, {t["id"]: t["status"] for t in r["must_hold"]}
    r, st = run("base", ng={"0": "0", "1": "3", "2": "2", "3": "1"})
    want = {"P0": "PASS", "MH1": "PASS", "MH2": "PASS", "MH8": "PASS", "AUD1": "PASS", "G5a.ROOTCFG": "PASS", "G5a.GTCOV": "PASS"}
    if st != want or r["gate"] != "PASS": fails.append(f"G5-0 baseline: gate {r['gate']} {st}")
    sw = r["report"]["switches_cand"]
    if sw.get("n") != 2 or sw.get("n_step2") != 0: fails.append(f"G5-0 switches (docx exchange inside seed segments) {sw.get('n')}/{sw.get('n_step2')}")
    if (r["report"]["switches_ref"] or {}).get("n") != 2: fails.append("G5-0 ref (v4v2p2 vs docx) switches")
    ms = r["cand"]["mapping_seed"]
    if ms.get("map") != {0: 0, 1: 1, 2: 2, 3: 3}: fails.append(f"G5-0 seed map {ms}")
    ng = r["report"]["nightgap_0423_0424"]
    if not ng.get("evaluable") or ng.get("agrees_with_v4v2_chain") is not True: fails.append(f"G5-0 night gap report {ng}")
    r, st = run("badperm", ng={"0": "0", "1": "1", "2": "2", "3": "3"})
    if r["report"]["nightgap_0423_0424"].get("agrees_with_v4v2_chain") is not False: fails.append("G5-1 night gap disagreement")
    if r["gate"] != "PASS": fails.append("G5-1 the night gap is reported, never gated")
    r, st = run("link", link=True)
    if st.get("MH2") != "FAIL" or st.get("MH1") != "PASS" or r["gate"] != "FAIL":
        fails.append(f"G5-2 exchange inside a dark_link segment: MH2 {st.get('MH2')} MH1 {st.get('MH1')} gate {r['gate']}")
    r, st = run("rcdiff", rc_widget="300")
    if st.get("G5a.ROOTCFG") != "FAIL" or r["gate"] != "FAIL": fails.append(f"G5-3 ROOT_CONFIG difference: {st.get('G5a.ROOTCFG')}")
    return "G5a evaluator: DocxGT == Tol_29 truth(); docx hop in a seed segment = reported switch, in a dark_link segment = " \
           "MH2 FAIL; ROOT_CONFIG identity; 04-23->24 night gap = agreement with the v4v2 chain (report only)"

def _st_d8_geometry_mem(fails):
    """finisher 2026-09-25: (1) MH4z, the D8 hard zero on FLAGGED + silent step-2 wrong time (MH3 sees silent only);
    (2) geometry errors counted only where v4v2 and the teacher agree (SPEC §6.3); (3) a delayed free re-measured by
    Tol_47's gc_recheck does not enter the event-based allocated peak; (4) host-RSS trend fields reported"""
    d = "20240423"
    def R_of(s2w, s2s):
        ident = dict(step2_wrong_min=s2w, step2_wrong_silent_min=s2s, strata=dict(main=dict(wrong_min=10.0)))
        C = dict(precondition=dict(frac=1.0), identity=ident, fills=dict(step2_wrong_s=0.0), teacher=dict(mode="transfer"),
                 labels=dict(cand_s=dict(all=dict(MULTI_out_gt_dark=0.0, NONE_out_gt_dark=0.0, NONE_in_gt_dark=0.0))))
        return {"cand": {d: C}, "ref": {d: dict(identity=dict(strata=dict(main=dict(wrong_min=10.0))))}, "telemetry": {}, "nightgap": {}}
    ns = argparse.Namespace(d6_fallback=False, cand=None)
    T, g = must_holds(R_of(1.5, 0.0), [d], ns); st = {t["id"]: t["status"] for t in T}
    if st.get("MH3") != "PASS" or st.get("MH4z") != "FAIL" or g != "FAIL":
        fails.append(f"D8-1 flagged step-2 wrong time must FAIL MH4z (MH3 silent-only PASS): {st} gate {g}")
    T, g = must_holds(R_of(0.0, 0.0), [d], ns); st = {t["id"]: t["status"] for t in T}
    if st.get("MH4z") != "PASS": fails.append(f"D8-2 zero step-2 wrong time: MH4z {st.get('MH4z')}")
    # (2) 10 rows, one GT slot in band; rows 0-4 teacher agrees (IoU 0.9), rows 5-8 disagrees (0.2), row 9 no teacher row
    n = 10
    A = dict(ts=np.arange(n) * 1.0 + ts_of(datetime(2024, 4, 23, 12, 0, 0)), has_gt=np.ones(n, bool),
             g_area=np.zeros((n, 4), np.int32), g_gg=np.zeros((n, 4), np.float32), g_stat=np.zeros((n, 4), np.int8),
             g_c=np.full((n, 4), -1, np.int8), c_iou=np.zeros((n, 4), np.float32), g_tiou=np.full((n, 4), -1.0, np.float32))
    A["g_area"][:, 0] = 100; A["g_stat"][:, 0] = G_ASSIGNED; A["g_c"][:, 0] = 0
    A["c_iou"][:, 0] = [0.8, 0.8, 0.8, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    A["g_stat"][4, 0] = G_UNCOVERED; A["g_c"][4, 0] = -1                          # row 4: no cand mask on the animal
    A["g_tiou"][:5, 0] = 0.9; A["g_tiou"][5:9, 0] = 0.2
    S = dict(A=A, phase=np.zeros(n, np.int8), geo_ex=np.zeros((n, 4), bool), dawn_mask=np.zeros(n, bool))
    ta = mask_fidelity(S, 100.0).get("day", {}).get("teacher_agree") or {}
    if (ta.get("n_teacher_seen"), ta.get("n_agree"), ta.get("n_assigned"), ta.get("geometry_errors")) != (9, 5, 4, 2):
        fails.append(f"D8-3 teacher-agreement geometry {ta}")
    # (3) delayed free: frame 250 read 6.5 GiB, gc_recheck 2.0 -> peak 3.0 (frame 500); a later assert fail counts
    ev = [dict(type="mem_check", si=250, allocated_gib=6.5, reserved_gib=7.0, processed_frames=2, clip="c.mp4", where="periodic"),
          dict(type="mem_check", si=250, allocated_gib=2.0, reserved_gib=7.0, processed_frames=2, clip="c.mp4", where="gc_recheck"),
          dict(type="mem_check", si=500, allocated_gib=3.0, reserved_gib=6.0, processed_frames=2, clip="c.mp4", where="periodic")]
    o = ops_metrics(dict(cev=ev, cand_dir=os.path.join(os.sep, "nonexistent_tol48_dir")))
    if o["peak_alloc_gib"] != 3.0 or o["peak_reserved_gib"] != 7.0 or o["mem_gc_rechecks"] != 1 or o["mem_checks"] != 2:
        fails.append(f"D8-4 delayed free: peak {o['peak_alloc_gib']} reserved {o['peak_reserved_gib']} rechecks {o['mem_gc_rechecks']}")
    ev.append(dict(type="mem_assert_fail", si=750, allocated_gib=6.2, reserved_gib=7.0, processed_frames=2, clip="c.mp4"))
    if ops_metrics(dict(cev=ev, cand_dir=os.path.join(os.sep, "nonexistent_tol48_dir")))["peak_alloc_gib"] != 6.2:
        fails.append("D8-5 a mem_assert_fail must count in the peak")
    return "MH4z (D8 hard zero incl. flagged), geometry only where v4v2 and the teacher agree, delayed free (gc_recheck) not a peak"

def _r1_seg(sid, slot, t0, t1, reason="seed_day", close="day_end", **kw):
    return dict(seg_id=sid, slot=slot, t_start_wall=t0, t_end_wall=t1, open_reason=reason, close_reason=close, n_frames=0, **kw)

def _r1_case(tmp, tag, day, hh, mm, gt_fn, gsegs, gasg, cand_fn, csegs, cev_extra=(), mode="g2", gev_extra=(), teacher_fn=None):
    """review r1 fixtures: 3 clips x 80 s (5 fps, 60x100 px) from day hh:mm. gt_fn / cand_fn(ci, i) -> {slot: mask|None};
    gsegs / csegs(t_, tend) -> segment dicts; gasg = GT segment -> day-local animal. mode 'g2' scores the window with
    g2_evaluate (all G2-only rows satisfiable), 'score' runs the full mode on the day (no Tol_30).
    teacher_fn(ci, i) (review r2): also write a teacher root (one seed segment per slot, transfer mode) and score with it."""
    import io, contextlib
    w0 = datetime.strptime(day, "%Y%m%d").replace(hour=hh, minute=mm)
    stems = [f"Tol_ch01-{day}-{(w0 + timedelta(seconds=80 * c)).strftime('%H%M%S')}-"
             f"{(w0 + timedelta(seconds=80 * (c + 1))).strftime('%H%M%S')}" for c in range(3)]
    t_ = lambda ci, i: (w0 + timedelta(seconds=80 * ci + i * 0.2)).isoformat()
    tend = (w0 + timedelta(seconds=240)).isoformat()
    bdir = os.path.join(tmp, "r1_" + tag)
    gs = gsegs(t_, tend); cs = csegs(t_, tend)
    def evs(segs):
        e = [dict(type="seeded", clip=stems[0] + ".mp4", t=0.0, slots=[0, 1, 2, 3], day=True)]
        return e + [dict(type="segment_open", seg_id=s["seg_id"], slot=s["slot"], wall=s["t_start_wall"], reason=s["open_reason"],
                         **({"via": s["via"]} if s.get("via") else {})) for s in segs]
    gt_root = os.path.join(bdir, "gt")
    _write_root(gt_root, day, stems, _fx_frames(stems, lambda s, i: gt_fn(stems.index(s), i)), gs,
                evs(gs) + [e(stems, t_) if callable(e) else e for e in gev_extra])
    gtd = os.path.join(bdir, "gtfiles"); os.makedirs(os.path.join(gtd, day), exist_ok=True)
    nm = "segment_assignments_reviewed.json" if day == "20240422" else "segment_assignments.json"
    json.dump({"assignments": gasg}, open(os.path.join(gtd, day, nm), "w"))
    json.dump({"global_ident_map": {day: {"0": 0, "1": 1, "2": 2, "3": 3}}}, open(os.path.join(gtd, "week_ident_map.json"), "w"))
    cev = evs(cs) + [e(stems, t_) if callable(e) else e for e in cev_extra]
    cev.append(dict(type="mem_check", si=250, allocated_gib=2.5, reserved_gib=5.0, processed_frames=2, clip=stems[0] + ".mp4",
                    host_rss_gib=3.0))
    cframes = _fx_frames(stems, lambda s, i: cand_fn(stems.index(s), i))
    tch = None
    if teacher_fn is not None:
        tsg = [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in range(N_SLOTS)]
        _write_root(os.path.join(bdir, "teacher"), day, stems, _fx_frames(stems, lambda s, i: teacher_fn(stems.index(s), i)),
                    tsg, evs(tsg))
        tch = [f"{day}={os.path.join(bdir, 'teacher', day)}"]
    if mode == "score":
        root = os.path.join(bdir, "cand"); _write_root(root, day, stems, cframes, cs, cev)
        ns = argparse.Namespace(cand=root, ref=gt_root, gt_root=None, gt_dir=gtd, days=day, cand_day_dir=None, ref_day_dir=None,
                                teacher=tch, teacher_rules_0422=None, run_tol30=False, tol30=None, tol30_sha=None,
                                week_map_gt=None, week_map_ref=None, week_map_cand=None, nightgap_cand=None, nightgap_ref=None,
                                telemetry=None, work=os.path.join(bdir, "work"), out=os.path.join(bdir, "m.json"), jobs=1,
                                train_summary=None)
        with contextlib.redirect_stdout(io.StringIO()): r = score(ns)
        return r, {t["id"]: t["status"] for t in r["must_hold"]}, r["cand"][day]
    root = os.path.join(bdir, "Tracking_v5a10_pilot")
    sub = f"{day}_w{w0.strftime('%H%M')}-{(w0 + timedelta(minutes=4)).strftime('%H%M')}"
    _write_root(root, sub, stems, cframes, cs, cev)
    for s in stems:
        with gzip.open(os.path.join(root, sub, f"detections_{s}.json.gz"), "wt") as f:
            json.dump({"clip": s, "detector": "rfdetr", "records": [dict(clip=s + ".mp4", frame_idx=75, t_sec=15.0, kind="pass",
                      n_inband=4, n_raw=4, det_ms=40.0, raw=[], kept_idx=[])]}, f)
    _fx_v5_extras(root, sub, stems)
    ns = argparse.Namespace(cand=root, ref=gt_root, gt_root=None, gt_dir=gtd, window=[f"{day}={sub}"], teacher=tch,
                            teacher_rules_0422=None, week_map_gt=None, telemetry=None, no_ref=True, jobs=1,
                            out=os.path.join(bdir, "verdict.json"), d6_fallback=False)
    with contextlib.redirect_stdout(io.StringIO()): r = g2_evaluate(ns)
    w = r["windows"][sub]
    return r, {t["id"]: t["status"] for t in w["must_hold"]}, w

def _st_review_r1(fails, tmp):
    """review r1 (2026-09-25): (1) ghost GT masks are not animals (n_dist / MULTI / MH7 / 5f trigger) and MH7 gates the
    MAIN stratum only; (2) step-2 verdicts from the nearest >= 10 s trusted run anywhere in the segment, an unknown
    verdict is REVIEW (never PASS), actions in the late / excluded strata are unknown; (3) the duplicate rule sees a
    second cand slot on the same GT mask whichever slot won the Hungarian; MH2 counts trusted main-stratum evidence"""
    H, W = 60, 100
    base = {k: _mk(H, W, [v]) for k, v in {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}.items()}
    R = lambda m: runs_of(rle_encode(m))
    # ---- associate: ghosts ----
    GR = [R(base[0]), R(base[1]), R(base[2]), R(_mk(H, W, [(22, 30, 70, 80)]))]          # GT 3 = a straw blob
    real3 = R(base[3])
    cs, cg, *_, nd = associate([GR[0], GR[1], GR[2], real3], GR, [False, False, False, True])
    if nd != 3 or cs[3] != C_NONE: fails.append(f"R1-A7 ghost is not an animal: nd={nd} cs={cs}")
    if associate([GR[0], GR[1], GR[2], real3], GR)[7] != 4: fails.append("R1-A7 without the ghost flag n_dist must be 4")
    big = R(_mk(H, W, [(20, 58, 2, 22)]))                                                   # covers GT 2 + a ghost above it
    GRg = [GR[0], GR[1], R(base[2]), R(_mk(H, W, [(20, 36, 2, 22)]))]
    if associate([None, None, big, None], GRg)[0][2] != C_MULTI: fails.append("R1-A8 fixture: without the flag it is MULTI")
    cs, *_ = associate([None, None, big, None], GRg, [False, False, False, True])
    if cs[2] != C_ASSIGNED: fails.append(f"R1-A8 a ghost never makes a cand mask MULTI: {cs}")
    # ---- _side_evidence: a 6 s first run then a qualifying run of another animal = inconsistent ----
    ts = np.arange(0, 60, 0.2); an = np.array([1] * 30 + [2] * 270)
    ev = _side_evidence(ts, an, 5.0, first=True)
    if ev["consistent"] or ev["ids"] != [1, 2]: fails.append(f"R1-S1 conflicting side evidence {ev}")
    ev = _side_evidence(ts, np.array([-1] * 100 + [2] * 200), 5.0, first=True)
    if not ev["consistent"] or ev["ids"] != [2] or ev["win"] is not None: fails.append(f"R1-S2 run-only evidence {ev}")
    # common GT / cand pieces (04-23 12:04 windows; the sentinel 12:05:23 lies inside)
    def gseed(t_, tend, dark23=None):
        g = [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1)]
        if dark23 is None: return g + [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (2, 3)]
        for k in (2, 3):
            g.append(_r1_seg(f"{k}_001", k, t_(0, 0), t_(1, 99), close="merge_ambiguous"))
            if dark23 == "reopen": g.append(_r1_seg(f"{k}_002", k, t_(2, 100), tend, reason="reopen"))
        return g
    asg = {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3, "2_002": 2, "3_002": 3}
    def link_ev(prev_sid, sid, slot):
        return lambda stems, t_: dict(type="dark_link", slot=slot, seg_id=sid, prev_seg_id=prev_sid, closed_occ=True, gap_s=10.0,
                                      ios=0.7, iou=0.6, det_area=300, conflict=0.0, cause="no_mask", n_dark=2, sole_dark=False,
                                      clip=stems[1] + ".mp4", t=30.0)
    def link_segs(t_, tend):
        return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1, 2)] + [
            _r1_seg("3_001", 3, t_(0, 0), t_(1, 150), close="dark_gap"),
            _r1_seg("3_002", 3, t_(1, 150), tend, reason="dark_link", link_from="3_001", cause="no_mask", sole_dark=False, n_dark=2)]
    # R1-a (finding 2, the reviewer's demo): a WRONG dark_link while the GT lacks both animals for the first 10 s -> FAIL
    def gt_a(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or (ci == 2 and i < 100): ms[2] = None; ms[3] = None
        return ms
    def cand_a(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[2] = None
        if ci == 1 and 100 <= i < 150: ms[3] = None
        if (ci == 1 and i >= 150) or ci == 2: ms[3] = base[2]
        return ms
    r, st, w = _r1_case(tmp, "a", "20240423", 12, 4, gt_a, lambda t_, e: gseed(t_, e, "reopen"), asg, cand_a, link_segs,
                        [link_ev("3_001", "3_002", 3)])
    a0 = (w["cand"]["step2_actions"]["list"] or [{}])[0]
    if st.get("MH1") != "FAIL" or r["gate"] != "FAIL" or a0.get("verdict") != "wrong" or a0.get("new_animal") != 2 \
            or a0.get("prev_animal") != 3 or "12:05:23" not in (a0.get("near_sentinel") or []):
        fails.append(f"R1-a wrong link judged from the first >= 10 s run: MH1 {st.get('MH1')} gate {r['gate']} {a0.get('verdict')} "
                     f"{a0.get('new_animal')}/{a0.get('prev_animal')} {a0.get('near_sentinel')}")
    # R1-b (finding 2): the GT never sees the linked animal again -> unknown -> MH1 REVIEW, gate REVIEW (never PASS)
    def gt_b(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[2] = None; ms[3] = None
        return ms
    r, st, w = _r1_case(tmp, "b", "20240423", 12, 4, gt_b, lambda t_, e: gseed(t_, e, "gone"), asg, cand_a, link_segs,
                        [link_ev("3_001", "3_002", 3)])
    sa = w["cand"]["step2_actions"]
    if st.get("MH1") != "REVIEW" or r["gate"] != "REVIEW" or sa.get("n_unknown") != 1 or sa.get("n_unknown_near_sentinel") != 1:
        fails.append(f"R1-b unverifiable link must be REVIEW: MH1 {st.get('MH1')} gate {r['gate']} {sa.get('n_unknown')}")
    # R1-c (finding 3): a dark-pass open whose mask DUPLICATES animal 2 while slot 2 keeps an offset mask of it -> FAIL
    off2 = _mk(H, W, [(40, 58, 4, 22)])
    def cand_c(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[2] = off2
        if ci == 1 and 100 <= i < 150: ms[3] = None
        if (ci == 1 and i >= 150) or ci == 2: ms[3] = base[2]
        return ms
    def open_segs(t_, tend):
        return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1, 2)] + [
            _r1_seg("3_001", 3, t_(0, 0), t_(1, 150), close="dark_gap"),
            _r1_seg("3_002", 3, t_(1, 150), tend, reason="reopen", via="dark_pass")]
    dopen = lambda stems, t_: dict(type="dark_open", slot=3, seg_id="3_002", clip=stems[1] + ".mp4", t=30.0)
    r, st, w = _r1_case(tmp, "c", "20240423", 12, 4, lambda ci, i: dict(base), lambda t_, e: gseed(t_, e), asg, cand_c,
                        open_segs, [dopen])
    a0 = (w["cand"]["step2_actions"]["list"] or [{}])[0]
    if st.get("MH1") != "FAIL" or a0.get("verdict") != "wrong" or "duplicate" not in str(a0.get("why")) \
            or a0.get("first_10s_status", {}).get("assigned", 0) == 0:
        fails.append(f"R1-c duplicate open (new slot won the Hungarian): MH1 {st.get('MH1')} {a0.get('verdict')} {a0.get('why')} "
                     f"{a0.get('first_10s_status')}")
    # R1-d (finding 2, no false REVIEW): a dark-pass open onto the calf the GT lacks for its first 60 s -> correct -> PASS
    def gt_d(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or (ci == 2 and i < 100): ms[3] = None
        return ms
    def cand_d(ci, i):
        ms = dict(base)
        if ci == 1 and 100 <= i < 150: ms[3] = None
        return ms
    def gseed_d(t_, tend):
        return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1, 2)] + [
            _r1_seg("3_001", 3, t_(0, 0), t_(1, 99), close="merge_ambiguous"), _r1_seg("3_002", 3, t_(2, 100), tend, reason="reopen")]
    r, st, w = _r1_case(tmp, "d", "20240423", 12, 4, gt_d, gseed_d, asg, cand_d, open_segs, [dopen])
    a0 = (w["cand"]["step2_actions"]["list"] or [{}])[0]
    if st.get("MH1") != "PASS" or r["gate"] != "PASS" or a0.get("verdict") != "correct" or a0.get("new_animal") != 3:
        fails.append(f"R1-d legit fill into GT-dark: MH1 {st.get('MH1')} gate {r['gate']} {a0.get('verdict')} {a0.get('why')}")
    # R1-e (finding 1): v4v2 has an UNASSIGNED straw blob as slot 3 for 70 s while the real calf is unmasked; the cand holds
    # the real calf -> GT-dark (ghost), NONE inside GT-dark, MH7 PASS, 5f fill records written; ghost time reported
    blob = _mk(H, W, [(22, 30, 70, 80)])
    def gt_e(ci, i):
        ms = dict(base)
        if ci == 1 and i < 350: ms[3] = blob
        return ms
    def gseg_e(t_, tend):
        return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1, 2)] + [
            _r1_seg("3_001", 3, t_(0, 0), t_(0, 399), close="absence_break"),
            _r1_seg("3_009", 3, t_(1, 0), t_(1, 349), reason="reopen", close="absence_break"),
            _r1_seg("3_010", 3, t_(1, 350), tend, reason="self_recovery")]
    asg_e = {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3, "3_009": None, "3_010": 3}
    r, st, c = _r1_case(tmp, "e", "20240423", 12, 0, gt_e, gseg_e, asg_e, lambda ci, i: dict(base),
                        lambda t_, e: [_r1_seg(f"{k}_001", k, t_(0, 0), e) for k in range(4)], mode="score")
    L = c["labels"]["cand_s"]
    if st.get("MH7") != "PASS" or L["main"]["NONE_out_gt_dark"] != 0 or abs(L["main"]["NONE_in_gt_dark"] - 70.0) > 0.01 \
            or abs(c["labels"]["gt_ghost_mask_s"]["main"] - 70.0) > 0.01 or abs(c["labels"]["gt_dark_rows_by_ghost_s"] - 70.0) > 0.01:
        fails.append(f"R1-e ghost GT mask: MH7 {st.get('MH7')} main {L.get('main', {}).get('NONE_out_gt_dark')} / "
                     f"{L.get('main', {}).get('NONE_in_gt_dark')} ghost {c['labels'].get('gt_ghost_mask_s')}")
    if c["teacher"]["n_fill_rows"] != 350: fails.append(f"R1-e 5f fill trigger on ghost rows: {c['teacher']['n_fill_rows']} != 350")
    # R1-f (finding 1): 04-22 21:00-21:04 lies inside the identity exclusion 20:59:51-21:27:09 -- a 70 s non-calf mask is
    # REPORTED (excluded), MH7 PASS; a correct-looking dark_link there is unknown -> MH1 REVIEW; a GT animal change
    # inside one cand segment is not a switch (untrusted evidence, MH2)
    worker = _mk(H, W, [(22, 36, 40, 60)])
    def cand_f(ci, i):
        ms = dict(base)
        if ci == 0 and 50 <= i < 400: ms[0] = worker
        if ci >= 1: ms[1] = base[2]; ms[2] = None
        if ci == 1 and 100 <= i < 150: ms[3] = None
        return ms
    r, st, c = _r1_case(tmp, "f", "20240422", 21, 0, lambda ci, i: dict(base), lambda t_, e: gseed(t_, e), asg, cand_f,
                        link_segs, [link_ev("3_001", "3_002", 3)], mode="score")
    L = c["labels"]["cand_s"]
    mh7 = [t for t in r["must_hold"] if t["id"] == "MH7"][0]["value"]["20240422"]
    if st.get("MH7") != "PASS" or L["main"]["NONE_out_gt_dark"] != 0 or abs(mh7["reported"]["excluded"]["none_out_s"] - 70.0) > 0.01:
        fails.append(f"R1-f excluded stratum NONE: MH7 {st.get('MH7')} {mh7}")
    sa = c["step2_actions"]
    if st.get("MH1") != "REVIEW" or sa.get("n_unknown_other_strata") != 1 or sa["list"][0].get("stratum") != "excluded":
        fails.append(f"R1-f link in the excluded stratum: MH1 {st.get('MH1')} {sa.get('list')}")
    if c["switches"]["n"] != 0: fails.append(f"R1-f untrusted GT change counted as a switch: {c['switches']}")
    # R1-g: the late stratum 04-23 23:31 -- a 70 s non-calf mask is reported (late), not gated
    r, st, c = _r1_case(tmp, "g", "20240423", 23, 31, lambda ci, i: dict(base), lambda t_, e: gseed(t_, e), asg,
                        lambda ci, i: {**base, **({0: worker} if ci == 0 and 50 <= i < 400 else {})},
                        lambda t_, e: [_r1_seg(f"{k}_001", k, t_(0, 0), e) for k in range(4)], mode="score")
    mh7 = [t for t in r["must_hold"] if t["id"] == "MH7"][0]["value"]["20240423"]
    if st.get("MH7") != "PASS" or abs(mh7["reported"]["late"]["none_out_s"] - 70.0) > 0.01 or mh7["none_out_s"] != 0:
        fails.append(f"R1-g late stratum NONE: MH7 {st.get('MH7')} {mh7}")
    # R1-h: the R1-c duplicate open inside the 04-22 identity exclusion -> unknown (cannot be shown), MH1 REVIEW, not FAIL
    r, st, c = _r1_case(tmp, "h", "20240422", 21, 4, lambda ci, i: dict(base), lambda t_, e: gseed(t_, e), asg, cand_c,
                        open_segs, [dopen], mode="score")
    a0 = (c["step2_actions"]["list"] or [{}])[0]
    if st.get("MH1") != "REVIEW" or a0.get("verdict") != "unknown" or a0.get("stratum") != "excluded" or not a0.get("dup_rows"):
        fails.append(f"R1-h duplicate open in the excluded stratum: MH1 {st.get('MH1')} {a0.get('verdict')} {a0.get('stratum')}")
    # R1-j: the GT itself swaps slots 2/3 for 7 s around a GT merge_resolved inside the first 10 s of a CORRECT link:
    # those rows are not identity-trusted (±5 s), so the verdict stays 'correct' (untrusted rows would make it unknown)
    def gt_j(ci, i):
        ms = dict(base)
        if ci == 1 and 150 <= i <= 185: ms[2], ms[3] = base[3], base[2]
        return ms
    def cand_j(ci, i):
        ms = dict(base)
        if ci == 1 and 100 <= i < 150: ms[3] = None
        return ms
    gmerge = lambda stems, t_: dict(type="merge_resolved", wall=t_(1, 160), jumpers=[2, 3], clip=stems[1] + ".mp4", t=32.0)
    r, st, w = _r1_case(tmp, "j", "20240423", 12, 4, gt_j, lambda t_, e: gseed(t_, e), asg, cand_j, link_segs,
                        [link_ev("3_001", "3_002", 3)], gev_extra=[gmerge])
    a0 = (w["cand"]["step2_actions"]["list"] or [{}])[0]
    if st.get("MH1") != "PASS" or a0.get("verdict") != "correct" or a0.get("new", {}).get("win_votes") != {3: 15}:
        fails.append(f"R1-j GT defect at a GT merge must not decide a verdict: MH1 {st.get('MH1')} {a0.get('verdict')} "
                     f"{a0.get('new')}")
    # R1-i: 5f fills in the late / excluded strata are reported, not gated (MH5 reads the main stratum only)
    t9 = ts_of(datetime(2024, 4, 22, 21, 0, 0))
    Sf = dict(A=dict(c_seg=np.zeros((3, 4), np.int32)), fps=5.0, tvotes={}, teacher_mode="transfer",
              cseg=[dict(idx=0, sid="0_002", slot=0)], strat=np.array([0, 2, 1], np.int8),
              fills=[dict(row=q, slot=0, t=t9 + q, tslot=0, t_ios=1.0, t_iou=1.0, tkey=None, t_on_gt=2, why=None) for q in range(3)])
    fm = fill_metrics(Sf, np.zeros((3, 4), np.int8), {0: 0}, {0: "dark_link"})
    if fm["step2_wrong_s"] != 0.2 or fm["other_strata"]["step2_wrong_s"] != 0.4:
        fails.append(f"R1-i fill strata: main {fm['step2_wrong_s']} other {fm['other_strata']['step2_wrong_s']}")
    return "review r1: ghost GT masks (n_dist / MULTI / MH7 / 5f), MH7 main stratum only, step-2 verdict from the nearest " \
           ">= 10 s trusted run (wrong link FAIL, unverifiable REVIEW, legit GT-dark fill PASS), late / excluded = unknown, " \
           "duplicate open whichever slot wins the Hungarian (unknown when excluded), MH2 on trusted evidence, " \
           "5f fills of the late / excluded strata reported, not gated"

def _st_review_r2(fails, tmp):
    """review r2 (2026-09-25): (1) the geometric step-2 rules (duplicate, split, non-calf) decide only on EVALUABLE rows --
    main stratum, the GT holding four real animals, outside the SPEC §6.3 geometry exclusions, the teacher not
    contradicting v4v2 -- else unknown; a duplicate on a GT-dark calf is caught by trusted identity (teacher fills); a GT
    mask over two disjoint cand masks names no identity. (2) MH7 counts MULTI / NONE only where v4v2 and the teacher agree
    or v4v2 is the only reference, outside the ±30 s GT-defect exclusions; the excused time is reported."""
    H, W = 60, 100
    base = {k: _mk(H, W, [v]) for k, v in {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}.items()}
    asg = {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3}
    seeds = lambda t_, e: [_r1_seg(f"{k}_001", k, t_(0, 0), e) for k in range(N_SLOTS)]
    def open_segs(slot, ci0, i_dark, i_open):
        def f(t_, tend):
            return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in range(N_SLOTS) if k != slot] + [
                _r1_seg(f"{slot}_001", slot, t_(0, 0), t_(ci0, i_dark), close="dark_gap"),
                _r1_seg(f"{slot}_002", slot, t_(ci0, i_open), tend, reason="reopen", via="dark_pass")]
        return f
    def dopen(slot, ci0, i_open):
        return lambda stems, t_: dict(type="dark_open", slot=slot, seg_id=f"{slot}_002", clip=stems[ci0] + ".mp4", t=i_open * 0.2)
    first = lambda w: (w["cand"]["step2_actions"]["list"] or [{}])[0] if "cand" in w else (w["step2_actions"]["list"] or [{}])[0]
    # R2-a (finding 1, the reviewer's case B): GT slot 2 over-extends over calves 2+3 and GT slot 3 is absent (GT-dark);
    # the cand keeps both calves and dark-pass-reopens slot 3 on calf 3 = a CORRECT open -> unknown (REVIEW), not wrong
    over = _mk(H, W, [(40, 58, 2, 22), (40, 58, 70, 95)])
    def gt_a(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[2] = over; ms[3] = None
        return ms
    def gsegs_a(t_, tend):
        return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in (0, 1, 2)] + [_r1_seg("3_001", 3, t_(0, 0), t_(1, 99), close="merge_ambiguous")]
    def cand_a(ci, i):
        ms = dict(base)
        if ci == 1 and 100 <= i < 150: ms[3] = None
        return ms
    r, st, w = _r1_case(tmp, "r2a", "20240423", 12, 4, gt_a, gsegs_a, asg, cand_a, open_segs(3, 1, 150, 150), [dopen(3, 1, 150)])
    a0 = first(w)
    if st.get("MH1") != "REVIEW" or r["gate"] == "FAIL" or a0.get("verdict") != "unknown" or not a0.get("dup_rows") \
            or a0.get("dup_rows_evaluable") != 0:
        fails.append(f"R2-a GT over two calves (dup on non-evaluable rows) -> unknown: MH1 {st.get('MH1')} gate {r['gate']} "
                     f"{a0.get('verdict')} {a0.get('why')} dup {a0.get('dup_rows')}/{a0.get('dup_rows_evaluable')}")
    # R2-b (the reviewer's case A, identity duplicate on a GT-dark calf): GT lacks calf 3, the teacher has it; slot 3 keeps
    # a fragment of calf 3 and slot 2 (which lost calf 2) dark-pass-opens onto calf 3 -> wrong (MH1 FAIL), via the fills
    frag3 = _mk(H, W, [(40, 50, 70, 80)])
    def gt_b(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[3] = None
        return ms
    def cand_b(ci, i):
        ms = dict(base)
        if (ci == 1 and i >= 100) or ci == 2: ms[3] = frag3
        if ci == 1 and 100 <= i < 150: ms[2] = None
        if (ci == 1 and i >= 150) or ci == 2: ms[2] = base[3]
        return ms
    gsegs_b = lambda t_, e: [_r1_seg(f"{k}_001", k, t_(0, 0), e) for k in (0, 1, 2)] + [
        _r1_seg("3_001", 3, t_(0, 0), t_(1, 99), close="absence_break")]
    r, st, c = _r1_case(tmp, "r2b", "20240423", 12, 4, gt_b, gsegs_b, asg, cand_b, open_segs(2, 1, 150, 150), [dopen(2, 1, 150)],
                        mode="score", teacher_fn=lambda ci, i: dict(base))
    a0 = first(c)
    if st.get("MH1") != "FAIL" or a0.get("verdict") != "wrong" or "trusted identity" not in str(a0.get("why")):
        fails.append(f"R2-b duplicate open onto a GT-dark calf: MH1 {st.get('MH1')} {a0.get('verdict')} {a0.get('why')} "
                     f"id_dup {a0.get('id_dup_rows')}")
    # R2-c (finding 1, geometry exclusion): R1-c's duplicate open with its first 10 s inside the 04-23 12:05:24 ±30 s
    # GT-defect exclusion (GT holds four animals) -> unknown (REVIEW)
    off2 = _mk(H, W, [(40, 58, 4, 22)])
    def cand_c(ci, i):
        ms = dict(base)
        if ci >= 1: ms[2] = off2
        if ci == 1 and i < 50: ms[3] = None
        if (ci == 1 and i >= 50) or ci == 2: ms[3] = base[2]
        return ms
    r, st, w = _r1_case(tmp, "r2c", "20240423", 12, 4, lambda ci, i: dict(base), seeds, asg, cand_c,
                        open_segs(3, 1, 0, 50), [dopen(3, 1, 50)])
    a0 = first(w)
    if st.get("MH1") != "REVIEW" or a0.get("verdict") != "unknown" or not a0.get("dup_rows") or a0.get("dup_rows_evaluable") != 0:
        fails.append(f"R2-c duplicate inside a GT-defect exclusion -> unknown: MH1 {st.get('MH1')} {a0.get('verdict')} "
                     f"{a0.get('why')} dup {a0.get('dup_rows')}/{a0.get('dup_rows_evaluable')}")
    # R2-d (non-calf): a dark-pass open onto a worker blob with the GT holding four animals -- at 14:04 -> wrong (control);
    # inside the 12:05:24 GT-defect exclusion -> unknown, even where the teacher agrees there is no animal (the ±30 s
    # windows are excluded outright, SPEC §6.3); at 14:04 with the teacher ALSO masking the blob -> unknown
    worker = _mk(H, W, [(22, 36, 40, 60)])
    def cand_d(ci, i):
        ms = dict(base)
        if ci == 1 and i < 50: ms[0] = None
        if (ci == 1 and i >= 50) or ci == 2: ms[0] = worker
        return ms
    for tag, hh, tf, want_v, want_mh1 in (("r2d0", 14, None, "wrong", "FAIL"), ("r2d1", 12, None, "unknown", "REVIEW"),
                                          ("r2d3", 12, lambda ci, i: dict(base), "unknown", "REVIEW"),
                                          ("r2d2", 14, lambda ci, i: {**base, 0: worker}, "unknown", "REVIEW")):
        r, st, c = _r1_case(tmp, tag, "20240423", hh, 4, lambda ci, i: dict(base), seeds, asg, cand_d, open_segs(0, 1, 0, 50),
                            [dopen(0, 1, 50)], mode="score", teacher_fn=tf)
        a0 = first(c)
        if st.get("MH1") != want_mh1 or a0.get("verdict") != want_v:
            fails.append(f"R2-d {tag} non-calf open: MH1 {st.get('MH1')} (want {want_mh1}) {a0.get('verdict')} (want {want_v}) "
                         f"{a0.get('why')} none {a0.get('none_out')}/{a0.get('none_not_evaluable')}")
    # R2-e (trusted_animal): two DISJOINT cand masks on one GT mask (a GT merge) name no identity; OVERLAPPING ones (a cand
    # duplicate on one animal) keep the assigned mask's identity
    n = 2
    Ae = dict(ts=np.arange(n, dtype=float), c_stat=np.array([[C_ASSIGNED, C_SPLIT, 0, 0]] * n, np.int8),
              c_gs=np.array([[0, 0, -1, -1]] * n, np.int8), g_anim=np.array([[2, 1, -2, -2]] * n, np.int8),
              g_idt=np.array([[True, True, False, False]] * n), c_area=np.array([[10, 10, 0, 0]] * n, np.int32),
              p_over=np.array([0, 1 << PAIR_IDX[(0, 1)]], np.uint8))
    ta = trusted_animal(dict(A=Ae, strat=np.zeros(n, np.int8)), np.arange(n), 0)
    if ta.tolist() != [-1, 2]: fails.append(f"R2-e shared GT mask identity (disjoint -> -1, overlapping -> kept): {ta.tolist()}")
    # R2-f (finding 2, MH7): v4v2 splits calf 2 into its slot-2 mask and a slot-0 fragment for 70 s at 14:04 (calf 0 unmasked);
    # a cand with the right geometry is MULTI on calf 2 and NONE on calf 0. No teacher -> v4v2 only -> counted -> FAIL;
    # the teacher = the right geometry -> it contradicts v4v2 -> excused -> PASS; the teacher = v4v2 -> agrees -> FAIL;
    # the same inside the 12:05:24 ±30 s GT-defect exclusion (no teacher) -> excused -> PASS
    g2p, g0f = _mk(H, W, [(40, 58, 2, 9)]), _mk(H, W, [(40, 58, 9, 22)])
    def gt_f(ival):
        def f(ci, i):
            ms = dict(base)
            if ival(ci, i): ms[2] = g2p; ms[0] = g0f
            return ms
        return f
    at14 = lambda ci, i: ci == 1 and i < 350
    at12 = lambda ci, i: (ci == 0 and i >= 300) or (ci == 1 and i < 100)      # 12:05:00-12:05:40 inside 12:04:54-12:05:54
    for tag, hh, ival, tf, want, key, secs in (
            ("r2f0", 14, at14, None, "FAIL", "counted_v4v2_only", 70.0),
            ("r2f1", 14, at14, lambda ci, i: dict(base), "PASS", "excused_teacher_contradicts", 70.0),
            ("r2f2", 14, at14, lambda ci, i: gt_f(at14)(ci, i), "FAIL", "counted_teacher_agrees", 70.0),
            ("r2f3", 12, at12, None, "PASS", "excused_gt_defect_window", 40.0)):
        r, st, c = _r1_case(tmp, tag, "20240423", hh, 4, gt_f(ival), seeds, asg, lambda ci, i: dict(base), seeds,
                            mode="score", teacher_fn=tf)
        v = [t for t in r["must_hold"] if t["id"] == "MH7"][0]["value"]["20240423"]
        gm = v.get("geometry_main") or {}
        got = [(gm.get(lb) or {}).get(key) for lb in ("MULTI", "NONE")]
        if st.get("MH7") != want or any(g is None or abs(g - secs) > 0.01 for g in got) \
                or abs(v["raw_out_gt_dark"]["multi_s"] - secs) > 0.01 or abs(v["raw_out_gt_dark"]["none_s"] - secs) > 0.01:
            fails.append(f"R2-f {tag} MH7 {st.get('MH7')} (want {want}) {key} MULTI/NONE {got} (want {secs}) raw {v['raw_out_gt_dark']}")
    # R2-g (per-slot exclusion 04-22 slot 2 13:47-15:58): a 70 s worker-blob NONE at 13:50 with the GT holding four animals.
    # No teacher -> excused (the excluded GT mask may be the animal) -> PASS; the teacher sees no animal there either
    # -> counted -> FAIL. The same blob as a dark-pass open: no teacher -> unknown (REVIEW); the teacher agrees -> wrong.
    asg22 = dict(asg)
    for tag, tf, want, key in (("r2g0", None, "PASS", "excused_gt_defect_window"),
                               ("r2g1", lambda ci, i: dict(base), "FAIL", "counted_teacher_agrees")):
        r, st, c = _r1_case(tmp, tag, "20240422", 13, 50, lambda ci, i: dict(base), seeds, asg22,
                            lambda ci, i: {**base, **({0: worker} if ci == 1 and i < 350 else {})}, seeds, mode="score", teacher_fn=tf)
        v = [t for t in r["must_hold"] if t["id"] == "MH7"][0]["value"]["20240422"]
        got = ((v.get("geometry_main") or {}).get("NONE") or {}).get(key)
        if st.get("MH7") != want or got is None or abs(got - 70.0) > 0.01:
            fails.append(f"R2-g {tag} per-slot exclusion NONE: MH7 {st.get('MH7')} (want {want}) {key} {got}")
    for tag, tf, want_v, want_mh1 in (("r2h0", None, "unknown", "REVIEW"), ("r2h1", lambda ci, i: dict(base), "wrong", "FAIL")):
        r, st, c = _r1_case(tmp, tag, "20240422", 13, 50, lambda ci, i: dict(base), seeds, asg22, cand_d, open_segs(0, 1, 0, 50),
                            [dopen(0, 1, 50)], mode="score", teacher_fn=tf)
        a0 = first(c)
        if st.get("MH1") != want_mh1 or a0.get("verdict") != want_v:
            fails.append(f"R2-h {tag} non-calf open on a per-slot exclusion row: MH1 {st.get('MH1')} (want {want_mh1}) "
                         f"{a0.get('verdict')} (want {want_v}) {a0.get('why')}")
    # R2-i: R2-c's duplicate open (both cand slots on GT slot 2) at 04-22 13:50, where GT slot 2's geometry is excluded
    # -> unknown (REVIEW), not wrong
    r, st, c = _r1_case(tmp, "r2i", "20240422", 13, 50, lambda ci, i: dict(base), seeds, asg22, cand_c, open_segs(3, 1, 0, 50),
                        [dopen(3, 1, 50)], mode="score")
    a0 = first(c)
    if st.get("MH1") != "REVIEW" or a0.get("verdict") != "unknown" or not a0.get("dup_rows") or a0.get("dup_rows_evaluable") != 0:
        fails.append(f"R2-i duplicate on the per-slot excluded GT mask -> unknown: MH1 {st.get('MH1')} {a0.get('verdict')} "
                     f"dup {a0.get('dup_rows')}/{a0.get('dup_rows_evaluable')}")
    return "review r2: step-2 geometric rules on evaluable rows only (GT over two calves / GT-defect exclusion / teacher " \
           "contradiction -> unknown; per-slot exclusion decided by the teacher), identity duplicate on a GT-dark calf -> " \
           "wrong, a GT mask over two disjoint cand masks names no identity, MH7 counted where v4v2 and the teacher agree " \
           "or v4v2 is the only reference (excused reported)"

def _st_review_r3(fails, tmp):
    """review r3 (2026-09-27; re-review r2 finding, repro build/rr48r2/rr_tmerge.py): a TEACHER mask over two calves
    names no 5f fill identity for a cand NONE mask -- neither the GT identity of the GT mask it is paired with (t_on_gt)
    nor the teacher-segment identity -- when (a) another cand mask, DISJOINT from it, holds that GT mask, (b) another
    disjoint cand mask lies inside the teacher mask (the GT-dark sibling), or (c) NO cand mask overlapping it holds that
    GT mask (the cand lost the GT-visible calf: a wrong link Y -> Z would otherwise PASS as 'same animal'). A correct
    dark-pass open / dark_link onto the calf the GT lost is then unknown (REVIEW), never wrong, and its fills are
    teacher_unknown (MH5 counts no wrong fill). Controls: a one-calf teacher mask names the calf (correct, PASS); a ghost
    GT mask under the teacher mask is no second animal (the teacher key decides, PASS); an OVERLAPPING cand mask holding
    the paired GT mask keeps t_on_gt (a duplicate -> wrong, MH1 FAIL); R2-b's overlapping duplicate stays wrong (r3f)."""
    H, W = 60, 100
    base = {k: _mk(H, W, [v]) for k, v in {0: (2, 18, 2, 22), 1: (2, 18, 40, 60), 2: (40, 58, 2, 22), 3: (40, 58, 70, 95)}.items()}
    merged = _mk(H, W, [(40, 58, 2, 95)])                  # teacher slot 2: calves 2+3 (and the straw between) in one mask
    asg = {"0_001": 0, "1_001": 1, "2_001": 2, "3_001": 3}
    first = lambda c: (c["step2_actions"]["list"] or [{}])[0]
    dark = lambda ci, i: (ci == 1 and i >= 100) or ci == 2          # the GT-dark stretch (and the teacher merge)
    gap = lambda ci, i: ci == 1 and 100 <= i < 150                   # the cand slot's own dark gap before the action
    after = lambda ci, i: (ci == 1 and i >= 150) or ci == 2          # the new cand segment
    def mk_gt(lost, extra=None):
        def f(ci, i):
            ms = dict(base)
            if dark(ci, i):
                for s in lost: ms[s] = None
                ms.update(extra or {})
            return ms
        return f
    def mk_gsegs(lost, extra_segs=()):
        return lambda t_, e: [_r1_seg(f"{k}_001", k, t_(0, 0), e) for k in range(N_SLOTS) if k not in lost] + [
            _r1_seg(f"{k}_001", k, t_(0, 0), t_(1, 99), close="absence_break") for k in lost] + [
            _r1_seg(sid, s, t_(1, 100), e) for sid, s in extra_segs]
    def mk_teacher(merge):
        def f(ci, i):
            ms = dict(base)
            if merge and dark(ci, i): ms[2] = merged; ms[3] = None
            return ms
        return f
    def mk_cand(slot, new_mask, lost_for_good=()):
        """cand slot `slot` goes dark over gap() and re-appears on new_mask after(); lost_for_good slots vanish at dark()"""
        def f(ci, i):
            ms = dict(base)
            for s in lost_for_good:
                if dark(ci, i): ms[s] = None
            if gap(ci, i): ms[slot] = None
            if after(ci, i): ms[slot] = new_mask
            return ms
        return f
    def mk_segs(slot, how, closed=()):
        """cand segments: `slot` closes at the gap and re-opens after it (how = 'open': dark-pass reopen, 'link':
        dark_link from its previous segment); `closed` slots end at the gap for good"""
        def f(t_, tend):
            nxt = dict(reason="reopen", via="dark_pass") if how == "open" else \
                dict(reason="dark_link", link_from=f"{slot}_001", cause="collapse")
            return [_r1_seg(f"{k}_001", k, t_(0, 0), tend) for k in range(N_SLOTS) if k != slot and k not in closed] + [
                _r1_seg(f"{k}_001", k, t_(0, 0), t_(1, 100), close="dark_gap") for k in closed] + [
                _r1_seg(f"{slot}_001", slot, t_(0, 0), t_(1, 100), close="dark_gap"),
                _r1_seg(f"{slot}_002", slot, t_(1, 150), tend, **nxt)]
        return f
    def mk_ev(slot, how):
        if how == "open":
            return lambda stems, t_: dict(type="dark_open", slot=slot, seg_id=f"{slot}_002", clip=stems[1] + ".mp4", t=30.0)
        return lambda stems, t_: dict(type="dark_link", slot=slot, seg_id=f"{slot}_002", prev_seg_id=f"{slot}_001",
                                      clip=stems[1] + ".mp4", t=30.0, cause="collapse")
    gfrag = _mk(H, W, [(40, 58, 85, 95)])                  # r3j: v4v2 keeps a GHOST fragment (unassigned 3_002) on calf 3's back
    cfront = _mk(H, W, [(40, 58, 70, 84)])                 # ... and the cand re-opens on calf 3's front (disjoint from it)
    gpart = _mk(H, W, [(40, 58, 70, 80)])                  # r3i: v4v2 holds only calf 3's front (assigned 3_001) ...
    cright = _mk(H, W, [(40, 58, 82, 95)])                 # ... slot 2 opens on calf 3's back, INSIDE slot 3's full mask
    # (tag, hh, GT: lost slots / extra masks / extra segs, teacher merges 2+3, cand slot, cand new mask, cand slots lost for
    #  good, open|link, want verdict, want MH1, want t_span via on the new mask's fills, want fill class)
    O, L = "open", "link"
    cases = (
        ("r3a14", 14, (3,), None, (), True, 3, base[3], (), O, "unknown", "REVIEW", "paired_gt_mask", "teacher_unknown"),  # repro
        ("r3a22", 22, (3,), None, (), True, 3, base[3], (), O, "unknown", "REVIEW", "paired_gt_mask", "teacher_unknown"),  # night
        ("r3b", 14, (3,), None, (), True, 3, base[3], (), L, "unknown", "REVIEW", "paired_gt_mask", "teacher_unknown"),    # link
        ("r3c", 14, (2, 3), None, (), True, 3, base[3], (), O, "unknown", "REVIEW", "teacher_mask", "teacher_unknown"),    # GT-dark 2+3
        # (c) the cand LOST calf 2 (no cand mask holds GT 2): a correct reopen of slot 3 on calf 3 -> unknown ...
        ("r3g", 14, (3,), None, (), True, 3, base[3], (2,), O, "unknown", "REVIEW", "paired_gt_mask_unheld", "teacher_unknown"),
        # ... and a WRONG dark_link of slot 2 from calf 2 onto calf 3 (slot 3 lost): without (c) the merged teacher mask
        # names calf 2 on both ends -> a false 'same animal' PASS; now unknown (REVIEW)
        ("r3h", 14, (3,), None, (), True, 2, base[3], (3,), L, "unknown", "REVIEW", "paired_gt_mask_unheld", "teacher_unknown"),
        ("r3d", 14, (3,), None, (), False, 3, base[3], (), O, "correct", "PASS", None, "correct"),    # control: one-calf teacher
        ("r3e", 14, (3,), None, (), False, 3, base[3], (), L, "correct", "PASS", None, "correct"),    # control: link
        # control: the teacher mask is paired with a GHOST GT fragment (not an animal) -> no second animal, the teacher key
        # names calf 3 -> correct
        ("r3j", 14, (3,), {3: gfrag}, (("3_002", 3),), False, 3, cfront, (), O, "correct", "PASS", None, "correct"),
        # control: an OVERLAPPING cand mask (slot 3, full calf 3) holds the paired GT mask (v4v2's front of calf 3) -> t_on_gt
        # kept = calf 3 -> slot 2's open onto calf 3 is a duplicate by trusted identity -> wrong (and its fill, labelled
        # calf 2, is a wrong step-2 fill)
        ("r3i", 14, (1,), {3: gpart}, (), False, 2, cright, (), O, "wrong", "FAIL", None, "wrong"))
    def scan_of(tag):                                      # the scan (with its 5f fill records) of one case
        return scan_day(dict(arm="cand", day="20240423", cand_dir=os.path.join(tmp, "r1_" + tag, "cand", "20240423"),
                             gt_dir=os.path.join(tmp, "r1_" + tag, "gt", "20240423"),
                             gt_assign=os.path.join(tmp, "r1_" + tag, "gtfiles", "20240423", "segment_assignments.json"),
                             teacher_dir=os.path.join(tmp, "r1_" + tag, "teacher", "20240423"), frag_lo=dict(FRAG_LO)))
    for tag, hh, lost, extra, xsegs, merge, slot, nmask, gone, how, want_v, want_mh1, via, want_fc in cases:
        r, st, c = _r1_case(tmp, tag, "20240423", hh, 4, mk_gt(lost, extra), mk_gsegs(lost, xsegs), asg,
                            mk_cand(slot, nmask, gone), mk_segs(slot, how, gone), [mk_ev(slot, how)], mode="score",
                            teacher_fn=mk_teacher(merge))
        a0 = first(c)
        if st.get("MH1") != want_mh1 or (r["gate"] == "FAIL" and want_mh1 != "FAIL") or a0.get("verdict") != want_v:
            fails.append(f"R3 {tag}: MH1 {st.get('MH1')} (want {want_mh1}) gate {r['gate']} {a0.get('verdict')} (want "
                         f"{want_v}) {a0.get('why')} new {a0.get('new_animal')} prev {a0.get('prev_animal')} id_dup "
                         f"{a0.get('id_dup_rows')}")
        if want_v == "correct" and a0.get("new_animal") != 3:
            fails.append(f"R3 {tag} control: the new mask is on calf 3, the evidence names {a0.get('new_animal')}")
        # scan level: which rule fired on the new mask's fills (its rows after the gap)
        S3 = scan_of(tag)
        t_new = ts_of(datetime(2024, 4, 23, hh, 4) + timedelta(seconds=110))
        fs = [f for f in S3["fills"] if f["slot"] == slot and f["t"] >= t_new - 1e-6]
        vias = Counter((f.get("t_span") or {}).get("via") for f in fs)
        if via is not None:
            bad = not fs or set(vias) != {via} or any(f["t_on_gt"] is not None or f["tkey"] is not None for f in fs)
        else:
            bad = not fs or set(vias) != {None} or any(tuple(f["tkey"] or ()) != ("seg", "3_001") for f in fs) \
                or (tag == "r3i" and any(f["t_on_gt"] != 3 for f in fs))
        if bad:
            fails.append(f"R3 {tag} span rule on slot {slot}'s fills: {dict(vias)} (want only {via}), n {len(fs)}, "
                         f"tkeys {dict(Counter(str(f['tkey']) for f in fs))} t_on_gt {dict(Counter(f['t_on_gt'] for f in fs))}")
        # MH5 level (5f): every cand slot's Tol_30 label is its seed calf (slot k -> calf k); the new mask's fills must be
        # teacher_unknown where the teacher mask spans two regions, correct / wrong on the controls
        lab = np.tile(np.arange(N_SLOTS, dtype=np.int8), (len(S3["A"]["ts"]), 1))
        mech = {r_["idx"]: seg_mech(r_, events_by_seg(S3["cev"])) for r_ in S3["cseg"]}
        fm = fill_metrics(dict(S3, fills=fs), lab, {k: k for k in range(N_SLOTS)}, mech)
        cls = {k_: v_ for k_, v_ in fm["by_class_s"].items() if v_ > 0}
        if set(cls) != {want_fc} or (fm["step2_wrong_s"] > 0) != (want_fc == "wrong"):
            fails.append(f"R3 {tag} MH5 fill classes of slot {slot}: {cls} (want only {want_fc}) step2_wrong_s "
                         f"{fm['step2_wrong_s']} sources {fm['by_source_s']}")
    dup3 = _mk(H, W, [(40, 58, 60, 95)])                   # the new slot 3 overlaps slot 2's mask on calf 3 (a cand duplicate)
    def cand_dup(ci, i):
        ms = dict(base)
        if dark(ci, i): ms[2] = base[3]
        if gap(ci, i): ms[3] = None
        if after(ci, i): ms[3] = dup3
        return ms
    r, st, c = _r1_case(tmp, "r3f", "20240423", 14, 4, mk_gt((2, 3)), mk_gsegs((2, 3)), asg, cand_dup, mk_segs(3, "open"),
                        [mk_ev(3, "open")], mode="score", teacher_fn=mk_teacher(False))
    a0 = first(c)
    if st.get("MH1") != "FAIL" or a0.get("verdict") != "wrong" or "trusted identity" not in str(a0.get("why")):
        fails.append(f"R3-f overlapping duplicate on a GT-dark calf must stay wrong: MH1 {st.get('MH1')} {a0.get('verdict')} "
                     f"{a0.get('why')} id_dup {a0.get('id_dup_rows')}")
    return "review r3: a teacher mask over two calves names no 5f fill identity (paired GT mask held by a disjoint cand " \
           "slot or by no overlapping cand mask, or a disjoint cand mask inside it) -> a correct open / link is unknown " \
           "(REVIEW), a wrong link no longer PASSes, its fills are teacher_unknown (no MH5 wrong fill); one-calf teacher, " \
           "ghost GT fragment and overlapping-holder controls keep their identity (correct / duplicate wrong); an " \
           "overlapping cand duplicate on a GT-dark calf stays wrong"

def selftest(data=None):
    fails = []; msgs = []
    import tempfile
    t0 = time.time()
    msgs.append(_st_rle_ops(fails))
    msgs.append(_st_associate(fails))
    msgs.append(_st_switches_gaps(fails))
    msgs.append(_st_d8_geometry_mem(fails))
    with tempfile.TemporaryDirectory() as tmp:
        msgs.append(_st_stream(fails, tmp))
        msgs.append(_st_end_to_end(fails, tmp))
        msgs.append(_st_gpu_telemetry(fails, tmp))
        msgs.append(_st_g2(fails, tmp))
        msgs.append(_st_g5a(fails, tmp))
        msgs.append(_st_review_r1(fails, tmp))
        msgs.append(_st_review_r2(fails, tmp))
        msgs.append(_st_review_r3(fails, tmp))
    for m in msgs: print("  ran:", m)
    if data:
        print(f"  reproducing the known v4v2 figures from {data} ...", flush=True)
        for day, exp in V4V2_KNOWN.items():
            dd = os.path.join(data, day)
            if not os.path.isdir(dd): print(f"  SKIPPED {day} (no data)"); continue
            S = scan_day(dict(arm="ref", day=day, cand_dir=dd, gt_dir=None, light=True))
            G = gap_metrics(S); SG = segment_metrics(S); FR = fragment_metrics(S)
            got = dict(nomask_h=round(G["hours"], 2), segments=SG["n"], merge_resolved=SG["merge_resolved"],
                       night_frag_pct=round(FR["by_phase"]["night"]["frag_half_own_pct"], 1))
            print(f"    {day}: {got}  expected {exp}  ({S['scan_s']} s)")
            for k, v in exp.items():
                if got[k] != v: fails.append(f"V4V2 {day} {k}: got {got[k]} expected {v}")
    else:
        print("  SKIPPED (no data): known-v4v2 reproduction -- G0 requires one run with --data")
    print(f"SELFTEST {'PASSED' if not fails else 'FAILED'} ({time.time() - t0:.0f} s)")
    for f in fails[:50]: print("   FAIL", f)
    return not fails

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--data", default=None)
    ap.add_argument("--cand"); ap.add_argument("--ref"); ap.add_argument("--gt_root", default=None,
                    help="root holding the GT masks/segments/events per day (default: --ref, i.e. Tracking_v4v2)")
    ap.add_argument("--gt_dir", help="dir with the GT assignment files + week_ident_map.json")
    ap.add_argument("--days", default="20240422,20240423")
    ap.add_argument("--cand_day_dir", action="append", help="DAY=subdir (e.g. 20240423=20240423_w1152-1259)")
    ap.add_argument("--ref_day_dir", action="append")
    ap.add_argument("--teacher", action="append", help="DAY=<teacher day dir>")
    ap.add_argument("--teacher_rules_0422", default=None, help="reid_ground_truth_teacher_0422.json (v2, user-verified)")
    ap.add_argument("--run_tol30", action="store_true"); ap.add_argument("--tol30", default=os.path.join(HERE, "Tol_30_periodic_identity.py"))
    ap.add_argument("--tol30_sha", default=None, help="pin: the Tol_30 sha256 (prefix) that must be scored with")
    ap.add_argument("--week_map_gt", default=None, help="week_ident_map.json for the GT (default <gt_dir>/week_ident_map.json)")
    ap.add_argument("--week_map_ref", default=None); ap.add_argument("--week_map_cand", default=None)
    ap.add_argument("--nightgap_cand", default=None); ap.add_argument("--nightgap_ref", default=None)
    ap.add_argument("--telemetry", action="append", help="ARM:DAY=<Metrics_track_*.json>")
    ap.add_argument("--train_summary", default=None)
    ap.add_argument("--work", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--d6_fallback", action="store_true",
                    help="addendum D6: the cand root ran on an A100 under mem_cap_gib=21.0 (MH8 then requires exactly that, labelled)")
    ap.add_argument("--g2", action="store_true", help="coded G2 pilot evaluator (verdict JSON)")
    ap.add_argument("--window", action="append", help="G2: DAY=SUBDIR of a pilot window, e.g. 20240423=20240423_w1152-1259")
    ap.add_argument("--no_ref", action="store_true", help="G2 / G5a: skip the (report-only) re-scored reference arm")
    ap.add_argument("--g5a", action="store_true", help="coded G5a evaluator for 04-24 (verdict JSON)")
    ap.add_argument("--gt24", default=None, help="G5a: gt_0424_from_user_notes.json (the user's docx interval GT)")
    ap.add_argument("--asg24", default=None, help="G5a: Tracking_v4v2p2_reid/20240424/segment_assignments.json")
    ap.add_argument("--g3_metrics", default=None, help="G5a: the G3 metrics JSON (cand 04-23 identity map, for the 04-23->24 report)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.data) else 1)
    if a.g2:
        for k in ("cand", "ref", "gt_dir"):
            if not getattr(a, k): ap.error(f"--g2: --{k} is required")
        a.out = a.out or "Metrics_v5_G2_verdict.json"
        r = g2_evaluate(a); sys.exit(0 if r["gate"] == "PASS" else (3 if r["gate"] == "REVIEW" else 1))
    if a.g5a:
        for k in ("cand", "gt_root", "gt24", "asg24"):
            if not getattr(a, k): ap.error(f"--g5a: --{k} is required")
        a.out = a.out or "Metrics_v5_G5a_verdict.json"
        r = g5a_evaluate(a); sys.exit(0 if r["gate"] == "PASS" else (3 if r["gate"] == "REVIEW" else 1))
    for k in ("cand", "ref", "gt_dir"):
        if not getattr(a, k): ap.error(f"--{k} is required")
    a.out = a.out or "Metrics_score_retrack.json"
    score(a)

if __name__ == "__main__":
    main()
