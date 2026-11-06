# Distillation code

Store SAM 3 targets, optionally mine extra frames, then train a student. The folder is flat because scripts import each other by file name, and `bench_core.py` and `farm_paths.py` from `benchmark/core/` sit beside them when they run.

| Step | Calves | PigTrack | Edinburgh |
|:---|:---|:---|:---|
| Targets | `extract_teacher_targets.py` | `extract_teacher_targets_pigtrack.py` | preprint notebook |
| Extra frames | `mine_targets.py` | `mine_targets_pigtrack.py` | `measure_edinburgh_hard_frames.py`, `edinburgh_frame_similarity.py`, `select_targets_edinburgh.py` |
| Training | `distill_student.py` | `distill_student_pigtrack.py`, then `distill_pigtrack_mined.py` for the mined student | `distill_edinburgh_mined.py` |

`distill_core.py` holds the shared helpers and `test_distill_core.py` tests them with numpy alone.

These are Databricks source-format notebooks. Set the storage paths to yours. Some constants pin the exact input files of our runs, so change those when you train on your own data. Steps are in [docs/REPRODUCE.md](../docs/REPRODUCE.md).
