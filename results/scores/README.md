# Raw score tables

These are the stored outputs of the runs, copied as they are. The tables in [docs/RESULTS.md](../../docs/RESULTS.md) and the CSV files one level up are the ones to quote.

Three fields in `efficiency/tab_efficiency.csv` and `efficiency/tab_efficiency_protD.csv` must not be read as they stand:

| Field | File | What it says | What it means |
|:---|:---|:---|:---|
| `streams_analytic` | `tab_efficiency.csv` | 67, 48, 75, 169, 70 | The stored count takes the leftover memory once per stream. It is biased. Use [`efficiency_streams.csv`](../efficiency_streams.csv), which gives 84, 86, 87, 246, 87 |
| `fits_a10` | `tab_efficiency_protD.csv` | `False` on both PigTrack runs | Not assessable. `pigtrack0002` has 985 frames, fewer than the 2,000 the check needs |
| `detector_weights_mib` | `tab_efficiency_protD.csv` | `0.0` | Not measured. The RF-DETR weights were not on the GPU at the reading |

Keys such as `T`, `S-PT` and `S21` are record keys. [docs/CHECKPOINTS.md](../../docs/CHECKPOINTS.md) maps the student keys to names. `T` is SAM 3, `T21` SAM 2.1-tiny, `S21` SAM 2.1-small, `ET` EdgeTAM and `ES-L` the EfficientSAM3 encoder.
