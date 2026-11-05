# Protocol files

Frozen values read by the benchmark code. Copy them to `paper1_distillation/protocol/` under your registry root.

| File | What it holds | SHA-256 as it ran | SHA-256 here |
|:---|:---|:---|:---|
| `es3_g2_v1.json` | Gate record of the EfficientSAM3 encoder. The code pins it by hash | `99b2fe0d6d1978a70df1a8b988ff5fcb52b17f814f5a113cc7583f465de4d257` | same |
| `protD_tau_v1.json` | Detector threshold of Protocol D, frozen on PigTrack validation | `a72b24970a25ea066dc27e7a734ccd00a788465dcba884476b29c44a5f659b8a` | `6d7c1385d801f437255ffc326971af9c2e036cbeb42637a882422cbb0c8eb25b` |
| `mot_xy_offset_v1_pigtrack.json` | MOT xy offset, PigTrack | `153e5e4e430ef03f39a3dd1aaca88c037e86a59529013b212a42d0708aba0821` | `d65f5bd167948c6008f75a55cdcc9a0e78abd9c9c72813fd96beeb5d5c16b181` |
| `mot_xy_offset_v1_edinburgh.json` | MOT xy offset, Edinburgh | `0fde808fe77376ce0da41136f569c9390438399716f49ee1a9a5a1006dab8759` | `d8393eb39ee4dce61e0cf1f1606f024153411968d9826f7a3624d943b920235d` |
| `hf_revisions_v1.json` | Pinned model revisions | not re-checked here | not re-checked here |

Where the hash differs, one workspace path line was changed to `<DATABRICKS_USER>`. No value changed.

`calf_windows_v1.json` and `calf_windows_v2.json` are not here. They hold the seed masks of the calf windows, including 23 April, and their release is an open question.
