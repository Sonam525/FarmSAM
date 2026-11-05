# Notice

## Code

Code in this repository is under the MIT licence in [LICENSE](LICENSE), unless a file or the list below says otherwise.

## Weights

No weights are in this repository. The student checkpoints are published separately on Hugging Face, and MIT does not apply to them.

The students are trained to match features of the SAM 3 image backbone. They hold TinyViT and head tensors, and they run inside SAM 3's tracker. They are distributed with a copy of the [SAM License](LICENSES/SAM_LICENSE.txt). SAM 3 itself is not redistributed here: you download it from Meta under its gate.

The two Edinburgh students were distilled on frames of the Edinburgh pig dataset (CC BY-NC) and are marked non-commercial, research only.

## Acknowledgement

This work uses SAM 3 by Meta. Trade control limits in the SAM License apply to its use.

## Data shown

Figures use PigTrack frames (Henrich et al., CC BY 4.0, DOI 10.25625/P7VQTP), with overlays added. No Edinburgh frame appears in this repository. Calf figures here are charts. They show no calf frame.

## Changes to third-party material

The students start from `timm/tiny_vit_21m_512.dist_in22k_ft_in1k` weights and change them through training. BatchNorm is replaced by GroupNorm.
