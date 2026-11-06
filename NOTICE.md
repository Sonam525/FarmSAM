# Notice

Licence holder: Haiyu Yang.

## Code

Code in this repository is under the MIT licence in [LICENSE](LICENSE), unless a file or the list below says otherwise.

**Exception.** `benchmark/core/es3_encoder.py` is outside the MIT grant for now. The file re-implements glue around the EfficientSAM3 encoder with the authors' module names, and that code sits in a fork of SAM 3 whose licence file is the SAM License. Check those terms before you reuse it.

## Weights

No weights are in this repository. The five student checkpoints are published on Hugging Face at [`Sonam5/FarmSAM`](https://huggingface.co/Sonam5/FarmSAM). MIT does not apply to them.

All five students are released under the [SAM License](LICENSES/SAM_LICENSE.txt). They are trained to match features of the SAM 3 image backbone, hold TinyViT and head tensors, and run inside SAM 3's tracker, so we apply its terms to be safe. Anyone who passes a student on must do so under those terms and include a copy of the licence. SAM 3 itself is not redistributed here: you download it from Meta under its gate.

**Non-commercial limit.** The two Edinburgh students may be used for non-commercial purposes only. They were distilled on frames of the Edinburgh pig dataset (CC BY-NC). This limit comes from the data licence and applies on top of the SAM License.

## Acknowledgement

This work uses SAM 3 by Meta. Clause 1.b.ii of the SAM License requires publications that use SAM 3 or these students to acknowledge it. Trade control limits in the SAM License apply to its use.

## Data shown

Figures use PigTrack frames (Henrich et al., CC BY 4.0, DOI 10.25625/P7VQTP), with overlays added. No Edinburgh frame appears in this repository. Calf figures here are charts. They show no calf frame.

## Changes to third-party material

The students start from `timm/tiny_vit_21m_512.dist_in22k_ft_in1k` weights and change them through training. BatchNorm is replaced by GroupNorm.

This notice is not legal advice.
