# AeroAD-CLIP

Official code release for **AeroAD-CLIP: A Semantics–Geometry Synergized Framework for Zero-Shot Aircraft Skin Defect Detection** (IEEE Transactions on Aerospace and Electronic Systems, under review).

AeroAD-CLIP detects aircraft skin defects **without any aircraft-specific training data**. The CLIP backbone is frozen; only three lightweight parts are trained, all on public industrial anomaly-detection data (MVTec AD / VisA):

- **MLA — Multi-Layer Adapter**: two MLP projectors that map the CLIP text anchors ("normal" / "anomalous") into the visual feature space of every selected ViT layer (6, 12, 18, 24).
- **SAQ — Semantic-Adaptive Querying**: four learnable queries that are propagated layer by layer through self- and cross-attention, so that the discriminative direction is conditioned both on the image and on the depth of the layer.
- **GCT — Geometric Consistency Tuning**: a bottleneck autoencoder trained on normal images only, whose reconstruction gradients are clustered into anomaly-manifold prototypes; the normal anchor is rotated toward these prototypes on the unit sphere (SLERP), and a patch is only accepted when several prototypes agree.

This repository contains the source code of the three modules, the checkpoints of **all trainable parts**, and the pre-computed **anomaly-manifold prototypes** in HDF5 format.

<p align="center">
  <img src="assets/framework.png" alt="Overview of AeroAD-CLIP" width="92%">
</p>

<p align="center">
  <em>Overview of AeroAD-CLIP. (1) Frozen CLIP backbone with the Multi-Layer Adapter (MLA) that maps text anchors into the visual space of layers 6/12/18/24; (2) Semantic-Adaptive Querying (SAQ), where four learnable queries are propagated across layers by self- and cross-attention; (3) Geometric Consistency Tuning (GCT), where reconstruction-gradient prototypes steer the normal anchor on the unit sphere by SLERP and the per-patch deviations are combined by consistency voting.</em>
</p>

---

## 1. What is included

| Component | Code | Weights | Status |
|---|---|---|---|
| Multi-Layer Adapters (MLA) | `models/mla.py` | `checkpoints/mvtec_ad/MLA/adapters_{6,12,18,24}.pth`<br>`checkpoints/visa/MLA/adapters_{6,12,18,24}.pth` | **released** |
| Semantic-Adaptive Querying (SAQ) | `models/saq.py` | `checkpoints/*/SAQ/*.pth` — **Release asset** | weights shipped in `saq_checkpoints.zip` (≈ 385 MB each, above GitHub's 100 MB file limit) |
| GCT bottleneck autoencoder | `models/gct_autoencoder.py` | `checkpoints/*/GCT/*.pth` — **Release asset** | weights shipped with the same release |
| Anomaly-manifold prototypes (GCT) | `models/gct_tuning.py` | `manifold/anomaly_manifold.h5` — **Release asset** | in preparation |
| Fusion / scoring heads | `models/scoring.py` | — | no trainable parameters |

**Not included**

- **CLIP ViT-L/14-336 weights.** Download `ViT-L/14-336px` from the official OpenAI CLIP release and place it so that `clip.load('ViT-L/14-336px')` can find it. The backbone is frozen in our pipeline and is never fine-tuned.
- The industrial benchmarks (MVTec AD, VisA, BTAD, MPDD, DAGM, KolektorSDD, KolektorSDD2). Please obtain them from their original sources and follow their licenses.
- The aircraft-skin benchmark (see §7).

---

## 2. Repository structure

```
AeroAD-CLIP/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── aero_adclip.yaml          # layers, image size, K, gamma, lambda
├── models/
│   ├── mla.py                    # Multi-Layer Adapter (two MLP projectors per layer)
│   ├── saq.py                    # Semantic-Adaptive Querying (queries + attn blocks)
│   ├── gct_autoencoder.py        # bottleneck AE used to obtain reconstruction gradients
│   ├── gct_tuning.py             # prototype clustering, SLERP, consistency voting
│   └── scoring.py                # heatmap fusion, top-k image score
├── checkpoints/
│   ├── mvtec_ad/MLA/adapters_{6,12,18,24}.pth    # trained on MVTec AD
│   ├── visa/MLA/adapters_{6,12,18,24}.pth        # trained on VisA
│   ├── mvtec_ad/SAQ/                             # via Releases (saq_checkpoints.zip)
│   └── visa/SAQ/                                 # via Releases (saq_checkpoints.zip)
├── manifold/
│   └── anomaly_manifold.h5       # prototypes mu_k^(l), K=16, l in {12,18,24}
├── scripts/
│   ├── infer_single.py           # single-image inference / heatmap export
│   ├── eval_dataset.py           # pixel AUROC / PRO / F1-max on a folder dataset
│   └── visualize.py              # heatmap overlay for figures
└── assets/
    └── framework.png             # Fig. 2 of the paper
```

---

## 3. Environment

```bash
conda create -n aeroadclip python=3.10 -y
conda activate aeroadclip

# PyTorch (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt`

```
numpy>=1.24
opencv-python>=4.8
h5py>=3.9
scipy>=1.10
scikit-learn>=1.3     # k-means for the GCT prototypes
tqdm
pyyaml
```

Everything reported in the paper was run on a single NVIDIA RTX 5070, PyTorch 2.x, FP16 inference.

---

## 4. Checkpoints

All checkpoints are plain `state_dict` files. They were obtained by training **only** on the public source-domain data stated below; no aircraft-skin image was ever used for training or adaptation.

Checkpoints are released for the **MLP adapters (MLA)**, separately for the two source domains: use `mvtec_ad/` for every target domain, and `visa/` only when MVTec AD itself is the target (as in the paper). One file per selected layer:

| File | Trained on | Contents |
|---|---|---|
| `checkpoints/mvtec_ad/MLA/adapters_6.pth`  | MVTec AD | `F_a^(l)`, `F_n^(l)` at layer l = 6 |
| `checkpoints/mvtec_ad/MLA/adapters_12.pth` | MVTec AD | layer l = 12 |
| `checkpoints/mvtec_ad/MLA/adapters_18.pth` | MVTec AD | layer l = 18 |
| `checkpoints/mvtec_ad/MLA/adapters_24.pth` | MVTec AD | layer l = 24 |
| `checkpoints/visa/MLA/adapters_{6,12,18,24}.pth` | VisA | same, for the MVTec-AD target row |

Loading:

```python
import torch
from models.mla import MultiLayerAdapter

for l in (6, 12, 18, 24):
    adapter = MultiLayerAdapter(...)
    adapter.load_state_dict(torch.load(f"checkpoints/mvtec_ad/MLA/adapters_{l}.pth", map_location="cpu"))
    adapter.eval()
```

The CLIP encoders and the MLA/SAQ modules are always evaluated with `torch.no_grad()`.

### Weights shipped through Releases

GitHub rejects any individual file larger than 100 MB, and one SAQ checkpoint is ≈ 385 MB. The SAQ weights are therefore **attached to the `saq-checkpoints` release** instead of being stored in the git history:

1. open the **Releases** page of this repository and download `saq_checkpoints.zip`;
2. unzip it at the repository root — the archive already contains the folders
   `checkpoints/mvtec_ad/SAQ/` and `checkpoints/visa/SAQ/`;
3. after unzipping the layout is

```
checkpoints/
├── mvtec_ad/SAQ/SAQ_with_iteration.pth      # full SAQ, 385 MB
├── mvtec_ad/SAQ/SAQ_without_iteration.pth   # fixed-query ablation, 128 MB
└── visa/SAQ/SAQ_with_iteration.pth          # full SAQ, 385 MB
```

The `without_iteration` checkpoints reproduce the layer-iteration ablation of the paper (released for MVTec AD); the `with_iteration` ones are the full SAQ used for all reported results.

---

## 5. Anomaly-manifold file (HDF5)

`manifold/anomaly_manifold.h5` stores the GCT prototypes: for each selected layer, the normalized reconstruction-gradient directions of normal training images are clustered with k-means (K = 16), and the cluster centres are stored.

**Layout**

```
anomaly_manifold.h5
├── layer_12/           # group per layer (l ∈ {12, 18, 24})
│   ├── prototypes      # (16, 1024) float32, unit-norm rows  mu_k^(l)
│   ├── kmeans_centers  # alias of `prototypes`, kept for readability
│   └── meta            # attrs: K=16, dim=1024, n_samples, seed
├── layer_18/
└── layer_24/
```

Reading:

```python
import h5py, numpy as np

with h5py.File("manifold/anomaly_manifold.h5", "r") as f:
    for l in (12, 18, 24):
        mu = f[f"layer_{l}/prototypes"][()]           # (16, 1024)
        mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
        print(f"layer {l}: {mu.shape}, norm={np.linalg.norm(mu, axis=1).mean():.4f}")
```

Notes

- Rows are unit-normalized so that they can be used directly in the SLERP update of Eq. (17).
- Layer 6 has no prototypes: GCT is applied to layers 12, 18 and 24 only, as in the paper.
- The file is produced by `scripts/build_manifold.py`; re-running it with another seed gives a set of prototypes that can be used to check the stability of the GCT prior.

---

## 6. Quick start

Single image:

```bash
python scripts/infer_single.py \
    --image path/to/skin.jpg \
    --clip ViT-L/14-336px \
    --mla checkpoints/mla.pth \
    --saq checkpoints/saq.pth \
    --gct checkpoints/gct_ae.pth \
    --manifold manifold/anomaly_manifold.h5 \
    --image-size 336 \
    --out outputs/heatmap.png
```

Minimal Python sketch of what the script does:

```python
# 1. frozen CLIP, multi-scale patch features
P = clip.visual.get_intermediate_layers(img, n=[6, 12, 18, 24])       # P^(l) in R^{N x d_v}

# 2. MLA: text anchors -> visual space (Eq. 4-6)
H_base = sum(alpha[l] * sim(A_a[l], P[l]) - sim(A_n[l], P[l]))        # Eq. 6-7

# 3. SAQ: queries evolved layer by layer (Eq. 8-10)
A_q = saq(P)                                                          # A^(l), 4 x d_v per layer
H_saq = sum(alpha[l] * sim(A_q[l], P[l]))                             # Eq. 11

# 4. GCT: rotate the normal anchor, vote, fuse (Eq. 17-23)
delta = gct.deviation(P, mu, saq_anchor=A_q)                          # per-patch delta, Eq. 19-22
H_hat = clip(H_saq + gamma * delta, 0, 1)                             # Eq. 23

# 5. final heatmap and image-level score
H = 0.5 * (H_base + H_hat)                                            # Eq. 24
score = topk_mean(H, k=...)
```

The numbers `alpha[l]`, `gamma`, `K` and the SLERP step `lambda` are read from `configs/aero_adclip.yaml`.

---

## 7. Data

- **Public industrial benchmarks** (MVTec AD, VisA, BTAD, MPDD, DAGM, KolektorSDD, KolektorSDD2) are used for training and for the generalisation table. They are *not* redistributed here; use the official links.
- **Aircraft-skin benchmark.** 1,476 test images with 3,246 bounding boxes over six damage categories (crack, dent, corrosion, scratch, missing rivet head, paint peel-off). The pixel-level masks were produced by seeding SAM inside each box and refining them by hand.
  <!-- Choose ONE and delete the others: -->
  <!-- (a) released as a GitHub Release asset: see Releases -> aircraft_skin_benchmark.zip -->
  <!-- (b) available from the corresponding author upon reasonable request -->
  <!-- (c) released at <DOI>, see the paper -->
- During every evaluation reported in the paper the aircraft data was used **for testing only**.

---

## 8. Results (as reported in the paper)

Zero-shot, CLIP frozen, trained on public industrial data only.

| Setting | Metric | Result |
|---|---|---|
| 7 industrial benchmarks (mean over datasets) | image AUROC | 92.2 |
| 7 industrial benchmarks (mean over datasets) | pixel AUROC | 96.6 |
| aircraft-skin benchmark (mean over 6 classes) | pixel AUROC / PRO / F1-max | 89.8 / 76.8 / 43.7 |

Latency: 28.5 ms per frame on a single RTX 5070 at 336 × 336 inference resolution (FP16 with kernel fusion).

---

## 9. Citation

```bibtex
@article{song2026aeroadclip,
  title   = {{AeroAD-CLIP}: A Semantics--Geometry Synergized Framework for Zero-Shot
             Aircraft Skin Defect Detection},
  author  = {Song, Xinze and Liu, Hu and Ma, Yunpeng and Zheng, Zewei},
  journal = {IEEE Transactions on Aerospace and Electronic Systems},
  year    = {2026},
  note    = {under review}
}
```

---

## 10. License and contact

- Code and checkpoints: **MIT License** (see `LICENSE`).
- The aircraft-skin benchmark and the CLIP backbone keep their own terms; please cite the original sources when you use them.
- CLIP is released by OpenAI under the MIT License; the frozen backbone is not redistributed here.

Questions: **Yunpeng Ma** — `myp@buaa.edu.cn`
Xinze Song — School of Aeronautic Science and Engineering, Beihang University.

---

## 11. Notes

- The prototypes in `manifold/anomaly_manifold.h5` are derived **only** from normal source-domain images; they encode the directions in which the normal manifold is most easily breached, not a distribution of anomalous samples.
- SLERP keeps the anchor on the unit sphere; `lambda` controls the geodesic step and is set in the config file.
- Reproducing the exact numbers of the paper requires the same source-domain split and the same inference resolution; both are stated in the config files.
