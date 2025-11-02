
# Controllable CLIP-Guided Flowers (Oxford-102)

Tiny, compute-friendly pipeline for **GAN-free** text→image generation and **text-guided edits** using **CLIP** on the Oxford-102 Flowers dataset. Includes prompt building, zero-shot sanity checks, simple generation via optimization, and an attribute editor baseline.

## Features

* **Prompt CSV builder:** base + attribute prompts → `{image_path, class_name, prompt}`
* **Zero-shot CLIP check:** quick baseline over Oxford-102 class names
* **Generation (no GAN):** optimize pixels (or a tiny decoder) to match text in CLIP space
* **Text-guided editing:** differentiable HSV+conv editor for color/texture tweaks
* **Eval hooks:** CLIPScore, R-precision, (FID/KID optional)

## Setup

```bash
git clone <your-repo> && cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch kornia lpips einops matplotlib pandas tqdm
```

## Quickstart

1. **Build prompts + CSV**

   ```bash
   python scripts/build_prompts.py
   ```

   Outputs:

   * `prepared/flowers102/{train,val,test}/...jpg`
   * `prompts_map.csv` with `image_path,class_name,prompt`

   *(Optional)* Add real class names by creating `class_names.json` with 102 names (list) in the project root.

2. **Zero-shot sanity** (small subset check)

   ```bash
   python scripts/zero_shot_eval.py --csv prompts_map.csv --samples 1000 --out runs/zero_shot.csv
   ```

3. **Text→image (pixel optimization)**

   ```bash
   python scripts/pixel_opt_generate.py \
     --prompt "a photo of a red tulip flower" --steps 500 --out figures/pixel_opt_grid.png
   ```

4. **Text-guided editing**

   ```bash
   python scripts/attribute_edit.py \
     --image prepared/flowers102/val/val_00010.jpg \
     --prompt "a photo of a tulip flower with a white margin" \
     --out figures/edit_before_after.png
   ```

5. **Metrics (early)**

   ```bash
   python scripts/metrics_eval.py --csv prompts_map.csv --gen_dir figures/ --out tables/metrics.csv
   ```

## Project Structure

```
.
├── scripts/
│   ├── build_prompts.py         # creates prepared/… and prompts_map.csv
│   ├── zero_shot_eval.py        # zero-shot CLIP baseline
│   ├── pixel_opt_generate.py    # CLIP-guided pixel optimization
│   ├── attribute_edit.py        # text-guided HSV+conv editor
│   └── metrics_eval.py          # CLIPScore, R-precision; FID/KID hooks
├── prepared/flowers102/...      # saved images per split (auto)
├── runs/                        # logs, intermediate results
├── figures/                     # sample grids, before/after panels
├── tables/                      # metrics CSVs
├── prompts_map.csv              # image_path, class_name, prompt
└── class_names.json             # (optional) 102 real class names (list)
```

## Tips

* Always **L2-normalize** CLIP features before cosine similarity.
* Use **mixed-size cutouts** (with moderate color jitter) for stability.
* Start at **64×64**; upscale later if needed.
* Keep early grids and logs—these are perfect for the mid-project report.

## References

* **CLIP:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (ICML 2021).


