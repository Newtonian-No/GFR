
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation / inference script for the VisionTransformer segmentation model.
This variant scans an input directory and processes all .dcm files whose
filenames start with a letter. There are no ground-truth labels.

For each DICOM file it:
 - reads the pixel data (using pydicom),
 - preprocesses to model input size,
 - runs the model to obtain a segmentation prediction,
 - saves the predicted mask (optional),
 - writes a line to an output txt (tab-separated) with:
     patient_name<TAB>main_predicted_class<TAB>unique_predicted_classes(comma separated)

Usage example:
  python back/validate.py --input-dir /path/to/dcm_folder \
                         --checkpoint /path/to/checkpoint.pth \
                         --model ViT-B_16 \
                         --img-size 224 \
                         --batch-size 4 \
                         --device cuda \
                         --save-preds ./preds \
                         --output-txt results.txt

Notes:
 - Requires pydicom (pip install pydicom).
 - If the DICOM has PatientName metadata it will be used as the patient name,
   otherwise the filename (stem) is used.
 - The script assumes the model's segmentation head outputs logits of shape
   (B, C, H, W). The script will take argmax across channel dim.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

# DICOM reader
try:
    import pydicom
except ImportError:
    pydicom = None

# Ensure repo root in path (so imports like back.networks... work when running from repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import model module from your repo
from back.networks.vit_seg_modeling_bimambaattention import VisionTransformer, CONFIGS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=str, required=True, help='Directory containing DICOM (.dcm) files.')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    p.add_argument('--model', type=str, default='ViT-B_16', choices=list(CONFIGS.keys()), help='CONFIGS key')
    p.add_argument('--img-size', type=int, default=224, help='Resize image to this square size (model input).')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    p.add_argument('--save-preds', type=str, default=None, help='If set, save predicted masks to this folder')
    p.add_argument('--output-txt', type=str, default='predictions.txt', help='Output txt path (tab-separated)')
    p.add_argument('--no-resize', action='store_true', help='Do not resize inputs (use original DICOM size)')
    return p.parse_args()


def is_target_dcm(p) -> bool:
    """
    Return True if the given path (Path or str) points to a file whose name
    starts with a letter and whose extension is .dcm (case-insensitive).

    This function is defensive: it accepts both pathlib.Path and str to avoid
    AttributeError when callers pass strings.
    """
    # Normalize to Path
    if not isinstance(p, Path):
        try:
            p = Path(p)
        except Exception:
            return False

    name = p.name
    if not name:
        return False
    # first character of the filename (not including directories)
    first_char = name[0]
    # Ensure suffix check is robust
    suffix = p.suffix or ''
    return first_char.isalpha() and suffix.lower() == '.dcm'



class DicomDataset(Dataset):
    def __init__(self, files: List[Path], img_size: int = 224, no_resize: bool = False):
        if pydicom is None:
            raise ImportError("pydicom is required to read DICOM files. Install with `pip install pydicom`.")
        self.files = files
        self.img_size = img_size
        self.no_resize = no_resize
        # transforms: to tensor and normalize (ImageNet-like). Adjust if your model used different normalization.
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def read_dicom_image(self, path: Path) -> Tuple[np.ndarray, dict]:
        ds = pydicom.dcmread(str(path))
        # Pixel array
        arr = ds.pixel_array.astype(np.float32)
        meta = {}
        # Try to get a patient name if available
        try:
            patient = ds.get('PatientName', None)
            if patient is not None:
                meta['patient_name'] = str(patient)
            else:
                meta['patient_name'] = path.stem
        except Exception:
            meta['patient_name'] = path.stem

        # Some DICOMs may have multiplicative rescale intercept/slope
        try:
            intercept = float(ds.get('RescaleIntercept', 0.0))
            slope = float(ds.get('RescaleSlope', 1.0))
            arr = arr * slope + intercept
        except Exception:
            pass

        # Normalize pixel values to 0-255 then to uint8 for PIL
        # handle constant images
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr_scaled = (arr - mn) / (mx - mn) * 255.0
        else:
            arr_scaled = np.clip(arr, 0, 255)
        arr_uint8 = np.clip(arr_scaled, 0, 255).astype(np.uint8)
        return arr_uint8, meta

    def __getitem__(self, idx):
        path = self.files[idx]
        arr_uint8, meta = self.read_dicom_image(path)
        # convert to PIL Image (grayscale). If model expects 3 channels, later we will repeat channels.
        img = Image.fromarray(arr_uint8).convert('L')  # single-channel
        if not self.no_resize:
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        # Convert to 3-channel tensor as model expects RGB (if single-channel, repeat)
        img_t = self.to_tensor(img)  # (1,H,W)
        img_t = img_t.repeat(3, 1, 1)  # (3,H,W)
        img_t = self.normalize(img_t)
        return img_t, meta, path.name  # tensor, metadata dict, filename


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    names = [b[2] for b in batch]
    return imgs, metas, names


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, device: str = 'cuda'):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        # support common keys
        if 'state_dict' in sd:
            state_dict = sd['state_dict']
        elif 'model' in sd and isinstance(sd['model'], dict):
            state_dict = sd['model']
        else:
            # assume sd is the state_dict itself
            state_dict = sd
    else:
        state_dict = sd

    # remove "module." if present
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("Missing keys when loading checkpoint:", missing)
    if unexpected:
        print("Unexpected keys when loading checkpoint:", unexpected)


def save_prediction_mask(pred_np: np.ndarray, save_dir: str, name: str):
    os.makedirs(save_dir, exist_ok=True)
    # pred_np assumed HxW integer class indexes. Convert to uint8 for saving (clamp if >255).
    arr = np.array(pred_np, dtype=np.int64)
    arr_clamped = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr_clamped)
    im.save(os.path.join(save_dir, f"{Path(name).stem}_pred.png"))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    inp = Path(args.input_dir)
    if not inp.exists() or not inp.is_dir():
        raise RuntimeError(f"Input directory not found: {inp}")

    # collect .dcm files starting with a letter
    files = sorted([p for p in inp.iterdir() if p.is_file() and is_target_dcm(p)])
    if len(files) == 0:
        raise RuntimeError(f"No matching DICOM (.dcm) files whose names start with a letter were found in {inp}")

    print(f"Found {len(files)} DICOM files to process.")

    cfg = CONFIGS[args.model]
    # num_classes is used only to interpret mask channels if needed; for this script we'll infer from logits.
    model = VisionTransformer(cfg, img_size=args.img_size, num_classes=1000, vis=False)
    model.to(device)

    print("Loading checkpoint:", args.checkpoint)
    load_checkpoint_to_model(model, args.checkpoint, device=str(device))
    model.eval()

    dataset = DicomDataset(files, img_size=args.img_size, no_resize=args.no_resize)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # open output txt and write header
    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_txt, 'w', encoding='utf-8')
    f_out.write("patient_name\tfile_name\tmain_pred_class\tunique_pred_classes\n")

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    with torch.no_grad():
        for imgs, metas, names in tqdm(dataloader, desc='Inference', unit='batch'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            # model.forward typically returns logits (B,C,H,W)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            # logits may already be (B, H, W) in degenerate case; handle common case (B,C,H,W)
            if logits.dim() == 3:
                # (B, H, W) -> treat as single-channel predictions
                preds = logits.cpu().numpy().astype(np.int64)
            else:
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()  # (B,H,W)

            B = preds.shape[0]
            for b in range(B):
                pred_mask = preds[b]  # H x W
                # flatten and compute main (mode) predicted class and unique set
                flat = pred_mask.ravel().astype(np.int64)
                # handle empty
                if flat.size == 0:
                    main_class = -1
                    unique_classes = []
                else:
                    vals, counts = np.unique(flat, return_counts=True)
                    # choose the class with max count as main predicted class
                    idx = np.argmax(counts)
                    main_class = int(vals[idx])
                    unique_classes = [int(v) for v in vals]

                patient_name = metas[b].get('patient_name', names[b])
                # write line: patient_name\tfile_name\tmain_class\tcomma_separated_unique_classes
                uniq_str = ",".join(str(x) for x in unique_classes)
                f_out.write(f"{patient_name}\t{names[b]}\t{main_class}\t{uniq_str}\n")

                if args.save_preds:
                    save_prediction_mask(pred_mask, args.save_preds, names[b])

    f_out.close()
    print(f"Finished. Results written to {out_txt}")
    if args.save_preds:
        print(f"Predicted masks saved to {args.save_preds}")


if __name__ == '__main__':
    main()
