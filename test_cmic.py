import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cmic_model.cmic_vAuxT import CMIC_AuxT

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found in: {image_dir}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.to_tensor(img)
        return img, str(path)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CMIC checkpoint(s) with forward and actual compress."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint file paths",
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--pad_multiple", type=int, default=64, help="Pad H/W to multiples of this value")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_padding_size(height, width, p=128):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - width
    padding_top = 0
    padding_bottom = new_h - height
    return padding_left, padding_right, padding_top, padding_bottom


def pad_image(x, p=128):
    _, _, h, w = x.shape
    padding_l, padding_r, padding_t, padding_b = get_padding_size(h, w, p)
    x_padded = F.pad(
        x,
        (padding_l, padding_r, padding_t, padding_b),
        mode="replicate",
    )
    return x_padded, (padding_l, padding_r, padding_t, padding_b)


def crop_image(x, padding):
    padding_l, padding_r, padding_t, padding_b = padding
    return F.pad(x, (-padding_l, -padding_r, -padding_t, -padding_b))


def psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def estimate_bpp_from_likelihoods(likelihoods_dict, num_pixels):
    total_bits = 0.0
    for likelihoods in likelihoods_dict.values():
        total_bits += torch.log(likelihoods).sum().item() / (-math.log(2))
    return total_bits / num_pixels


def count_bytes_in_strings(strings):
    total_bytes = 0
    for group in strings:
        if isinstance(group, (list, tuple)):
            for s in group:
                total_bytes += len(s)
        else:
            total_bytes += len(group)
    return total_bytes


def load_state_dict_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict


@torch.no_grad()
def evaluate_forward(model, dataloader, device, pad_multiple):
    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()

    for x, path in tqdm(dataloader, desc="Forward", leave=False):
        x = x.to(device)

        x_padded, padding = pad_image(x, p=pad_multiple)
        out = model(x_padded)
        x_hat = crop_image(out["x_hat"], padding).clamp(0.0, 1.0)

        _, _, h, w = x.shape
        num_pixels = h * w

        cur_psnr = psnr(x, x_hat)
        cur_bpp = estimate_bpp_from_likelihoods(out["likelihoods"], num_pixels)

        avg_psnr.update(cur_psnr)
        avg_bpp.update(cur_bpp)

    return avg_bpp.avg, avg_psnr.avg


@torch.no_grad()
def evaluate_compress(model, dataloader, device, pad_multiple):
    avg_psnr = AverageMeter()
    avg_bpp = AverageMeter()

    for x, path in tqdm(dataloader, desc="Compress", leave=False):
        x = x.to(device)

        x_padded, padding = pad_image(x, p=pad_multiple)
        enc = model.compress(x_padded)
        dec = model.decompress(enc["strings"], enc["shape"])
        x_hat = crop_image(dec["x_hat"], padding).clamp(0.0, 1.0)

        _, _, h, w = x.shape
        num_pixels = h * w

        cur_psnr = psnr(x, x_hat)
        total_bytes = count_bytes_in_strings(enc["strings"])
        cur_bpp = total_bytes * 8.0 / num_pixels

        avg_psnr.update(cur_psnr)
        avg_bpp.update(cur_bpp)

    return avg_bpp.avg, avg_psnr.avg


def main():
    args = parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    device = torch.device(args.device)

    dataset = ImageDataset(args.image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_list = args.checkpoint


    forward_bpp_list = []
    forward_psnr_list = []
    compress_bpp_list = []
    compress_psnr_list = []

    for ckpt_path in tqdm(checkpoint_list, desc="Checkpoints"):
        print(f"\nLoading checkpoint: {ckpt_path}")

        model = CMIC_AuxT().to(device)

        state_dict = load_state_dict_from_checkpoint(ckpt_path, device)
        model.load_state_dict(state_dict, strict=True)
        model.update()
        model.eval()

        forward_bpp, forward_psnr = evaluate_forward(
            model=model,
            dataloader=dataloader,
            device=device,
            pad_multiple=args.pad_multiple,
        )

        actual_bpp, actual_psnr = evaluate_compress(
            model=model,
            dataloader=dataloader,
            device=device,
            pad_multiple=args.pad_multiple,
        )

        forward_bpp_list.append(forward_bpp)
        forward_psnr_list.append(forward_psnr)
        compress_bpp_list.append(actual_bpp)
        compress_psnr_list.append(actual_psnr)

        print("========== Current Result ==========")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Forward  : BPP = {forward_bpp:.6f}, PSNR = {forward_psnr:.4f}")
        print(f"Compress : BPP = {actual_bpp:.6f}, PSNR = {actual_psnr:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n========== Final Results ==========")
    print("checkpoint_list =", checkpoint_list)
    print("forward_bpp_list =", forward_bpp_list)
    print("forward_psnr_list =", forward_psnr_list)
    print("compress_bpp_list =", compress_bpp_list)
    print("compress_psnr_list =", compress_psnr_list)


if __name__ == "__main__":
    main()