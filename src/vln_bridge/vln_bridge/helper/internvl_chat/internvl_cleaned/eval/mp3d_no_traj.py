import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial
import numpy as np
import torch
from internvl_cleaned.model import load_model_and_tokenizer, load_pixel_model_and_tokenizer
from internvl_cleaned.dataset.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
import sys



import math

IMG_H, IMG_W = 256+16, 256+16  # Input image size after padding to square

PIXEL_EVAL_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5]
pixel_eval_stats = {
    "total": 0,
    "valid": 0,
    "hits": {thr: 0 for thr in PIXEL_EVAL_THRESHOLDS},
}

TEXT_ACTION_CHOICES = ["SELECT_PIXEL", "TURN_LEFT", "TURN_RIGHT", "STOP"]
TEXT_PRED_LABELS = TEXT_ACTION_CHOICES + ["UNKNOWN"]
TEXT_ACTION_TO_IDX = {label: idx for idx, label in enumerate(TEXT_ACTION_CHOICES)}
TEXT_PRED_TO_IDX = {label: idx for idx, label in enumerate(TEXT_PRED_LABELS)}
text_eval_stats = {
    "total": 0,
    "valid": 0,
    "correct": 0,
    "confusion": [[0 for _ in TEXT_PRED_LABELS] for _ in TEXT_ACTION_CHOICES],
}

text_pixel_distance_stats = {
    "total": 0,
    "valid": 0,
    "sum_distance": 0.0,
}


import re
from typing import Tuple


def visualize_pixel_tensors(pixel_values, sample_id, out_dir, normalize_type="imagenet", num_patches_list=None):
    """
    Visualize pixel tensors by denormalizing and saving as images.
    pixel_values: tensor of shape (N, C, H, W) where N is total number of tiles
    num_patches_list: list of how many tiles per original image (e.g., [1, 6, 6] for BEV=1, RGB=6, SEG=6)
    """

    vis_dir = os.path.join(out_dir, "vis_pixel_tensors")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get normalization stats
    if normalize_type == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif normalize_type == "clip":
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
    elif normalize_type == "siglip":
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy and denormalize
    pixel_np = pixel_values.cpu().float().numpy()  # (N, C, H, W)
    
    image_names = ["BEV", "RGB", "SEG"]  # Assuming 3 images
    
    if num_patches_list is None:
        num_patches_list = [1] * pixel_np.shape[0]
    
    tile_idx = 0
    for img_idx, num_patches in enumerate(num_patches_list):
        name = image_names[img_idx] if img_idx < len(image_names) else f"img_{img_idx}"
        for patch_idx in range(num_patches):
            img = pixel_np[tile_idx]  # (C, H, W)
            img = img.transpose(1, 2, 0)  # (H, W, C)
            
            # Denormalize
            img = img * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Save image with tile index if multiple tiles
            if num_patches > 1:
                save_path = os.path.join(vis_dir, f"{sample_id}_{name}_tile{patch_idx}.png")
            else:
                save_path = os.path.join(vis_dir, f"{sample_id}_{name}.png")
            Image.fromarray(img).save(save_path)
            tile_idx += 1
    
    print(f"[Vis] Saved {pixel_np.shape[0]} tiles to {vis_dir} for sample {sample_id}")
    print(f"      num_patches_list: {num_patches_list}")

def parse_coords(obj):
    """Parse normalized (x, y) from various GT formats (list of strings, CSV text, or JSON list)."""
    # If already a list/tuple of numbers or numeric strings
    if isinstance(obj, (list, tuple)):
        vals = []
        for x in obj:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                continue
        if len(vals) >= 2:
            return vals[0], vals[1]
        return None

    if isinstance(obj, str):
        s = obj.strip()
        # Try JSON-style list
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                return parse_coords(parsed)
            except Exception:
                pass
        # Fallback: extract first two floats (supports scientific notation)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(nums) >= 2:
            try:
                return float(nums[0]), float(nums[1])
            except ValueError:
                return None
        return None

    return None


def extract_text_value(value):
    """Return the first non-empty string-like value from nested containers."""
    if isinstance(value, (list, tuple)):
        for candidate in value:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_pixel_coordinates(text_value, width, height):
    coords = parse_coords(text_value) if text_value is not None else None
    if coords is None or width <= 0 or height <= 0:
        return None
    x, y = coords
    norm_x = max(0.0, min(1.0, float(x) / float(width)))
    norm_y = max(0.0, min(1.0, float(y) / float(height)))
    return norm_x, norm_y


def update_text_pixel_distance(pred_text, gt_text, width, height):
    text_pixel_distance_stats["total"] += 1
    pred_norm = normalize_pixel_coordinates(pred_text, width, height)
    gt_norm = normalize_pixel_coordinates(gt_text, width, height)
    if pred_norm is None or gt_norm is None:
        return None
    dx = pred_norm[0] - gt_norm[0]
    dy = pred_norm[1] - gt_norm[1]
    dist = math.sqrt(dx * dx + dy * dy)
    text_pixel_distance_stats["valid"] += 1
    text_pixel_distance_stats["sum_distance"] += dist
    return dist


def summarize_text_pixel_distance():
    stats_vector = [
        float(text_pixel_distance_stats["total"]),
        float(text_pixel_distance_stats["valid"]),
        float(text_pixel_distance_stats["sum_distance"]),
    ]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(stats_vector, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        stats_vector = tensor.cpu().tolist()

    total = int(stats_vector[0])
    valid = int(stats_vector[1])
    sum_distance = float(stats_vector[2])
    avg_distance = (sum_distance / valid) if valid > 0 else 0.0
    return {
        "total": total,
        "valid": valid,
        "sum_distance": sum_distance,
        "avg_distance": avg_distance,
    }

class MP3DDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        annotation_jsonl: str,
        input_size: int = 224,
        dynamic_image_size: bool = False,
        use_thumbnail: bool = False,
        max_num: int = 6,
        pad2square=True,
        normalize_type="imagenet"
    ):
        self.root = root
        self.annotation_jsonl = annotation_jsonl
        with open(annotation_jsonl, "r") as f:
            self.data = [json.loads(line) for line in f]

        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num

        self.pad2square = pad2square
        self.normalize_type = normalize_type
        self.transform = build_transform(
            is_train=False,
            input_size=input_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )

    def __len__(self):
        return len(self.data)

    def _load_one_image(self, rel_path: str):
        img_path = os.path.join(self.root, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.dynamic_image_size:
            # dynamic_preprocess returns a list of PIL images (tiles/aug views)
            # For evaluation here we keep a single view: take first tile
            tiles = dynamic_preprocess(
                img, image_size=self.input_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num
            )
            img = tiles[0] if isinstance(tiles, list) and len(tiles) > 0 else img
        return self.transform(img)
        # return np.array(img)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item["conversations"][0]["value"].strip()
        gt = item["conversations"][1]["value"].strip()
        pixel_label = None
        for i in range(len(item["conversations"])):
            if item["conversations"][i]["from"] == "pixel_label":
                pixel_label = item["conversations"][i]["value"]
                break
        rel_paths = item["image"]
        # Expect exactly 3 images [BEV, RGB, SEG]
        pixel_tensors = [self._load_one_image(p) for p in rel_paths]
        # Stack to shape (num_images, C, H, W)
        pixel_values = torch.stack(pixel_tensors, dim=0)
        return {
            "id": item["id"],
            "question": q,
            "gt_action": gt,
            "pixel_values": pixel_values,
            "pixel_labels": pixel_label,
        }


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0)
    questions = [b["question"] for b in batches]
    gts = [b["gt_action"] for b in batches]
    ids = [b["id"] for b in batches]
    pixel_labels = [b["pixel_labels"] for b in batches]
    return pixel_values, questions, gts, ids, pixel_labels

class InferenceSampler(torch.utils.data.Sampler):
    def __init__(self, size):
        self._size = int(size); assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)
    @staticmethod
    def _get_local_indices(total, world, rank):
        shard = total // world; left = total % world
        sizes = [shard + int(r < left) for r in range(world)]
        begin = sum(sizes[:rank]); end = min(sum(sizes[:rank+1]), total)
        return range(begin, end)
    def __iter__(self): yield from self._local_indices
    def __len__(self): return len(self._local_indices)


# # evaluate MLP head pixel prediction
# def evaluate_pixel_pred_number(pixel_pred_out, gts):
#     """Evaluate pixel prediction output against GTs."""
#     pixel_eval_stats["total"] += 1
#     if pixel_pred_out is None or gts is None:
#         return None

#     if isinstance(pixel_pred_out, torch.Tensor):
#         pred_vec = pixel_pred_out.detach().float().view(-1)
#     else:
#         pred_vec = torch.tensor(pixel_pred_out, dtype=torch.float32).view(-1)
#     if pred_vec.numel() < 2:
#         return None
#     pred_xy = torch.clamp(pred_vec[:2], 0.0, 1.0)

#     gt_tensor = torch.tensor(gts, dtype=torch.float32).view(-1)
#     if gt_tensor.numel() < 2:
#         return None
#     gt_xy = torch.clamp(gt_tensor[:2], 0.0, 1.0)

#     dist = torch.linalg.norm(pred_xy - gt_xy).item()
#     pixel_eval_stats["valid"] += 1
#     for thr in PIXEL_EVAL_THRESHOLDS:
#         if dist <= thr:
#             pixel_eval_stats["hits"][thr] += 1

#     return {
#         "pred_ratio": (float(pred_xy[0]), float(pred_xy[1])),
#         "gt_ratio": (float(gt_xy[0]), float(gt_xy[1])),
#         "distance": dist,
#     }

# def summarize_pixel_eval_stats():
#     """Aggregate pixel-prediction accuracies across all ranks."""
#     stats_vector = [
#         float(pixel_eval_stats["total"]),
#         float(pixel_eval_stats["valid"]),
#         *[float(pixel_eval_stats["hits"][thr]) for thr in PIXEL_EVAL_THRESHOLDS],
#     ]

#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
#         tensor = torch.tensor(stats_vector, dtype=torch.float64, device=device)
#         torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
#         stats_vector = tensor.cpu().tolist()

#     total = int(stats_vector[0])
#     valid = int(stats_vector[1])
#     hit_values = [int(v) for v in stats_vector[2:]]
#     accuracy = {
#         thr: (hit_values[idx] / valid) if valid > 0 else 0.0
#         for idx, thr in enumerate(PIXEL_EVAL_THRESHOLDS)
#     }
#     hits = {thr: hit_values[idx] for idx, thr in enumerate(PIXEL_EVAL_THRESHOLDS)}
#     return {
#         "total": total,
#         "valid": valid,
#         "hits": hits,
#         "accuracy": accuracy,
#     }


def extract_action_label(text):
    """Map raw text to one of the predefined action labels."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    normalized = text.upper().replace("-", "_")
    normalized = normalized.replace(" ", "_")
    for label in TEXT_ACTION_CHOICES:
        if label in normalized:
            return label
    return None


# def evaluate_text_action(pred_text, gt_text):
#     """Update running classification stats for action text outputs."""
#     text_eval_stats["total"] += 1
#     gt_label = extract_action_label(gt_text)
#     if gt_label is None:
#         return None

#     text_eval_stats["valid"] += 1
#     pred_label = extract_action_label(pred_text)
#     if pred_label is None:
#         pred_label = "UNKNOWN"

#     if pred_label == gt_label:
#         text_eval_stats["correct"] += 1

#     gt_idx = TEXT_ACTION_TO_IDX[gt_label]
#     pred_idx = TEXT_PRED_TO_IDX[pred_label]
#     text_eval_stats["confusion"][gt_idx][pred_idx] += 1
#     return {"gt": gt_label, "pred": pred_label, "correct": pred_label == gt_label}


def summarize_text_eval_stats():
    """Aggregate text-action classification stats across all ranks."""
    num_gt = len(TEXT_ACTION_CHOICES)
    num_pred = len(TEXT_PRED_LABELS)
    flat_confusion = []
    for row in text_eval_stats["confusion"]:
        flat_confusion.extend(row)

    stats_vector = [
        float(text_eval_stats["total"]),
        float(text_eval_stats["valid"]),
        float(text_eval_stats["correct"]),
        *map(float, flat_confusion),
    ]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(stats_vector, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        stats_vector = tensor.cpu().tolist()

    total = int(stats_vector[0])
    valid = int(stats_vector[1])
    correct = int(stats_vector[2])
    conf_values = stats_vector[3:]
    matrix = []
    idx = 0
    for _ in range(num_gt):
        row = []
        for _ in range(num_pred):
            row.append(int(conf_values[idx]))
            idx += 1
        matrix.append(row)

    accuracy = (correct / valid) if valid > 0 else 0.0
    return {
        "total": total,
        "valid": valid,
        "correct": correct,
        "accuracy": accuracy,
        "matrix": matrix,
    }




def evaluate_actions(args, model, tokenizer, image_size, use_thumbnail):
    random.seed(args.seed)

    dataset = MP3DDataset(
        root=args.root,
        annotation_jsonl=args.annotation,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        pad2square=args.pad2square,
        normalize_type=args.normalize_type,   # <= important
    )

    if args.limit > 0 and len(dataset) > args.limit:
        dataset.data = random.sample(dataset.data, args.limit)
        print(f"[Eval] Randomly selected {len(dataset)} samples for evaluation.")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    vis_count = 0   # jsut for visualization
    with torch.no_grad():
        for _, (pixel_values, questions, gts, ids, pixel_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # print(questions)
            if args.write_visualize and vis_count < 1:
                visualize_pixel_tensors(pixel_values, ids[0], args.out_dir, args.normalize_type)
                vis_count += 1

            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
            )
            assert len(questions) == 1 == len(gts) == len(ids), "Use --batch-size 1 with model.chat"
            if args.if_pixel_head:
                response_text, pixel_pred_number, pooled_sample_ids = model.chat_with_pixel(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    verbose=False,
                )
            elif args.if_pixel_head == False:
                response_text = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    verbose=False,
                )
                pixel_pred_number = None

            # # gts: ['555, 650']
            # # response_text: ['558, 672']
            # pixel_GT = pixel_labels[0]
            # pixel_eval = evaluate_pixel_pred_number(pixel_pred_number, pixel_GT)

            pred_text = extract_text_value(response_text)
            gt_text = extract_text_value(gts[0] if gts else None)
            # evaluate_text_action(pred_text, gt_text)
            update_text_pixel_distance(
                pred_text,
                gt_text,
                width=args.coord_width,
                height=args.coord_height,
            )
    
    # pixel_eval_summary = summarize_pixel_eval_stats()
    text_eval_summary = summarize_text_eval_stats()
    text_distance_summary = summarize_text_pixel_distance()
    
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("-" * 60)
        print("TEXT-ACTION METRICS")
        print("-" * 60)
        print(f"Total samples: {text_eval_summary['total']}")
        print(f"Evaluated samples: {text_eval_summary['valid']}")
        if text_eval_summary['valid'] > 0:
            print(
                f"Accuracy: {text_eval_summary['accuracy']:.4f} "
                f"({text_eval_summary['correct']}/{text_eval_summary['valid']})"
            )
        else:
            print("Accuracy: n/a (no valid samples)")

        header = ["GT\\Pred", *TEXT_PRED_LABELS]
        col_width = 14
        header_line = "".join(f"{col:>{col_width}}" for col in header)
        print(header_line)
        for gt_label, row in zip(TEXT_ACTION_CHOICES, text_eval_summary["matrix"]):
            row_str = [str(val) for val in row]
            line = f"{gt_label:>{col_width}}" + "".join(f"{val:>{col_width}}" for val in row_str)
            print(line)
        print("-" * 60 + "\n")

        print("-" * 60)
        print("TEXT-COORD L2 DISTANCE")
        print("-" * 60)
        print(f"Total samples: {text_distance_summary['total']}")
        print(f"Evaluated samples: {text_distance_summary['valid']}")
        if text_distance_summary["valid"] > 0:
            print(
                f"Avg L2 distance (normalized coords): {text_distance_summary['avg_distance']:.4f} "
                f"(sum={text_distance_summary['sum_distance']:.4f})"
            )
        else:
            print("Avg L2 distance: n/a (no valid samples)")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    # argparse (near other args)
    parser.add_argument("--pad2square", action="store_true", help="Pad to square before resizing.")
    parser.set_defaults(pad2square=True)

    parser.add_argument("--normalize-type", type=str, default="imagenet",
                        choices=["imagenet", "clip", "siglip"],
                        help="Normalization used in training (default: imagenet).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Evaluate only first N samples (for debugging)")
    parser.add_argument("--write_visualize", type=bool, default=True, help="whether visualize pixel tensor")
    parser.add_argument("--coord-width", type=float, default=1024.0,
                        help="Width used to normalize textual pixel outputs (default: 1024)")
    parser.add_argument("--coord-height", type=float, default=1024.0,
                        help="Height used to normalize textual pixel outputs (default: 1024)")
    parser.add_argument("--if-pixel-head", default=False, required=True, help="previous code inlcude pixel head, this is used to control load model")

    

    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))


    # 
    if args.if_pixel_head:
        model, tokenizer = load_pixel_model_and_tokenizer(args)
    else:
        model, tokenizer = load_model_and_tokenizer(args)


    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = getattr(model.config, "use_thumbnail", False)

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f"[eval] total_params: {total_params:.2f}B -> forcing num_beams=1")
    else:
        print(f"[eval] total_params: {total_params:.2f}B")
    print(f"[eval] image_size: {image_size}  dynamic={args.dynamic}  use_thumbnail={use_thumbnail}")
    assert args.batch_size == 1, "Only batch size 1 is supported with model.chat"

    evaluate_actions(args, model, tokenizer, image_size, use_thumbnail)
