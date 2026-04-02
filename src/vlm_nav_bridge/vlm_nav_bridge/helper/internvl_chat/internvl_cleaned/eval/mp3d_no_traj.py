import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
import numpy as np
from internvl_cleaned.model import load_model_and_tokenizer
from internvl_cleaned.dataset.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

# ImageNet normalization stats (default)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

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
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif normalize_type == "clip":
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
    elif normalize_type == "siglip":
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

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

class MP3DDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        annotation_jsonl,  # Can be a single path (str) or list of paths
        input_size: int = 224,
        dynamic_image_size: bool = False,
        use_thumbnail: bool = False,
        max_num: int = 6,
        pad2square=True,
        normalize_type="imagenet"
    ):
        self.root = root
        self.annotation_jsonl = annotation_jsonl
        
        # Handle single path or list of paths
        if isinstance(annotation_jsonl, str):
            # Check if it's a JSON array string
            if annotation_jsonl.startswith('['):
                annotation_paths = json.loads(annotation_jsonl)
            else:
                annotation_paths = [annotation_jsonl]
        else:
            annotation_paths = annotation_jsonl
        
        # Load data from all annotation files, tracking which file each sample came from
        self.data = []
        self.sample_to_anno_dir = []  # Track the annotation directory for each sample
        for anno_path in annotation_paths:
            anno_dir = os.path.dirname(anno_path)  # e.g., ".../few_debug"
            with open(anno_path, "r") as f:
                file_data = [json.loads(line) for line in f]
                print(f"[Dataset] Loaded {len(file_data)} samples from {anno_path}")
                self.data.extend(file_data)
                self.sample_to_anno_dir.extend([anno_dir] * len(file_data))
        print(f"[Dataset] Total samples loaded: {len(self.data)}")

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
        """Load and process one image. Returns (list of tensors, num_patches)."""
        img_path = os.path.join(self.root, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.dynamic_image_size:
            # dynamic_preprocess returns a list of PIL images (tiles)
            tiles = dynamic_preprocess(
                img, image_size=self.input_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num
            )
            # Transform each tile and return all of them
            pixel_tensors = [self.transform(tile) for tile in tiles]
            return pixel_tensors, len(tiles)
        else:
            return [self.transform(img)], 1

    def __getitem__(self, idx):
        item = self.data[idx]
        # Parse id: format is "{scene_id}-{ep_id}-{object_category}-{step_idx}"
        sample_id = item['id']
        parts = sample_id.rsplit('-', 3)  # Split from right to get last two parts
        if len(parts) == 4:
            scene_id, ep_id, object_category, step_idx_str = parts
            step_idx = int(step_idx_str)
        else:
            raise ValueError(f"Invalid sample_id format: {sample_id}")

        # go find the episode info based on scene_id and ep_id
        # Use the tracked annotation directory for this sample
        anno_dir = self.sample_to_anno_dir[idx]
        ep_info_path = os.path.join(
            anno_dir,
            "annotations",
            scene_id,
            f"{ep_id}_{object_category}.json"
        )
        with open(ep_info_path, "r") as f:
            ep_info = json.load(f)
        target_found = ep_info['steps'][step_idx]['target_found']

        q = item["conversations"][0]["value"].strip()
        gt = item["conversations"][1]["value"].strip()
        rel_paths = item["image"]
        # Load all images - each may have multiple tiles
        all_tensors = []
        num_patches_list = []
        for p in rel_paths:
            tensors, num_patches = self._load_one_image(p)
            all_tensors.extend(tensors)
            num_patches_list.append(num_patches)
        # Stack all tiles to shape (total_patches, C, H, W)
        pixel_values = torch.stack(all_tensors, dim=0)
        return {
            "id": item["id"],
            "question": q,
            "gt_action": gt,
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "target_found": target_found,
        }


def collate_fn(batches, tokenizer):
    """
    merge multiple data samples into a single batch for the dataloader.

        Concatenates the images, tokens, and other required inputs.
        Uses a tokenizer to convert input texts to token IDs.

    """
    pixel_values = torch.cat([b["pixel_values"] for b in batches], dim=0)
    questions = [b["question"] for b in batches]
    gts = [b["gt_action"] for b in batches]
    ids = [b["id"] for b in batches]
    target_founds = [b["target_found"] for b in batches]
    # Flatten num_patches_list from all batches
    num_patches_list = []
    for b in batches:
        num_patches_list.extend(b["num_patches_list"])
    return pixel_values, questions, gts, ids, num_patches_list, target_founds

class InferenceSampler(torch.utils.data.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total, world, rank):
        shard = total // world; left = total % world
        sizes = [shard + int(r < left) for r in range(world)]
        begin = sum(sizes[:rank]); end = min(sum(sizes[:rank+1]), total)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



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

    # Filter to specific episode if requested
    if args.specific_episode:
        dataset.data = [d for d in dataset.data if args.specific_episode in d["id"]]
        if len(dataset.data) == 0:
            raise ValueError(f"No sample found containing: {args.specific_episode}")
        print(f"[Eval] Filtering to {len(dataset.data)} samples containing: {args.specific_episode}")

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

    local_results = []
    local_correct = 0
    local_total = 0
    # Separate counters for target_found=True and target_found=False
    local_correct_found = 0
    local_total_found = 0
    local_correct_notfound = 0
    local_total_notfound = 0
    vis_count = 0  # Counter for visualization

    for _, (pixel_values, questions, gts, ids, num_patches_list, target_founds) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Visualize first 5 samples (show all tiles) if --write-visualize is set
        if args.write_visualize and vis_count < 5:
            visualize_pixel_tensors(pixel_values, ids[0], args.out_dir, args.normalize_type, num_patches_list)
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
        pred_text = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=questions[0],
            generation_config=generation_config,
            num_patches_list=num_patches_list,  # Tell model how tiles are grouped per image
            verbose=False,
        )

        # Direct string comparison (strip whitespace for fair comparison)
        pred = pred_text.strip()
        gt = gts[0].strip()
        is_correct = (pred == gt)
        target_found = target_founds[0]

        # Detailed logging for specific episode evaluation
        if args.specific_episode:
            print("\n" + "="*80)
            print(f"[Sample] ID: {ids[0]}")
            print(f"[Sample] Target Found: {target_found}")
            print("-"*80)
            print(f"[Conversation/Question]:\n{questions[0]}")
            print("-"*80)
            print(f"[Ground Truth]: {gt}")
            print(f"[VLM Output]:   {pred}")
            print(f"[Correct]: {is_correct}")
            print("="*80 + "\n")

        if is_correct:
            local_correct += 1
        local_total += 1

        # Track by target_found status
        if target_found:
            local_total_found += 1
            if is_correct:
                local_correct_found += 1
        else:
            local_total_notfound += 1
            if is_correct:
                local_correct_notfound += 1

        local_results.append({
            "id": ids[0],
            "question": questions[0],
            "gt": gt,
            "pred": pred,
            "correct": is_correct,
            "target_found": target_found,
        })

    torch.distributed.barrier()
    world = torch.distributed.get_world_size()

    buf = [None for _ in range(world)]
    torch.distributed.all_gather_object(buf, json.dumps(local_results))
    merged = list(itertools.chain.from_iterable(json.loads(x) for x in buf))

    # Gather correct/total counts (overall and split by target_found)
    counts = [None for _ in range(world)]
    torch.distributed.all_gather_object(counts, (
        local_correct, local_total,
        local_correct_found, local_total_found,
        local_correct_notfound, local_total_notfound
    ))
    total_correct = sum(c[0] for c in counts)
    total_samples = sum(c[1] for c in counts)
    total_correct_found = sum(c[2] for c in counts)
    total_samples_found = sum(c[3] for c in counts)
    total_correct_notfound = sum(c[4] for c in counts)
    total_samples_notfound = sum(c[5] for c in counts)

    if torch.distributed.get_rank() == 0:
        ts = time.strftime("%y%m%d%H%M%S", time.localtime())
        os.makedirs(args.out_dir, exist_ok=True)
        out_json = os.path.join(args.out_dir, f"cotnav_eval_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"[Eval] Saved per-sample results to {out_json}")

        # Compute accuracies
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        acc_found = total_correct_found / total_samples_found if total_samples_found > 0 else 0.0
        acc_notfound = total_correct_notfound / total_samples_notfound if total_samples_notfound > 0 else 0.0

        # Find some examples of mismatches
        incorrect_samples = [s for s in merged if not s["correct"]][:10]

        out_txt = os.path.join(args.out_dir, f"cotnav_eval_{ts}.txt")
        with open(out_txt, "w") as f:
            f.write(f"=" * 60 + "\n")
            f.write(f"OVERALL RESULTS\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Correct: {total_correct}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")

            f.write(f"=" * 60 + "\n")
            f.write(f"TARGET FOUND (target visible in view)\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Total samples: {total_samples_found}\n")
            f.write(f"Correct: {total_correct_found}\n")
            f.write(f"Accuracy: {acc_found:.4f}\n\n")

            f.write(f"=" * 60 + "\n")
            f.write(f"TARGET NOT FOUND (target not visible)\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Total samples: {total_samples_notfound}\n")
            f.write(f"Correct: {total_correct_notfound}\n")
            f.write(f"Accuracy: {acc_notfound:.4f}\n\n")

            if incorrect_samples:
                f.write("Example mismatches (first 10):\n")
                f.write("="*80 + "\n")
                for i, sample in enumerate(incorrect_samples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"  ID: {sample['id']}\n")
                    f.write(f"  Target Found: {sample.get('target_found', 'N/A')}\n")
                    f.write(f"  Ground Truth: {sample['gt']}\n")
                    f.write(f"  Prediction:   {sample['pred']}\n")
                    f.write("-"*80 + "\n")

        print(f"[Eval] Summary written to {out_txt}")

        # Print metrics to console
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nOVERALL:")
        print(f"  Total samples: {total_samples}")
        print(f"  Correct: {total_correct}")
        print(f"  Accuracy: {acc:.4f}")

        print(f"\nTARGET FOUND (target visible):")
        print(f"  Total samples: {total_samples_found}")
        print(f"  Correct: {total_correct_found}")
        print(f"  Accuracy: {acc_found:.4f}")

        print(f"\nTARGET NOT FOUND (target not visible):")
        print(f"  Total samples: {total_samples_notfound}")
        print(f"  Correct: {total_correct_notfound}")
        print(f"  Accuracy: {acc_notfound:.4f}")

        if incorrect_samples:
            print(f"\nShowing {len(incorrect_samples)} example mismatches:")
            for i, sample in enumerate(incorrect_samples, 1):
                print(f"\n  [{i}] ID: {sample['id']}")
                print(f"      Target Found: {sample.get('target_found', 'N/A')}")
                print(f"      GT:   '{sample['gt']}'")
                print(f"      Pred: '{sample['pred']}'")

        print("="*60 + "\n")

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
    parser.add_argument("--write-visualize", action="store_true",
                        help="Write visualization images for first 5 samples.")
    parser.add_argument("--specific-episode", type=str, default=None,
                        help="Evaluate only the sample with this specific ID.")

    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

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
