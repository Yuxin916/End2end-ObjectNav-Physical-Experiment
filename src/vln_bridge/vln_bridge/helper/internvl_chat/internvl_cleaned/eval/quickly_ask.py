import argparse
import os
from typing import List

import torch
from PIL import Image, ImageDraw

from internvl_cleaned.dataset.dataset import build_transform, dynamic_preprocess
from internvl_cleaned.model import load_pixel_model_and_tokenizer


# offline
# QUESTION_PROMPT = 'Imagine you are an autonomous robot in an indoor environment for instruction following task. You should follow the instruction and then predict a navigable goal pixel ratio in the image. \nInstruction: Walk through the game room past the table tennis game. Walk into the next room through the arches and past the desks towards the couches in the corner of the room. \n\nInput: - Current Step egocentric RGB image <image> , where the left/right gray padding indicates left/right turns, and the bottom padding indicates stop. \n- History Images: Previous 5 egocentric RGB keyframes from the past trajectory, with blue trajectory showing movement trajectory: <image>(latest), <image>, <image>, <image>, <image>(oldest)\n\nOutput: \n<GOAL_PIXEL_X>, <GOAL_PIXEL_Y> a\n'
# IMAGE_LIST: List[str] = [
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/1.png",
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/2.png",
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/3.png",
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/4.png",
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/5.png",
#     "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/6.png",
# ]
# SAVE_PATH = "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors"


# online
QUESTION_PROMPT = 'Imagine you are an autonomous robot in an indoor environment for instruction following task. You should follow the instruction and then predict a navigable goal pixel ratio in the image. \nInstruction: Walk through the game room past the table tennis game. Walk into the next room through the arches and past the desks towards the couches in the corner of the room. \n\nInput: - Current Step egocentric RGB image <image> , where the left/right gray padding indicates left/right turns, and the bottom padding indicates stop. \n- History Images: Previous 5 egocentric RGB keyframes from the past trajectory, with blue trajectory showing movement trajectory: <image>(latest), <image>, <image>, <image>, <image>(oldest)\n\nOutput: \n<GOAL_PIXEL_X>, <GOAL_PIXEL_Y>\n'
IMAGE_LIST: List[str] = [
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0000.jpg",
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0001.jpg",
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0002.jpg",
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0003.jpg",
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0004.jpg",
    "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/0005.jpg",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/1.png",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/2.png",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/3.png",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/4.png",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/5.png",
    # "/home/all/muyi/CL_CoTNav/all_log/my_experiments/pixel/offline_eval/vis_pixel_tensors/6.png",
]
SAVE_PATH = "/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/pixel/train_eval50_visualization/temp/"

def _prepare_image(path: str, args, image_size: int, use_thumbnail: bool, transform) -> torch.Tensor:
    """Load one image from disk and convert it to the model's tensor format."""
    img = Image.open(path).convert("RGB")
    if args.dynamic:
        tiles = dynamic_preprocess(
            img,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
        )
        if isinstance(tiles, list) and len(tiles) > 0:
            img = tiles[0]
        else:
            img = tiles
    return transform(img)


def run_single_query(
    args,
    model,
    tokenizer,
    question: str,
    image_paths: List[str],
    image_size: int,
    use_thumbnail: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    transform = build_transform(
        is_train=False,
        input_size=image_size,
        pad2square=args.pad2square,
        normalize_type=args.normalize_type,
    )

    tensors: List[torch.Tensor] = []
    resolved_paths: List[str] = []
    for rel_path in image_paths:
        full_path = rel_path if os.path.isabs(rel_path) or args.root == "" else os.path.join(args.root, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        tensor = _prepare_image(full_path, args, image_size, use_thumbnail, transform)
        tensors.append(tensor)
        resolved_paths.append(full_path)

    pixel_values = torch.stack(tensors, dim=0).to(device=device, dtype=tensor_dtype)
    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
    )

    response_text, pixel_pred, _ = model.chat_with_pixel(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        verbose=False,
    )

    print("\n[Interactive] Images used:")
    for idx, rp in enumerate(resolved_paths):
        print(f"  [{idx}] {rp}")
    print(f"[Interactive] Model response: {response_text}")

    if pixel_pred is not None and resolved_paths:
        pred_vec = pixel_pred.detach().float().view(-1)
        if pred_vec.numel() >= 2:
            px, py = float(pred_vec[0]), float(pred_vec[1])
            h, w = pixel_values.shape[-2], pixel_values.shape[-1]
            abs_x = px * (w - 1)
            abs_y = py * (h - 1)
            print(f"[Interactive] Pixel head (normalized): ({px:.4f}, {py:.4f})")
            print(f"[Interactive] Pixel head (pixels): ({abs_x:.1f}, {abs_y:.1f}) on {w}x{h}")

            base_img = Image.open(resolved_paths[0]).convert("RGB")
            width, height = base_img.size
            abs_x_img = px * (width - 1)
            abs_y_img = py * (height - 1)
            radius = max(3, int(min(width, height) * 0.01))
            draw = ImageDraw.Draw(base_img)
            draw.ellipse(
                (
                    abs_x_img - radius,
                    abs_y_img - radius,
                    abs_x_img + radius,
                    abs_y_img + radius,
                ),
                outline="red",
                width=max(2, radius // 2),
            )

            save_target = SAVE_PATH
            # Allow users to provide a directory or a path without extension.
            if save_target.endswith(os.sep) or (os.path.isdir(save_target) and os.path.exists(save_target)):
                os.makedirs(save_target, exist_ok=True)
                save_target = os.path.join(save_target, "prediction_circle.png")
            else:
                save_dir = os.path.dirname(save_target)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                root, ext = os.path.splitext(save_target)
                if ext == "":
                    save_target = root + ".png"

            base_img.save(save_target)
            print(f"[Interactive] Saved annotated prediction to {save_target}")
    else:
        print("[Interactive] Pixel head did not emit coordinates.")


def main():
    parser = argparse.ArgumentParser(description="Manually chat with InternVL pixel head")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--root", type=str, default="", help="Base path for relative image inputs")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic tiling (matches training)")
    parser.add_argument("--max-num", type=int, default=6, help="Max dynamic tiles when --dynamic is set")
    parser.add_argument("--pad2square", action="store_true", help="Pad images to square before resize")
    parser.set_defaults(pad2square=True)
    parser.add_argument(
        "--normalize-type",
        type=str,
        default="imagenet",
        choices=["imagenet", "clip", "siglip"],
        help="Normalization stats to match training",
    )
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Let transformers shard layers across GPUs")

    args = parser.parse_args()

    model, tokenizer = load_pixel_model_and_tokenizer(args)
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
        print(f"[Interactive] total_params: {total_params:.2f}B -> forcing num_beams=1")
    else:
        print(f"[Interactive] total_params: {total_params:.2f}B")
    print(f"[Interactive] image_size={image_size} dynamic={args.dynamic} use_thumbnail={use_thumbnail}")

    question = QUESTION_PROMPT.strip()
    if not question:
        raise ValueError("QUESTION_PROMPT is empty; please provide a prompt.")
    if not IMAGE_LIST:
        raise ValueError("IMAGE_LIST is empty; provide at least one image path.")

    run_single_query(args, model, tokenizer, question, IMAGE_LIST, image_size, use_thumbnail)


if __name__ == "__main__":
    main()
