"""
Template preparation helpers for RGB-centric VLM prompts.

This module keeps prompt construction local to vln_bridge so custom templates
such as "RGB_HisKFSingleColor" or a plain single-RGB template can be added
without depending on old external reconstruction scripts.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


PIXEL_OUTPUT_MODE = 'pixelintext'
ACTION_OUTPUT_MODE = 'action'
ACTION_PIXEL_OUTPUT_MODE = 'action_pixel'

RGB_TEMPLATE_ALIASES = {
    'SingleRGB': 'RGB',
    'Single_RGB': 'RGB',
    'SingleRGBVLM': 'RGB',
    'Single RGB': 'RGB',
    'RGB_HisKFSingle Color': 'RGB_HisKFSingleColor',
}

SUPPORTED_RGB_TEMPLATES = {
    'RGB',
    'RGB_HisKFSingleColor',
    'RGB_His1Interval8',
    'RGB_HisKFSingleColor_HisAction',
}


@dataclass(frozen=True)
class PreparedTemplate:
    question: str
    images: List[Image.Image]


def normalize_template_name(template: str) -> str:
    if template in RGB_TEMPLATE_ALIASES:
        return RGB_TEMPLATE_ALIASES[template]
    return template


def is_rgb_template(template: str) -> bool:
    return normalize_template_name(template) in SUPPORTED_RGB_TEMPLATES


def prepare_rgb_template(
    template: str,
    instruction: str,
    current_rgb: Image.Image,
    current_rgb_padded: Optional[Image.Image] = None,
    history_keyframes_single_color: Optional[Sequence[Image.Image]] = None,
    history_interval_images: Optional[Sequence[Image.Image]] = None,
    history_action_text: Optional[str] = None,
    output_mode: str = PIXEL_OUTPUT_MODE,
) -> PreparedTemplate:
    """
    Build the prompt text plus ordered image list for RGB-based templates.

    Image order in the returned list must match the order of ``<image>`` tokens
    in the prompt. The current RGB frame is always the first image.
    """
    template = normalize_template_name(template)
    if template not in SUPPORTED_RGB_TEMPLATES:
        raise ValueError(
            f'Unsupported RGB template "{template}". '
            f'Available: {sorted(SUPPORTED_RGB_TEMPLATES)}'
        )

    current = _ensure_pil_rgb(current_rgb_padded or current_rgb)
    images: List[Image.Image] = [current]
    sections = [_base_intro(instruction), 'Input:\n']

    if template == 'RGB':
        sections.append('- Current egocentric RGB image: <image>\n')

    elif template == 'RGB_HisKFSingleColor':
        history = _require_images(
            history_keyframes_single_color,
            'history_keyframes_single_color',
            template,
        )
        images.extend(history)
        sections.append('- Current egocentric RGB image: <image>\n')
        sections.append(_render_history_section(
            history,
            label_prefix='Historical keyframe with single-color trajectory overlay',
        ))

    elif template == 'RGB_His1Interval8':
        history = _require_images(
            history_interval_images,
            'history_interval_images',
            template,
        )
        images.extend(history)
        sections.append('- Current egocentric RGB image: <image>\n')
        sections.append(_render_history_section(
            history,
            label_prefix='Historical interval frame',
        ))

    elif template == 'RGB_HisKFSingleColor_HisAction':
        history = _require_images(
            history_keyframes_single_color,
            'history_keyframes_single_color',
            template,
        )
        if not history_action_text:
            raise ValueError(
                f'history_action_text must be provided for template "{template}".'
            )
        images.extend(history)
        sections.append('- Current egocentric RGB image: <image>\n')
        sections.append(_render_history_section(
            history,
            label_prefix='Historical keyframe with single-color trajectory overlay',
        ))
        sections.append(
            f'- Historical action summary: {history_action_text.strip()}\n'
        )

    sections.append(_render_output_block(output_mode))
    return PreparedTemplate(question=''.join(sections), images=images)


def _base_intro(instruction: str) -> str:
    return (
        'Imagine you are an autonomous indoor robot.\n'
        'Use the provided egocentric observations to decide the next navigation '
        'target for the instruction below.\n'
        f'Instruction: {instruction}\n\n'
    )


def _render_history_section(
    images: Sequence[Image.Image],
    label_prefix: str,
) -> str:
    lines = []
    for idx, _ in enumerate(images, start=1):
        lines.append(f'- {label_prefix} {idx}: <image>\n')
    return ''.join(lines)


def _render_output_block(output_mode: str) -> str:
    if output_mode == PIXEL_OUTPUT_MODE:
        return 'Output: a pixel coordinate pair in the format XXX, YYY\n'
    if output_mode == ACTION_OUTPUT_MODE:
        return 'Output: one action from {LEFT, RIGHT, FORWARD, STOP}\n'
    if output_mode == ACTION_PIXEL_OUTPUT_MODE:
        return (
            'Output: one action from {LEFT, RIGHT, FORWARD, STOP} and a pixel '
            'coordinate pair in the format ACTION, XXX, YYY\n'
        )
    raise ValueError(
        f'Unsupported output_mode "{output_mode}". '
        f'Available: {PIXEL_OUTPUT_MODE}, {ACTION_OUTPUT_MODE}, '
        f'{ACTION_PIXEL_OUTPUT_MODE}'
    )


def _require_images(
    images: Optional[Sequence[Image.Image]],
    field_name: str,
    template: str,
) -> List[Image.Image]:
    if not images:
        raise ValueError(
            f'{field_name} must be provided for template "{template}".'
        )
    return [_ensure_pil_rgb(img) for img in images]


def _ensure_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f'Expected an RGB-like image, got array with shape {arr.shape}.'
        )
    return Image.fromarray(arr, mode='RGB')
