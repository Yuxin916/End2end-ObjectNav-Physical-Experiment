"""
ROS-native trajectory history utilities for RGB-based VLM prompting.

This implementation is intentionally closer to helper/History/version1 than
the earlier lightweight approximation:

- each history entry stores pose + RGB + aligned depth snapshot
- keyframes are chosen with a visibility / look-ahead check inspired by the
  original helper instead of only distance / time thresholds
- FPV history overlays are drawn only for points that are actually visible from
  the keyframe image according to the stored depth map

The public ROS-facing API remains small:
``reset(...)``, ``update(...)``, ``get_hist_img(...)``, ``get_interval_img(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class HistoryEntry:
    """One egocentric RGB observation aligned with a camera pose."""

    x: float
    y: float
    z: float
    yaw_rad: float
    stamp_s: float
    rgb: np.ndarray
    depth: Optional[np.ndarray]


@dataclass
class VisibilityResult:
    pixel: Optional[Tuple[int, int]]
    visible: bool
    known: bool = True


class History:
    """
    ROS adaptation of helper/History/version1 for egocentric RGB prompts.

    Compared with the previous simplified version, this class tracks depth for
    visibility checks and uses a version1-like look-ahead heuristic for
    selecting history keyframes.
    """

    def __init__(
        self,
        *,
        hfov_deg: float,
        image_height: int,
        image_width: int,
        sensor_height_m: float = 0.55,
        camera_elevation_deg: float = 0.0,
        keyframe_translation_m: float = 0.30,
        keyframe_rotation_deg: float = 12.0,
        keyframe_time_sec: float = 0.75,
        dedup_translation_m: float = 0.03,
        dedup_rotation_deg: float = 2.0,
        dedup_time_sec: float = 0.20,
        max_entries: int = 1024,
        look_ahead: int = 2,
        visibility_distance_threshold_m: float = 0.60,
        visibility_patch_radius_px: int = 1,
        blind_radius_m: Optional[float] = None,
        storage_merge_distance_m: float = 0.10,
        draw_merge_distance_m: float = 0.10,
        depth_min_m: float = 0.05,
        depth_max_m: float = 10.0,
    ):
        self.hfov_deg = float(hfov_deg)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.sensor_height_m = float(sensor_height_m)
        self.camera_elevation_deg = float(camera_elevation_deg)
        self.keyframe_translation_m = float(keyframe_translation_m)
        self.keyframe_rotation_rad = math.radians(float(keyframe_rotation_deg))
        self.keyframe_time_sec = float(keyframe_time_sec)
        self.dedup_translation_m = float(dedup_translation_m)
        self.dedup_rotation_rad = math.radians(float(dedup_rotation_deg))
        self.dedup_time_sec = float(dedup_time_sec)
        self.max_entries = max(8, int(max_entries))
        self.look_ahead = max(1, int(look_ahead))
        self.visibility_distance_threshold_m = float(
            visibility_distance_threshold_m
        )
        self.visibility_patch_radius_px = max(0, int(visibility_patch_radius_px))
        if blind_radius_m is None:
            blind_radius_m = sensor_height_m
        self.blind_radius_m = max(0.0, float(blind_radius_m))
        self.storage_merge_distance_m = max(0.0, float(storage_merge_distance_m))
        self.draw_merge_distance_m = max(0.0, float(draw_merge_distance_m))
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)

        self._base_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        self.entries: List[HistoryEntry] = []
        self.keyframe_indices: List[int] = []
        self._keyframes_dirty: bool = True
        self._recompute_intrinsics()

    def _recompute_intrinsics(self):
        cx = (float(self.image_width) - 1.0) * 0.5
        cy = (float(self.image_height) - 1.0) * 0.5
        hfov_rad = math.radians(max(1e-3, float(self.hfov_deg)))
        fx = cx / max(math.tan(hfov_rad * 0.5), 1e-6)
        vfov = 2.0 * math.atan(
            math.tan(hfov_rad * 0.5)
            * (float(self.image_height) / max(float(self.image_width), 1.0))
        )
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

    def reset(
        self,
        start_pos: Sequence[float],
        start_yaw_rad: float,
        rgb,
        *,
        depth=None,
        stamp_s: float = 0.0,
    ):
        """Reset the trajectory history with the first RGB/depth observation."""
        self.entries = [
            self._make_entry(start_pos, start_yaw_rad, rgb, depth, stamp_s)
        ]
        self.keyframe_indices = [0]
        self._keyframes_dirty = False

    def update(
        self,
        pos: Sequence[float],
        yaw_rad: float,
        rgb,
        *,
        depth=None,
        stamp_s: float = 0.0,
    ) -> bool:
        """
        Append or refresh one trajectory observation.

        Returns ``True`` when recomputing keyframes would likely add a new one.
        """
        if not self.entries:
            self.reset(pos, yaw_rad, rgb, depth=depth, stamp_s=stamp_s)
            return True

        old_keyframes = list(self.keyframe_indices)
        new_entry = self._make_entry(pos, yaw_rad, rgb, depth, stamp_s)
        last_entry = self.entries[-1]
        if self._is_near_duplicate(last_entry, new_entry):
            self.entries[-1] = new_entry
            self._keyframes_dirty = True
            self._refresh_keyframes()
            return len(self.keyframe_indices) > len(old_keyframes)

        if self._should_merge_into_storage_tail(new_entry):
            self.entries[-1] = new_entry
            self._keyframes_dirty = True
            self._refresh_keyframes()
            return len(self.keyframe_indices) > len(old_keyframes)

        self.entries.append(new_entry)
        self._trim_history()
        self._keyframes_dirty = True
        self._refresh_keyframes()
        return len(self.keyframe_indices) > len(old_keyframes)

    def get_hist_img(self, length: int = 8) -> Dict[str, List[np.ndarray]]:
        """
        Render historical FPV images in the same spirit as version1 History.

        Returned order is chronological, oldest to newest. Callers that want
        ``latest -> oldest`` can reverse the lists after receiving them.
        """
        if not self.entries:
            return {
                'fpv_imgs_multi_colored': [],
                'fpv_imgs_blue': [],
                'fpv_imgs_no_draw': [],
            }

        self._refresh_keyframes()
        indices = self._select_history_indices(length)
        if not indices:
            fallback = self.entries[-1].rgb.copy()
            return {
                'fpv_imgs_multi_colored': [fallback],
                'fpv_imgs_blue': [fallback],
                'fpv_imgs_no_draw': [fallback],
            }

        return {
            'fpv_imgs_multi_colored': [
                self.get_ks_fpv(idx, draw_mode='multi_color') for idx in indices
            ],
            'fpv_imgs_blue': [
                self.get_ks_fpv(
                    idx,
                    draw_mode='single_color',
                    single_color=(0, 0, 255),
                )
                for idx in indices
            ],
            'fpv_imgs_no_draw': [
                self.get_ks_fpv(idx, draw_mode='no_draw') for idx in indices
            ],
        }

    def get_interval_img(self, length: int = 8) -> List[np.ndarray]:
        """Return raw interval images for templates such as RGB_His1Interval8."""
        if not self.entries:
            return []
        if len(self.entries) <= 1:
            return [self.entries[-1].rgb.copy()]

        usable = self.entries[:-1]
        target = max(1, int(length))
        if len(usable) <= target:
            return [entry.rgb.copy() for entry in usable]

        sample_pos = np.linspace(0, len(usable) - 1, num=target, dtype=int)
        dedup_indices: List[int] = []
        for idx in sample_pos.tolist():
            if not dedup_indices or idx != dedup_indices[-1]:
                dedup_indices.append(idx)
        return [usable[idx].rgb.copy() for idx in dedup_indices]

    def get_action_summary(self, recent_n: int = 5) -> str:
        """Infer a compact action history string from odometry deltas."""
        if len(self.entries) <= 1:
            return 'none action has been executed'

        actions: List[str] = []
        for prev, cur in zip(self.entries[:-1], self.entries[1:]):
            delta_yaw = self._wrap_to_pi(cur.yaw_rad - prev.yaw_rad)
            distance = self._horizontal_distance(prev, cur)
            if abs(delta_yaw) >= math.radians(10.0) and distance < 0.20:
                actions.append('TURN_LEFT' if delta_yaw > 0.0 else 'TURN_RIGHT')
            elif distance >= 0.10:
                actions.append('MOVE_FORWARD')

        if not actions:
            return 'none action has been executed'
        return ', '.join(actions[-max(1, int(recent_n)):])

    def get_ks_fpv(
        self,
        entry_idx: int,
        *,
        draw_mode: str = 'multi_color',
        single_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Render one history image aligned to a stored trajectory keyframe.

        ``draw_mode`` supports:
        - ``multi_color``
        - ``single_color``
        - ``no_draw``
        """
        if entry_idx < 0 or entry_idx >= len(self.entries):
            raise ValueError(f'entry_idx out of range: {entry_idx}')

        entry = self.entries[entry_idx]
        base = entry.rgb.copy()
        if draw_mode == 'no_draw':
            return base

        image = Image.fromarray(base).convert('RGBA')
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        future_indices = self._trajectory_draw_indices(entry_idx)
        future_entries = [self.entries[idx] for idx in future_indices]

        prev_pixel: Optional[Tuple[int, int]] = None
        visible_rank = 0
        started_segment = False
        for hist_entry in future_entries:
            # Drawing the trajectory is only a geometric overlay.  Keyframe
            # selection still uses depth-based visibility, but the visual prompt
            # should not disappear just because depth is sparse, mismatched, or
            # inside the depth blind zone.
            pixel = self._project_entry_into_view(hist_entry, observer=entry)
            if pixel is None:
                if started_segment:
                    break
                prev_pixel = None
                continue

            u, v = pixel
            started_segment = True
            if draw_mode == 'single_color':
                color_rgb = single_color or (0, 0, 255)
            else:
                palette_idx = min(
                    len(self._base_colors) - 1,
                    visible_rank % len(self._base_colors),
                )
                color_rgb = self._base_colors[palette_idx]
            color_rgba = (
                int(color_rgb[0]),
                int(color_rgb[1]),
                int(color_rgb[2]),
                255,
            )

            radius = 4 if visible_rank == 0 else 3
            draw.ellipse(
                [(u - radius, v - radius), (u + radius, v + radius)],
                fill=color_rgba,
            )
            if prev_pixel is not None:
                draw.line([prev_pixel, (u, v)], fill=color_rgba, width=2)
            prev_pixel = (u, v)
            visible_rank += 1

        return np.asarray(Image.alpha_composite(image, overlay).convert('RGB'))

    def _refresh_keyframes(self):
        if not self._keyframes_dirty:
            return

        if not self.entries:
            self.keyframe_indices = []
            self._keyframes_dirty = False
            return

        if len(self.entries) == 1:
            self.keyframe_indices = [0]
            self._keyframes_dirty = False
            return

        distinct_indices = self._distinct_entry_indices()
        if not distinct_indices:
            self.keyframe_indices = []
            self._keyframes_dirty = False
            return

        keyframes = [distinct_indices[0]]
        cursor = 1
        while cursor < len(distinct_indices):
            last_key_idx = keyframes[-1]
            cur_idx = distinct_indices[cursor]
            if not self._is_past_keyframe_blind_zone(last_key_idx, cur_idx):
                cursor += 1
                continue
            visibility = self._visibility_state(last_key_idx, cur_idx)
            if visibility is True:
                cursor += 1
                continue
            if visibility is None:
                cursor += 1
                continue

            found_new_keyframe = False
            search_pos = cursor
            while search_pos < len(distinct_indices):
                cand_idx = distinct_indices[search_pos]
                if not self._is_past_keyframe_blind_zone(last_key_idx, cand_idx):
                    search_pos += 1
                    continue
                if not self._passes_reference_motion(last_key_idx, cand_idx):
                    search_pos += 1
                    continue
                candidate_visibility = self._visibility_state(last_key_idx, cand_idx)
                if candidate_visibility is True:
                    search_pos += 1
                    continue
                if candidate_visibility is None:
                    search_pos += 1
                    continue

                candidate_result = self._looks_like_keyframe(cand_idx)
                if candidate_result is None:
                    break
                if candidate_result:
                    keyframes.append(cand_idx)
                    cursor = self._jump_after_keyframe(
                        cand_idx,
                        distinct_indices,
                        search_pos + 1,
                    )
                    found_new_keyframe = True
                    break
                search_pos += 1

            if not found_new_keyframe:
                break

        if not keyframes:
            keyframes = [0]
        self.keyframe_indices = keyframes
        self._keyframes_dirty = False

    def _select_history_indices(self, length: int) -> List[int]:
        if not self.entries:
            return []

        candidates = list(self.keyframe_indices)
        if not candidates and len(self.entries) > 1:
            return [len(self.entries) - 2]

        length = max(1, int(length))
        return candidates[-length:]

    def _passes_reference_motion(self, ref_idx: int, cur_idx: int) -> bool:
        ref_entry = self.entries[ref_idx]
        cur_entry = self.entries[cur_idx]
        distance = self._horizontal_distance(ref_entry, cur_entry)
        return distance >= self.keyframe_translation_m

    def _looks_like_keyframe(self, idx: int) -> Optional[bool]:
        observer = self.entries[idx]
        future_indices = self._distinct_entry_indices(start_idx=idx + 1)
        future = [self.entries[i] for i in future_indices]
        if not future:
            return False
        if len(future) < self.look_ahead:
            return None

        look_ahead_entries = future[: self.look_ahead]
        if not self._in_fov_cone(observer, look_ahead_entries):
            return False

        if observer.depth is None:
            return False

        far_future = [
            entry
            for entry in future
            if self._horizontal_distance(observer, entry)
            > max(self.blind_radius_m, self.keyframe_translation_m)
        ]
        if not far_future:
            # Match the original helper intent more closely: if we only have
            # future waypoints inside the blind zone, we do not have enough
            # evidence yet to confirm a new keyframe. Recompute later when more
            # trajectory is available.
            return None

        vis_results = self._visibility_results(
            observer,
            far_future[: max(self.look_ahead * 3, 6)],
        )
        known_results = [result for result in vis_results if result.known]
        if not known_results:
            return None
        if any(result.visible for result in known_results):
            return True

        return False

    def _visibility_state(
        self,
        observer_idx: int,
        target_idx: int,
    ) -> Optional[bool]:
        observer = self.entries[observer_idx]
        target = self.entries[target_idx]
        if self._horizontal_distance(observer, target) <= float(self.blind_radius_m):
            return True
        vis_results = self._visibility_results(observer, [target])
        if not vis_results or not vis_results[0].known:
            return None
        return bool(vis_results[0].visible)

    def _is_visible_from(self, observer_idx: int, target_idx: int) -> bool:
        return self._visibility_state(observer_idx, target_idx) is True

    def _jump_after_keyframe(
        self,
        keyframe_idx: int,
        distinct_indices: Sequence[int],
        start_pos: int,
    ) -> int:
        start_pos = max(0, int(start_pos))
        for pos in range(start_pos, len(distinct_indices)):
            state = self._visibility_state(keyframe_idx, distinct_indices[pos])
            if state is True:
                return pos
        return start_pos

    def _is_past_keyframe_blind_zone(self, observer_idx: int, target_idx: int) -> bool:
        observer = self.entries[observer_idx]
        target = self.entries[target_idx]
        min_keyframe_distance = float(self.blind_radius_m) + float(
            self.keyframe_translation_m
        )
        return self._horizontal_distance(observer, target) > min_keyframe_distance

    def _visibility_results(
        self,
        observer: HistoryEntry,
        targets: Sequence[HistoryEntry],
    ) -> List[VisibilityResult]:
        results: List[VisibilityResult] = []
        for target in targets:
            if self._horizontal_distance(observer, target) <= float(self.blind_radius_m):
                results.append(VisibilityResult(pixel=None, visible=False, known=False))
                continue
            pixel = self._project_entry_into_view(target, observer=observer)
            if pixel is None:
                results.append(VisibilityResult(pixel=None, visible=False, known=True))
                continue
            if observer.depth is None:
                results.append(VisibilityResult(pixel=pixel, visible=False, known=False))
                continue

            depth_m = self._lookup_depth(observer.depth, pixel)
            if depth_m is None:
                results.append(VisibilityResult(pixel=pixel, visible=False, known=False))
                continue

            reproj = self._pixel_to_agent_from_depth(pixel, depth_m)
            if reproj is None:
                results.append(VisibilityResult(pixel=pixel, visible=False, known=False))
                continue

            right, forward, up = self._world_to_agent(target, observer=observer)
            target_agent = np.array([right, forward, up], dtype=np.float32)
            delta = reproj - target_agent
            dist_3d = float(np.linalg.norm(delta))
            forward_err = abs(float(delta[1]))
            lateral_err = float(np.linalg.norm(delta[[0, 2]]))
            tol = float(self.visibility_distance_threshold_m)
            visible = dist_3d < tol
            results.append(VisibilityResult(pixel=pixel, visible=bool(visible), known=True))
        return results

    def _lookup_depth(
        self,
        depth: np.ndarray,
        pixel_xy: Tuple[int, int],
    ) -> Optional[float]:
        if depth is None:
            return None
        h, w = depth.shape[:2]
        u = int(pixel_xy[0])
        v = int(pixel_xy[1])
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        base_radius = self.visibility_patch_radius_px
        search_radii = [base_radius]
        if base_radius < 2:
            search_radii.append(2)
        if base_radius < 3:
            search_radii.append(3)

        for radius in search_radii:
            u0 = max(0, u - radius)
            u1 = min(w, u + radius + 1)
            v0 = max(0, v - radius)
            v1 = min(h, v + radius + 1)
            patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float32)
            valid = np.isfinite(patch)
            valid &= patch >= float(self.depth_min_m)
            valid &= patch <= float(self.depth_max_m)
            if np.any(valid):
                return float(np.median(patch[valid]))
        return None

    def _in_fov_cone(
        self,
        observer: HistoryEntry,
        targets: Sequence[HistoryEntry],
    ) -> bool:
        half_fov = 0.5 * math.radians(float(self.hfov_deg))
        for target in targets:
            _, forward, _ = self._world_to_agent(target, observer=observer)
            if forward <= 0.0:
                return False
            right, _, _ = self._world_to_agent(target, observer=observer)
            if abs(math.atan2(right, forward)) > half_fov:
                return False
        return True

    def _world_to_agent(
        self,
        target: HistoryEntry,
        *,
        observer: HistoryEntry,
    ) -> Tuple[float, float, float]:
        dx = float(target.x) - float(observer.x)
        dy = float(target.y) - float(observer.y)
        dz = float(target.z) - float(observer.z)

        cos_yaw = math.cos(observer.yaw_rad)
        sin_yaw = math.sin(observer.yaw_rad)

        forward = cos_yaw * dx + sin_yaw * dy
        right = sin_yaw * dx - cos_yaw * dy
        up = dz
        return right, forward, up

    def _project_entry_into_view(
        self,
        world_entry: HistoryEntry,
        *,
        observer: HistoryEntry,
    ) -> Optional[Tuple[int, int]]:
        right, forward, up = self._world_to_agent(world_entry, observer=observer)
        if forward <= 0.05:
            return None

        half_hfov = 0.5 * math.radians(self.hfov_deg)
        if abs(math.atan2(right, forward)) > half_hfov:
            return None

        pitch = math.radians(self.camera_elevation_deg)
        rel_up = up - self.sensor_height_m
        forward_rot = forward * math.cos(pitch) + rel_up * math.sin(pitch)
        up_rot = -forward * math.sin(pitch) + rel_up * math.cos(pitch)
        if forward_rot <= 1e-3:
            return None

        u = self.fx * (right / forward_rot) + self.cx
        v = self.cy - self.fy * (up_rot / forward_rot)
        if (
            u < 0.0
            or u >= float(self.image_width)
            or v < 0.0
            or v >= float(self.image_height)
        ):
            return None

        return int(round(u)), int(round(v))

    def _pixel_to_agent_from_depth(
        self,
        pixel_xy: Tuple[int, int],
        depth_m: float,
    ) -> Optional[np.ndarray]:
        if depth_m <= 0.0 or not np.isfinite(depth_m):
            return None
        u = float(pixel_xy[0])
        v = float(pixel_xy[1])
        x_img = (u - self.cx) / max(self.fx, 1e-6)
        y_img = (self.cy - v) / max(self.fy, 1e-6)
        pitch = math.radians(float(self.camera_elevation_deg))

        ray_right = float(x_img)
        ray_up = float(y_img * math.cos(pitch) + math.sin(pitch))
        ray_forward = float(-y_img * math.sin(pitch) + math.cos(pitch))
        if ray_forward <= 1e-6:
            return None

        scale = float(depth_m) / ray_forward
        right = ray_right * scale
        forward = ray_forward * scale
        up = self.sensor_height_m + ray_up * scale
        return np.array([right, forward, up], dtype=np.float32)

    def _make_entry(
        self,
        pos: Sequence[float],
        yaw_rad: float,
        rgb,
        depth,
        stamp_s: float,
    ) -> HistoryEntry:
        rgb_arr = np.asarray(rgb)
        if rgb_arr.dtype != np.uint8:
            rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)
        if rgb_arr.ndim == 2:
            rgb_arr = np.stack([rgb_arr, rgb_arr, rgb_arr], axis=-1)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError(f'Expected RGB image with shape HxWx3, got {rgb_arr.shape}')

        if (
            rgb_arr.shape[0] != self.image_height
            or rgb_arr.shape[1] != self.image_width
        ):
            self.image_height = int(rgb_arr.shape[0])
            self.image_width = int(rgb_arr.shape[1])
            self._recompute_intrinsics()

        depth_arr: Optional[np.ndarray] = None
        if depth is not None:
            depth_arr = np.asarray(depth, dtype=np.float32)
            if depth_arr.ndim != 2:
                raise ValueError(f'Expected depth image with shape HxW, got {depth_arr.shape}')
            if depth_arr.shape[:2] != rgb_arr.shape[:2]:
                raise ValueError(
                    f'RGB/depth shape mismatch: rgb={rgb_arr.shape[:2]}, '
                    f'depth={depth_arr.shape[:2]}'
                )
            depth_arr = depth_arr.copy()

        px = float(pos[0]) if len(pos) > 0 else 0.0
        py = float(pos[1]) if len(pos) > 1 else 0.0
        pz = float(pos[2]) if len(pos) > 2 else 0.0
        return HistoryEntry(
            x=px,
            y=py,
            z=pz,
            yaw_rad=float(yaw_rad),
            stamp_s=float(stamp_s),
            rgb=rgb_arr.copy(),
            depth=depth_arr,
        )

    def _trim_history(self):
        if len(self.entries) <= self.max_entries:
            return

        excess = len(self.entries) - self.max_entries
        self.entries = self.entries[excess:]
        self.keyframe_indices = [
            idx - excess for idx in self.keyframe_indices if idx - excess >= 0
        ]
        if not self.keyframe_indices and self.entries:
            self.keyframe_indices = [0]
        self._keyframes_dirty = True

    def _should_merge_into_storage_tail(self, new_entry: HistoryEntry) -> bool:
        if self.storage_merge_distance_m <= 0.0 or not self.entries:
            return False

        last_idx = len(self.entries) - 1
        if last_idx in set(self.keyframe_indices):
            return False

        last_entry = self.entries[last_idx]
        distance = self._horizontal_distance(last_entry, new_entry)
        return distance < self.storage_merge_distance_m

    def _trajectory_draw_indices(self, start_idx: int) -> List[int]:
        end_idx = len(self.entries)
        sorted_keyframes = sorted(idx for idx in self.keyframe_indices if idx >= 0)
        if start_idx in sorted_keyframes:
            for key_idx in sorted_keyframes:
                if key_idx > start_idx:
                    end_idx = key_idx
                    break

        indices = [
            idx
            for idx in self._distinct_entry_indices(start_idx=start_idx)
            if idx < end_idx
        ]
        if not indices or self.draw_merge_distance_m <= 0.0:
            return indices

        keyframe_set = set(self.keyframe_indices)
        merged: List[int] = [indices[0]]
        for idx in indices[1:]:
            if idx in keyframe_set:
                merged.append(idx)
                continue

            last_kept_idx = merged[-1]
            if (
                self._horizontal_distance(self.entries[last_kept_idx], self.entries[idx])
                < self.draw_merge_distance_m
            ):
                continue
            merged.append(idx)
        return merged

    def _distinct_entry_indices(self, start_idx: int = 0) -> List[int]:
        if not self.entries:
            return []
        start_idx = max(0, int(start_idx))
        if start_idx >= len(self.entries):
            return []

        distinct = [start_idx]
        last_idx = start_idx
        for idx in range(start_idx + 1, len(self.entries)):
            if self._horizontal_distance(self.entries[last_idx], self.entries[idx]) < self.dedup_translation_m:
                continue
            distinct.append(idx)
            last_idx = idx
        return distinct

    def _previous_distinct_index(self, idx: int) -> int:
        idx = max(0, min(int(idx), len(self.entries) - 1))
        for prev in range(idx - 1, -1, -1):
            if self._horizontal_distance(self.entries[prev], self.entries[idx]) >= self.dedup_translation_m:
                return prev
        return 0

    def _horizontal_distance(self, a: HistoryEntry, b: HistoryEntry) -> float:
        return math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))

    def _is_near_duplicate(self, prev: HistoryEntry, cur: HistoryEntry) -> bool:
        distance = self._horizontal_distance(prev, cur)
        delta_yaw = abs(self._wrap_to_pi(cur.yaw_rad - prev.yaw_rad))
        delta_t = max(0.0, float(cur.stamp_s) - float(prev.stamp_s))
        return (
            distance < self.dedup_translation_m
            and delta_yaw < self.dedup_rotation_rad
            and delta_t < self.dedup_time_sec
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
