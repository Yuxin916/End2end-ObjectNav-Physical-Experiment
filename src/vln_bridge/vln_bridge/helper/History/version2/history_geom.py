import math
import numpy as np


class HistoryGeomMixin:
    def _get_map_offset(self, planner_pose):
        """Compute (or reuse) a constant offset from world to map coordinates.

        Args:
            planner_pose: Planner pose containing map-space origin (x, y).

        Returns:
            (x_offset, y_offset) tuple or None if not computable.
        """
        # Fast path: reuse cached offset.
        if self._map_offset is not None:
            return self._map_offset
        # Guard against missing pose or history.
        if planner_pose is None or len(planner_pose) < 2 or not self.pos_hist:
            return None
        # Anchor offset using current planner_pose and latest world position.
        start_x_m = float(planner_pose[0])
        start_y_m = float(planner_pose[1])
        cur_wp = np.asarray(self.pos_hist[-1], dtype=np.float32)
        # Use world frame (x,z) if available, otherwise fallback to 2D input.
        if cur_wp.shape[0] >= 3:
            self._map_offset = (
                start_x_m + float(cur_wp[2]),
                start_y_m + float(cur_wp[0]),
            )
        else:
            self._map_offset = (
                start_x_m - float(cur_wp[0]),
                start_y_m - float(cur_wp[1]),
            )
        return self._map_offset

    def _get_world_to_map_transform(self, planner_pose):
        """Estimate a linear transform from world XZ to map XY.

        Args:
            planner_pose: Planner pose with map-frame (x, y, yaw_deg).

        Returns:
            (transform_matrix, offset) or None if insufficient info.
        """
        # Fast path: reuse cached transform.
        if self._map_transform is not None:
            return self._map_transform
        # Require map pose with yaw and a valid world pose history.
        if planner_pose is None or len(planner_pose) < 3:
            return None
        if not self.pos_hist or not self.rot_hist:
            return None

        # Use the latest world pose to align basis vectors.
        world_pos = np.asarray(self.pos_hist[-1], dtype=np.float32)
        world_rot = self.rot_hist[-1]
        try:
            import quaternion
        except Exception:
            return None

        # Build world basis (right, forward) from quaternion.
        R_world = quaternion.as_rotation_matrix(world_rot)
        right_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fwd_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        right_world = R_world @ right_local
        fwd_world = R_world @ fwd_local
        right_world_xz = np.array([right_world[0], right_world[2]], dtype=np.float32)
        fwd_world_xz = np.array([fwd_world[0], fwd_world[2]], dtype=np.float32)

        # Reject degenerate basis vectors.
        right_norm = np.linalg.norm(right_world_xz)
        fwd_norm = np.linalg.norm(fwd_world_xz)
        if right_norm < 1e-6 or fwd_norm < 1e-6:
            return None
        right_world_xz /= right_norm
        fwd_world_xz /= fwd_norm

        # Build map-frame basis from planner yaw.
        yaw_map_deg = float(planner_pose[2])
        yaw_map_rad = math.radians(yaw_map_deg)
        fwd_map = np.array([math.sin(yaw_map_rad), math.cos(yaw_map_rad)], dtype=np.float32)
        right_map = np.array([math.cos(yaw_map_rad), -math.sin(yaw_map_rad)], dtype=np.float32)

        # Solve linear mapping between world XZ and map XY bases.
        world_basis = np.stack([right_world_xz, fwd_world_xz], axis=1)
        map_basis = np.stack([right_map, fwd_map], axis=1)

        det = float(np.linalg.det(world_basis))
        # If the basis is singular, mapping is unstable.
        if abs(det) < 1e-6:
            return None

        # Solve for transform and offset (map = T * world + offset).
        transform = map_basis @ np.linalg.inv(world_basis)
        world_xz = np.array([world_pos[0], world_pos[2]], dtype=np.float32)
        map_xy = np.array([float(planner_pose[0]), float(planner_pose[1])], dtype=np.float32)
        offset = map_xy - transform @ world_xz
        self._map_transform = (transform, offset)
        return self._map_transform

    def _world_to_map_xy(self, world_pos, *, transform=None, offset=None):
        """Convert world position to map XY using a transform or offset.

        Args:
            world_pos: World position (x,y,z) or (x,y).
            transform: Optional (matrix, offset) tuple.
            offset: Optional constant offset (x, y).

        Returns:
            (x_m, y_m) in map coordinates, or (None, None) on failure.
        """
        # If a full transform is provided, apply it.
        if transform is not None:
            mat, off = transform
            wp = np.asarray(world_pos, dtype=np.float32)
            # Use world XZ if available, otherwise use 2D input.
            if wp.shape[0] >= 3:
                world_xz = np.array([wp[0], wp[2]], dtype=np.float32)
            else:
                world_xz = np.array([wp[0], wp[1]], dtype=np.float32)
            map_xy = mat @ world_xz + off
            return float(map_xy[0]), float(map_xy[1])
        # If no transform, fall back to a constant offset mapping.
        if offset is None:
            return None, None
        wp = np.asarray(world_pos, dtype=np.float32)
        # Apply the offset with convention from planner pose to world axes.
        if wp.shape[0] >= 3:
            x_m = -float(wp[2]) + float(offset[0])
            y_m = -float(wp[0]) + float(offset[1])
        else:
            x_m = float(wp[0]) + float(offset[0])
            y_m = float(wp[1]) + float(offset[1])
        return x_m, y_m