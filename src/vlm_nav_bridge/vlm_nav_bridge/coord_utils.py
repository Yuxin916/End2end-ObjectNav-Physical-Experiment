"""
Coordinate conversion utilities between:
  - ROS map frame (metres, X east / Y north / Z up)
  - Global BEV grid  (integer cells, origin at map_origin_x/y)
  - Local BEV image  (output_size×output_size pixels, robot at centre)

BEV layout (matches VLN_CL_CoTNav convention):
  - Rows increase downward  → corresponds to  -Y  in map frame
  - Cols increase rightward → corresponds to  +X  in map frame
  - The global map is axis-aligned with the ROS map frame (no rotation).

Local crop: ±(output_size/2) cells around the robot, 1 cell = 1 pixel (no resize).
  output_size = 448 → crop is 448×448 cells = 22.4 m × 22.4 m at 0.05 m/cell.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Global map <-> world (metres)
# ---------------------------------------------------------------------------

def world_to_global_cell(x: float, y: float,
                          map_origin_x: float, map_origin_y: float,
                          resolution: float):
    """
    World position (x, y) in metres → global map cell (col, row) as ints.

    col  = (x - map_origin_x) / resolution
    row  = (y - map_origin_y) / resolution   (y↑ → row↑, but BEV rows go downward,
                                               so the full_map stores row=0 at lowest-y)
    We keep numpy array indexing as [row, col] throughout.
    """
    col = (x - map_origin_x) / resolution
    row = (y - map_origin_y) / resolution
    return int(col), int(row)


def global_cell_to_world(col: int, row: int,
                          map_origin_x: float, map_origin_y: float,
                          resolution: float):
    """Global map cell → world centre of that cell in metres."""
    x = map_origin_x + (col + 0.5) * resolution
    y = map_origin_y + (row + 0.5) * resolution
    return x, y


# ---------------------------------------------------------------------------
# Local BEV image <-> world (metres)
# ---------------------------------------------------------------------------

def local_pixel_to_world(pixel_row: float, pixel_col: float,
                          robot_x: float, robot_y: float,
                          output_size: int = 448,
                          resolution: float = 0.05) -> tuple:
    """
    Convert a pixel in the local BEV image to world (x, y) in metres.

    The local BEV is a window of ±(output_size/2) cells around the robot,
    with 1 cell = 1 pixel (no resize).

    Robot is at pixel centre (output_size/2, output_size/2).

    Convention (same as VLN training code):
      pixel_col offset → +X world
      pixel_row offset → -Y world  (rows increase downward = south)
    """
    centre = output_size / 2.0

    dcol = pixel_col - centre   # cells east (+X)
    drow = pixel_row - centre   # cells south (-Y)

    world_x = robot_x + dcol * resolution
    world_y = robot_y - drow * resolution        # minus: south → -Y
    return world_x, world_y


def world_to_local_pixel(world_x: float, world_y: float,
                          robot_x: float, robot_y: float,
                          output_size: int = 448,
                          resolution: float = 0.05) -> tuple:
    """Inverse of local_pixel_to_world.  Returns (pixel_row, pixel_col)."""
    centre = output_size / 2.0

    dx = world_x - robot_x
    dy = world_y - robot_y

    dcol = dx / resolution        # cells east
    drow = -dy / resolution       # cells south (flip sign)

    pixel_col = centre + dcol
    pixel_row = centre + drow
    return float(pixel_row), float(pixel_col)


# ---------------------------------------------------------------------------
# Global map crop → local pixel (used inside LidarBEVMapper)
# ---------------------------------------------------------------------------

def global_cell_to_local_pixel(g_col: int, g_row: int,
                                robot_g_col: int, robot_g_row: int,
                                output_size: int = 448) -> tuple:
    """
    Convert a global map cell to a pixel in the local BEV image.
    Returns (pixel_row, pixel_col).  May be outside [0, output_size).

    Local BEV convention:
      - col increases rightward  -> +X
      - row increases downward   -> -Y
    Since global g_row increases northward (+Y), row offset is flipped here.
    """
    centre = output_size / 2.0

    dcol = g_col - robot_g_col
    drow = g_row - robot_g_row

    pixel_col = centre + dcol
    pixel_row = centre - drow
    return float(pixel_row), float(pixel_col)


# ---------------------------------------------------------------------------
# Yaw helpers
# ---------------------------------------------------------------------------

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (radians, CCW from +X) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_bev_degrees(yaw_rad: float) -> float:
    """
    Convert ROS yaw (CCW from +X, radians) to the VLN BEV convention
    (degrees, 0° = east, increasing counter-clockwise).

    In VLN training code agent_yaw_deg is measured CCW from east in the
    BEV image (matching Habitat's convention).  ROS yaw is the same, just
    in radians → convert to degrees.
    """
    return math.degrees(yaw_rad)
