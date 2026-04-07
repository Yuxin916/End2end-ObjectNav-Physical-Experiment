#!/usr/bin/env zsh
set -e

mkdir -p bags

ros2 bag record \
/registered_scan \
/path \
/free_paths \
/way_point \
/fake_way_point \
/navigation_boundary \
/overall_map \
/trajectory \
/sam2_detection_debug \
/egocentric_rgb \
/fov \
/vlm_bev_debug \
/frontier_rgb_debug \
/tf \
/tf_static \
-o bags/run_$(date +%Y%m%d_%H%M%S) --storage mcap --compression-mode file --compression-format zstd
