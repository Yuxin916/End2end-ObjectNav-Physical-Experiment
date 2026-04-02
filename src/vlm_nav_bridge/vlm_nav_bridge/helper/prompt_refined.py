from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Tuple

# General descriptions
INTRO = "Imagine you are an autonomous robot in an indoor habitat environment.\nInputs:\n"
GOAL_TPL = "- Goal: search for and navigate to **{goal}**.\n"

# Input description
BEV_FT = ("- BEV grid map <image> showing free (white), occupied (black), unexplored (gray), "
         "frontier candidates (green dots), robot pose/heading (red arrow), and past trajectory (blue line).\n")
BEV_FT_FOV = ("- BEV grid map <image> showing free (white), occupied (black), unexplored (gray), "
         "frontier candidates (green dots), robot pose/heading (red arrow), past trajectory (blue line), "
         "and egocentric camera field of view (yellow cone). "
         "If an orange dot appears, it indicates the goal object has been detected - navigate directly to it.\n")
BEV_FT_FOV_CANDIDATE = ("- BEV grid map <image_bev> showing free (white), occupied (black), unexplored (gray), "
         "frontier candidates (green dots), robot pose/heading (red arrow), past trajectory (blue line), "
         "and egocentric camera field of view (yellow cone). "
         "An orange dot may appear on the BEV map indicating the detected goal location.\n")
RGB = "- Current Step egocentric RGB image <image>\n"
SEG = "- Current Step Segmentation mask <image>\n"
PAST5 = "- Past 5 frames of egocentric RGB images <image> <image> <image> <image> <image>.\n"

# Output description
DECIDE_ACTION_FIRST = (
    "\nDecide the next step:\n"
    "- Action in {LEFT, RIGHT, FORWARD, STOP}\n"
    "- Frontier index in numbered frontiers in the BEV map\n"
)
DECIDE_FRONTIER_FIRST = (
    "\nDecide the next step:\n"
    "- Frontier index in numbered frontiers in the BEV map\n"
    "- Action in {LEFT, RIGHT, FORWARD, STOP}\n"
)
DECIDE_FRONTIER_ONLY = (
    "\nDecide the next step:\n"
    "- Frontier index in numbered frontiers in the BEV map\n"
)
OUTPUT_FRONTIER_ONLY = "\nOutput:\n<FRONTIER_NUMBER>\n"
OUTPUT_ACTION_FRONTIER = "\nOutput:\n<ACTION>, <FRONTIER_NUMBER>\n"
OUTPUT_FRONTIER_ACTION = "\nOutput:\n<FRONTIER_NUMBER>, <ACTION>\n"

DECIDE_ACTION_ONLY = (
    "\nDecide the next step:\n"
    "- Action in {LEFT, RIGHT, FORWARD, STOP}\n"
)
DECIDE_FRONTIER_PIXEL_ONLY = (
    "\nDecide the next step:\n"
    "- Frontier pixel coordinates (x,y) on the BEV map from the following candidates: {}\n"
)
DECIDE_FRONTIER_PIXEL_NUMBER_ONLY = (
    "\nDecide the next step:\n"
    "- Choose a frontier index from the following candidates (coordinates are on the BEV map): {}\n"
)
DECIDE_FRONTIER_PIXEL_NUMBER_ONLY_WITH_FRONTIER_SEMANTICS = (
    "\nDecide the next step:\n"
    "- Choose a frontier index from the following candidates. Each frontier shows [index: BEV_map_pixel_coordinates (nearby_objects)]. Consider which frontier is most likely to lead you toward your goal: {}\n"
)
DECIDE_FRONTIER_PIXEL_NUMBER_ONLY_WITH_FRONTIER_SEMANTICS_NO_PIXEL = (
    "\nDecide the next step:\n"
    "- Choose a frontier index from the following candidates. Each frontier shows [index: (nearby_objects)]. Use the spatial information from the BEV map and position embeddings to determine which frontier is most likely to lead you toward your goal: {}\n"
)
OUTPUT_ACTION_ONLY = "\nOutput:\n<ACTION>\n"
OUTPUT_FRONTIER_PIXEL_ONLY = "\nOutput:\n<FRONTIER_BEV_MAP_PIXEL_COORDINATES>\n"
OUTPUT_FRONTIER_PIXEL_NUMBER_ONLY = "\nOutput:\n<FRONTIER_INDEX>\n"

DECIDE_OUTPUT_FRONTIER_ID_ONLY = "\nChoose one frontier id. Output only the integer id.\n"
DECIDE_OUTPUT_CANDIDATE_ID_ONLY = "\nChoose one candidate id. Output only the integer id.\n"
DECIDE_OUTPUT_CANDIDATE_TOKEN_ONLY = "\nChoose one candidate token. Output only one token in the form <id_k>.\n"

Action = Literal["LEFT", "RIGHT", "FORWARD", "STOP"]
OutputField = Literal["ACTION", "FRONTIER", "FRONTIER_PIXEL", "FRONTIER_PIXEL_NUMBER"]
OutputOrder = Literal["action_first", "frontier_first"]
ImageModality = Literal["BEVft", "BEVftFOV", "RGB", "Seg", "Past5RGB"]

MODALITY_TO_TEXT = {
    "BEVft": BEV_FT,
    "BEVftFOV": BEV_FT_FOV,
    "RGB": RGB,
    "Seg": SEG,
    "Past5RGB": PAST5,
}


@dataclass(frozen=True)
class InputBundle:
    modalities: Tuple[ImageModality, ...]


@dataclass(frozen=True)
class OutputSpec:
    fields: Tuple[OutputField, ...]
    order: OutputOrder = "action_first"


def _render_inputs_text(bundle: InputBundle) -> str:
    lines = [INTRO]
    for m in bundle.modalities:
        lines.append(MODALITY_TO_TEXT[m])
    return "".join(lines)


def _render_decide_and_output(
    spec: OutputSpec,
    local_frontiers: Optional[List[Tuple[int, int]]] = None,
    semantic_labels: Optional[List[List[str]]] = None,
    hide_coordinates: bool = False,
) -> str:
    fieldset = set(spec.fields)
    if fieldset == {"ACTION"} and len(spec.fields) == 1:
        return DECIDE_ACTION_ONLY + OUTPUT_ACTION_ONLY
    if fieldset == {"FRONTIER"} and len(spec.fields) == 1:
        return DECIDE_FRONTIER_ONLY + OUTPUT_FRONTIER_ONLY
    if fieldset == {"FRONTIER_PIXEL"} and len(spec.fields) == 1:
        if local_frontiers is not None:
            frontier_list_str = ", ".join([f"{coord}" for coord in local_frontiers])
            return DECIDE_FRONTIER_PIXEL_ONLY.format(frontier_list_str) + OUTPUT_FRONTIER_PIXEL_ONLY
        return DECIDE_FRONTIER_PIXEL_ONLY.format("[list not provided]") + OUTPUT_FRONTIER_PIXEL_ONLY

    if fieldset == {"FRONTIER_PIXEL_NUMBER"} and len(spec.fields) == 1:
        if local_frontiers is not None:
            if semantic_labels is not None:
                if hide_coordinates:
                    frontier_list_str = ", ".join([f"{i}: ({semantic_labels[i]})" for i in range(len(local_frontiers))])
                    return DECIDE_FRONTIER_PIXEL_NUMBER_ONLY_WITH_FRONTIER_SEMANTICS_NO_PIXEL.format(frontier_list_str) + OUTPUT_FRONTIER_PIXEL_NUMBER_ONLY
                frontier_list_str = ", ".join([f"{i}: {coord} ({semantic_labels[i]})" for i, coord in enumerate(local_frontiers)])
                return DECIDE_FRONTIER_PIXEL_NUMBER_ONLY_WITH_FRONTIER_SEMANTICS.format(frontier_list_str) + OUTPUT_FRONTIER_PIXEL_NUMBER_ONLY
            frontier_list_str = ", ".join([f"{i}: {coord}" for i, coord in enumerate(local_frontiers)])
            return DECIDE_FRONTIER_PIXEL_NUMBER_ONLY.format(frontier_list_str) + OUTPUT_FRONTIER_PIXEL_NUMBER_ONLY
        return DECIDE_FRONTIER_PIXEL_NUMBER_ONLY.format("[list not provided]") + OUTPUT_FRONTIER_PIXEL_NUMBER_ONLY

    if fieldset == {"ACTION", "FRONTIER"} and len(spec.fields) == 2:
        if spec.order == "action_first":
            return DECIDE_ACTION_FIRST + OUTPUT_ACTION_FRONTIER
        return DECIDE_FRONTIER_FIRST + OUTPUT_FRONTIER_ACTION
    return DECIDE_ACTION_ONLY + OUTPUT_ACTION_ONLY


def _format_reply(
    spec: OutputSpec,
    step_action: Optional[Action],
    frontier_index: Optional[int],
    selected_frontier: Optional[Tuple[int, int]] = None,
) -> str:
    a = step_action if step_action is not None else "STOP"
    f = "None" if frontier_index is None else str(frontier_index)
    fieldset = set(spec.fields)
    if fieldset == {"ACTION"} and len(spec.fields) == 1:
        return f"{a}"
    if fieldset == {"FRONTIER"} and len(spec.fields) == 1:
        return f"{f}"
    if fieldset == {"FRONTIER_PIXEL"} and len(spec.fields) == 1:
        return f"{selected_frontier}" if selected_frontier is not None else "None"
    if fieldset == {"FRONTIER_PIXEL_NUMBER"} and len(spec.fields) == 1:
        return f"{f}"
    if spec.order == "action_first":
        return f"{a}, {f}"
    return f"{f}, {a}"


def _render_semantic_candidates(position_info: Dict, show_semantics: bool = True) -> str:
    def _sem_str(semantic) -> str:
        if isinstance(semantic, list):
            return str(semantic)
        return "[]" if semantic is None else str([semantic])

    # Fast path: use pre-built (possibly shuffled) candidates list from dataset reconstruction
    if "candidates" in position_info:
        lines = []
        for cand in position_info["candidates"]:
            r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
            if show_semantics:
                sem = _sem_str(cand.get('semantic'))
                sem_part = f" semantic={sem}" if sem != "[]" else ""
            else:
                sem_part = ""
            lines.append(f"<cand> id={cand['id']} type={cand['type']} pos=({r}, {c}){sem_part} <e_cand>")
        if not lines:
            return ""
        return "Candidates:\n" + "\n".join(lines) + "\n"

    # Fallback: build from frontier_positions + target_position (eval-time, no shuffle)
    frontier_positions = position_info.get("frontier_positions") or []
    frontier_semantic_labels = position_info.get("frontier_semantic_labels") or []
    target_position = position_info.get("target_position")
    target_semantic = position_info.get("target_semantic")

    lines = []
    for i, pos in enumerate(frontier_positions):
        r, c = int(round(pos[0])), int(round(pos[1]))
        if show_semantics:
            semantic = frontier_semantic_labels[i] if i < len(frontier_semantic_labels) else []
            sem = _sem_str(semantic)
            sem_part = f" semantic={sem}" if sem != "[]" else ""
        else:
            sem_part = ""
        lines.append(f"<cand> id={i} type=frontier pos=({r}, {c}){sem_part} <e_cand>")

    if target_position is not None:
        r, c = int(round(target_position[0])), int(round(target_position[1]))
        target_id = len(frontier_positions)
        if show_semantics:
            sem = _sem_str(target_semantic)
            sem_part = f" semantic={sem}" if sem != "[]" else ""
        else:
            sem_part = ""
        lines.append(f"<cand> id={target_id} type=target pos=({r}, {c}){sem_part} <e_cand>")

    if not lines:
        return ""
    return "Candidates:\n" + "\n".join(lines) + "\n"


def _validate(bundle: InputBundle, spec: OutputSpec) -> None:
    needs_frontier = "FRONTIER" in spec.fields
    has_frontier_map = any(m in ("BEVft", "BEVgt") for m in bundle.modalities)
    if needs_frontier and not has_frontier_map:
        raise ValueError("FRONTIER requested but no BEVft/BEVgt modality provided.")


def make_conversation(
    goal_object: str,
    bundle: InputBundle,
    spec: OutputSpec,
    *,
    step_action: Optional[Action] = None,
    local_frontiers: Optional[List[Tuple[int, int]]] = None,
    semantic_labels: Optional[List[List[str]]] = None,
    selected_frontier: Optional[Tuple[int, int]] = None,
    frontier_index: Optional[int] = None,
    eval_mode: bool = False,
    position_info: Optional[Dict] = None,
    action_history: Optional[List[str]] = None,
    include_heading_in_state: bool = False,
    use_candidate_semantics: bool = False,
    show_candidate_semantics: bool = True,
) -> List[Dict[str, str]]:
    _validate(bundle, spec)

    state_section = ""
    frontier_section = ""
    target_section = ""
    candidates_section = ""
    action_history_section = ""
    if action_history is not None and len(action_history) > 0:
        action_str = " -> ".join(action_history)
        action_history_section = f"\n<action_history> Turns at this position: {action_str}\n"

    if position_info is not None:
        agent_pos = position_info.get("agent_pos")
        agent_yaw_deg = position_info.get("agent_yaw_deg")
        frontier_positions = position_info.get("frontier_positions")
        target_position = position_info.get("target_position")

        _state_label = "State:" if use_candidate_semantics else "<state>"
        _state_end = " <e_s>" if use_candidate_semantics else ""
        if agent_pos is not None:
            r, c = int(round(agent_pos[0])), int(round(agent_pos[1]))
            if include_heading_in_state and agent_yaw_deg is not None:
                state_section = f"\n{_state_label} <s> pos=({r}, {c}) yaw_deg={float(agent_yaw_deg):.1f}{_state_end}\n"
            else:
                state_section = f"\n{_state_label} <s> pos=({r}, {c}){_state_end}\n"
        else:
            if include_heading_in_state and agent_yaw_deg is not None:
                state_section = f"\n{_state_label} <s> pos=(unknown) yaw_deg={float(agent_yaw_deg):.1f}{_state_end}\n"
            else:
                state_section = f"\n{_state_label} <s>{_state_end}\n"

        if frontier_positions is not None:
            if not use_candidate_semantics:
                parts = []
                for pos in frontier_positions:
                    r, c = int(round(pos[0])), int(round(pos[1]))
                    parts.append(f"<f> ({r}, {c})")
                frontier_section = f"\n<frontier> " + " ".join(parts) + "\n"

        if target_position is not None:
            if not use_candidate_semantics:
                r, c = int(round(target_position[0])), int(round(target_position[1]))
                target_parts = [f"({r}, {c})"]
                if "target_semantic" in position_info and position_info["target_semantic"] is not None:
                    target_semantic = position_info["target_semantic"]
                    if isinstance(target_semantic, list):
                        target_parts.append(f"target_semantic={target_semantic}")
                    else:
                        target_parts.append(f"target_semantic=[{target_semantic}]")
                else:
                    target_parts.append("goal object detected")
                target_section = f"\n<target> <t> " + ", ".join(target_parts) + "\n"

        if use_candidate_semantics:
            candidates_section = "\n" + _render_semantic_candidates(position_info, show_semantics=show_candidate_semantics)

    hide_coordinates = position_info is not None
    if use_candidate_semantics:
        decide_output = DECIDE_OUTPUT_CANDIDATE_ID_ONLY
    else:
        decide_output = _render_decide_and_output(spec, local_frontiers, semantic_labels, hide_coordinates=hide_coordinates)
    human_prompt = (
        _render_inputs_text(bundle)
        + GOAL_TPL.format(goal=goal_object)
        + action_history_section
        + state_section
        + candidates_section
        + frontier_section
        + target_section
        + decide_output
    )
    if eval_mode:
        return human_prompt
    gpt_reply = _format_reply(spec, step_action, frontier_index, selected_frontier)
    return [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": gpt_reply}]


def RGB_Seg__Action(goal_object: str, step_action: Action = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("RGB", "Seg"))
    spec = OutputSpec(("ACTION",), order="action_first")
    return make_conversation(goal_object, bundle, spec, step_action=step_action, frontier_index=None, eval_mode=eval_mode)


def Past5RGB_RGB__Action(goal_object: str, step_action: Action = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("Past5RGB", "RGB"))
    spec = OutputSpec(("ACTION",), order="action_first")
    return make_conversation(goal_object, bundle, spec, step_action=step_action, frontier_index=None, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg__Frontier_Action(goal_object: str, step_action: Action = None, frontier_index: Optional[int] = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("FRONTIER", "ACTION"), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, step_action=step_action, frontier_index=frontier_index, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg__Action(goal_object: str, step_action: Action = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("ACTION",), order="action_first")
    return make_conversation(goal_object, bundle, spec, step_action=step_action, frontier_index=None, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg__FRONTIER_PIXEL_ONLY(goal_object: str, local_frontiers, selected_frontier=None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("FRONTIER_PIXEL",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, selected_frontier=selected_frontier, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, eval_mode=eval_mode)


def BEVftFOV_RGB__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB"))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=semantic_labels, eval_mode=eval_mode)


def BEVftFOV_Sem__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV",))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=semantic_labels, eval_mode=eval_mode)


def BEVftFOV_RGB_Seg_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV", "RGB", "Seg"))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=semantic_labels, eval_mode=eval_mode, position_info=position_info, include_heading_in_state=True)


def BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, action_history: Optional[List[str]] = None) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV",))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=semantic_labels, eval_mode=eval_mode, position_info=position_info, action_history=action_history, include_heading_in_state=True, use_candidate_semantics=True)


def BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, action_history: Optional[List[str]] = None) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV",))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=None, eval_mode=eval_mode, position_info=position_info, action_history=action_history, include_heading_in_state=True, use_candidate_semantics=True, show_candidate_semantics=False)


def BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, action_history: Optional[List[str]] = None) -> List[Dict[str, str]]:
    bundle = InputBundle(("BEVftFOV",))
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    return make_conversation(goal_object, bundle, spec, local_frontiers=local_frontiers, frontier_index=frontier_index, semantic_labels=semantic_labels, eval_mode=eval_mode, position_info=position_info, action_history=action_history, include_heading_in_state=True, use_candidate_semantics=True)


def BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, num_frontier_images: int = 0) -> List[Dict[str, str]]:
    """PosA: explicit text coords + <e_s>/<e_cand> tokens; candidates from shuffled list when available."""
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    state_section = ""
    candidates_section = ""
    if position_info is not None:
        agent_pos = position_info.get("agent_pos")
        agent_yaw_deg = position_info.get("agent_yaw_deg")
        # State section with <e_s> closing token
        if agent_pos is not None:
            r, c = int(round(agent_pos[0])), int(round(agent_pos[1]))
            yaw_str = f" yaw_deg={agent_yaw_deg:.1f}" if agent_yaw_deg is not None else ""
            state_section = f"\n<state> <s> pos=({r}, {c}){yaw_str} <e_s>\n\n"
        else:
            state_section = "\n<state> <s> <e_s>\n\n"
        # Candidates: use shuffled list from position_info['candidates'] if available (training),
        # otherwise build from frontier_positions + target_position (eval fallback)
        if "candidates" in position_info and position_info["candidates"]:
            cands = position_info["candidates"]
            if num_frontier_images > 0:
                lines = []
                for cand in cands:
                    r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
                    ctype = cand["type"]
                    view_label = "Target view" if ctype == "target" else "Frontier view"
                    lines.append(f"<cand> id={cand['id']} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
            else:
                lines = []
                for cand in cands:
                    r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
                    ctype = cand["type"]
                    lines.append(f"<cand> id={cand['id']} type={ctype} pos=({r}, {c}) <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
        else:
            # Eval fallback: build unshuffled list from frontier_positions + target_position
            frontier_positions = position_info.get("frontier_positions", [])
            target_position = position_info.get("target_position")
            candidate_positions = [[float(p[0]), float(p[1])] for p in frontier_positions]
            candidate_types = ["frontier"] * len(candidate_positions)
            if target_position is not None:
                candidate_positions.append([float(target_position[0]), float(target_position[1])])
                candidate_types.append("target")
            if candidate_positions:
                if num_frontier_images > 0:
                    lines = []
                    for i, p in enumerate(candidate_positions):
                        r, c = int(round(p[0])), int(round(p[1]))
                        ctype = candidate_types[i]
                        view_label = "Target view" if ctype == "target" else "Frontier view"
                        lines.append(f"<cand> id={i} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
                else:
                    lines = []
                    for i, p in enumerate(candidate_positions):
                        r, c = int(round(p[0])), int(round(p[1]))
                        ctype = candidate_types[i]
                        lines.append(f"<cand> id={i} type={ctype} pos=({r}, {c}) <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
    inputs_text = INTRO + BEV_FT_FOV_CANDIDATE + "\n"
    decide_output = DECIDE_OUTPUT_CANDIDATE_ID_ONLY
    human_prompt = inputs_text + GOAL_TPL.format(goal=goal_object) + state_section + candidates_section + decide_output
    if eval_mode:
        return human_prompt
    gpt_reply = _format_reply(spec, None, frontier_index, None)
    return [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": gpt_reply}]


def BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, num_frontier_images: int = 0) -> List[Dict[str, str]]:
    """PosB: NO text coords in state (embeddings only); <e_s>/<e_cand> tokens; candidates from shuffled list when available."""
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    state_section = ""
    candidates_section = ""
    if position_info is not None:
        # PosB: no text coordinates — position info conveyed via embeddings only
        state_section = "\n<state> <s> <e_s>\n\n"
        # Candidates: use shuffled list from position_info['candidates'] if available (training),
        # otherwise build from frontier_positions + target_position (eval fallback)
        if "candidates" in position_info and position_info["candidates"]:
            cands = position_info["candidates"]
            if num_frontier_images > 0:
                lines = []
                for cand in cands:
                    ctype = cand["type"]
                    view_label = "Target view" if ctype == "target" else "Frontier view"
                    lines.append(f"<cand> id={cand['id']} type={ctype} {view_label} <image_ego> <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
            else:
                lines = []
                for cand in cands:
                    ctype = cand["type"]
                    lines.append(f"<cand> id={cand['id']} type={ctype} <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
        else:
            # Eval fallback: build unshuffled list from frontier_positions + target_position
            frontier_positions = position_info.get("frontier_positions", [])
            target_position = position_info.get("target_position")
            candidate_types = ["frontier"] * len(frontier_positions)
            if target_position is not None:
                candidate_types.append("target")
            if candidate_types:
                if num_frontier_images > 0:
                    lines = []
                    for i, ctype in enumerate(candidate_types):
                        view_label = "Target view" if ctype == "target" else "Frontier view"
                        lines.append(f"<cand> id={i} type={ctype} {view_label} <image_ego> <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
                else:
                    lines = []
                    for i, ctype in enumerate(candidate_types):
                        lines.append(f"<cand> id={i} type={ctype} <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
    inputs_text = INTRO + BEV_FT_FOV_CANDIDATE + "\n"
    decide_output = DECIDE_OUTPUT_CANDIDATE_ID_ONLY
    human_prompt = inputs_text + GOAL_TPL.format(goal=goal_object) + state_section + candidates_section + decide_output
    if eval_mode:
        return human_prompt
    gpt_reply = _format_reply(spec, None, frontier_index, None)
    return [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": gpt_reply}]


def BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, num_frontier_images: int = 0) -> List[Dict[str, str]]:
    """PosC: explicit text coords (same as PosA); model uses PairwiseSpatialEncoder for <cand> injection.
    Standalone function — does NOT delegate to PosB."""
    spec = OutputSpec(("FRONTIER_PIXEL_NUMBER",), order="frontier_first")
    state_section = ""
    candidates_section = ""
    if position_info is not None:
        agent_pos = position_info.get("agent_pos")
        agent_yaw_deg = position_info.get("agent_yaw_deg")
        # State section with <e_s> closing token (same as PosA)
        if agent_pos is not None:
            r, c = int(round(agent_pos[0])), int(round(agent_pos[1]))
            yaw_str = f" yaw_deg={agent_yaw_deg:.1f}" if agent_yaw_deg is not None else ""
            state_section = f"\n<state> <s> pos=({r}, {c}){yaw_str} <e_s>\n\n"
        else:
            state_section = "\n<state> <s> <e_s>\n\n"
        # Candidates: use shuffled list from position_info['candidates'] if available (training),
        # otherwise build from frontier_positions + target_position (eval fallback)
        if "candidates" in position_info and position_info["candidates"]:
            cands = position_info["candidates"]
            if num_frontier_images > 0:
                lines = []
                for cand in cands:
                    r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
                    ctype = cand["type"]
                    view_label = "Target view" if ctype == "target" else "Frontier view"
                    lines.append(f"<cand> id={cand['id']} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
            else:
                lines = []
                for cand in cands:
                    r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
                    ctype = cand["type"]
                    lines.append(f"<cand> id={cand['id']} type={ctype} pos=({r}, {c}) <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
        else:
            # Eval fallback: build unshuffled list from frontier_positions + target_position
            frontier_positions = position_info.get("frontier_positions", [])
            target_position = position_info.get("target_position")
            candidate_positions = [[float(p[0]), float(p[1])] for p in frontier_positions]
            candidate_types = ["frontier"] * len(candidate_positions)
            if target_position is not None:
                candidate_positions.append([float(target_position[0]), float(target_position[1])])
                candidate_types.append("target")
            if candidate_positions:
                if num_frontier_images > 0:
                    lines = []
                    for i, p in enumerate(candidate_positions):
                        r, c = int(round(p[0])), int(round(p[1]))
                        ctype = candidate_types[i]
                        view_label = "Target view" if ctype == "target" else "Frontier view"
                        lines.append(f"<cand> id={i} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
                else:
                    lines = []
                    for i, p in enumerate(candidate_positions):
                        r, c = int(round(p[0])), int(round(p[1]))
                        ctype = candidate_types[i]
                        lines.append(f"<cand> id={i} type={ctype} pos=({r}, {c}) <e_cand>")
                    candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
    inputs_text = INTRO + BEV_FT_FOV_CANDIDATE + "\n"
    decide_output = DECIDE_OUTPUT_CANDIDATE_ID_ONLY
    human_prompt = inputs_text + GOAL_TPL.format(goal=goal_object) + state_section + candidates_section + decide_output
    if eval_mode:
        return human_prompt
    gpt_reply = _format_reply(spec, None, frontier_index, None)
    return [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": gpt_reply}]


def BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY(goal_object: str, local_frontiers, frontier_index: Optional[int] = None, semantic_labels: Optional[List[List[str]]] = None, eval_mode: bool = False, position_info: Optional[Dict] = None, num_frontier_images: int = 0) -> List[Dict[str, str]]:
    """PosD: same architecture as PosC, but output symbol is candidate token <id_k>."""
    state_section = ""
    candidates_section = ""
    if position_info is not None:
        agent_pos = position_info.get("agent_pos")
        agent_yaw_deg = position_info.get("agent_yaw_deg")
        if agent_pos is not None:
            r, c = int(round(agent_pos[0])), int(round(agent_pos[1]))
            yaw_str = f" yaw_deg={agent_yaw_deg:.1f}" if agent_yaw_deg is not None else ""
            state_section = f"\n<state> <s> pos=({r}, {c}){yaw_str} <e_s>\n\n"
        else:
            state_section = "\n<state> <s> <e_s>\n\n"

        if "candidates" in position_info and position_info["candidates"]:
            cands = position_info["candidates"]
            lines = []
            for cand in cands:
                cid = int(cand["id"])
                id_tok = f"<id_{cid}>"
                ctype = cand["type"]
                r, c = int(round(cand["pos"][0])), int(round(cand["pos"][1]))
                if num_frontier_images > 0:
                    view_label = "Target view" if ctype == "target" else "Frontier view"
                    lines.append(f"<cand> id_token={id_tok} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                else:
                    lines.append(f"<cand> id_token={id_tok} type={ctype} pos=({r}, {c}) <e_cand>")
            candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
        else:
            frontier_positions = position_info.get("frontier_positions", [])
            target_position = position_info.get("target_position")
            candidate_positions = [[float(p[0]), float(p[1])] for p in frontier_positions]
            candidate_types = ["frontier"] * len(candidate_positions)
            if target_position is not None:
                candidate_positions.append([float(target_position[0]), float(target_position[1])])
                candidate_types.append("target")
            if candidate_positions:
                lines = []
                for i, p in enumerate(candidate_positions):
                    id_tok = f"<id_{i}>"
                    r, c = int(round(p[0])), int(round(p[1]))
                    ctype = candidate_types[i]
                    if num_frontier_images > 0:
                        view_label = "Target view" if ctype == "target" else "Frontier view"
                        lines.append(f"<cand> id_token={id_tok} type={ctype} pos=({r}, {c}) {view_label} <image_ego> <e_cand>")
                    else:
                        lines.append(f"<cand> id_token={id_tok} type={ctype} pos=({r}, {c}) <e_cand>")
                candidates_section = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"

    inputs_text = INTRO + BEV_FT_FOV_CANDIDATE + "\n"
    human_prompt = inputs_text + GOAL_TPL.format(goal=goal_object) + state_section + candidates_section + DECIDE_OUTPUT_CANDIDATE_TOKEN_ONLY
    if eval_mode:
        return human_prompt
    target_id = 0 if frontier_index is None else int(frontier_index)
    gpt_reply = f"<id_{target_id}>"
    return [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": gpt_reply}]


__all__ = [
    "RGB_Seg__Action",
    "Past5RGB_RGB__Action",
    "BEVftFOV_RGB_Seg__Frontier_Action",
    "BEVftFOV_RGB_Seg__Action",
    "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_ONLY",
    "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_RGB__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_Sem__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_RGB_Seg_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY",
    "BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY",
]