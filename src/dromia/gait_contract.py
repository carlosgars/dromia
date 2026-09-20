"""Version and column order for the public biomechanics contract."""

from typing import Any

SCHEMA_VERSION = 13
EVENT_COLUMNS = (
    "runner_id",
    "event_id",
    "side",
    "source_fps",
    "real_world_fps",
    "landing_frame",
    "knee_alignment_frame",
    "takeoff_frame",
    "contact_frames",
    "contact_time_ms",
    "flexion_braking_time_ms",
    "impulse_propulsion_time_ms",
    "global_flight_time_ms",
    "same_foot_flight_time_ms",
    "step_length_cm",
    "stride_length_cm",
    "foot_strike",
    "landing_knee_angle_deg",
    "landing_tibia_horizontal_angle_deg",
    "landing_foot_tibia_angle_deg",
    "landing_torso_lean_deg",
    "takeoff_knee_angle_deg",
    "takeoff_tibia_horizontal_angle_deg",
    "takeoff_foot_tibia_angle_deg",
    "takeoff_torso_lean_deg",
)


def event_csv_row(event: dict[str, Any]) -> dict[str, Any]:
    """Put canonical units first; internal timing aliases remain in JSON/NPZ."""
    primary = {field: event.get(field) for field in EVENT_COLUMNS}
    secondary = {
        field: value
        for field, value in event.items()
        if field not in primary
        and not field.endswith("_s")
        and field not in {"braking_time_ms", "propulsion_time_ms", "strike_type"}
    }
    return {**primary, **secondary}
