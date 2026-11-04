"""Descrittori per fase, episodi continui e contributi geometrici individuali."""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from .config import PHASES
from .context import phase_mask


EPISODE_COLUMNS = ["entity_id", "role_group", "period_id", "kind", "start_frame", "end_frame",
                   "duration_s", "peak", "phase_base", "phase_detail", "live_segment_id"]
TRANSITION_COLUMNS = ["entity_id", "role_group", "period_id", "kind", "transition_id",
                      "start_frame", "end_frame", "duration_s", "progress_m", "live_segment_id"]


def detect_episodes(df, config):
    """Ogni episodio è contato una volta, nella fase del suo inizio."""
    df = df.sort_values("frame_index").reset_index(drop=True)
    n = len(df)
    if not n:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    boundary = np.ones(n, dtype=bool)
    boundary[1:] = (np.diff(df.frame_index) != 1) | (np.diff(df.period_id) != 0)
    for key in ("segment_id", "role_window_id", "live_segment_id"):
        boundary[1:] |= df[key].to_numpy()[1:] != df[key].to_numpy()[:-1]
    eligible = df.analysis_interval.to_numpy() & df.speed.ge(config.episode_min_speed_m_s).to_numpy()
    records = []
    for kind, condition in (("acceleration", df.tangential_acceleration.ge(config.acceleration_threshold_m_s2)),
                            ("braking", df.tangential_acceleration.le(-config.braking_threshold_m_s2))):
        active = eligible & condition.to_numpy()
        starts = np.flatnonzero(active & (boundary | ~np.r_[False, active[:-1]]))
        ends = np.flatnonzero(active & (np.r_[boundary[1:], True] | ~np.r_[active[1:], False])) + 1
        for start, end in zip(starts, ends):
            duration = float(df.delta_t_s.iloc[start:end].sum())
            if duration + 1e-10 < config.episode_min_duration_s:
                continue
            row = df.iloc[start]
            tangent = df.tangential_acceleration.iloc[start:end]
            records.append({"entity_id": row.entity_id, "role_group": row.role_group,
                            "period_id": int(row.period_id), "kind": kind,
                            "start_frame": int(row.frame_index), "end_frame": int(df.frame_index.iloc[end - 1]),
                            "duration_s": duration, "peak": float(tangent.max() if kind == "acceleration" else tangent.min()),
                            "phase_base": row.phase_base, "phase_detail": row.phase_detail,
                            "live_segment_id": int(row.live_segment_id)})
    return pd.DataFrame(records, columns=EPISODE_COLUMNS)


def summarize_transitions(df, config):
    records = []
    by_frame = df.set_index("frame_index", drop=False)
    selected = df.loc[df.analysis_interval & df.transition_id.ge(0)]
    for (tid, window), group in selected.groupby(["transition_id", "role_window_id"], sort=False):
        group = group.sort_values("frame_index")
        elapsed = group.transition_elapsed_s.to_numpy()
        dt = group.delta_t_s.to_numpy()
        exposure = float(dt.sum())
        # Richiede la finestra intera, medesimo ruolo e nessun frame intermedio perso.
        if not np.isclose(elapsed[0], dt[0], atol=1e-8, rtol=0) or not np.isclose(elapsed[-1], config.transition_window_s, atol=1e-8, rtol=0):
            continue
        if not np.isclose(exposure, config.transition_window_s, atol=1e-8, rtol=0) or not np.all(np.diff(group.frame_index) == 1):
            continue
        if group.live_segment_id.nunique() != 1 or group.segment_id.nunique() != 1:
            continue
        row = group.iloc[0]
        onset_frame = int(row.frame_index) - 1
        if onset_frame not in by_frame.index:
            continue
        onset = by_frame.loc[onset_frame]
        if any(onset[key] != row[key] for key in ("period_id", "role_window_id", "segment_id", "live_segment_id")):
            continue
        kind = "attack" if row.phase_detail == "attacking_transition" else "defense" if row.phase_detail == "defensive_transition" else None
        if kind is None:
            continue
        progress = float((group.vx * group.delta_t_s * group.attack_sign).sum())
        records.append({"entity_id": row.entity_id, "role_group": row.role_group,
                        "period_id": int(row.period_id), "kind": kind, "transition_id": int(tid),
                        "start_frame": onset_frame, "end_frame": int(group.frame_index.iloc[-1]),
                        "duration_s": exposure, "progress_m": progress,
                        "live_segment_id": int(row.live_segment_id)})
    return pd.DataFrame(records, columns=TRANSITION_COLUMNS)


def hull_area(xy):
    if len(np.unique(xy, axis=0)) < 3:
        return np.nan
    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return np.nan


def add_spatial_features(samples, config):
    """Campionamento esplicito; portieri esclusi da tutti i riferimenti geometrici."""
    players = samples.loc[~samples.is_ball & ~samples.is_goalkeeper].copy().reset_index(drop=True)
    for key in ("relative_lateral_abs_m", "relative_forward_m", "hull_share", "team_hull_area",
                "centroid_x", "centroid_y", "valid_outfield_count"):
        players[key] = np.nan
    for _, indices in players.groupby(["team_id", "period_id", "frame_index"], sort=False).indices.items():
        count = len(indices)
        players.loc[indices, "valid_outfield_count"] = count
        if count < config.min_outfield_players:
            continue
        group = players.iloc[indices]
        xy = group[["x_smooth", "y_smooth"]].to_numpy()
        signs = group.attack_sign.to_numpy()
        other_centroid = (xy.sum(axis=0) - xy) / (count - 1)
        area = hull_area(xy)
        players.loc[indices, "relative_forward_m"] = (xy[:, 0] - other_centroid[:, 0]) * signs
        players.loc[indices, "relative_lateral_abs_m"] = np.abs(xy[:, 1] - other_centroid[:, 1])
        players.loc[indices, "centroid_x"] = xy[:, 0].mean()
        players.loc[indices, "centroid_y"] = xy[:, 1].mean()
        players.loc[indices, "team_hull_area"] = area
        if not np.isfinite(area) or area <= 1e-12:
            continue
        for offset, index in enumerate(indices):
            reduced = hull_area(np.delete(xy, offset, axis=0))
            if np.isfinite(reduced):
                share = (area - reduced) / area
                if not -1e-9 <= share <= 1 + 1e-9:
                    raise ValueError("Contributo Convex Hull fuori dal dominio geometrico")
                players.loc[index, "hull_share"] = np.clip(share, 0, 1)
    return players


def build_player_profiles(df, spatial, episodes, transitions, config):
    """Profili di un solo giocatore-partita, complessivi e per periodo."""
    match_count = int(df.match_id.nunique())
    if match_count != 1 or df.match_id.isna().any() or df.entity_id.nunique(dropna=False) != 1:
        raise ValueError("Ogni profilo deve riferirsi a un solo giocatore e una sola partita")
    records = []
    periods = [0] + sorted(df.period_id.unique().astype(int).tolist())
    for role, role_data in df.groupby("role_group", sort=False):
        role_spatial = spatial.loc[spatial.role_group.eq(role)]
        role_episodes = episodes.loc[episodes.role_group.eq(role)]
        role_transitions = transitions.loc[transitions.role_group.eq(role)]
        for period in periods:
            data = role_data if period == 0 else role_data.loc[role_data.period_id.eq(period)]
            space = role_spatial if period == 0 else role_spatial.loc[role_spatial.period_id.eq(period)]
            events = role_episodes if period == 0 else role_episodes.loc[role_episodes.period_id.eq(period)]
            opportunities = role_transitions if period == 0 else role_transitions.loc[role_transitions.period_id.eq(period)]
            for phase in PHASES:
                selected = data.loc[data.analysis_interval & phase_mask(data, phase)]
                duration = float(selected.delta_t_s.sum())
                if duration <= 0:
                    continue
                row = selected.iloc[0]
                minutes = duration / 60
                e = events.loc[phase_mask(events, phase)]
                acceleration_count = int(e.kind.eq("acceleration").sum())
                braking_count = int(e.kind.eq("braking").sum())
                high_speed = float(selected.loc[selected.speed.ge(config.high_speed_threshold_m_s), "delta_distance"].sum())
                sp = space.loc[phase_mask(space, phase)]
                attack = opportunities.loc[opportunities.kind.eq("attack")] if phase not in ("out_of_possession", "defensive_transition") else opportunities.iloc[:0]
                defense = opportunities.loc[opportunities.kind.eq("defense")] if phase not in ("in_possession", "attacking_transition") else opportunities.iloc[:0]
                record = {
                    "entity_id": row.entity_id, "role_group": role, "period_id": period, "phase": phase,
                    "role_verified": bool(selected.role_verified.all()),
                    "role_usable": bool(selected.role_usable.all()),
                    "role_documented_duration_s": float(selected.loc[selected.role_verified, "delta_t_s"].sum()),
                    "role_assigned_duration_s": float(selected.loc[~selected.role_verified & selected.role_usable, "delta_t_s"].sum()),
                    "role_assigned_minutes": float(selected.loc[~selected.role_verified & selected.role_usable, "delta_t_s"].sum()) / 60,
                    "role_assignment_sources": " | ".join(sorted(set(selected.role_assignment_basis))),
                    "context_role_groups": " | ".join(sorted(set(selected.context_role_group))),
                    "role_details": " | ".join(sorted(set(selected.position + " — " + selected.position_side))),
                    "position_sides": " | ".join(sorted(set(selected.position_side) - {""})),
                    "observed_duration_s": duration, "observed_minutes": minutes, "valid_intervals": len(selected),
                    "match_count": match_count,
                    "known_possession_fraction": float(selected.phase_base.ne("unknown").mean()),
                    "total_distance_m": float(selected.delta_distance.sum()), "speed_p95": float(selected.speed.quantile(.95)),
                    "speed_max": float(selected.speed.max()), "high_speed_distance_m": high_speed,
                    "high_speed_m_per_min": high_speed / minutes,
                    "acceleration_count": acceleration_count, "braking_count": braking_count,
                    "acceleration_events_per_match": acceleration_count / match_count,
                    "braking_events_per_match": braking_count / match_count,
                    "spatial_samples": len(sp),
                    "mean_valid_outfield_count": float(sp.valid_outfield_count.mean()) if len(sp) else np.nan,
                    "attack_opportunities": len(attack), "defense_opportunities": len(defense),
                    "attack_progress_m_per_opportunity": float(attack.progress_m.mean()) if len(attack) >= config.transition_min_opportunities else np.nan,
                    "defense_retreat_m_per_opportunity": -float(defense.progress_m.mean()) if len(defense) >= config.transition_min_opportunities else np.nan,
                }
                for output, source in (("relative_lateral_abs_m", "relative_lateral_abs_m"),
                                       ("relative_forward_m", "relative_forward_m"), ("hull_share_mean", "hull_share")):
                    values = sp[source].dropna()
                    record[output] = float(values.mean()) if len(values) >= config.min_spatial_samples else np.nan
                records.append(record)
    return pd.DataFrame(records)
