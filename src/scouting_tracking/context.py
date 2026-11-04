"""Possesso e finestre di transizione: nessun attraversamento di pause o ignoti."""

import numpy as np
import pandas as pd


def build_context(ball, team_ids, config):
    df = ball.sort_values(["period", "frame"]).reset_index(drop=True).copy()
    if df.frame.duplicated().any():
        raise ValueError("Il dataset del progetto richiede frame globali univoci")
    df["live_observed"] = np.isfinite(df[["x", "y", "video_ms"]]).all(axis=1)
    for flag in ("is_interpolated", "jump_anomaly", "speed_anomaly", "out_of_bounds"):
        if flag in df:
            df["live_observed"] &= ~df[flag].astype(bool)
    dt = df.video_ms.diff() / 1000
    same_period = df.period.eq(df.period.shift())
    continuous = same_period & df.frame.diff().eq(1) & dt.gt(0) & dt.le(.2)
    continuous &= df.live_observed & df.live_observed.shift(fill_value=False)
    df["live_interval"] = continuous
    df["live_segment_id"] = (~continuous).cumsum().astype(int)
    df["context_dt_s"] = dt.where(continuous)
    typical_dt = df.context_dt_s.median()
    if np.isfinite(typical_dt) and not np.isclose(config.transition_window_s / typical_dt,
                                                 round(config.transition_window_s / typical_dt), atol=1e-7, rtol=0):
        raise ValueError("La finestra di transizione deve essere un multiplo del passo temporale osservato")
    df["possession_known"] = df.live_observed & df.team_in_poss.isin(team_ids)
    owner = df.team_in_poss.where(df.possession_known, "")
    run_start = ~continuous | owner.ne(owner.shift()) | ~df.possession_known
    df["possession_run_id"] = run_start.cumsum()
    df["transition_id"] = -1
    df["gaining_team"] = ""
    df["losing_team"] = ""
    df["transition_elapsed_s"] = np.nan
    run_data = []
    for _, indices in df.groupby("possession_run_id", sort=False).indices.items():
        start, end = indices[0], indices[-1]
        if not df.possession_known.iloc[start]:
            run_data.append(None)
            continue
        duration = (df.video_ms.iloc[end] - df.video_ms.iloc[start]) / 1000
        run_data.append((start, end, str(owner.iloc[start]), duration, int(df.live_segment_id.iloc[start])))
    opportunities = []
    transition_id = 0
    for previous, current in zip(run_data, run_data[1:]):
        if previous is None or current is None:
            continue
        pstart, pend, old_owner, pduration, psegment = previous
        start, end, new_owner, duration, segment = current
        if old_owner == new_owner or segment != psegment or start != pend + 1:
            continue
        if min(duration, pduration) + 1e-10 < config.possession_min_duration_s:
            continue
        elapsed = (df.loc[start:end, "video_ms"] - df.video_ms.iloc[start]) / 1000
        # vx(t) descrive l'intervallo (t-dt, t]: t=0 includerebbe il tratto
        # precedente al recupero. Le finestre usano quindi 0 < t <= window.
        selected = elapsed.index[(elapsed > 1e-10) & (elapsed <= config.transition_window_s + 1e-10)]
        if len(selected) == 0:
            continue
        df.loc[selected, "transition_id"] = transition_id
        df.loc[selected, "gaining_team"] = new_owner
        df.loc[selected, "losing_team"] = old_owner
        df.loc[selected, "transition_elapsed_s"] = elapsed.loc[selected]
        opportunities.append({"transition_id": transition_id, "period_id": int(df.period.iloc[start]),
                              "start_frame": int(df.frame.iloc[start]), "end_frame": int(df.frame.loc[selected].iloc[-1]),
                              "gaining_team": new_owner, "losing_team": old_owner,
                              "live_segment_id": segment, "available_s": float(elapsed.loc[selected].iloc[-1]),
                              "window_s": config.transition_window_s})
        transition_id += 1
    keep = ["frame", "period", "video_ms", "team_in_poss", "live_observed", "live_interval",
            "live_segment_id", "context_dt_s", "possession_known", "possession_run_id",
            "transition_id", "gaining_team", "losing_team", "transition_elapsed_s"]
    return df[keep], pd.DataFrame(opportunities)


def attach_context(metrics, context):
    frame_context = context.set_index("frame").reindex(metrics.frame_index)
    frame_context.index = metrics.index
    df = metrics.copy()
    for column in frame_context:
        df[column] = frame_context[column]
    df["phase_base"] = "unknown"
    known = df.possession_known & df.is_valid
    df.loc[known & df.team_id.eq(df.team_in_poss), "phase_base"] = "in_possession"
    df.loc[known & ~df.team_id.eq(df.team_in_poss), "phase_base"] = "out_of_possession"
    df["phase_detail"] = df.phase_base
    df.loc[known & df.transition_id.ge(0) & df.team_id.eq(df.gaining_team), "phase_detail"] = "attacking_transition"
    df.loc[known & df.transition_id.ge(0) & df.team_id.eq(df.losing_team), "phase_detail"] = "defensive_transition"
    df["analysis_interval"] = df.is_valid & df.valid_interval & df.live_interval
    return df


def phase_mask(df, phase):
    if phase == "all":
        return pd.Series(True, index=df.index)
    return df.phase_base.eq(phase) if phase in ("in_possession", "out_of_possession") else df.phase_detail.eq(phase)
