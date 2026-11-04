"""Figure locali per l'esplorazione dei profili e delle sequenze."""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, QhullError

from .config import FEATURES, FAMILIES

COLORS = ["#61DDB1", "#F2A17B", "#7DA6FF", "#D5ADFF"]
BACKGROUND = "#101824"


def style(figure, height=400):
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Arial, sans-serif", color="#DCE6EE"), height=height,
                         margin=dict(l=20, r=20, t=35, b=30), legend=dict(orientation="h", y=1.12))
    figure.update_xaxes(gridcolor="#243143", zerolinecolor="#3C4A5C")
    figure.update_yaxes(gridcolor="#243143", zerolinecolor="#3C4A5C")
    return figure


def landscape(profiles, target_id=None):
    fig = go.Figure()
    for index, (match, group) in enumerate(profiles.groupby("match_id", sort=True)):
        hover = group.player_name + " · " + group.team_name
        fig.add_trace(go.Scatter(x=group.relative_lateral_abs_m, y=group.high_speed_m_per_min,
                                mode="markers", name=match.upper(), text=hover,
                                customdata=np.column_stack([group.observed_minutes, group.score]),
                                marker=dict(size=np.clip(np.sqrt(group.observed_minutes) * 2.8, 7, 23), color=COLORS[index % 4], opacity=.8,
                                            line=dict(width=1, color="#E4F2EA")),
                                hovertemplate="%{text}<br>Scostamento laterale: %{x:.1f} m<br>HSR: %{y:.1f} m/min<br>Minuti fase: %{customdata[0]:.1f}<br>Compatibilità: %{customdata[1]:.1f}<extra></extra>"))
    if target_id is not None:
        row = profiles.loc[profiles.profile_id.eq(target_id)]
        if len(row):
            fig.add_trace(go.Scatter(x=row.relative_lateral_abs_m, y=row.high_speed_m_per_min, mode="markers",
                                    marker=dict(symbol="star", size=22, color="#F7E5A9", line=dict(width=1, color="#FFFFFF")), name="Riferimento"))
    fig.update_layout(xaxis_title="Scostamento laterale dai compagni (m)", yaxis_title="Distanza ad alta velocità (m/min)")
    return style(fig)


def compatibility_bars(ranked):
    data = ranked.dropna(subset=["score"]).head(12).iloc[::-1]
    labels = data.player_name + " · " + data.match_id.str.replace("wc", "", regex=False)
    fig = go.Figure(go.Bar(x=data.score, y=labels, orientation="h", marker_color=COLORS[0],
                           text=data.score.round(1), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Compatibilità: %{x:.1f}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 109], title="Compatibilità descrittiva · 0–100"), yaxis_title=None)
    return style(fig, max(260, 28 * len(data) + 60))


def mismatch_chart(details):
    data = details.sort_values("mismatch_share")
    fig = go.Figure(go.Bar(x=data.mismatch_share * 100, y=[FEATURES[k][0] for k in data.feature],
                           orientation="h", marker_color=[COLORS[list(FAMILIES).index(family)] for family in data.family],
                           hovertemplate="%{y}<br>Quota della distanza quadratica: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Quota della distanza quadratica (%)", yaxis_title=None)
    return style(fig, 330)


def replay_figure(samples, entity_id, teams, sign=1):
    """Campioni a 5 Hz: un solo periodo e segmento osservato per sequenza."""
    if samples.empty:
        return go.Figure()
    unique_frames = sorted(samples.frame_index.unique())
    team_ids = sorted(samples.loc[~samples.is_ball, "team_id"].dropna().unique())
    time_origin = float(samples.timestamp_ms.min())
    plotted_frames = []
    for frame in unique_frames:
        data = samples.loc[samples.frame_index.eq(frame)]
        traces = []
        for index, team in enumerate(team_ids):
            group = data.loc[data.team_id.eq(team) & ~data.is_ball]
            hover = group.get("player_name", group.entity_id)
            selected = group.entity_id.eq(entity_id)
            traces.append(go.Scatter(x=group.x_smooth * sign, y=group.y_smooth * sign, mode="markers",
                                     name=teams.get(team, team[:8]), text=hover,
                                     marker=dict(color=COLORS[index % 4], size=np.where(selected, 19, 10),
                                                 symbol=np.where(group.is_goalkeeper, "triangle-up", np.where(selected, "diamond", "circle")),
                                                 line=dict(width=np.where(selected, 3, 1), color="white")),
                                     hovertemplate="%{text}<extra></extra>"))
        ball = data.loc[data.is_ball]
        traces.append(go.Scatter(x=ball.x_smooth * sign, y=ball.y_smooth * sign, mode="markers", name="Palla",
                                 marker=dict(size=8, color="#FAF7E8", line=dict(color="#101824", width=1)), hoverinfo="skip"))
        for index, team in enumerate(team_ids):
            points = data.loc[data.team_id.eq(team) & ~data.is_ball & ~data.is_goalkeeper, ["x_smooth", "y_smooth"]].to_numpy() * sign
            hull_x, hull_y = [], []
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    vertices = np.r_[hull.vertices, hull.vertices[0]]
                    hull_x, hull_y = points[vertices, 0], points[vertices, 1]
                except QhullError:
                    pass
            traces.append(go.Scatter(x=hull_x, y=hull_y, mode="lines", fill="toself", opacity=.2,
                                     line=dict(color=COLORS[index % 4], width=2), showlegend=False, hoverinfo="skip"))
            mean = points.mean(axis=0) if len(points) else [np.nan, np.nan]
            traces.append(go.Scatter(x=[mean[0]], y=[mean[1]], mode="markers", marker=dict(symbol="x", size=12, color=COLORS[index % 4]),
                                     showlegend=False, hoverinfo="skip"))
        trail = samples.loc[samples.entity_id.eq(entity_id) & samples.frame_index.le(frame) & samples.frame_index.ge(frame - 50)]
        traces.append(go.Scatter(x=trail.x_smooth * sign, y=trail.y_smooth * sign, mode="lines",
                                 line=dict(color="#F7E5A9", width=3), showlegend=False, hoverinfo="skip"))
        elapsed = (float(data.timestamp_ms.iloc[0]) - time_origin) / 1000
        plotted_frames.append(go.Frame(data=traces, name=str(frame), layout=go.Layout(title_text=f"+{elapsed:.2f} s · frame {frame}")))
    fig = go.Figure(data=plotted_frames[0].data, frames=plotted_frames)
    fig.update_layout(
        shapes=[dict(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34, line=dict(color="#81958E", width=2), fillcolor="#173B32", layer="below"),
                dict(type="line", x0=0, x1=0, y0=-34, y1=34, line=dict(color="#81958E")),
                dict(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15, line=dict(color="#81958E"))],
        xaxis=dict(range=[-56, 56], title="Attacco del giocatore selezionato →", showgrid=False, zeroline=False),
        yaxis=dict(range=[-37, 37], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-.08, buttons=[
            dict(label="▶ Riproduci", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]),
            dict(label="Ⅱ Pausa", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
        sliders=[dict(active=0, x=.28, len=.72, y=-.04, currentvalue=dict(prefix="Frame "),
                      steps=[dict(label=str(frame), method="animate", args=[[str(frame)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]) for frame in unique_frames])],
    )
    return style(fig, 520)
