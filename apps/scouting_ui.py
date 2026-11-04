"""Presentazione della dashboard scouting, separata dai calcoli e dai manifest."""

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PAGE = "#0F1420"
SURFACE = "#171E2E"
GRID = "#263042"
INK = "#E5E9F0"
MUTED = "#A6B2C7"
AMBER = "#F5A524"
MATCH_COLORS = {"wc2022": "#3987E5", "wc2026": "#E66767"}
COLORS = [*MATCH_COLORS.values(), "#199E70", "#9085E9"]


def inject_styles(root):
    assets = Path(root) / "assets"
    st.markdown("<style>" + (assets / "fonts.css").read_text()
                + (assets / "scouting.css").read_text() + "</style>", unsafe_allow_html=True)


def render_intro():
    st.markdown("""<div class="scouting-intro">
      <p>Scegli il profilo che cerchi, confronta i giocatori delle due finali e scopri quali comportamenti li avvicinano al tuo riferimento.</p>
    </div>""", unsafe_allow_html=True)


def render_reference(target, role, phase, *, heading="Riferimento del confronto"):
    st.markdown(f"""<div class="scouting-reference">
      <div><span class="scouting-label">{escape(heading)}</span><strong>{escape(target)}</strong></div>
      <div class="scouting-context"><span>{escape(role)}</span><span>{escape(phase)}</span></div>
    </div>""", unsafe_allow_html=True)


def style(figure, height=420):
    """Solo layout: nessuna trasformazione dei valori delle figure."""
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family="Fira Sans, Arial, sans-serif", color=INK, size=12),
        height=height, margin=dict(l=12, r=22, t=55, b=20),
        colorway=COLORS, separators=",.",
        legend=dict(orientation="h", y=1.14, x=0, font=dict(color=MUTED, size=11),
                    bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        hoverlabel=dict(bgcolor="#1E2740", bordercolor="#3B4C67", font=dict(family="Fira Sans, Arial, sans-serif", size=13, color=INK)),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=AMBER),
    )
    figure.update_xaxes(gridcolor=GRID, gridwidth=1, zerolinecolor="#46536B", linecolor=GRID,
                        tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12),
                        title_standoff=14, automargin=True)
    figure.update_yaxes(gridcolor=GRID, gridwidth=1, zerolinecolor="#46536B", linecolor=GRID,
                        tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12),
                        title_standoff=14, automargin=True)
    return figure


def compatibility_bars(ranked, *, score_label="Compatibilità"):
    """Stessi primi dodici punteggi, con identità cromatica stabile per finale."""
    data = ranked.dropna(subset=["score"]).head(12).iloc[::-1]
    labels = data.player_name + " · " + data.match_id.str.replace("wc", "", regex=False)
    figure = go.Figure(go.Bar(
        x=data.score.tolist(), y=labels.tolist(), orientation="h",
        marker=dict(color=[MATCH_COLORS.get(match, COLORS[0]) for match in data.match_id],
                    cornerradius=4),
        width=.58, text=data.score.round(1).tolist(), textposition="outside", cliponaxis=False,
        textfont=dict(family="Fira Code, monospace", color=INK, size=12),
        hovertemplate="%{y}<br>" + score_label + ": %{x:.1f}<extra></extra>",
    ))
    title = "Compatibilità descrittiva" if score_label == "Compatibilità" else score_label
    figure.update_layout(showlegend=False, xaxis=dict(range=[0, 109], title=title + " · 0–100"), yaxis_title=None)
    style(figure, max(420, 31 * len(data) + 100))
    figure.update_yaxes(showgrid=False)
    figure.update_xaxes(dtick=25)
    figure.update_layout(margin=dict(t=24))
    return figure


def match_legend():
    st.markdown('<div class="scouting-legend"><span><i class="final-2022"></i>Finale 2022</span>'
                '<span><i class="final-2026"></i>Finale 2026</span></div>', unsafe_allow_html=True)


def shortlist_columns(*, score_label="Compatibilità"):
    description = ("Somiglianza al riferimento su scala 0–100. Non è una valutazione della qualità del giocatore."
                   if score_label == "Compatibilità" else "Media pesata dei percentili nelle misure e nei versi scelti, su scala 0–100.")
    return {
        "Giocatore": st.column_config.TextColumn("Giocatore", width="medium", pinned=True),
        score_label: st.column_config.ProgressColumn(score_label, min_value=0, max_value=100, format="%.1f", help=description),
        **{name: st.column_config.NumberColumn(name, format="%.1f", help="Punteggio nella dimensione, su scala 0–100.")
           for name in ("Cinematica", "Spazio", "Transizioni", "Distribuzione e progressione",
                        "Creazione offensiva", "Recupero e opposizione", "Protezione della porta")},
        "Valore di mercato (M€)": st.column_config.NumberColumn("Valore di mercato (M€)", format="%.1f"),
    }
