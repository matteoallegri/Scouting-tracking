"""Configurazione esplicita del prototipo; nessun addestramento implicito."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

PHASES = {
    "all": "Tutto il gioco attivo osservato",
    "in_possession": "In possesso",
    "out_of_possession": "Non in possesso",
    "attacking_transition": "Transizione offensiva",
    "defensive_transition": "Transizione difensiva",
}
FAMILIES = {"motion": "Cinematica", "space": "Spazio", "transitions": "Transizioni"}
FEATURES = {
    "speed_p95": ("Velocità · 95° percentile", "m/s", "motion"),
    "high_speed_m_per_min": ("Distanza ad alta velocità", "m/min", "motion"),
    "acceleration_events_per_match": ("Episodi di accelerazione", "episodi/partita", "motion"),
    "braking_events_per_match": ("Episodi di frenata", "episodi/partita", "motion"),
    "relative_lateral_abs_m": ("Scostamento laterale dai compagni", "m", "space"),
    "relative_forward_m": ("Posizione avanzata rispetto ai compagni", "m", "space"),
    "hull_share_mean": ("Contributo geometrico al Convex Hull", "quota 0–1", "space"),
    "attack_progress_m_per_opportunity": ("Avanzamento nella finestra offensiva", "m/opportunità", "transitions"),
    "defense_retreat_m_per_opportunity": ("Arretramento nella finestra difensiva", "m/opportunità", "transitions"),
}


def features_for_phase(phase):
    return [key for key in FEATURES if not (
        key.startswith("attack_") and phase in ("out_of_possession", "defensive_transition")
        or key.startswith("defense_") and phase in ("in_possession", "attacking_transition")
    )]


@dataclass(frozen=True)
class ScoutingConfig:
    matches: tuple[str, ...] = ("wc2022", "wc2026")
    high_speed_threshold_m_s: float = 25.0 / 3.6
    acceleration_threshold_m_s2: float = 2.0
    braking_threshold_m_s2: float = 2.0
    episode_min_duration_s: float = 0.2
    episode_min_speed_m_s: float = 0.5
    transition_window_s: float = 5.0
    possession_min_duration_s: float = 0.2
    transition_min_opportunities: int = 3
    min_observed_minutes: float = 0.0
    min_transition_minutes: float = 0.0
    min_spatial_samples: int = 10
    spatial_step_frames: int = 25
    replay_step_frames: int = 5
    min_outfield_players: int = 8
    family_weights: dict = field(default_factory=lambda: {"motion": .4, "space": .35, "transitions": .25})
    thresholds_note: str = "Parametri operativi esplorativi, non standard ufficiali."

    def __post_init__(self):
        for name in ("high_speed_threshold_m_s", "acceleration_threshold_m_s2", "braking_threshold_m_s2",
                     "episode_min_duration_s", "transition_window_s", "possession_min_duration_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve essere positivo e finito")
        for name in ("min_observed_minutes", "min_transition_minutes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} deve essere non negativo e finito")
        if not math.isfinite(self.episode_min_speed_m_s) or self.episode_min_speed_m_s < 0:
            raise ValueError("episode_min_speed_m_s non valido")
        for name in ("transition_min_opportunities", "min_spatial_samples", "spatial_step_frames", "replay_step_frames", "min_outfield_players"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} deve essere un intero positivo")
        if self.min_outfield_players < 4:
            raise ValueError("Il contributo hull richiede almeno 4 giocatori di movimento")
        if self.spatial_step_frames % self.replay_step_frames:
            raise ValueError("spatial_step_frames deve essere un multiplo di replay_step_frames")
        if set(self.family_weights) != set(FAMILIES) or any(not math.isfinite(v) or v < 0 for v in self.family_weights.values()) or sum(self.family_weights.values()) <= 0:
            raise ValueError("Pesi delle famiglie non validi")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        data["matches"] = tuple(data.get("matches", ("wc2022", "wc2026")))
        return cls(**data)
