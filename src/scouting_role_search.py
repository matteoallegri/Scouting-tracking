"""Ruoli come perimetro flessibile di ricerca, senza rietichettare i giocatori.

Le associazioni sotto definiscono gruppi affini da esplorare, non nuovi ruoli
certificati o dedotti dai dati. Le scale usano sempre il gruppo esteso iniziale;
modificare i filtri dei candidati non cambia il metro comune di confronto.
"""

import hashlib
from pathlib import Path

from src.scouting_events.catalog import features_for_phase
from src.scouting_tracking.ranking import fit_scalers

DEFAULT_SEARCH_ROLES = {
    "Difensore centrale": ("Difensore centrale", "Esterno difensivo"),
    "Esterno difensivo": ("Esterno difensivo", "Difensore centrale", "Esterno offensivo"),
    "Centrocampista centrale": ("Centrocampista centrale", "Trequartista / seconda punta"),
    "Trequartista / seconda punta": ("Trequartista / seconda punta", "Esterno offensivo", "Attaccante centrale", "Centrocampista centrale"),
    "Esterno offensivo": ("Esterno offensivo", "Trequartista / seconda punta", "Attaccante centrale"),
    "Attaccante centrale": ("Attaccante centrale", "Trequartista / seconda punta", "Esterno offensivo"),
}


def policy_version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def role_mask(profiles, selected_roles):
    selected = set(selected_roles)
    primary = profiles.scouting_primary_role.isin(selected)
    secondary = profiles.scouting_secondary_roles.fillna("").map(
        lambda value: bool({item.strip() for item in value.split("|") if item.strip()} & selected))
    return primary | secondary


def comparison_view(profiles, reference_role):
    """Etichetta tecnica del confronto; identità, ruoli reali e misure intatti."""
    result = profiles.copy()
    result["role_group"] = reference_role
    return result


def fit_search_scalers(profiles, config):
    scalers = {}
    for role, related in DEFAULT_SEARCH_ROLES.items():
        group = comparison_view(profiles.loc[role_mask(profiles, related)], role)
        scalers.update(fit_scalers(group, config, phase_features=features_for_phase))
    return scalers
