"""Indice descrittivo di indicatori orientati, su una popolazione fissa.

Ogni percentile usa ranghi medi riscalati: 100*(L+E/2-1/2)/(n-1), dove L è
il numero di valori strettamente inferiori ed E quello dei valori uguali.
Il risultato è limitato a [0,100]; per il verso decrescente si usa 100-p.
Non è una stima del talento né un confronto con un giocatore ideale.
"""

import numpy as np
import pandas as pd

from src.scouting_events.catalog import FEATURES, FAMILIES

DEFAULT_DIRECTIONS = {key: 0 if family == "space" else 1 for key, (_, _, family) in FEATURES.items()}
DIRECTION_LABELS = {1: "Premia valori più alti", -1: "Premia valori più bassi", 0: "Non usare"}


def ranking_features(scaler, weights, directions):
    if set(weights) != set(FAMILIES) or any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Pesi delle dimensioni non validi")
    keys = []
    for key, item in scaler["features"].items():
        direction = directions.get(key, 0)
        if direction not in (-1, 0, 1):
            raise ValueError("Verso della misura non valido: " + key)
        if item["available"] and weights[FEATURES[key][2]] > 0 and direction != 0:
            keys.append(key)
    return keys


def percentiles(values, reference):
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) < 3 or not np.isfinite(reference).all() or reference[0] == reference[-1]:
        raise ValueError("Popolazione percentile insufficiente o costante")
    values = np.asarray(values, dtype=float)
    less = np.searchsorted(reference, values, side="left")
    equal = np.searchsorted(reference, values, side="right") - less
    result = np.clip(100 * (less + equal / 2 - .5) / (len(reference) - 1), 0, 100)
    return np.where(np.isfinite(values), result, np.nan)


def rank_profiles(profiles, population, scaler, weights, directions):
    if not profiles.role_group.eq(scaler["role_group"]).all() or not profiles.phase.eq(scaler["phase"]).all():
        raise ValueError("Scale e profili devono condividere gruppo e fase")
    keys = ranking_features(scaler, weights, directions)
    if not keys:
        raise ValueError("Nessuna misura variabile con peso e verso attivi")
    ids = scaler["profile_ids"]
    if population.profile_id.duplicated().any() or len(ids) != len(set(ids)):
        raise ValueError("Popolazione duplicata")
    index = population.set_index("profile_id")
    if not set(ids).issubset(index.index):
        raise ValueError("Popolazione di riferimento incompleta")
    training = index.loc[ids]
    if not training.phase.eq(scaler["phase"]).all() or not training.period_id.eq(0).all():
        raise ValueError("Popolazione di un'altra fase o periodo")
    result = profiles.copy()
    result["score"] = np.nan
    result["distance"] = np.nan
    result["score_status"] = "Disponibile"
    complete = np.isfinite(result[keys].to_numpy(dtype=float)).all(axis=1)
    result.loc[~complete, "score_status"] = "Caratteristiche insufficienti"
    result.loc[result.observed_minutes.lt(0), "score_status"] = "Esposizione insufficiente"
    usable = result["role_usable"] if "role_usable" in result else result["role_verified"]
    result.loc[~usable, "score_status"] = "Ruolo non assegnabile"
    eligible = result.score_status.eq("Disponibile")
    families = {family: [key for key in keys if FEATURES[key][2] == family] for family in FAMILIES}
    denominator = sum(weights[family] for family, columns in families.items() if columns)
    total = pd.Series(0., index=result.index)
    details = []
    for family, columns in families.items():
        result[family + "_score"] = np.nan
        if not columns:
            continue
        points = pd.Series(0., index=result.index)
        for key in columns:
            reference = training[key].to_numpy(dtype=float)
            reference = reference[np.isfinite(reference)]
            if len(reference) != scaler["features"][key]["n"]:
                raise ValueError("Conteggio della popolazione non coerente: " + key)
            pct = percentiles(result[key], reference)
            if directions[key] == -1:
                pct = 100 - pct
            contribution = pct * weights[family] / denominator / len(columns)
            points += pct / len(columns)
            details.append(pd.DataFrame({"profile_id": result.profile_id, "feature": key, "family": family,
                                         "value": result[key], "direction": directions[key],
                                         "percentile": pct, "weighted_points": contribution}))
        total += points * weights[family] / denominator
        result.loc[eligible, family + "_score"] = points.loc[eligible]
    result.loc[eligible, "score"] = total.loc[eligible]
    detail = pd.concat(details, ignore_index=True)
    detail = detail.loc[detail.profile_id.isin(set(result.loc[eligible, "profile_id"]))].copy()
    # A parità di indice usiamo il tempo da referto, poi ordinamento stabile.
    order = [key for key in ("score", "minutes_played_match", "player_name", "match_id", "profile_id") if key in result]
    result = result.sort_values(order, ascending=[key != "score" and key != "minutes_played_match" for key in order], na_position="last")
    return result, detail
