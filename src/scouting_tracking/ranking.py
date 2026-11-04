"""Compatibilità descrittiva, scale congelate e contributi spiegabili."""

import numpy as np
import pandas as pd

from .config import FEATURES, FAMILIES, features_for_phase


def minimum_minutes(phase, config):
    return config.min_transition_minutes if "transition" in phase else config.min_observed_minutes


def fit_scalers(profiles, config, *, phase_features=None):
    result = {}
    base = profiles.loc[profiles.period_id.eq(0) & profiles.get("role_usable", profiles.role_verified)]
    for (role, phase), group in base.groupby(["role_group", "phase"], sort=True):
        training = group.loc[group.observed_minutes.ge(minimum_minutes(phase, config))]
        features = {}
        for key in (features_for_phase(phase) if phase_features is None else phase_features(phase)):
            values = training[key].dropna().to_numpy(dtype=float)
            info = {"n": len(values), "available": False, "reason": "campione insufficiente (<3)"}
            if len(values) >= 3:
                q25, median, q75 = np.quantile(values, [.25, .5, .75])
                scale = float(q75 - q25)
                method = "IQR"
                if scale <= 1e-12:
                    scale = float(np.std(values))
                    method = "std (IQR nullo)"
                info.update({"median": float(median), "q25": float(q25), "q75": float(q75),
                             "min": float(values.min()), "max": float(values.max()), "scale": scale,
                             "method": method, "available": scale > 1e-12,
                             "reason": "" if scale > 1e-12 else "nessuna variabilità nel campione"})
            features[key] = info
        result[role + "|" + phase] = {"role_group": role, "phase": phase,
                                     "population_size": len(training), "profile_ids": training.profile_id.tolist(),
                                     "minimum_minutes": minimum_minutes(phase, config), "features": features}
    return result


def active_features(scaler, weights, *, feature_catalog=None):
    catalog = FEATURES if feature_catalog is None else feature_catalog
    return [key for key, item in scaler["features"].items() if item["available"] and weights[catalog[key][2]] > 0]


def score_profiles(profiles, target, scaler, weights, min_minutes=None, *, feature_catalog=None, family_catalog=None):
    catalog = FEATURES if feature_catalog is None else feature_catalog
    family_catalog = FAMILIES if family_catalog is None else family_catalog
    if set(weights) != set(family_catalog) or any(not np.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Scegliere almeno un peso positivo e finito")
    if not profiles.role_group.eq(scaler["role_group"]).all() or not profiles.phase.eq(scaler["phase"]).all():
        raise ValueError("Scale e profili devono avere lo stesso ruolo e la stessa fase")
    keys = active_features(scaler, weights, feature_catalog=catalog)
    if not keys:
        raise ValueError("Nessuna caratteristica variabile disponibile per i pesi selezionati")
    missing_target = [key for key in keys if key not in target or not np.isfinite(target[key])]
    if missing_target:
        raise ValueError("Riferimento incompleto: " + ", ".join(catalog[key][0] for key in missing_target))
    families = {family: [key for key in keys if catalog[key][2] == family] for family in family_catalog}
    denominator = sum(weights[family] for family, columns in families.items() if columns)
    result = profiles.copy()
    result["score"] = np.nan
    result["distance"] = np.nan
    result["score_status"] = "Disponibile"
    exposure = scaler["minimum_minutes"] if min_minutes is None else min_minutes
    if not np.isfinite(exposure) or exposure < 0:
        raise ValueError("Esposizione minima non valida")
    incomplete = ~np.isfinite(result[keys].to_numpy(dtype=float)).all(axis=1)
    result.loc[incomplete, "score_status"] = "Caratteristiche insufficienti"
    result.loc[result.observed_minutes.lt(exposure), "score_status"] = "Esposizione insufficiente"
    result.loc[~result.get("role_usable", result.role_verified), "score_status"] = "Ruolo non assegnabile"
    eligible = result.score_status.eq("Disponibile")
    total_cost = pd.Series(0.0, index=result.index)
    contributions = []
    for family, columns in families.items():
        result[family + "_score"] = np.nan
        if not columns:
            continue
        family_cost = pd.Series(0.0, index=result.index)
        for key in columns:
            raw_delta = result[key] - float(target[key])
            scaled_delta = raw_delta / scaler["features"][key]["scale"]
            cost = scaled_delta ** 2
            family_cost += cost / len(columns)
            weighted = cost * weights[family] / denominator / len(columns)
            contributions.append(pd.DataFrame({"profile_id": result.profile_id, "feature": key,
                                                "family": family, "value": result[key],
                                                "target": float(target[key]), "delta": raw_delta,
                                                "scaled_delta": scaled_delta, "weighted_cost": weighted}))
        total_cost += family_cost * weights[family] / denominator
        result.loc[eligible, family + "_score"] = 100 / (1 + np.sqrt(family_cost.loc[eligible]))
    result.loc[eligible, "distance"] = np.sqrt(total_cost.loc[eligible])
    result.loc[eligible, "score"] = 100 / (1 + result.loc[eligible, "distance"])
    detail = pd.concat(contributions, ignore_index=True)
    valid_ids = set(result.loc[eligible, "profile_id"])
    detail = detail.loc[detail.profile_id.isin(valid_ids)].copy()
    totals = detail.groupby("profile_id").weighted_cost.transform("sum")
    detail["mismatch_share"] = np.where(totals > 0, detail.weighted_cost / totals.where(totals > 0), 0.0)
    return result.sort_values(["score", "observed_minutes"], ascending=[False, False], na_position="last"), detail


def feature_correlations(profiles, config):
    records = []
    base = profiles.loc[profiles.period_id.eq(0) & profiles.get("role_usable", profiles.role_verified)]
    for (role, phase), group in base.groupby(["role_group", "phase"]):
        group = group.loc[group.observed_minutes.ge(minimum_minutes(phase, config))]
        keys = features_for_phase(phase)
        for index, left in enumerate(keys):
            for right in keys[index + 1:]:
                selected = group[[left, right]].dropna()
                if len(selected) < 3 or selected[left].nunique() < 2 or selected[right].nunique() < 2:
                    continue
                records.append({"role_group": role, "phase": phase, "feature_a": left,
                                "feature_b": right, "n": len(selected),
                                "spearman": float(selected.rank().corr().iloc[0, 1])})
    return pd.DataFrame(records)
