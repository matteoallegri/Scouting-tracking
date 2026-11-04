"""Tracking verificato → profili, contesti, sequenze e scale di confronto."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .config import PHASES
from .context import build_context, attach_context
from .features import detect_episodes, summarize_transitions, add_spatial_features, build_player_profiles
from .metadata import discover_templates, discover_player_defaults, load_match_metadata, attach_roles
from .ranking import fit_scalers, feature_correlations, score_profiles, active_features
from .role_catalog import CATALOG_PATH, load_role_catalog
from .selection import PROJECT_VALUES_PATH, VERIFIED_VALUES_PATH

METRIC_COLUMNS = {
    "entity_id", "team_id", "is_ball", "is_goalkeeper", "period_id", "segment_id", "frame_index",
    "timestamp_ms", "delta_t_s", "is_valid", "valid_interval", "x_smooth", "y_smooth", "vx", "vy",
    "speed", "tangential_acceleration", "delta_distance",
}
REPLAY_COLUMNS = ["entity_id", "team_id", "is_ball", "is_goalkeeper", "period_id", "frame_index",
                  "timestamp_ms", "x_smooth", "y_smooth", "role_group", "position_side", "role_verified",
                  "role_usable", "phase_base", "phase_detail", "live_segment_id", "attack_sign"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_blocks(path, chunksize=150_000):
    pending = None
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=lambda key: key in METRIC_COLUMNS, low_memory=False):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        entity, period = chunk.entity_id.to_numpy(), chunk.period_id.to_numpy()
        cuts = np.r_[0, np.flatnonzero((entity[1:] != entity[:-1]) | (period[1:] != period[:-1])) + 1, len(chunk)]
        for start, end in zip(cuts[:-2], cuts[1:-1]):
            yield chunk.iloc[start:end].reset_index(drop=True)
        pending = chunk.iloc[cuts[-2]:].reset_index(drop=True)
    if pending is not None:
        yield pending


def load_ball(path):
    use = ["frame", "period", "video_ms", "x", "y", "is_ball", "team_in_poss",
           "is_interpolated", "jump_anomaly", "speed_anomaly", "out_of_bounds"]
    return pd.concat([chunk.loc[chunk.is_ball] for chunk in pd.read_csv(path, usecols=use, chunksize=250_000, low_memory=False)], ignore_index=True)


def write_csv(df, path):
    if str(path).endswith(".gz"):
        df.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0}, float_format="%.10g")
    else:
        df.to_csv(path, index=False)


def evidence_table(episodes, transitions, spatial):
    records = []
    for row in episodes.itertuples():
        records.append({"entity_id": row.entity_id, "role_group": row.role_group, "period_id": row.period_id,
                        "kind": row.kind, "start_frame": row.start_frame, "end_frame": row.end_frame,
                        "live_segment_id": row.live_segment_id, "phase_base": row.phase_base,
                        "phase_detail": row.phase_detail, "indicator_value": row.peak,
                        "indicator_unit": "m/s²", "duration_s": row.duration_s})
    for row in transitions.itertuples():
        records.append({"entity_id": row.entity_id, "role_group": row.role_group, "period_id": row.period_id,
                        "kind": "attacking_transition" if row.kind == "attack" else "defensive_transition",
                        "start_frame": row.start_frame, "end_frame": row.end_frame,
                        "live_segment_id": row.live_segment_id,
                        "phase_base": "in_possession" if row.kind == "attack" else "out_of_possession",
                        "phase_detail": "attacking_transition" if row.kind == "attack" else "defensive_transition",
                        "indicator_value": row.progress_m if row.kind == "attack" else -row.progress_m,
                        "indicator_unit": "m", "duration_s": row.duration_s})
    for _, group in spatial.dropna(subset=["hull_share"]).groupby(["entity_id", "role_group", "phase_detail"], sort=False):
        row = group.loc[group.hull_share.idxmax()]
        records.append({"entity_id": row.entity_id, "role_group": row.role_group, "period_id": int(row.period_id),
                        "kind": "spatial", "start_frame": int(row.frame_index), "end_frame": int(row.frame_index),
                        "live_segment_id": int(row.live_segment_id), "phase_base": row.phase_base,
                        "phase_detail": row.phase_detail, "indicator_value": float(row.hull_share),
                        "indicator_unit": "quota 0–1", "duration_s": 0.0})
    result = pd.DataFrame(records)
    if len(result):
        result["sequence_id"] = ["seq_" + str(i) for i in range(len(result))]
    return result


def validate_build(profiles, spatial, replay, roles, expected_seconds, expected_distance, episodes):
    base = profiles.loc[profiles.period_id.eq(0) & profiles.phase.eq("all")]
    checks = {
        "unique_profiles": not profiles.duplicated(["entity_id", "role_group", "period_id", "phase"]).any(),
        "duration_conservation": bool(np.isclose(base.observed_duration_s.sum(), expected_seconds, rtol=1e-9, atol=1e-7)),
        "distance_conservation": bool(np.isclose(base.total_distance_m.sum(), expected_distance, rtol=1e-9, atol=1e-7)),
        "episode_conservation": int(base.acceleration_count.sum() + base.braking_count.sum()) == len(episodes),
        "role_exposure_conservation": bool(np.allclose(
            profiles.role_documented_duration_s + profiles.role_assigned_duration_s,
            profiles.observed_duration_s, rtol=1e-9, atol=1e-7)),
        "all_observed_roles_usable": bool(profiles.role_usable.all()),
        "catalog_matches_profiles": bool(profiles.role_group.eq(profiles.scouting_primary_role).all()),
        "one_primary_role_per_player": bool(profiles.groupby("entity_id").role_group.nunique().eq(1).all()),
        "official_minutes_available": bool((np.isfinite(profiles.minutes_played_match) & profiles.minutes_played_match.ge(0)).all()),
        "quoted_values_valid": bool((np.isfinite(profiles.loc[profiles.market_value_status.eq("quoted"), "market_value_eur"])
                                      & profiles.loc[profiles.market_value_status.eq("quoted"), "market_value_eur"].ge(0)).all()),
        "positive_durations": bool(profiles.observed_duration_s.gt(0).all()),
        "nonnegative_distances": bool(profiles.total_distance_m.ge(0).all()),
        "high_speed_subset": bool(profiles.high_speed_distance_m.le(profiles.total_distance_m + 1e-7).all()),
        "p95_below_max": bool(profiles.speed_p95.le(profiles.speed_max + 1e-9).all()),
        "spatial_hull_domain": bool(spatial.hull_share.dropna().between(0, 1).all()),
        "no_goalkeepers_in_spatial": not bool(spatial.is_goalkeeper.any()),
        "finite_replay_positions": bool(np.isfinite(replay[["x_smooth", "y_smooth"]]).all().all()),
    }
    for _, group in roles.groupby("entity_id"):
        ordered = group.sort_values("start_frame")
        if len(ordered) > 1 and (ordered.end_frame.to_numpy()[:-1] > ordered.start_frame.to_numpy()[1:]).any():
            checks["nonoverlapping_role_windows"] = False
            break
    checks.setdefault("nonoverlapping_role_windows", True)
    for _, group in profiles.groupby(["entity_id", "role_group", "period_id"]):
        seconds = group.set_index("phase").observed_duration_s
        all_seconds = seconds.get("all", 0)
        checks.setdefault("phase_duration_subsets", True)
        if seconds.get("in_possession", 0) + seconds.get("out_of_possession", 0) > all_seconds + 1e-6:
            checks["phase_duration_subsets"] = False
        for transition, parent in (("attacking_transition", "in_possession"), ("defensive_transition", "out_of_possession")):
            if seconds.get(transition, 0) > seconds.get(parent, 0) + 1e-6:
                checks["phase_duration_subsets"] = False
    return {"passed": all(checks.values()), "checks": checks,
            "analysis_player_seconds": expected_seconds, "analysis_distance_m": expected_distance}


def build_match(root, match, output_root, config, templates, player_defaults, role_catalog):
    started = time.monotonic()
    metrics_dir = root / "outputs/tracking_metrics" / match
    metrics_manifest_path = metrics_dir / "manifest.json"
    source_manifest = json.loads(metrics_manifest_path.read_text())
    audit = json.loads((metrics_dir / "independent_audit.json").read_text())
    if not source_manifest["validation_passed"] or not source_manifest["config"]["exclude_interpolated"]:
        raise ValueError("Rigenerare prima il tracking verificato con interpolazioni escluse")
    if not audit["passed"] or audit["manifest_sha256"] != sha256(metrics_manifest_path):
        raise ValueError("L'audit delle metriche di input non corrisponde al manifest")
    if audit["output_sha256"]["entities_frame.csv.gz"] != sha256(metrics_dir / "entities_frame.csv.gz"):
        raise ValueError("Le metriche per frame sono cambiate dopo l'audit")
    raw_path = root / "dataframes" / match / "tracking_positions_clean.csv"
    team_ids = pd.read_csv(raw_path.parent / "players_meta.csv").team_id.unique()
    context, opportunities = build_context(load_ball(raw_path), team_ids, config)
    players, roles, directions, teams, frame_recoveries = load_match_metadata(root, match, context, templates, player_defaults, role_catalog)
    period_origins = context.groupby("period").frame.min().to_dict()
    player_parts, replay_parts = defaultdict(list), []
    expected_seconds = expected_distance = 0.0
    seen = set()
    for block in metric_blocks(metrics_dir / "entities_frame.csv.gz"):
        entity = str(block.entity_id.iloc[0])
        period = int(block.period_id.iloc[0])
        key = (entity, period)
        if key in seen or block.frame_index.duplicated().any():
            raise ValueError("Blocco metrico duplicato")
        seen.add(key)
        data = attach_context(block, context)
        data = attach_roles(data, roles.loc[roles.entity_id.eq(entity)])
        ball = bool(data.is_ball.iloc[0])
        sign = 1 if ball else directions[(str(data.team_id.iloc[0]), period)]
        data["attack_sign"] = sign
        valid_position = data.is_valid & data.live_observed
        valid = data.loc[valid_position].copy()
        sampled = valid.frame_index.sub(period_origins[period]).mod(config.replay_step_frames).eq(0)
        replay_parts.append(valid.loc[sampled, REPLAY_COLUMNS])
        if not ball and not bool(data.is_goalkeeper.iloc[0]):
            player_parts[entity].append(valid)
            intervals = valid.loc[valid.analysis_interval]
            expected_seconds += float(intervals.delta_t_s.sum())
            expected_distance += float(intervals.delta_distance.sum())
    print(f"{match}: contesti e {len(player_parts)} giocatori di movimento caricati ({time.monotonic()-started:.1f}s)", flush=True)
    missing_roles = set(player_parts) - set(role_catalog.player_id)
    if missing_roles:
        raise ValueError("Completare il catalogo dei ruoli per i giocatori osservati: " + ", ".join(sorted(missing_roles)))
    replay = pd.concat(replay_parts, ignore_index=True).sort_values(["period_id", "frame_index", "entity_id"])
    sampled_spatial = replay.frame_index.sub(replay.period_id.map(period_origins)).mod(config.spatial_step_frames).eq(0)
    spatial = add_spatial_features(replay.loc[sampled_spatial], config)
    profile_parts, episode_parts, transition_parts = [], [], []
    for entity, parts in player_parts.items():
        data = pd.concat(parts, ignore_index=True).sort_values("frame_index").reset_index(drop=True)
        data["match_id"] = match
        player_spatial = spatial.loc[spatial.entity_id.eq(entity)]
        episodes = detect_episodes(data, config)
        transitions = summarize_transitions(data, config)
        profile_parts.append(build_player_profiles(data, player_spatial, episodes, transitions, config))
        episode_parts.append(episodes)
        transition_parts.append(transitions)
    profiles = pd.concat(profile_parts, ignore_index=True)
    episodes = pd.concat(episode_parts, ignore_index=True)
    transitions = pd.concat(transition_parts, ignore_index=True)
    sequences = evidence_table(episodes, transitions, spatial)
    profiles = profiles.merge(players.drop(columns="match_id"), on="entity_id", how="left", validate="many_to_one")
    for table in (profiles, episodes, transitions, spatial, replay, sequences):
        table["match_id"] = match
    profiles["profile_id"] = profiles.match_id + "|" + profiles.entity_id + "|" + profiles.role_group + "|" + profiles.phase + "|" + profiles.period_id.astype(str)
    validation = validate_build(profiles, spatial, replay, roles, expected_seconds, expected_distance, episodes)
    if not validation["passed"]:
        raise ValueError("Verifiche scouting fallite: " + str(validation["checks"]))
    players["has_outfield_tracking"] = players.entity_id.isin(player_parts)
    inputs = [metrics_manifest_path, metrics_dir / "entities_frame.csv.gz", metrics_dir / "independent_audit.json",
              raw_path, raw_path.parent / "players_meta.csv", root / "data/task2_wc2026/player_market_values.csv",
              root / CATALOG_PATH,
              root / VERIFIED_VALUES_PATH,
              root / "outputs/tracking_metrics/diagnostics" / (match + "_verified_event_links.csv")]
    verified_names = root / "data/scouting_tracking/player_names_verified.csv"
    if verified_names.exists():
        inputs.append(verified_names)
    raw_folder = next((root / "data/task1_finals").glob(match + "_*"))
    inputs += sorted(raw_folder.glob("*.json")) + sorted(raw_folder.glob("*.xml"))
    # Le etichette per la stessa identità possono provenire dall'altra finale.
    # Si registrano anche queste fonti, senza usare le sue prestazioni nel profilo.
    for source_match in config.matches:
        directory = next((root / "data/task1_finals").glob(source_match + "_*"))
        inputs += sorted(directory.glob("info_stats_*.json")) + sorted(directory.glob("events_*.json"))
    inputs = list(dict.fromkeys(inputs))
    manifest = {
        "match_id": match, "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
        "teams": teams, "validation": validation, "profile_rows": len(profiles), "outfield_players": len(player_parts),
        "role_windows": len(roles), "role_event_frame_recoveries": frame_recoveries,
        "unknown_role_player_seconds": float(profiles.loc[profiles.period_id.eq(0) & profiles.phase.eq("all") & ~profiles.role_usable, "observed_duration_s"].sum()),
        "assigned_role_player_seconds": float(profiles.loc[profiles.period_id.eq(0) & profiles.phase.eq("all"), "role_assigned_duration_s"].sum()),
        "documented_role_player_seconds": float(profiles.loc[profiles.period_id.eq(0) & profiles.phase.eq("all"), "role_documented_duration_s"].sum()),
        "transition_starts": len(opportunities), "complete_player_transitions": len(transitions), "episodes": len(episodes),
        "spatial_sampling_step_frames": config.spatial_step_frames, "replay_sampling_step_frames": config.replay_step_frames,
        "inputs": {str(path.relative_to(root)): sha256(path) for path in inputs},
        "directions": [{"team_id": team, "period_id": period, "attack_sign": sign} for (team, period), sign in directions.items()],
        "conventions": {
            "context": "Coordinate valide della palla, senza interpolazioni/flag esclusi; possesso noto solo quando presente. Nessun fill attraverso pause o ignoti.",
            "transitions": "Cambio fra possessi stabili nello stesso segmento osservato. Intervalli 0 < elapsed <= window: nessuna inclusione del tratto prima del recupero. Finestra interrotta al successivo cambio/ignoto/pausa; descrittore solo su finestre intere per giocatore e ruolo.",
            "phases": "IP/OP comprendono le rispettive transizioni. Le viste di transizione sono sottoinsiemi, non addendi da sommare a IP/OP.",
            "episodes": "Soglia tangenziale sostenuta per durata minima e velocità minima. Conteggio assegnato alla fase di inizio; nessun attraversamento di gap/ruolo/segmento. Accelerazioni e frenate sono episodi per partita rappresentata nel profilo (match_count=1 anche nelle viste per periodo), senza divisione per minuti o proiezione a gara intera.",
            "space": "Campioni espliciti, portieri esclusi. Baricentro di riferimento esclude anche il giocatore. Delta hull a compagni fissi è geometrico, non causale.",
            "roles": "Categoria primaria stabile dal catalogo curato per identità; secondari descrittivi, senza duplicazione. Ruoli/posizioni del match e loro disponibilità conservati separatamente come contesto. Nessun dato fisico modificato dalla classificazione.",
            "selection": "Minuti giocati dal campo minsPlayed del referto, distinti dal tempo osservato nella fase. Il filtro somma per ID i minuti delle partite selezionate, contando ogni partita una sola volta indipendentemente da fase, periodo o disponibilità delle metriche; i profili tracking restano distinti per incontro. Budget massimo per singolo giocatore come vincolo sul valore noto: ignoti/ritirati esclusi dalla selezione economica, senza prezzo zero e senza alterare la somiglianza tracking. Fonte/data delle quote conservate.",
            "names": "Mapping esistente e integrazioni curate da referto FIFA con evidenza locale verificata. Lookup limitato alla rosa del match, senza aggiungere candidati; altrimenti ID abbreviato.",
            "scope": "Prototipo descrittivo su due finali, non previsione del successo futuro.",
        }, "runtime_s": round(time.monotonic() - started, 2),
    }
    destination = output_root / match
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scouting_" + match + "_", dir=output_root) as temp:
        stage = Path(temp)
        for filename, table in (("profiles.csv", profiles), ("players.csv", players), ("roles.csv", roles),
                                ("frame_context.csv.gz", context), ("possession_transitions.csv", opportunities),
                                ("episodes.csv", episodes), ("player_transitions.csv", transitions),
                                ("spatial_samples.csv.gz", spatial), ("replay.csv.gz", replay), ("sequences.csv", sequences)):
            write_csv(table, stage / filename)
        manifest["outputs_sha256"] = {path.name: sha256(path) for path in stage.iterdir()}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        for path in stage.iterdir():
            path.replace(destination / path.name)
    print(f"{match}: {len(profiles)} profili, {len(episodes)} episodi, {len(transitions)} transizioni complete; controlli OK ({time.monotonic()-started:.1f}s)", flush=True)
    return profiles, manifest


def build_all(root, output_root, config):
    output_root.mkdir(parents=True, exist_ok=True)
    templates = discover_templates(root, config.matches)
    player_defaults = discover_player_defaults(root, config.matches, templates)
    role_catalog = load_role_catalog(root)
    parts, manifests = [], []
    for match in config.matches:
        profiles, manifest = build_match(root, match, output_root, config, templates, player_defaults, role_catalog)
        parts.append(profiles)
        manifests.append(manifest)
    combined = pd.concat(parts, ignore_index=True)
    scalers = fit_scalers(combined, config)
    write_csv(combined, output_root / "profiles.csv")
    (output_root / "reference_scalers.json").write_text(json.dumps(scalers, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(feature_correlations(combined, config), output_root / "feature_correlations.csv")
    baselines = []
    for key, scaler in scalers.items():
        group = combined.loc[combined.period_id.eq(0) & combined.role_group.eq(scaler["role_group"]) & combined.phase.eq(scaler["phase"])]
        keys = active_features(scaler, config.family_weights)
        if not keys:
            continue
        complete = group.loc[group.observed_minutes.ge(scaler["minimum_minutes"]) & group[keys].notna().all(axis=1)]
        if complete.empty:
            continue
        target = complete.sort_values("observed_minutes", ascending=False).iloc[0]
        ranked, _ = score_profiles(group, target.to_dict(), scaler, config.family_weights)
        ranked["reference_profile_id"] = target.profile_id
        baselines.append(ranked)
    write_csv(pd.concat(baselines, ignore_index=True) if baselines else pd.DataFrame(columns=["profile_id", "score"]), output_root / "baseline_compatibility.csv")
    code = sorted((root / "src/scouting_tracking").glob("*.py")) + [root / "commands/build_tracking_scouting.py"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
              "matches": list(config.matches), "validation_passed": all(m["validation"]["passed"] for m in manifests),
              "code_sha256": {str(path.relative_to(root)): sha256(path) for path in code},
              "profiles": len(combined), "reference_cohorts": len(scalers),
              "role_catalog": {"path": CATALOG_PATH, "sha256": sha256(root / CATALOG_PATH)},
              "selection_inputs_sha256": {name: sha256(root / name) for name in (PROJECT_VALUES_PATH, VERIFIED_VALUES_PATH)},
              "outputs_sha256": {name: sha256(output_root / name) for name in (
                  "profiles.csv", "reference_scalers.json", "baseline_compatibility.csv", "feature_correlations.csv")},
              "match_manifests_sha256": {m: sha256(output_root / m / "manifest.json") for m in config.matches},
              "note": "Scale congelate sul campione dichiarato; filtri di visualizzazione non rifittano la normalizzazione. Compatibilità 100/(1+d), non probabilità."}
    (output_root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
