"""Referti Opta → una riga per giocatore-partita presente nel tracking.

Le fasi sono categorie funzionali dell'azione, non un join temporale al possesso
del tracking: un recupero è difensivo anche se al suo istante il possesso cambia.
I conteggi del referto comprendono le riprese di gioco e non vanno interpretati
come eventi limitati ai secondi di tracking valido. Non stimiamo una fase o una
durata mancante. Gli aggregati non vengono copiati nei tempi o nelle transizioni.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import EVENT_FEATURES, FAMILY_PHASE, STAT_FIELDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(root, matches):
    paths = []
    for match in matches:
        folder = next((Path(root) / "data/task1_finals").glob(match + "_*"))
        paths.extend([next(folder.glob("info_stats_*.json")), next(folder.glob("events_*.json"))])
    return paths


def read_stats(player):
    items = player.get("stat", [])
    if not items or len({s["type"] for s in items}) != len(items):
        raise ValueError("Referto giocatore mancante o con statistiche duplicate")
    stats = {s["type"]: float(s["value"]) for s in items}
    if any(not np.isfinite(value) or value < 0 for value in stats.values()):
        raise ValueError("Statistiche non finite o negative nel referto")
    return stats


def defensive_duels(events):
    """Sottoinsieme esplicito: tackles difensivi e aerei difensivi vinti.

    Un tackle (7) sottrae legalmente la palla in entrambi gli outcome: 0 significa
    che la squadra non conserva il possesso, non che il contrasto sia perso.
    Escludiamo i 50/50: il provider li conta separatamente dai duelWon.
    """
    ids = [int(event["id"]) for event in events]
    if len(set(ids)) != len(ids):
        raise ValueError("ID evento duplicati")
    rows = []
    for event in events:
        typ = int(event["typeId"])
        if (int(event["periodId"]) not in (1, 2, 3, 4) or typ not in (7, 44)
                or event.get("isDeleted", False) or not event.get("playerId")):
            continue
        qualifiers = {int(q["qualifierId"]) for q in event.get("qualifier", [])}
        if 285 not in qualifiers:
            continue
        if 286 in qualifiers:
            raise ValueError("Duello con qualifica offensiva e difensiva simultanee")
        outcome = event.get("outcome")
        if outcome not in (0, 1):
            raise ValueError("Esito del duello non disponibile")
        if typ == 44 and outcome != 1:
            continue
        rows.append({"event_id": int(event["id"]), "entity_id": event["playerId"],
                     "team_id": event["contestantId"], "period_id": int(event["periodId"]),
                     "type_id": typ, "outcome": outcome})
    return pd.DataFrame(rows, columns=["event_id", "entity_id", "team_id", "period_id", "type_id", "outcome"])


def match_features(info, events, tracked, match):
    roster = {}
    for team in info["liveData"]["lineUp"]:
        for player in team["player"]:
            pid = player["playerId"]
            if pid in roster:
                raise ValueError("Giocatore duplicato nella rosa")
            roster[pid] = (team["contestantId"], read_stats(player))
    required = set(STAT_FIELDS.values()) | {"accuratePass", "totalPass", "minsPlayed", "duelWon"}
    available = {key for _, stats in roster.values() for key in stats}
    if required - available:
        raise ValueError("Il referto non documenta le misure richieste: " + ", ".join(sorted(required - available)))
    duels = defensive_duels(events)
    counts = Counter(duels.entity_id)
    rows = []
    for row in tracked.itertuples(index=False):
        if row.entity_id not in roster:
            raise ValueError("Giocatore del tracking assente nel referto")
        team, stats = roster[row.entity_id]
        if team != row.team_id or stats.get("minsPlayed", 0) <= 0:
            raise ValueError("Squadra o partecipazione non coerente fra tracking e referto")
        # Gli omissioni sono zeri solo per conteggi documentati nello schema del
        # match e giocatori effettivamente entrati, con referto completo presente.
        result = {key: stats.get(field, 0.0) for key, field in STAT_FIELDS.items()}
        total, accurate = stats.get("totalPass", 0.0), stats.get("accuratePass", 0.0)
        if accurate > total:
            raise ValueError("Passaggi riusciti superiori ai tentativi")
        count_values = [*result.values(), total, accurate]
        if any(value != int(value) for value in count_values):
            raise ValueError("Conteggi evento non interi")
        result["event_pass_accuracy_pct"] = accurate / total * 100 if total > 0 else np.nan
        result["event_defensive_duels_won"] = counts[row.entity_id]
        own_duels = duels.loc[duels.entity_id.eq(row.entity_id)]
        if not own_duels.team_id.eq(team).all() or counts[row.entity_id] > stats.get("duelWon", 0):
            raise ValueError("Duelli difensivi non coerenti con squadra o duelli totali")
        rows.append({"match_id": match, "entity_id": row.entity_id, "team_id": team,
                     "event_pass_attempts": total, "event_pass_completed": accurate, **result})
    return pd.DataFrame(rows), duels.loc[duels.entity_id.isin(tracked.entity_id)].assign(match_id=match)


def build_bundle(root, profiles, output):
    root, output = Path(root), Path(output)
    tracked = profiles.loc[profiles.period_id.eq(0) & profiles.phase.eq("all"),
                           ["match_id", "entity_id", "team_id"]]
    if tracked.duplicated(["match_id", "entity_id"]).any():
        raise ValueError("Più profili base per giocatore-partita")
    matches = sorted(tracked.match_id.unique())
    parts, evidence = [], []
    for match in matches:
        paths = source_paths(root, [match])
        info = json.loads(paths[0].read_text())
        events = json.loads(paths[1].read_text())["liveData"]["event"]
        data, duels = match_features(info, events, tracked.loc[tracked.match_id.eq(match)], match)
        parts.append(data)
        evidence.append(duels)
    data, duels = pd.concat(parts, ignore_index=True), pd.concat(evidence, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "player_match_events.csv", index=False, float_format="%.15g")
    duels.to_csv(output / "defensive_duels.csv", index=False)
    paths = source_paths(root, matches) + [root / "outputs/scouting_tracking/profiles.csv"]
    code = sorted((root / "src/scouting_events").glob("*.py")) + [root / "commands/build_scouting_events.py"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "matches": matches,
        "player_matches": len(data), "validation_passed": True,
        "inputs_sha256": {str(path.relative_to(root)): digest(path) for path in paths},
        "code_sha256": {str(path.relative_to(root)): digest(path) for path in code},
        "outputs_sha256": {name: digest(output / name) for name in ("player_match_events.csv", "defensive_duels.csv")},
        "phase_policy": "Categoria funzionale del gesto, non assegnazione temporale dal tracking. Referti intera partita, supplementari inclusi; rigori di spareggio esclusi. Conteggi per partita; nessuna ripartizione nei tempi o nelle transizioni.",
        "zero_policy": "Conteggi omessi = 0 solo per giocatori entrati con referto presente e campo documentato nel match. Precisione senza passaggi = mancante.",
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def bundle_version(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text())
    paths = [folder / "manifest.json"]
    paths.extend(folder / path for path in manifest["outputs_sha256"])
    paths.extend(root / path for group in ("inputs_sha256", "code_sha256") for path in manifest[group])
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def load_verified_bundle(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("validation_passed"):
        raise ValueError("Verifica eventi non superata")
    required_outputs = {"player_match_events.csv", "defensive_duels.csv"}
    if not required_outputs.issubset(manifest.get("outputs_sha256", {})) or not manifest.get("inputs_sha256") or not manifest.get("code_sha256"):
        raise ValueError("Provenienza degli eventi incompleta")
    for group in ("inputs_sha256", "code_sha256", "outputs_sha256"):
        for path, expected in manifest[group].items():
            if digest((folder if group == "outputs_sha256" else root) / path) != expected:
                raise ValueError("Risultati evento da aggiornare: " + path)
    return pd.read_csv(folder / "player_match_events.csv")


def attach_events(profiles, events):
    original = profiles.index
    result = profiles.merge(events.drop(columns="team_id"), on=["match_id", "entity_id"], how="left", validate="many_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Metriche evento mancanti per un giocatore del tracking")
    result = result.drop(columns="_merge")
    result.index = original
    for key, (_, _, family) in EVENT_FEATURES.items():
        valid = result.period_id.eq(0) & result.phase.isin(["all", FAMILY_PHASE[family]])
        result.loc[~valid, key] = np.nan
    return result
