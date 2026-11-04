"""Nomi disponibili e ruoli per finestre, con provenienza esplicita."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .selection import match_minutes, enrich_market_values

ROLE_CLASSES = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Striker"}


def load_player_names(root, match, info, events):
    """Etichette anagrafiche limitate alla rosa del match; non crea candidati.

    Le integrazioni curate citano il referto ufficiale e gli eventi locali che
    consentono di risalire all'ID. Non si deducono nomi dal ruolo o dall'ordine.
    """
    roster = {player["playerId"]: team["contestantId"]
              for team in info["liveData"]["lineUp"] for player in team["player"]}
    market = pd.read_csv(root / "data/task2_wc2026/player_market_values.csv")
    available = market.loc[market.player_id.isin(roster), ["player_id", "player_name"]].drop_duplicates()
    if available.player_id.duplicated().any():
        raise ValueError("Nomi contraddittori nel mapping esistente")
    names = available.set_index("player_id").player_name.to_dict()
    sources = {pid: "player_market_values.csv" for pid in names}
    path = root / "data/scouting_tracking/player_names_verified.csv"
    if not path.exists():
        return names, sources
    verified = pd.read_csv(path, dtype=str, keep_default_na=False)
    selected = verified.loc[verified.match_id.eq(match)]
    if selected.player_id.duplicated().any():
        raise ValueError("ID duplicato nell'anagrafica verificata del match")
    indexed_events = {int(event["id"]): event for event in events}
    for row in selected.itertuples(index=False):
        if roster.get(row.player_id) != row.team_id:
            raise ValueError("L'anagrafica verificata contiene un giocatore esterno alla rosa/squadra del match")
        if not row.player_name.strip() or not row.source_url.strip():
            raise ValueError("Nome o fonte mancanti nell'anagrafica verificata")
        event = indexed_events.get(int(row.event_id))
        if event is None or event["contestantId"] != row.team_id:
            raise ValueError("Evento anagrafico assente o di un'altra squadra")
        if row.evidence_kind == "substitution":
            paired = indexed_events.get(int(row.paired_event_id))
            if (event.get("playerId") != row.player_id or paired is None
                    or paired["contestantId"] != row.team_id
                    or {int(event["typeId"]), int(paired["typeId"])} != {18, 19}):
                raise ValueError("Evidenza di sostituzione non coerente con il giocatore")
            def related(item):
                return next((int(q["value"]) for q in item.get("qualifier", []) if int(q["qualifierId"]) == 55), None)
            if related(event) != int(paired["eventId"]) or related(paired) != int(event["eventId"]):
                raise ValueError("La coppia di sostituzione non è collegata reciprocamente")
        elif row.evidence_kind == "initial_goalkeeper":
            if int(event["typeId"]) != 34 or not any(pid == row.player_id and cls == 1 for pid, cls, slot in snapshot(event)[1]):
                raise ValueError("Il portiere non compare nella formazione iniziale documentata")
        else:
            raise ValueError("Tipo di evidenza anagrafica non supportato")
        if row.player_id in names and names[row.player_id] != row.player_name:
            raise ValueError("Nome verificato in conflitto con il mapping esistente")
        names[row.player_id] = row.player_name
        sources[row.player_id] = f"player_names_verified.csv; FIFA match report, pag. {row.source_page}"
    return names, sources


def position_class(position):
    name = position.lower()
    return 1 if "goalkeeper" in name else 2 if "defender" in name else 3 if "midfielder" in name else 4 if "striker" in name or "attacker" in name else 5


def role_group(position, side, verified=True):
    if position_class(position) == 1:
        return "Portiere"
    if not verified:
        return "Ruolo non verificato"
    cls = position_class(position)
    lateral = side in ("Left", "Right")
    if cls == 2:
        return "Esterno difensivo" if lateral else "Difensore centrale"
    if cls == 3:
        return "Esterno offensivo" if lateral else "Centrocampista centrale"
    if cls == 4:
        return "Esterno offensivo" if "Left" in side or "Right" in side else "Attaccante centrale"
    return "Ruolo non verificato"


def snapshot(event):
    q = {int(item["qualifierId"]): item.get("value", "") for item in event.get("qualifier", [])}
    if not all(key in q for key in (30, 44, 130, 131)):
        raise ValueError(f"Snapshot incompleto: {event['id']}")
    players = [value.strip() for value in q[30].split(",")]
    classes = [int(value.strip()) for value in q[44].split(",")]
    slots = [int(value.strip()) for value in q[131].split(",")]
    if len({len(players), len(classes), len(slots)}) != 1:
        raise ValueError("Vettori della formazione non allineati")
    return int(q[130]), [(p, c, s) for p, c, s in zip(players, classes, slots) if c in ROLE_CLASSES and s > 0]


def discover_templates(root, matches):
    """I lati sono dedotti solo dai moduli iniziali documentati nei JSON."""
    templates = {}
    for match in matches:
        directory = next((root / "data/task1_finals").glob(match + "_*"))
        info = json.loads(next(directory.glob("info_stats_*.json")).read_text())
        events = json.loads(next(directory.glob("events_*.json")).read_text())["liveData"]["event"]
        initial = {e["contestantId"]: snapshot(e)[0] for e in events if int(e["typeId"]) == 34}
        for team in info["liveData"]["lineUp"]:
            enum = initial[team["contestantId"]]
            for player in team["player"]:
                slot = int(player.get("formationPlace", 0))
                if slot <= 0 or player["position"] == "Substitute":
                    continue
                value = (player["position"], player.get("positionSide", ""), str(team["formationUsed"]))
                key = (enum, slot)
                if key in templates and templates[key] != value:
                    raise ValueError(f"Template di ruolo contraddittorio per {key}")
                templates[key] = value
    return templates


def resolve_role_event_frames(events, links):
    """Recupera solo uscite/entrate con coppia reciproca allo stesso tempo.

    Nessun ricorso al frame più vicino. Il frame derivato resta dichiarato
    separatamente dai collegamenti diretti agli eventi.
    """
    if links.id.duplicated().any():
        raise ValueError("Collegamenti evento/frame duplicati")
    linked = links.set_index("id").verified_frame
    by_local = {(event["contestantId"], int(event["eventId"])): event for event in events}
    changes, recovered = [], []
    for event in events:
        typ = int(event["typeId"])
        if typ not in (18, 19, 40):
            continue
        frame = linked.get(int(event["id"]), np.nan)
        if pd.isna(frame) and typ in (18, 19):
            def related(item):
                return next((int(q["value"]) for q in item.get("qualifier", []) if int(q["qualifierId"]) == 55), None)
            pair = by_local.get((event["contestantId"], related(event)))
            if (pair is not None and {typ, int(pair["typeId"])} == {18, 19}
                    and related(pair) == int(event["eventId"])
                    and all(event[key] == pair[key] for key in ("periodId", "timeMin", "timeSec"))):
                frame = linked.get(int(pair["id"]), np.nan)
                if pd.notna(frame):
                    recovered.append({"event_id": int(event["id"]), "paired_event_id": int(pair["id"]),
                                      "frame": int(frame), "method": "reciprocal_substitution_same_clock"})
        if pd.isna(frame):
            raise ValueError(f"Evento di ruolo senza collegamento verificabile: {event['id']}")
        changes.append((int(frame), typ, event))
    return changes, recovered


def discover_player_defaults(root, matches, templates):
    """Etichette per identità: formazione iniziale, poi primo snapshot documentato.

    Non legge risultati di tracking/scouting né aggiunge giocatori alle rose.
    Il match corrente ha priorità sugli altri incontri per la stessa identità.
    """
    defaults = {}
    for match in matches:
        directory = next((root / "data/task1_finals").glob(match + "_*"))
        info = json.loads(next(directory.glob("info_stats_*.json")).read_text())
        events = json.loads(next(directory.glob("events_*.json")).read_text())["liveData"]["event"]
        for team in info["liveData"]["lineUp"]:
            for player in team["player"]:
                if player["position"] != "Substitute" and player.get("positionSide"):
                    defaults[(match, player["playerId"])] = {
                        "position": player["position"], "position_side": player["positionSide"],
                        "role_group": role_group(player["position"], player["positionSide"]),
                        "source": f"{match}:initial_lineup"}
        for event in events:
            if int(event["typeId"]) != 40:
                continue
            enum, active = snapshot(event)
            for pid, cls, slot in active:
                template = templates.get((enum, slot))
                if template is not None and position_class(template[0]) == cls:
                    defaults.setdefault((match, pid), {
                        "position": template[0], "position_side": template[1],
                        "role_group": role_group(template[0], template[1]),
                        "source": f"{match}:snapshot:{event['id']}"})
    return defaults


def assign_player_roles(roles, roster, match, defaults, substitutions):
    """Completa le sole finestre prive di dettaglio, mantenendo la provenienza.

    Un ruolo assegnato è utilizzabile nel confronto, senza diventare verificato
    nel tratto. Nessuna modifica ai confini delle finestre o ai dati fisici.
    """
    result = roles.copy()
    result["role_original_source"] = result.role_source
    result["role_usable"] = result.role_verified
    result["role_assignment_basis"] = "documented_window"

    def documented(pid):
        if (match, pid) in defaults:
            return defaults[(match, pid)]
        for (other_match, player), value in sorted(defaults.items()):
            if player == pid:
                return value
        return None

    for index, row in result.loc[~result.role_verified].iterrows():
        prior = documented(row.entity_id)
        basis = "player_documented_role"
        if prior is None:
            outgoing = substitutions.get(row.entity_id)
            replacement = documented(outgoing) if outgoing else None
            nominal_class = position_class(roster[row.entity_id]["nominal_position"])
            if replacement is not None and (
                    nominal_class == position_class(replacement["position"])
                    or nominal_class == 4 and replacement["role_group"] == "Esterno offensivo"):
                prior = {**replacement, "source": f"replacement:{outgoing};{replacement['source']}"}
                basis = "compatible_substitution"
        if prior is None:
            nominal = roster[row.entity_id]["nominal_position"]
            cls = position_class(nominal)
            if cls not in ROLE_CLASSES:
                result.loc[index, "role_assignment_basis"] = "unresolved"
                continue
            # La categoria generica ha un default centrale esplicito, senza
            # inventare un lato destro/sinistro. È un'assunzione di confronto.
            prior = {"position": nominal, "position_side": "Centre",
                     "role_group": role_group(nominal, "Centre"), "source": "roster_nominal_central_default"}
            basis = "nominal_category_default"
        for field in ("position", "position_side", "role_group"):
            result.loc[index, field] = prior[field]
        result.loc[index, "role_usable"] = True
        result.loc[index, "role_source"] = prior["source"]
        result.loc[index, "role_assignment_basis"] = basis
    return result


def load_match_metadata(root: Path, match, frame_context, templates, player_defaults=None, role_catalog=None):
    directory = next((root / "data/task1_finals").glob(match + "_*"))
    info = json.loads(next(directory.glob("info_stats_*.json")).read_text())
    events = json.loads(next(directory.glob("events_*.json")).read_text())["liveData"]["event"]
    links = pd.read_csv(root / "outputs/tracking_metrics/diagnostics" / (match + "_verified_event_links.csv"))
    changes, frame_recoveries = resolve_role_event_frames(events, links)
    names, name_sources = load_player_names(root, match, info, events)
    teams = {item["id"]: item["name"] for item in info["matchInfo"]["contestant"]}
    roster = {}
    initial_enums = {e["contestantId"]: snapshot(e)[0] for e in events if int(e["typeId"]) == 34}
    first, last = int(frame_context.frame.min()), int(frame_context.frame.max()) + 1
    states = {}
    records = []

    def make_role(player, team, enum, slot, position, side, verified, source):
        return {"entity_id": player, "team_id": team, "formation_enum": enum,
                "formation_place": slot, "position": position, "position_side": side,
                "role_verified": verified, "role_group": role_group(position, side, verified),
                "role_source": source}

    def close(player, frame):
        old = states.pop(player, None)
        if old is not None and frame > old["start_frame"]:
            records.append({**old, "end_frame": frame})

    def begin(role, frame):
        close(role["entity_id"], frame)
        states[role["entity_id"]] = {**role, "start_frame": frame}

    for team in info["liveData"]["lineUp"]:
        tid = team["contestantId"]
        for player in team["player"]:
            pid = player["playerId"]
            nominal = player.get("subPosition", "") if player["position"] == "Substitute" else player["position"]
            roster[pid] = {"entity_id": pid, "team_id": tid, "team_name": teams[tid],
                           "player_name": names.get(pid, "Giocatore " + pid[:8]),
                           "name_available": pid in names, "nominal_position": nominal or "Non disponibile",
                           "minutes_played_match": match_minutes(player),
                           "name_source": name_sources.get(pid, "ID sorgente")}
            slot = int(player.get("formationPlace", 0))
            if slot > 0 and player["position"] != "Substitute":
                begin(make_role(pid, tid, initial_enums[tid], slot, player["position"], player.get("positionSide", ""), True, "initial_lineup"), first)
    for frame, typ, event in sorted(changes, key=lambda item: (item[0], {18: 0, 19: 1, 40: 2}[item[1]])):
        if typ == 18:
            close(event["playerId"], frame)
        elif typ == 19:
            pid = event["playerId"]
            nominal = roster[pid]["nominal_position"]
            begin(make_role(pid, event["contestantId"], 0, 0, nominal, "", False, "substitution_without_verified_side"), frame)
        else:
            enum, active = snapshot(event)
            tid = event["contestantId"]
            active_ids = {item[0] for item in active}
            for pid, state in list(states.items()):
                if state["team_id"] == tid and pid not in active_ids:
                    close(pid, frame)
            for pid, cls, slot in active:
                template = templates.get((enum, slot))
                verified = template is not None and position_class(template[0]) == cls
                position, side = (template[0], template[1]) if verified else (ROLE_CLASSES[cls], "")
                begin(make_role(pid, tid, enum, slot, position, side, verified, "snapshot_template" if verified else "snapshot_side_unknown"), frame)
    for pid in list(states):
        close(pid, last)
    roles = pd.DataFrame(records).sort_values(["entity_id", "start_frame"]).reset_index(drop=True)
    roles["role_window_id"] = np.arange(len(roles))
    roles["match_id"] = match
    by_local = {(e["contestantId"], int(e["eventId"])): e for e in events}
    substitutions = {}
    for event in events:
        if int(event["typeId"]) != 19:
            continue
        paired_id = next((int(q["value"]) for q in event.get("qualifier", []) if int(q["qualifierId"]) == 55), None)
        partner = by_local.get((event["contestantId"], paired_id))
        reciprocal = partner is not None and any(int(q["qualifierId"]) == 55 and int(q["value"]) == int(event["eventId"]) for q in partner.get("qualifier", []))
        if reciprocal and int(partner["typeId"]) == 18:
            substitutions[event["playerId"]] = partner["playerId"]
    if player_defaults is None:
        player_defaults = discover_player_defaults(root, (match,), templates)
    roles = assign_player_roles(roles, roster, match, player_defaults, substitutions)
    players = pd.DataFrame(roster.values())
    players["match_id"] = match
    if role_catalog is not None:
        from .role_catalog import apply_role_catalog
        roles, players = apply_role_catalog(roles, players, role_catalog)
    players = enrich_market_values(root, players)
    directions = {}
    xml = ET.parse(next(directory.glob("tracking_metadata_*.xml")))
    for parent in xml.iter():
        if parent.tag.split("}")[-1] != "DirectionsOfPlay":
            continue
        for period in parent:
            for team in period:
                direction = next(child.text for child in team if child.tag.split("}")[-1] == "DirectionOfPlay")
                directions[(team.attrib["id"], int(period.attrib["id"]))] = 1 if direction == "LeftToRight" else -1
    return players, roles, directions, teams, frame_recoveries


def attach_roles(metrics, roles):
    df = metrics.copy()
    fields = ["role_window_id", "role_group", "role_verified", "position", "position_side", "role_source"]
    if "role_usable" in roles:
        fields += ["role_usable", "role_assignment_basis"]
    if "context_role_group" in roles:
        fields += ["context_role_group"]
    df["role_window_id"] = -1
    df["role_group"] = "Ruolo non verificato"
    df["role_verified"] = False
    df["role_usable"] = False
    df["role_assignment_basis"] = "unresolved"
    df["context_role_group"] = "Ruolo non verificato"
    df["position"] = "Non disponibile"
    df["position_side"] = ""
    df["role_source"] = "no_verified_window"
    for role in roles.itertuples():
        mask = df.frame_index.ge(role.start_frame) & df.frame_index.lt(role.end_frame)
        for field in fields:
            df.loc[mask, field] = getattr(role, field)
    return df
