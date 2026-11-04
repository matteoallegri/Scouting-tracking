"""Minuti da referto e quotazioni per una selezione con vincoli espliciti."""

import math
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PROJECT_VALUES_PATH = "data/task2_wc2026/player_market_values.csv"
VERIFIED_VALUES_PATH = "data/scouting_tracking/market_values_verified.csv"


def match_minutes(player):
    values = [float(item["value"]) for item in player.get("stat", []) if item.get("type") == "minsPlayed"]
    if len(values) > 1 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Minuti del referto duplicati o non validi")
    return values[0] if values else np.nan


def enrich_market_values(root, players):
    """Lookup per ID; le integrazioni non sovrascrivono quotazioni già presenti."""
    project = pd.read_csv(root / PROJECT_VALUES_PATH)
    if project.player_id.duplicated().any():
        raise ValueError("ID duplicato nella fonte dei valori esistenti")
    extra = pd.read_csv(root / VERIFIED_VALUES_PATH, keep_default_na=False)
    if extra.player_id.duplicated().any():
        raise ValueError("ID duplicato nelle integrazioni Transfermarkt")
    numeric = pd.to_numeric(extra.market_value_eur, errors="coerce")
    if not extra.currency.eq("EUR").all() or not extra.status.isin(["quoted", "retired", "unavailable"]).all():
        raise ValueError("Valuta o stato delle quotazioni non supportati")
    quoted = extra.status.eq("quoted")
    if not (np.isfinite(numeric[quoted]) & numeric[quoted].ge(0)).all() or numeric[~quoted].notna().any():
        raise ValueError("Quotazione mancante, non valida o associata a uno stato non quotato")
    if any(not urlparse(url).hostname or not urlparse(url).hostname.startswith("www.transfermarkt.") for url in extra.source_url):
        raise ValueError("Fonte Transfermarkt mancante o non coerente")
    dates = pd.to_datetime(extra.loc[quoted, "value_date"], format="%Y-%m-%d", errors="coerce")
    checked = pd.to_datetime(extra.checked_on, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or checked.isna().any() or (dates > checked.loc[quoted]).any():
        raise ValueError("Date delle quotazioni mancanti o incoerenti")
    source = project.set_index("player_id")
    supplements = extra.set_index("player_id")
    records = []
    for player in players.itertuples(index=False):
        pid = player.entity_id
        record = {"entity_id": pid, "market_value_eur": np.nan, "market_value_origin": "Non disponibile",
                  "market_value_status": "unavailable", "market_value_date": "", "market_value_checked_on": "",
                  "market_value_source": "", "market_value_note": "Quotazione non disponibile"}
        existing = source.loc[pid, "market_value_eur"] if pid in source.index else np.nan
        if pid in supplements.index:
            row = supplements.loc[pid]
            if row.player_name != player.player_name or row.team_name != player.team_name:
                raise ValueError("Integrazione Transfermarkt non coerente con identità o squadra")
            if pd.notna(existing):
                raise ValueError("Un'integrazione non può sovrascrivere un valore già disponibile")
            record.update(market_value_eur=float(row.market_value_eur) if row.status == "quoted" else np.nan,
                          market_value_origin="Transfermarkt", market_value_status=row.status,
                          market_value_date=row.value_date, market_value_checked_on=row.checked_on,
                          market_value_source=row.source_url, market_value_note=row.evidence_note)
        elif pd.notna(existing):
            if not math.isfinite(float(existing)) or float(existing) < 0:
                raise ValueError("Valore di mercato esistente non valido")
            record.update(market_value_eur=float(existing), market_value_origin="Dataset del progetto",
                          market_value_status="quoted", market_value_source=PROJECT_VALUES_PATH,
                          market_value_note="Data della quotazione non disponibile nella fonte locale; valore del dataset, non rivalutato online.")
        records.append(record)
    return players.merge(pd.DataFrame(records), on="entity_id", how="left", validate="one_to_one")


def selected_match_minutes(profiles, selected_matches):
    """Somma per identità, contando ogni partita selezionata una sola volta.

    Le righe per fase, periodo e ruolo ripetono il dato del referto. Un referto
    mancante rende ignoto il totale; valori discordanti non sono sommabili.
    """
    selected = profiles.loc[profiles.match_id.isin(selected_matches),
                            ["entity_id", "match_id", "minutes_played_match"]]
    grouped = selected.groupby(["entity_id", "match_id"], sort=False)
    if grouped.minutes_played_match.nunique(dropna=False).gt(1).any():
        raise ValueError("Minuti da referto discordanti per giocatore e partita")
    matches = selected.drop_duplicates(["entity_id", "match_id"])
    minutes = matches.minutes_played_match
    if (minutes.notna() & (~np.isfinite(minutes) | minutes.lt(0))).any():
        raise ValueError("Minuti da referto non validi")
    by_player = matches.groupby("entity_id", sort=False).minutes_played_match
    totals = by_player.sum(min_count=1)
    totals.loc[by_player.count().ne(by_player.size())] = np.nan
    return totals.rename("minutes_played_selected_matches")


def filter_candidates(profiles, selected_matches, minimum_played_minutes, maximum_market_value_eur, *, minutes_source=None):
    """Tetto per singolo giocatore, inclusivo; quotazioni ignote sempre escluse.

    Non modifica punteggi, riferimento o popolazione di normalizzazione.
    """
    if any(not math.isfinite(value) or value < 0 for value in (minimum_played_minutes, maximum_market_value_eur)):
        raise ValueError("Minuti e budget devono essere non negativi e finiti")
    totals = selected_match_minutes(profiles if minutes_source is None else minutes_source, selected_matches)
    minutes = profiles.entity_id.map(totals)
    price = profiles.market_value_eur
    mask = (profiles.match_id.isin(selected_matches) & np.isfinite(minutes) & minutes.ge(minimum_played_minutes)
            & np.isfinite(price) & price.ge(0) & price.le(maximum_market_value_eur)
            & profiles.market_value_status.eq("quoted"))
    result = profiles.loc[mask].copy()
    result["minutes_played_selected_matches"] = minutes.loc[mask]
    return result
