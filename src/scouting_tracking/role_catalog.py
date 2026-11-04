"""Categorie di scouting curate per identità, separate dalle etichette del match."""

import pandas as pd

SCOUTING_ROLES = (
    "Difensore centrale", "Esterno difensivo", "Centrocampista centrale",
    "Trequartista / seconda punta", "Esterno offensivo", "Attaccante centrale",
)
CATALOG_PATH = "data/scouting_tracking/player_roles_reviewed.csv"
CATALOG_COLUMNS = ("player_id", "player_name", "team_name", "primary_role", "secondary_roles",
                   "role_detail", "classification_note", "source_reference", "reviewed_on")


def load_role_catalog(root):
    catalog = pd.read_csv(root / CATALOG_PATH, dtype=str, keep_default_na=False)
    if not set(CATALOG_COLUMNS).issubset(catalog.columns):
        raise ValueError("Catalogo ruoli incompleto")
    if catalog.player_id.duplicated().any():
        raise ValueError("Giocatore duplicato nel catalogo ruoli")
    required = [column for column in CATALOG_COLUMNS if column != "secondary_roles"]
    if catalog[required].apply(lambda values: values.str.strip().eq("")).any().any():
        raise ValueError("Etichetta, motivazione o fonte del ruolo mancante")
    if not catalog.primary_role.isin(SCOUTING_ROLES).all():
        raise ValueError("Categoria di scouting non supportata")
    for row in catalog.itertuples(index=False):
        secondary = [value.strip() for value in row.secondary_roles.split("|") if value.strip()]
        if (len(secondary) != len(set(secondary)) or row.primary_role in secondary
                or not set(secondary).issubset(SCOUTING_ROLES)):
            raise ValueError("Ruoli secondari incoerenti")
    return catalog


def apply_role_catalog(roles, players, catalog):
    """Un unico gruppo primario per giocatore; nessuna duplicazione per secondari.

    role_verified e relative esposizioni continuano a descrivere il dettaglio
    tattico disponibile nelle finestre. Non attestano il ruolo primario curato.
    """
    selected = catalog.loc[catalog.player_id.isin(players.entity_id)].set_index("player_id")
    lookup = players.set_index("entity_id")
    for pid, row in selected.iterrows():
        if row.player_name != lookup.loc[pid, "player_name"] or row.team_name != lookup.loc[pid, "team_name"]:
            raise ValueError("Catalogo ruoli non coerente con identità o squadra del match")
    fields = {"primary_role": "scouting_primary_role", "secondary_roles": "scouting_secondary_roles",
              "role_detail": "scouting_role_detail", "classification_note": "scouting_role_note",
              "source_reference": "scouting_role_source"}
    players = players.merge(selected[list(fields)].rename(columns=fields), left_on="entity_id", right_index=True,
                            how="left", validate="one_to_one")
    roles = roles.copy()
    roles["context_role_group"] = roles.role_group
    primary = roles.entity_id.map(selected.primary_role)
    available = primary.notna()
    roles.loc[available, "role_group"] = primary.loc[available]
    roles.loc[available, "role_usable"] = True
    roles["scouting_role_source"] = roles.entity_id.map(selected.source_reference).fillna("")
    return roles, players
