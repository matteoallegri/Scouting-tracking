"""Gruppi espliciti di variabili; non fattori latenti stimati sul campione."""

from src.scouting_tracking.config import FEATURES as TRACKING_FEATURES, FAMILIES as TRACKING_FAMILIES, features_for_phase as tracking_features

EVENT_FAMILIES = {
    "distribution": "Distribuzione e progressione",
    "creation": "Creazione offensiva",
    "recovery": "Recupero e opposizione",
    "protection": "Protezione della porta",
}
FAMILY_PHASE = {"distribution": "in_possession", "creation": "in_possession",
                "recovery": "out_of_possession", "protection": "out_of_possession"}
EVENT_FEATURES = {
    "event_pass_accuracy_pct": ("Precisione dei passaggi", "%", "distribution"),
    "event_final_third_passes": ("Passaggi riusciti nell'ultimo terzo", "azioni/partita", "distribution"),
    "event_progressive_carries": ("Conduzioni progressive", "azioni/partita", "distribution"),
    "event_shot_assists": ("Passaggi che portano al tiro", "azioni/partita", "creation"),
    "event_big_chances_created": ("Grandi occasioni create", "azioni/partita", "creation"),
    "event_penalty_area_entries": ("Ingressi del pallone in area", "azioni/partita", "creation"),
    "event_box_touches": ("Tocchi in area avversaria", "tocchi/partita", "creation"),
    "event_recoveries": ("Recuperi di palla", "azioni/partita", "recovery"),
    "event_interceptions": ("Intercetti", "azioni/partita", "recovery"),
    "event_tackles": ("Contrasti", "azioni/partita", "recovery"),
    "event_defensive_duels_won": ("Duelli difensivi vinti", "azioni/partita", "recovery"),
    "event_clearances": ("Respinte difensive", "azioni/partita", "protection"),
    "event_shot_blocks": ("Tiri avversari bloccati", "azioni/partita", "protection"),
    "event_cross_blocks": ("Cross avversari bloccati", "azioni/partita", "protection"),
}
STAT_FIELDS = {
    "event_final_third_passes": "successfulFinalThirdPasses",
    "event_progressive_carries": "progressiveCarries",
    "event_shot_assists": "shotCreated",
    "event_big_chances_created": "bigChanceCreated",
    "event_penalty_area_entries": "penAreaEntries",
    "event_box_touches": "touchesInOppBox",
    "event_recoveries": "ballRecovery",
    "event_interceptions": "interception",
    "event_tackles": "totalTackle",
    "event_clearances": "totalClearance",
    "event_shot_blocks": "outfielderBlock",
    "event_cross_blocks": "blockedCross",
}
FEATURES = {**TRACKING_FEATURES, **EVENT_FEATURES}
FAMILIES = {**TRACKING_FAMILIES, **EVENT_FAMILIES}


def families_for_phase(phase):
    return [*TRACKING_FAMILIES, *(key for key in EVENT_FAMILIES
                                if phase == "all" or FAMILY_PHASE[key] == phase)]


def features_for_phase(phase):
    # I referti hanno granularità partita, non finestre di transizione.
    events = [key for key, (_, _, family) in EVENT_FEATURES.items()
              if phase in ("all", FAMILY_PHASE[family])]
    return tracking_features(phase) + events


FAMILY_HELP = {
    "distribution": "Confronta precisione dei passaggi (riusciti / tentati × 100), passaggi riusciti con arrivo nell'ultimo terzo e conduzioni progressive definite dal fornitore.",
    "creation": "Confronta passaggi che portano al tiro (assist inclusi), grandi occasioni create, passaggi da fuori a dentro l'area e tocchi in area avversaria. Gli ingressi riguardano il pallone tramite passaggio, non le corse senza palla.",
    "recovery": "Confronta recuperi, intercetti, contrasti e duelli difensivi vinti. Questi ultimi comprendono contrasti con qualifica difensiva e duelli aerei difensivi vinti; escludono dribbling e duelli offensivi. I contrasti rientrano anche nei duelli: le misure possono essere correlate.",
    "protection": "Confronta respinte difensive, tiri avversari bloccati dal difendente e cross avversari bloccati. I tiri del giocatore bloccati da un avversario sono esclusi. Più interventi non significa automaticamente difendere meglio.",
}
