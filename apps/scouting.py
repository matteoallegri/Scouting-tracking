"""Dashboard locale: profili e compatibilità da tracking e azioni evento.

Avvio: venv/bin/python -m streamlit run apps/scouting.py
"""

from pathlib import Path as _Path
import sys as _sys

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import importlib
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit può rieseguire questa pagina mantenendo i moduli importati in memoria.
# Ricarica prima il catalogo e poi i suoi utilizzatori: dati e definizioni devono
# condividere la stessa versione anche dopo un aggiornamento delle metriche.
# Aggiorna anche la presentazione, evitando che restino etichette già rimosse.
for _module_name in ("src.scouting_tracking.config", "src.scouting_events.catalog", "src.scouting_events.data",
                     "src.scouting_tracking.ranking", "src.scouting_tracking.plots",
                     "src.scouting_role_search",
                     "src.scouting_general_ranking",
                     "apps.scouting_ui"):
    importlib.reload(importlib.import_module(_module_name))

from src.scouting_tracking.config import ScoutingConfig, PHASES
from src.scouting_tracking.ranking import active_features, score_profiles
from src.scouting_events.catalog import FEATURES, FAMILIES, EVENT_FEATURES, EVENT_FAMILIES, FAMILY_HELP, families_for_phase
from src.scouting_events.data import attach_events, bundle_version as event_bundle_version, load_verified_bundle as load_events
from apps.scouting_ui import AMBER, COLORS, MATCH_COLORS, style, compatibility_bars, inject_styles, render_intro, render_reference, match_legend, shortlist_columns
from src.scouting_data_access import bundle_version, load_verified_bundle
from src.scouting_tracking.selection import filter_candidates, selected_match_minutes
from src.scouting_role_search import DEFAULT_SEARCH_ROLES, comparison_view, fit_search_scalers, policy_version, role_mask
from src.scouting_general_ranking import DEFAULT_DIRECTIONS, DIRECTION_LABELS, rank_profiles, ranking_features

ROOT = _PROJECT_ROOT
OUTPUT = ROOT / "outputs/scouting_tracking"
EVENT_OUTPUT = ROOT / "outputs/scouting_events"

st.set_page_config(page_title="Tracking metrics a supporto dello scouting", page_icon="⚽", layout="wide")
inject_styles(ROOT)


@st.cache_data(show_spinner=False)
def load_data(version, events_version, role_policy):
    manifest, profiles, _, correlations = load_verified_bundle(OUTPUT, ROOT)
    profiles = attach_events(profiles, load_events(EVENT_OUTPUT, ROOT))
    config = ScoutingConfig(**{**manifest["config"], "matches": tuple(manifest["config"]["matches"])})
    scalers = fit_search_scalers(profiles, config)
    return manifest, profiles, scalers, correlations


FEATURE_TITLES = {
    "speed_p95": "Velocità nelle corse più rapide",
    "high_speed_m_per_min": "Corsa ad alta velocità",
    "acceleration_events_per_match": "Accelerazioni sostenute",
    "braking_events_per_match": "Frenate sostenute",
    "relative_lateral_abs_m": "Distanza laterale dai compagni",
    "relative_forward_m": "Posizione avanzata rispetto ai compagni",
    "hull_share_mean": "Contributo allo spazio occupato",
    "attack_progress_m_per_opportunity": "Avanzamento dopo il recupero",
    "defense_retreat_m_per_opportunity": "Arretramento dopo la perdita",
}
FEATURE_TITLES.update({key: value[0] for key, value in EVENT_FEATURES.items()})
DASHBOARD_PHASES = {"in_possession": PHASES["in_possession"], "out_of_possession": PHASES["out_of_possession"], "all": "Tutte le fasi"}
DATA_SOURCES = {"tracking": "Solo tracking", "events": "Solo eventi", "both": "Tracking ed eventi"}
SOURCE_DESCRIPTIONS = {"tracking": "dati di tracking", "events": "dati evento", "both": "dati di tracking ed evento"}


def remember_weight(family):
    # Lo stato indipendente conserva il peso anche quando il widget è nascosto.
    st.session_state.setdefault("scouting_dimension_weights", {})[family] = st.session_state["weight_" + family]


def remember_direction(feature):
    st.session_state.setdefault("scouting_ranking_directions", {})[feature] = st.session_state["ranking_direction_" + feature]


def team_name(value):
    return {"France": "Francia", "Spain": "Spagna"}.get(value, value)


def match_name(value):
    return {"wc2022": "Finale 2022", "wc2026": "Finale 2026"}.get(value, "Partita del dataset")


def period_name(value):
    return {0: "Intera partita", 1: "Primo tempo", 2: "Secondo tempo",
            3: "Primo supplementare", 4: "Secondo supplementare"}.get(int(value), f"Periodo {int(value)}")


def position_name(value):
    positions = {"Attacker": "Attaccante", "Striker": "Attaccante", "Defender": "Difensore",
                 "Midfielder": "Centrocampista", "Attacking Midfielder": "Trequartista",
                 "Defensive Midfielder": "Mediano", "Goalkeeper": "Portiere", "Non disponibile": "Ruolo da verificare"}
    sides = {"Centre": "centro", "Left": "sinistra", "Right": "destra"}
    result = []
    for detail in str(value).split(" | "):
        position, _, side = detail.partition(" — ")
        side_label = " / ".join(sides.get(item.strip(), "da verificare") for item in side.split("/")) if side.strip() else "lato da verificare"
        result.append(f"{positions.get(position.strip(), 'Ruolo da verificare')} · {side_label}")
    return "; ".join(result)


def readable_table(data, columns):
    result = data[list(columns)].copy()
    for column, formatter in (("team_name", team_name), ("match_id", match_name), ("period_id", period_name),
                              ("role_details", position_name), ("phase", {**PHASES, **DASHBOARD_PHASES}.get)):
        if column in result:
            result[column] = result[column].map(formatter)
    if "score_status" in result:
        reasons = {"Esposizione insufficiente": "Minuti osservati insufficienti nella fase scelta",
                   "Caratteristiche insufficienti": "Mancano una o più misure necessarie al confronto",
                   "Ruolo non verificato": "Il ruolo non è ricostruibile con sufficiente certezza"}
        reasons["Ruolo non assegnabile"] = "Non è disponibile una categoria di ruolo utilizzabile"
        result["score_status"] = result.score_status.map(lambda value: reasons.get(value, value))
    return result.rename(columns=columns)


def unavailable_reason(reason):
    if "<3" in reason:
        return "Meno di tre osservazioni valide nel gruppo di confronto."
    return "I valori sono uguali nel gruppo: questa misura non aiuta a distinguere i profili."


def label(row):
    return f"{row.player_name} · {team_name(row.team_name)} · {match_name(row.match_id)}"


def axis_label(feature):
    return f"{FEATURE_TITLES[feature]} ({FEATURES[feature][1]})"


def profile_scatter(profiles, x_feature, y_feature, reference=None, *, events_only=False, score_label="Compatibilità"):
    figure = go.Figure()
    minutes_column = "minutes_played_match" if events_only else "observed_minutes"
    minutes_label = "Minuti giocati nella partita" if events_only else "Minuti osservati"
    x_label, y_label = axis_label(x_feature), axis_label(y_feature)
    coordinates_hover = f"<br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}"
    for index, (match, group) in enumerate(profiles.groupby("match_id", sort=True)):
        figure.add_trace(go.Scatter(
            x=group[x_feature].tolist(), y=group[y_feature].tolist(), mode="markers", name=match_name(match),
            text=(group.player_name + " · " + group.team_name.map(team_name)).tolist(),
            customdata=group[[minutes_column, "score"]].to_numpy().tolist(),
            marker=dict(size=(group[minutes_column].pow(.5) * 2.8).clip(7, 23).tolist(),
                        color=MATCH_COLORS.get(match, COLORS[index % len(COLORS)]), opacity=.85, line=dict(width=1, color="#D6E3F5")),
            hovertemplate="%{text}" + coordinates_hover + f"<br>{minutes_label}: %{{customdata[0]:.1f}}<br>{score_label}: %{{customdata[1]:.1f}}<extra></extra>",
        ))
    if reference is not None:
        figure.add_trace(go.Scatter(
            x=[reference[x_feature]], y=[reference[y_feature]], mode="markers", name="Riferimento",
            text=[f"{reference['player_name']} · {team_name(reference['team_name'])} · {match_name(reference['match_id'])}"],
            marker=dict(symbol="star", size=22, color=AMBER, line=dict(width=1, color="#FFFFFF")),
            hovertemplate="%{text}<br>Giocatore di riferimento" + coordinates_hover + "<extra></extra>",
        ))
    figure.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return style(figure)


def render_method(base, scaler, config, phase, selected_role, minimum_played, budget_eur, data_source, candidate_roles, *, general_mode=False, directions=None, keys=()):
    st.subheader("Dal movimento osservato a una shortlist di scouting")
    st.write("Questo progetto affianca alle posizioni dei giocatori le azioni registrate negli eventi per descrivere come i calciatori si muovono, occupano il campo e partecipano al gioco. L'obiettivo è individuare, fra i giocatori disponibili, quelli il cui comportamento assomiglia al profilo che stiamo cercando.")
    st.write("Un esempio: cerchiamo un esterno che mantenga ampiezza, accompagni il recupero della palla e rientri dopo una perdita. Scegliamo un riferimento, osserviamo queste caratteristiche e decidiamo quanto devono pesare nel confronto.")
    st.write("Sono disponibili tre modalità: somiglianza a un giocatore osservato, somiglianza a un profilo ideale manuale e Classifica generale. Quest'ultima ordina le prestazioni secondo indicatori, versi da premiare e pesi scelti, senza richiedere un giocatore di riferimento.")

    st.markdown("### 0. Il campione analizzato e l'estensione del metodo")
    st.write("Questa versione analizza tracking e statistiche evento Opta delle finali 2022 e 2026 presenti nel dataset e confronta i giocatori osservati in queste due partite. Il metodo può essere esteso a intere stagioni alimentando la pipeline con nuove gare, eventi e relativi dati su giocatori, minuti giocati e valori di mercato, organizzati in un formato coerente e distinti per stagione e competizione. Le logiche di calcolo e confronto restano le stesse: profili, scale di confronto e punteggi vengono ricalcolati sul nuovo campione.")

    st.markdown("### 1. Quali dati abbiamo")
    observed = base.loc[base.phase.eq("all")].drop_duplicates(["match_id", "entity_id"])
    st.write("Il tracking registra dove si trovano giocatori e palla sul campo, 25 volte al secondo. Le posizioni sono espresse in metri e descrivono il movimento sul piano del terreno. Usiamo le traiettorie già pulite e le informazioni su squadre, formazioni e cambi.")
    st.write(f"Il punto di partenza è quindi costituito da {len(observed)} osservazioni giocatore-partita, contando anche i subentrati. Un calciatore presente in entrambe le finali ha due osservazioni distinte. I filtri selezionano una parte di questo insieme; nessun giocatore esterno alle due finali entra nella shortlist.")
    st.info("Per le misure di tracking analizziamo soltanto il gioco attivo osservato. Le pause e i movimenti a palla ferma sono esclusi intenzionalmente: distanze e minuti osservati si riferiscono a questi tratti. Le statistiche evento descrivono invece le azioni registrate nell'intera partita, comprese le riprese di gioco.")
    st.write("Per la selezione sommiamo, per ciascun giocatore, i minuti da referto delle partite selezionate, contando ogni partita una sola volta. Se sono selezionate entrambe le finali, Messi ha 120 + 120 = 240 minuti; scegliendo una sola finale ne ha 120. Con un dataset stagionale, il totale sarà distinto per stagione e competizione. Questi minuti sono distinti dai minuti di tracking osservati nella fase: usiamo questi ultimi per rapportare la distanza ad alta velocità.")
    market = observed.drop_duplicates("entity_id")
    st.write("I dati sui valori di mercato dei giocatori provengono da Transfermarkt.")
    with st.expander("Fonti e disponibilità dei valori di mercato"):
        st.dataframe(readable_table(market.sort_values(["team_name", "player_name"]), {
            "player_name": "Giocatore", "team_name": "Nazionale", "market_value_m_eur": "Valore di mercato (M€)"}),
            hide_index=True, width="stretch")
        st.caption("Un valore mancante non diventa zero. Un giocatore ritirato può restare un riferimento storico di movimento; senza una quotazione corrente non entra nei candidati selezionati con il budget.")

    st.markdown("### 2. Come passiamo dalle posizioni al confronto")
    steps = [("Posizioni", "Dove si trovano giocatori e palla nel tempo."),
             ("Misure", "Quanto corrono e come occupano lo spazio."),
             ("Profili", "Riepiloghi per giocatore, ruolo e fase di gioco."),
             ("Criterio", "Somiglianza a un riferimento oppure indicatori da premiare."),
             ("Risultato", "Profili ordinati secondo la modalità e i pesi scelti.")]
    for index, (column, (title, text)) in enumerate(zip(st.columns(5), steps), start=1):
        with column:
            with st.container(border=True):
                st.markdown(f"**{index}. {title}**")
                st.caption(text)
    st.write("Un profilo riassume il comportamento di un giocatore in una partita, in un ruolo e in una fase. Le viste dei singoli tempi servono ad approfondire quel comportamento: più righe dello stesso giocatore non rappresentano nuovi candidati.")
    st.write("Dallo spostamento fra posizioni consecutive e dal tempo trascorso ricaviamo la velocità. Dal cambiamento della velocità ricaviamo accelerazioni e frenate; dalle posizioni simultanee dei compagni ricaviamo la forma della squadra. Le traiettorie vengono regolarizzate per attenuare le oscillazioni di misura prima di calcolare queste grandezze.")
    st.write("Per confrontare attacchi in direzioni opposte, consideriamo sempre positivo il verso in cui attacca la squadra del giocatore. Se una traiettoria si interrompe, non calcoliamo velocità o spostamenti attraverso il tratto non osservato.")

    st.markdown("### 3. Le nove caratteristiche del profilo")
    st.write("Le due fasi condividono quattro misure di corsa e tre di spazio. A queste si aggiunge una misura di transizione: avanzamento dopo il recupero in possesso, arretramento dopo la perdita in non possesso. Ogni fase usa al massimo otto caratteristiche. La vista Tutte le fasi usa fino a nove caratteristiche, includendo entrambe le misure di transizione.")
    st.markdown("**Cinematica: intensità e cambi di velocità**")
    motion = [
        {"Misura": FEATURE_TITLES["speed_p95"], "Che cosa significa": "È la soglia superata solo nel 5% circa delle osservazioni più veloci. Descrive le corse rapide senza dipendere da un singolo picco.", "Unità": "metri al secondo"},
        {"Misura": FEATURE_TITLES["high_speed_m_per_min"], "Che cosa significa": f"Metri percorsi ad almeno {config.high_speed_threshold_m_s * 3.6:g} km/h ({config.high_speed_threshold_m_s:.4f} metri al secondo), sprint inclusi, per ogni minuto osservato nella fase scelta.", "Unità": "metri al minuto"},
        {"Misura": FEATURE_TITLES["acceleration_events_per_match"], "Che cosa significa": f"Conta, per partita, gli episodi nella fase scelta in cui aumenta la velocità a un ritmo di almeno {config.acceleration_threshold_m_s2:g} metri al secondo per secondo, per almeno {config.episode_min_duration_s:g} secondi.", "Unità": "episodi per partita"},
        {"Misura": FEATURE_TITLES["braking_events_per_match"], "Che cosa significa": f"Conta, per partita, gli episodi nella fase scelta in cui diminuisce la velocità a un ritmo di almeno {config.braking_threshold_m_s2:g} metri al secondo per secondo, per almeno {config.episode_min_duration_s:g} secondi.", "Unità": "episodi per partita"},
    ]
    st.table(pd.DataFrame(motion), hide_index=True, width="stretch")
    st.caption(f"Per accelerazioni e frenate contiamo episodi continui, non singole rilevazioni; la velocità deve essere almeno {config.episode_min_speed_m_s:g} metri al secondo. Ogni episodio è assegnato alla fase in cui inizia.")
    st.write("Accelerazioni e frenate sono espresse come episodi divisi per il numero di partite rappresentate nel profilo. Nel prototipo ogni profilo riguarda una sola partita, quindi il valore coincide con gli episodi osservati nella fase scelta di quella gara. Anche la vista di un singolo tempo conta una partita e comprende solo gli episodi di quel tempo. Un giocatore presente in entrambe le finali mantiene due profili distinti. La misura risente del tempo effettivamente osservato: chi gioca meno ha meno occasioni di accumulare episodi.")
    st.markdown("**Spazio: posizione del giocatore rispetto ai compagni**")
    space = [
        {"Misura": FEATURE_TITLES["relative_lateral_abs_m"], "Che cosa significa": "Quanto si trova lateralmente distante dal centro medio degli altri compagni. Può aiutare a descrivere l'ampiezza del suo posizionamento.", "Unità": "metri"},
        {"Misura": FEATURE_TITLES["relative_forward_m"], "Che cosa significa": "Quanto è avanti o indietro rispetto al centro medio degli altri compagni. Un valore positivo indica una posizione più avanzata nel verso d'attacco.", "Unità": "metri"},
        {"Misura": FEATURE_TITLES["hull_share_mean"], "Che cosa significa": "Quanto si riduce l'area che racchiude la squadra se togliamo quel giocatore, lasciando gli altri fermi. La misura descrive il peso geometrico della sua posizione.", "Unità": "quota da 0 a 1"},
    ]
    st.table(pd.DataFrame(space), hide_index=True, width="stretch")
    st.write("Il Convex Hull è il più piccolo poligono convesso che contiene le posizioni dei giocatori di movimento della squadra in un dato istante. Il contributo individuale è la riduzione relativa della sua area quando si esclude quel giocatore, mantenendo fisse le posizioni degli altri: (area della squadra − area senza il giocatore) / area della squadra. Una riduzione del 10% corrisponde a un contributo di 0,10. Il valore del profilo è la media dei contributi nei campioni validi.")
    st.caption(f"I portieri sono esclusi da area e centro della squadra. Il centro usato per il singolo giocatore esclude anche lui stesso. Le misure spaziali sono campionate ogni {config.spatial_step_frames / 25:g} secondi: servono almeno {config.min_outfield_players} giocatori di movimento validi della stessa squadra, incluso quello analizzato, e almeno {config.min_spatial_samples} valori per una media. Un'area più grande non è automaticamente un vantaggio tattico.")
    st.markdown("**Transizioni: risposta al cambio di possesso**")
    transitions = [
        {"Misura": FEATURE_TITLES["attack_progress_m_per_opportunity"], "Che cosa significa": f"Spostamento medio verso la porta avversaria nei {config.transition_window_s:g} secondi successivi a un recupero. Un valore negativo indica uno spostamento nella direzione opposta.", "Unità": "metri per opportunità"},
        {"Misura": FEATURE_TITLES["defense_retreat_m_per_opportunity"], "Che cosa significa": f"Spostamento medio verso la propria porta nei {config.transition_window_s:g} secondi successivi a una perdita. Un valore negativo indica uno spostamento nel verso d'attacco.", "Unità": "metri per opportunità"},
    ]
    st.table(pd.DataFrame(transitions), hide_index=True, width="stretch")
    st.write(f"Una opportunità è una finestra intera di {config.transition_window_s:g} secondi osservata per quel giocatore nello stesso ruolo, senza interruzioni o un nuovo cambio di possesso. Servono almeno {config.transition_min_opportunities} opportunità per calcolare la media. Dividiamo per le opportunità realmente osservate per quel giocatore, non per tutti i recuperi o le perdite della squadra.")
    st.caption("Le transizioni sono sottoinsiemi delle due fasi: in possesso usiamo l'avanzamento dopo il recupero; in non possesso usiamo l'arretramento dopo la perdita. Nella vista Tutte le fasi usiamo entrambe le misure. I minuti delle transizioni sono già compresi nelle fasi e non si sommano una seconda volta.")

    st.markdown("**Le dimensioni evento: che cosa fa il giocatore con e contro il pallone**")
    st.write("Ai movimenti affianchiamo le statistiche Opta disponibili per entrambe le finali. Distribuzione e progressione e Creazione offensiva descrivono le azioni in possesso; Recupero e opposizione e Protezione della porta descrivono le azioni difensive. Nella barra dei pesi compaiono solo le dimensioni pertinenti alla fase scelta; Tutte le fasi le comprende tutte e quattro, insieme alle tre dimensioni di tracking.")
    st.table(pd.DataFrame([{"Dimensione": FAMILIES[key], "Che cosa descrive": description}
                           for key, description in FAMILY_HELP.items()]), hide_index=True, width="stretch")
    st.write("Per gli eventi la fase è una categoria funzionale del gesto: un recupero viene classificato come azione difensiva, anche se consegna il possesso alla propria squadra. Usiamo i totali di partita del referto, supplementari inclusi e rigori di spareggio esclusi. Questi totali comprendono anche le riprese di gioco e non sono ritagliati sui soli secondi di tracking valido. Non attribuiamo ai referti una precisione temporale che non hanno: le due fonti descrivono lo stesso giocatore nella stessa gara, con coperture diverse.")
    st.write("I conteggi evento sono espressi per partita, senza dividerli per i minuti della fase. La precisione dei passaggi è passaggi riusciti / tentati × 100; senza tentativi resta mancante. Un conteggio omesso nel referto vale zero soltanto per un giocatore effettivamente entrato, con statistiche presenti e misura documentata nel file della partita. I valori di partita non sono ripartiti nei singoli tempi o nelle finestre di transizione.")
    st.write("I passaggi che portano al tiro includono gli assist, secondo shotCreated. Le grandi occasioni e le conduzioni progressive seguono la definizione del fornitore. Gli ingressi in area sono passaggi da fuori a dentro l'area, anche non riusciti; i tocchi sono nella sola area avversaria. I duelli difensivi comprendono i contrasti con qualifica difensiva e gli aerei difensivi vinti. Per il contrasto, anche un esito in cui la squadra non conserva la palla indica che il pallone è stato sottratto legalmente all'avversario. Escludiamo duelli offensivi, dribbling e 50/50, che il fornitore registra separatamente.")
    st.caption("Alcune misure si sovrappongono: i contrasti sono anche parte dei duelli difensivi, e le grandi occasioni possono essere parte dei passaggi che portano al tiro. Le dimensioni sono gruppi interpretabili di variabili, non fattori latenti appresi statisticamente. Cercare somiglianza non equivale a premiare il numero più alto di interventi.")

    st.markdown("### 4. Perché separiamo ruoli e fasi di gioco")
    st.write("Un esterno e un difensore centrale svolgono compiti diversi; anche lo stesso giocatore può muoversi in modo diverso quando la squadra ha la palla o deve recuperarla. Separiamo le fasi, ma usiamo i ruoli come un perimetro flessibile di ricerca. Il Ruolo osservato serve a scegliere il riferimento; Ruoli dei candidati permette di allargare il confronto ad altre categorie, considerando sia i ruoli primari sia quelli secondari.")
    st.write("La ricerca parte da gruppi ampi: per un esterno offensivo comprende esterni, trequartisti e attaccanti; per un trequartista comprende anche i centrocampisti; per un esterno difensivo comprende centrali e giocatori offensivi di fascia. Puoi modificare le categorie o selezionarle tutte. Sono ambiti da esplorare, non nuovi ruoli certificati per ogni giocatore. Per esempio, Messi può uscire cercando alternative a Yamal, anche se il suo ruolo primario è trequartista/seconda punta.")
    st.write("Il ruolo non aggiunge bonus o penalità alla somiglianza. Tutti i candidati usano le stesse scale del gruppo esteso iniziale del riferimento e della fase. Cambiare i ruoli dei candidati filtra l'elenco senza ricalcolare le scale. Un giocatore compatibile con più categorie compare una sola volta per partita, mantenendo ruolo primario, identità e metriche originali.")
    st.table(pd.DataFrame([
        {"Fase": PHASES["in_possession"], "Quali tratti comprende": "La squadra del giocatore ha il pallone; include le transizioni offensive."},
        {"Fase": PHASES["out_of_possession"], "Quali tratti comprende": "La palla è della squadra avversaria; include le transizioni difensive."},
        {"Fase": DASHBOARD_PHASES["all"], "Quali tratti comprende": "Tutto il gioco attivo osservato del giocatore, con e senza palla; include entrambe le transizioni."},
    ]), hide_index=True, width="stretch")
    st.write("Tutte le fasi ricostruisce il profilo usando insieme tutti gli intervalli validi osservati. Distanze, tempi ed episodi sono aggregati; la corsa ad alta velocità usa il tempo osservato totale, accelerazioni e frenate il numero di partite, il percentile di velocità è ricalcolato su tutti i campioni e le medie spaziali usano tutti i rispettivi campioni validi. Non facciamo la media semplice dei due profili di fase e non sommiamo i loro punteggi. Il confronto usa scale proprie del totale, nello stesso ruolo. Le pause restano escluse.")
    st.caption(f"Nei tratti di possesso appena recuperato osserviamo la transizione offensiva; nei tratti di non possesso appena iniziati osserviamo quella difensiva, fino a {config.transition_window_s:g} secondi dal cambio. Per riconoscere un cambio richiediamo un possesso noto e stabile almeno {config.possession_min_duration_s:g} secondi prima e dopo, senza pause. La finestra si interrompe se cambia ancora il possesso, diventa ignoto o il gioco si ferma. I tratti con possesso ignoto sono esclusi da entrambe le fasi.")
    st.write("Ogni giocatore conserva una categoria primaria, assegnata con una revisione delle informazioni disponibili e dei casi dubbi, e gli eventuali ruoli secondari. La categoria rimane la stessa nelle due finali. Distinguiamo difensori centrali, esterni difensivi, centrocampisti centrali, trequartisti o seconde punte, esterni offensivi e attaccanti centrali. Le misure di movimento continuano a provenire dal tracking.")
    st.write("Per esempio, Acuña rientra negli esterni difensivi, De Paul nei centrocampisti e Álvarez negli attaccanti. Messi, Dybala, Griezmann e Dani Olmo appartengono al gruppo dei trequartisti e delle seconde punte. Sono categorie operative per lo scouting, non ruoli esclusivi o ricostruzioni automatiche del compito svolto a ogni istante.")
    st.write("Conserviamo separatamente il contesto della partita. Camavinga è un centrocampista di riferimento, ma nella finale 2022 è stato impiegato da terzino: questo va considerato leggendo il suo profilo. Anche Thuram, Kolo Muani e gli altri giocatori duttili possono avere un compito nel match diverso dal gruppo primario. I ruoli secondari descrivono questa possibilità; non duplicano il giocatore, i minuti o il punteggio in altri gruppi.")
    st.write("La distanza ad alta velocità viene rapportata ai minuti validi della fase scelta: per esempio, 120 metri in 4 minuti osservati diventano 30 metri al minuto. Accelerazioni e frenate sono invece rapportate alle partite: 12 accelerazioni nella fase scelta di una gara valgono 12 episodi per partita. Pochi minuti rimangono una base fragile per il confronto.")

    if general_mode:
        st.markdown("### 5. Come nasce il punteggio della classifica generale")
        st.write("Questa modalità non usa un riferimento individuale. Per ogni misura scegliamo se premiare valori più alti, valori più bassi oppure escluderla. Inizialmente premiamo valori più alti per cinematica, transizioni ed eventi; le tre misure di spazio sono escluse finché non ne scegliamo il verso. Questi criteri descrivono una preferenza operativa: più corsa, più arretramento o più interventi non sono automaticamente sinonimo di una prestazione migliore.")
        st.markdown("**Dai valori ai percentili.** Confrontiamo ogni valore con quelli disponibili nel gruppo fisso dello stesso ruolo di ricerca e della stessa fase. Assegniamo il rango medio ai valori uguali e riportiamo i ranghi su scala 0–100: 100 × (rango medio − 1) / (numero di valori validi − 1). Un minimo unico vale 0 e un massimo unico vale 100; i pari merito ricevono lo stesso valore. Se premiamo valori più bassi, usiamo 100 meno il percentile. Un candidato esterno al gruppo fisso viene collocato rispetto a quei valori, limitando il risultato fra 0 e 100.")
        st.markdown("**Dai percentili al totale.** Facciamo la media semplice dei percentili favorevoli delle misure attive dentro ogni dimensione; poi la media pesata delle dimensioni. I pesi sono riportati al 100% soltanto fra le dimensioni con almeno una misura utilizzabile e un verso attivo. Per esempio, una dimensione con peso effettivo del 40% e quattro misure attive assegna a ciascuna il 10% del totale. Il punteggio complessivo è su scala 0–100; non è necessariamente raggiunto da un giocatore del campione.")
        st.write("Misure costanti o con meno di tre valori validi nel gruppo sono escluse. Un candidato deve avere tutte le misure attive disponibili: un dato mancante non diventa zero. Disattivare una dimensione o escludere una misura elimina il relativo requisito. Il dettaglio del giocatore mostra valore originale, percentile favorevole e contributo in punti: la somma dei contributi restituisce il punteggio complessivo, salvo gli arrotondamenti di visualizzazione.")
        st.info("La classifica descrive le prestazioni nelle singole finali secondo i criteri impostati. Non stima il talento o la probabilità di successo; i conteggi per partita risentono dei minuti e delle opportunità disponibili. Un giocatore presente in entrambe le finali conserva due righe distinte.")
        if keys:
            st.write("**Criteri attivi:** " + "; ".join(f"{FEATURE_TITLES[key]}: {DIRECTION_LABELS[directions[key]].lower()}" for key in keys) + ".")
    else:
        render_similarity_method()

    st.markdown("### 6. Il confronto che hai impostato adesso")
    st.write("**Ruoli ammessi per i candidati:** " + (", ".join(candidate_roles) if candidate_roles else "nessuno") + ". È sufficiente una corrispondenza con il ruolo primario o con uno dei secondari.")
    st.write(f"**Dati selezionati: {DATA_SOURCES[data_source]}.** Il punteggio, i pesi, le caratteristiche del confronto e gli assi del grafico usano soltanto le fonti selezionate. Le dimensioni escluse per fonte o fase hanno peso effettivo zero; non richiedono misure complete per ottenere un punteggio. Cambiare fonte non ricostruisce le scale di confronto.")
    st.write(f"Il ruolo di ricerca è **{selected_role.lower()}**, nella vista **{DASHBOARD_PHASES[phase].lower()}**. Il gruppo esteso fisso usato per mettere le misure sulla stessa scala contiene **{scaler['population_size']} profili giocatore-partita**.")
    st.write("Le scale usano i profili dei ruoli affini previsti all'apertura per quel ruolo di ricerca, nella stessa fase e senza un taglio fisso sui minuti. I filtri, compresi i ruoli dei candidati, non ricostruiscono questo gruppo e non cambiano il metro di confronto.")
    if not general_mode:
        st.write("Il riferimento può restare sopra budget o avere un'esposizione diversa: serve a cercare alternative fra i candidati che rispettano i vincoli.")
    meaning = "segue i percentili e i criteri impostati" if general_mode else "descrive la somiglianza"
    st.write(f"La selezione attuale richiede almeno {minimum_played:g} minuti giocati complessivi nelle partite selezionate e un valore di mercato noto non superiore a {budget_eur / 1_000_000:g} milioni di euro per giocatore. Il totale dei minuti non dipende dalla fase, dai pesi o dalla disponibilità delle metriche di confronto. Il prezzo e i minuti giocati filtrano i candidati; il punteggio {meaning} nei {SOURCE_DESCRIPTIONS[data_source]} selezionati. Il limite di budget è incluso: un valore uguale al tetto è ammesso.")
    st.caption("I pesi della barra laterale sono proporzioni: vengono riportati a un totale del 100% fra le dimensioni attive. Un peso a zero disattiva quella dimensione. In possesso, Transizioni usa l'avanzamento dopo il recupero; in non possesso usa l'arretramento dopo la perdita. In Tutte le fasi usa entrambe, distribuendo il peso della dimensione fra le misure attive.")
    unavailable = [{"Misura non utilizzabile nel gruppo": FEATURE_TITLES[key], "Motivo": unavailable_reason(item["reason"])}
                   for key, item in scaler["features"].items() if not item["available"]]
    if unavailable:
        st.table(pd.DataFrame(unavailable), hide_index=True, width="stretch")

    st.markdown("### Come leggere le altre due schede")
    if general_mode:
        st.write("Nella Classifica trovi posizione, punteggio totale e punteggi delle dimensioni, con gli stessi filtri nei grafici, nella tabella e nel CSV. A parità di punteggio l'ordine considera i minuti da referto, poi nome e partita. Nel Profilo del giocatore trovi le misure e i contributi che spiegano il totale.")
    else:
        st.write("Nella Shortlist trovi i candidati più vicini al riferimento, i punteggi per dimensione e i minuti a supporto. Nel Confronto del profilo trovi la tabella delle caratteristiche, con unità, valori del candidato e del riferimento e differenze. Una differenza ampia richiede un'interpretazione rispetto al compito tattico desiderato.")
    st.write("Nel grafico Esplora le caratteristiche scegli due misure diverse fra quelle attive. Le opzioni seguono la fonte, la fase e le dimensioni con peso attivo; nella classifica seguono anche i versi scelti. Il grafico mostra i valori originali delle due misure; la vicinanza visiva non coincide necessariamente con il punteggio, che usa tutte le caratteristiche attive e i relativi pesi. Cambiare gli assi non modifica i filtri o i punteggi. Per il grafico servono almeno due caratteristiche attive. La stella compare soltanto nella modalità Giocatore osservato.")


def render_similarity_method():
    st.markdown("### 5. Come nasce il punteggio di compatibilità")
    st.markdown("**Scegliamo il riferimento.** Può essere il profilo di un giocatore in una delle due finali oppure una combinazione di obiettivi inseriti manualmente. I valori iniziali del profilo manuale sono quelli centrali del gruppo: sono un punto di partenza da modificare in base al compito tattico.")
    st.markdown("**Rendiamo confrontabili le unità.** Metri, velocità e conteggi hanno scale diverse. Per ogni misura guardiamo quanto variano i valori nella metà centrale del gruppo di confronto e usiamo questa variabilità come metro comune. Se quei valori coincidono, consideriamo la variabilità dell'intero gruppo. Se non c'è alcuna variabilità o abbiamo meno di tre valori, la misura viene esclusa.")
    st.markdown("**Misuriamo le differenze e applichiamo i pesi.** Per ogni caratteristica calcoliamo la differenza dal riferimento sulla scala comune. Le differenze vengono elevate al quadrato, mediate dentro ciascuna dimensione e combinate secondo i pesi scelti. Infine prendiamo la radice del risultato, ottenendo una distanza complessiva.")
    st.markdown("**Come pesano le singole variabili.** Il peso di ciascuna dimensione viene diviso in parti uguali fra le sue variabili attive. Nella cinematica, velocità al 95° percentile, distanza ad alta velocità, accelerazioni e frenate hanno quindi lo stesso peso fra loro, dopo averle portate sulla scala comune. Per esempio, se il peso effettivo della cinematica è il 40% e tutte e quattro le misure sono attive, ciascuna ha un coefficiente del 10% nel calcolo dello scostamento complessivo. Il contributo effettivo dipende anche da quanto il candidato si discosta dal riferimento su quella misura.")
    st.write("Aumentare il peso di una dimensione aumenta insieme l'importanza delle sue variabili rispetto alle altre, mantenendo uguale la ripartizione interna. La stessa regola vale per le dimensioni evento: per esempio, il peso di Creazione offensiva è diviso in parti uguali tra le sue misure attive. I pesi vengono riportati al 100% fra le dimensioni utilizzabili. Le dimensioni nascoste per la fase scelta hanno peso effettivo zero, anche se avevano un peso positivo in un'altra vista. Il confronto cerca somiglianza al riferimento: un peso maggiore non premia automaticamente chi corre di più o accumula più eventi.")
    st.markdown("**Trasformiamo la distanza in un punteggio leggibile.** Dividiamo 100 per uno più la distanza complessiva. Profili coincidenti ottengono 100; una distanza di 1 produce 50; una distanza di 2 produce circa 33. Più il punteggio è alto, più il profilo osservato è vicino al riferimento nelle misure attive.")
    st.info("Un punteggio di 80 non significa l'80% di probabilità di successo. Indica somiglianza al riferimento scelto. Cambiando riferimento o pesi può cambiare l'ordine dei candidati; più corsa o più ampiezza non sono sempre preferibili.")
    st.write("Tutte le dimensioni di tracking ed evento applicano la stessa logica alle rispettive caratteristiche. Il totale è calcolato dalle differenze pesate: non è la media semplice dei punteggi mostrati. Una dimensione senza variabilità nel gruppo non entra nel totale. La tabella del confronto affianca le misure del candidato e del riferimento, con la loro differenza nelle unità originali.")

st.title("Tracking metrics a supporto dello scouting")
render_intro()
st.caption("La shortlist include esclusivamente giocatori con tracking osservato nelle due finali del dataset, con minuti giocati e valore di mercato entro i criteri selezionati.")

if not (OUTPUT / "manifest.json").exists():
    st.info("Le analisi non sono ancora disponibili. La dashboard potrà mostrare i confronti quando sarà completata la preparazione dei dati del progetto.")
    st.stop()

try:
    manifest, profiles, scalers, _correlations = load_data(bundle_version(OUTPUT, ROOT), event_bundle_version(EVENT_OUTPUT, ROOT), policy_version())
except (OSError, ValueError, KeyError):
    st.error("I risultati devono essere aggiornati o verificati prima di mostrare il confronto. La preparazione dei dati del progetto è incompleta oppure i file non corrispondono alla versione di calcolo utilizzata.")
    st.stop()
config = ScoutingConfig(**{**manifest["config"], "matches": tuple(manifest["config"]["matches"])})
profiles["market_value_m_eur"] = profiles.market_value_eur / 1_000_000
base = profiles.loc[profiles.period_id.eq(0)].copy()
base["display_name"] = base.apply(label, axis=1)

with st.sidebar:
    st.markdown("### Costruisci il confronto")
    roles = sorted({item["role_group"] for item in scalers.values()})
    default_role = roles.index("Esterno offensivo") if "Esterno offensivo" in roles else 0
    selected_role = st.selectbox("Ruolo osservato", roles, index=default_role)
    phase = st.selectbox("Fase di gioco", list(DASHBOARD_PHASES), format_func=DASHBOARD_PHASES.get)
    mode = st.radio("Tipo di riferimento", ["Giocatore osservato", "Profilo ideale manuale", "Classifica generale"])
    general_mode = mode == "Classifica generale"
    score_label = "Punteggio" if general_mode else "Compatibilità"
    reference_slot = st.container() if mode == "Giocatore osservato" else None
    selected_matches = st.multiselect("Partite dei candidati", manifest["matches"], default=manifest["matches"], format_func=match_name)
    candidate_roles = st.multiselect("Ruoli dei candidati", roles, default=list(DEFAULT_SEARCH_ROLES[selected_role]),
                                    key="candidate_roles_" + selected_role,
                                    help="Ricerca estesa ai ruoli affini: basta che uno dei ruoli selezionati sia primario o secondario del giocatore. Puoi selezionare tutti i ruoli. Nessuna penalità di ruolo entra nel punteggio.")

scaler = scalers.get(selected_role + "|" + phase, {"role_group": selected_role, "phase": phase,
    "population_size": 0, "minimum_minutes": config.min_observed_minutes,
    "features": {}})
reference_cohort = base.loc[base.role_group.eq(selected_role) & base.phase.eq(phase)].copy()
cohort = comparison_view(base.loc[base.phase.eq(phase) & role_mask(base, candidate_roles)], selected_role)
with st.sidebar:
    st.markdown("### Criteri di selezione")
    # Intervallo fisso sul dataset: cambiare partite non deve azzerare o ridurre
    # automaticamente il requisito scelto (identità del widget Streamlit).
    played_totals = selected_match_minutes(base, manifest["matches"])
    maximum_minutes = max(1, int(math.ceil(played_totals.max()))) if played_totals.notna().any() else 1
    minimum_played = st.slider("Minuti giocati minimi nella competizione", 0, maximum_minutes, 0, step=1,
                               key="minimum_played_minutes",
                               help="Somma i minuti da referto delle partite selezionate, contando ogni gara una sola volta. La scala del cursore resta fissa sul dataset: la soglia scelta non cambia quando filtri le partite. Se supera i minuti disponibili, nessun candidato soddisfa il requisito.")
    maximum_budget_millions = float(math.ceil(base.market_value_eur.max() / 1_000_000))
    budget_millions = st.slider("Budget massimo per giocatore (milioni €)", 0.0, maximum_budget_millions, maximum_budget_millions, step=.1)
    budget_eur = round(budget_millions * 1_000_000, 2)
    data_source = st.radio("Dati per lo scouting", list(DATA_SOURCES), index=2,
                           format_func=DATA_SOURCES.get, key="scouting_data_source",
                           help="Scegli le fonti usate per il confronto. I pesi e le caratteristiche disponibili seguono sia questa scelta sia la fase di gioco.")
    source_families = {family for family in FAMILIES
                       if data_source == "both" or (family in EVENT_FAMILIES) == (data_source == "events")}
    source_features = [key for key, (_, _, family) in FEATURES.items() if family in source_families]
    st.markdown("### Peso delle dimensioni")
    weight_help = ("\n\n**Effetto del peso:** aumentarlo dà più importanza ai percentili di questa dimensione nella classifica. Il verso premiato si imposta nei Criteri della classifica. Con peso 0 la dimensione non entra nel punteggio."
                   if general_mode else "\n\n**Effetto del peso:** aumentarlo dà più importanza alla somiglianza in questa dimensione rispetto alle altre. Non premia automaticamente valori più alti. Con peso 0, la dimensione non partecipa al confronto.")
    transition_description = {
        "in_possession": "Nella fase in possesso considera l'avanzamento verso la porta avversaria dopo un recupero.",
        "out_of_possession": "Nella fase di non possesso considera l'arretramento verso la propria porta dopo una perdita.",
        "all": "Nella vista Tutte le fasi considera entrambe le misure: avanzamento verso la porta avversaria dopo un recupero e arretramento verso la propria porta dopo una perdita. Il peso di Transizioni è distribuito fra le due misure attive.",
    }[phase]
    dimension_help = {
        "motion": (
            "**Cinematica · Come si muove il giocatore**\n\n"
            "Confronta quattro misure nella fase scelta:\n"
            "- **Velocità P95:** la soglia superata nel 5% circa delle osservazioni più veloci.\n"
            f"- **Corsa ad alta velocità:** metri percorsi ad almeno {config.high_speed_threshold_m_s:g} m/s "
            f"({config.high_speed_threshold_m_s * 3.6:.1f} km/h) per minuto osservato.\n"
            f"- **Accelerazioni e frenate:** episodi di aumento o diminuzione della velocità di almeno "
            f"{config.acceleration_threshold_m_s2:g} e {config.braking_threshold_m_s2:g} m/s² rispettivamente, "
            f"sostenuti per almeno {config.episode_min_duration_s:g} secondi, "
            f"a velocità di almeno {config.episode_min_speed_m_s:g} m/s. Sono contati per partita, nella fase scelta; ogni profilo rappresenta una sola gara."
            + weight_help
        ),
        "space": (
            "**Spazio · Dove si posiziona rispetto alla squadra**\n\n"
            "Confronta tre misure medie nella fase scelta:\n"
            "- **Distanza laterale:** quanto si allontana lateralmente dal centro degli altri compagni.\n"
            "- **Posizione avanzata:** quanto è avanti o indietro rispetto agli altri compagni, nel verso d'attacco.\n"
            "- **Contributo al Convex Hull:** la quota di area che la squadra perderebbe togliendo quel giocatore "
            "e lasciando gli altri fermi. Descrive il peso geometrico della sua posizione.\n\n"
            "Il Convex Hull è l'area racchiusa dai giocatori più esterni. "
            "I portieri sono esclusi. Il centro di riferimento esclude anche il giocatore analizzato."
            + weight_help
        ),
        "transitions": (
            "**Transizioni · Come reagisce al cambio di possesso**\n\n"
            + transition_description
            + f" Misura lo spostamento longitudinale medio nei {config.transition_window_s:g} secondi successivi "
            "al cambio, in metri per opportunità. Un valore negativo indica movimento nel verso opposto.\n\n"
            f"Usiamo solo finestre intere e continue, senza un nuovo cambio di possesso; "
            f"servono almeno {config.transition_min_opportunities} opportunità osservate per giocatore. "
            "Le transizioni sono sottoinsiemi della fase selezionata."
            + weight_help
        ),
    }
    dimension_help.update({key: f"**{FAMILIES[key]}**\n\n{description}\n\n"
                           "Le statistiche evento descrivono le azioni nell'intera partita, supplementari inclusi. "
                           "La fase indica la funzione del gesto; i conteggi non sono limitati ai secondi di tracking osservato. "
                           "I conteggi sono per partita; la precisione è una percentuale." + weight_help
                           for key, description in FAMILY_HELP.items()})
    visible_families = [family for family in families_for_phase(phase) if family in source_families]
    weights = {key: 0 for key in FAMILIES}
    for key in visible_families:
        saved = st.session_state.get("scouting_dimension_weights", {})
        initial = st.session_state.get("weight_" + key, saved.get(key, int(round(config.family_weights.get(key, .25) * 100))))
        weights[key] = st.slider(FAMILIES[key], 0, 100, initial,
                       key="weight_" + key, help=dimension_help[key], on_change=remember_weight, args=(key,))
    st.caption("I pesi sono proporzioni: il loro totale viene riportato al 100%. La scala di confronto resta fissa quando filtri i candidati.")

# Vista locale del catalogo: nessun refit e nessuna modifica alle scale in cache.
scaler = {**scaler, "features": {key: item for key, item in scaler["features"].items()
                                 if FEATURES[key][2] in visible_families}}
keys = active_features(scaler, weights, feature_catalog=FEATURES)
directions = DEFAULT_DIRECTIONS.copy()
if general_mode:
    with st.expander("Criteri della classifica"):
        st.write("Scegli che cosa premiare per ogni misura: valori più alti, più bassi oppure esclusione. Inizialmente premiamo i valori più alti di cinematica, transizioni ed eventi. Le misure di spazio partono escluse perché dipendono dal compito tattico: puoi attivarle scegliendo un verso. Le misure costanti nel gruppo non entrano nel punteggio.")
        criterion_columns = st.columns(3)
        for index, key in enumerate(keys):
            saved = st.session_state.get("scouting_ranking_directions", {})
            initial = st.session_state.get("ranking_direction_" + key, saved.get(key, DEFAULT_DIRECTIONS[key]))
            directions[key] = criterion_columns[index % 3].selectbox(
                FEATURE_TITLES[key], list(DIRECTION_LABELS), index=list(DIRECTION_LABELS).index(initial),
                format_func=DIRECTION_LABELS.get, key="ranking_direction_" + key,
                on_change=remember_direction, args=(key,),
                help=f"Unità: {FEATURES[key][1]}. Il verso orienta il percentile; le misure attive della stessa dimensione hanno peso uguale.")
    keys = ranking_features(scaler, weights, directions)
    excluded_dimensions = [FAMILIES[family] for family in visible_families
                           if weights[family] > 0 and not any(FEATURES[key][2] == family for key in keys)]
    if excluded_dimensions:
        st.caption("Dimensioni senza misure attive nella classifica: " + ", ".join(excluded_dimensions)
                   + ". Il loro peso non entra nel totale. Puoi attivare una misura nei Criteri della classifica.")
if not keys:
    st.info("Seleziona una dimensione con almeno una caratteristica variabile disponibile." +
            (" Per la classifica serve anche un verso attivo nei Criteri della classifica." if general_mode else ""))

target_id = None
target = None
target_label = "Giocatori ordinati per indicatori e pesi" if general_mode else "Da impostare"
if mode == "Giocatore osservato":
    references = reference_cohort.loc[reference_cohort[keys].notna().all(axis=1)]
    if data_source == "events":
        references = references.sort_values(["minutes_played_match", "player_name", "match_id"], ascending=[False, True, True])
    else:
        references = references.sort_values("observed_minutes", ascending=False)
    if references.empty:
        reference_slot.info("Nessun riferimento ha tutte le caratteristiche necessarie. Puoi ridurre i pesi delle dimensioni non disponibili o usare obiettivi manuali.")
    else:
        reference_index = references.set_index("profile_id")
        reference_ids = references.profile_id.tolist()
        if st.session_state.get("reference_player") not in reference_ids:
            st.session_state["reference_player"] = reference_ids[0]
        reference_labels = reference_index.display_name.to_dict()
        target_id = reference_slot.selectbox("Giocatore di riferimento", reference_ids, format_func=reference_labels.get, key="reference_player")
        target = reference_index.loc[target_id].to_dict()
        target_label = target["display_name"]
elif mode == "Profilo ideale manuale" and keys:
    target = {}
    target_label = "Obiettivi manuali"
    with st.expander("Definisci il profilo ideale", expanded=True):
        st.caption("I valori iniziali sono quelli centrali del gruppo. Modificali in base al comportamento che cerchi: non sono obiettivi ideali scelti automaticamente dal modello.")
        columns = st.columns(3)
        for index, key in enumerate(keys):
            info = scaler["features"][key]
            _, unit, family = FEATURES[key]
            title = FEATURE_TITLES[key]
            bounds = {}
            if key in ("speed_p95", "high_speed_m_per_min", "acceleration_events_per_match", "braking_events_per_match", "relative_lateral_abs_m", "hull_share_mean"):
                bounds["min_value"] = 0.0
            if key == "hull_share_mean":
                bounds["max_value"] = 1.0
            if key in EVENT_FEATURES:
                bounds["min_value"] = 0.0
                if key == "event_pass_accuracy_pct":
                    bounds["max_value"] = 100.0
            target[key] = columns[index % 3].number_input(f"{title} ({unit})", value=float(info["median"]), key=f"ideal_{selected_role}_{phase}_{key}", **bounds)

ranking_detail = pd.DataFrame()
if general_mode and keys:
    target_label = "Giocatori ordinati per indicatori e pesi"
    ranked, ranking_detail = rank_profiles(cohort, base, scaler, weights, directions)
elif target is not None and keys:
    ranked, _ = score_profiles(cohort, target, scaler, weights, min_minutes=0.0,
                               feature_catalog=FEATURES, family_catalog=FAMILIES)
    if data_source == "events":
        ranked = ranked.sort_values(["score", "minutes_played_match", "player_name", "match_id"],
                                    ascending=[False, False, True, True], na_position="last")
else:
    ranked = cohort.copy()
    for column in ("score", "distance", *(key + "_score" for key in FAMILIES)):
        ranked[column] = float("nan")
    ranked["score_status"] = "Attiva almeno una misura e un verso" if general_mode else "Imposta un riferimento e almeno una dimensione utilizzabile"
visible = filter_candidates(ranked, selected_matches, minimum_played, budget_eur, minutes_source=base)
if target_id is not None:
    visible = visible.loc[~visible.profile_id.eq(target_id)]
eligible = visible.loc[visible.score.notna()].copy()
if general_mode:
    eligible["ranking_position"] = range(1, len(eligible) + 1)

with st.container(key="scouting-metrics"):
    metrics = st.columns(3)
metrics[0].metric("Profili in classifica" if general_mode else "Profili confrontabili", len(eligible), help="Giocatori-partita che soddisfano i filtri e hanno tutte le misure necessarie.")
metrics[1].metric("Campione di riferimento", scaler["population_size"], help="Profili usati per fissare la scala di confronto. Questo numero non cambia quando filtri le partite.")
metrics[2].metric("Caratteristiche attive", len(keys), help="Misure di tracking e statistiche evento delle dimensioni attive. La fase selezionata determina quali dimensioni evento partecipano; le misure costanti o con meno di tre valori sono escluse.")
render_reference(target_label, selected_role, f"{DASHBOARD_PHASES[phase]} · {DATA_SOURCES[data_source]}",
                 heading="Classifica generale" if general_mode else "Riferimento del confronto")
if general_mode:
    st.caption("Il punteggio è la media pesata dei percentili favorevoli nelle misure scelte, su scala 0–100. Un punteggio più alto indica risultati più alti secondo i criteri impostati. Ogni riga descrive un giocatore in una finale.")
else:
    st.caption("La compatibilità descrive somiglianza nelle caratteristiche selezionate; 100 indica coincidenza, non qualità assoluta o probabilità di successo.")

tabs = st.tabs(["Classifica" if general_mode else "Shortlist", "Profilo del giocatore" if general_mode else "Confronto del profilo", "Metodo e dati"])

with tabs[0]:
    left, right = st.columns([.9, 1.2], border=True)
    with left, st.container(key="scouting-ranking-panel"):
        st.subheader("Classifica generale" if general_mode else "Compatibilità con il riferimento")
        st.caption("I giocatori con il punteggio più alto nei criteri selezionati." if general_mode else "I candidati più vicini al profilo, ordinati per somiglianza.")
        match_legend()
        if eligible.empty:
            st.info("Nessun candidato soddisfa i criteri attuali. Controlla partite, minuti giocati, budget e caratteristiche disponibili.")
        else:
            st.plotly_chart(compatibility_bars(eligible, score_label=score_label), width="stretch", key="compatibility_bars", theme=None)
            if len(eligible) > 12:
                st.caption("Il grafico mostra i primi 12 candidati; la tabella contiene l'elenco completo.")
    with right, st.container(key="scouting-landscape-panel"):
        st.subheader("Esplora le caratteristiche")
        if len(keys) < 2:
            st.info("Per disegnare il grafico servono almeno due caratteristiche attive. Puoi attivare un'altra dimensione usando i pesi.")
        else:
            x_slot, y_slot = st.columns(2)
            if st.session_state.get("scatter_x_feature") not in keys:
                st.session_state["scatter_x_feature"] = "relative_lateral_abs_m" if "relative_lateral_abs_m" in keys else keys[0]
            x_feature = x_slot.selectbox("Asse orizzontale", keys, format_func=axis_label, key="scatter_x_feature",
                                       help="Scegli una delle caratteristiche attive nel confronto. La scelta modifica la vista del grafico.")
            y_options = [key for key in keys if key != x_feature]
            if st.session_state.get("scatter_y_feature") not in y_options:
                st.session_state["scatter_y_feature"] = "high_speed_m_per_min" if "high_speed_m_per_min" in y_options else y_options[0]
            y_feature = y_slot.selectbox("Asse verticale", y_options, format_func=axis_label, key="scatter_y_feature",
                                       help="Scegli una seconda caratteristica, diversa da quella dell'asse orizzontale.")
            valid_coordinates = eligible[[x_feature, y_feature]].apply(lambda column: column.map(math.isfinite)).all(axis=1)
            chart_data = eligible.loc[valid_coordinates]
            reference_plotted = (target_id is not None and target is not None
                                 and all(math.isfinite(target.get(key, float("nan"))) for key in (x_feature, y_feature)))
            figure = profile_scatter(chart_data, x_feature, y_feature, target if reference_plotted else None,
                                     events_only=data_source == "events", score_label=score_label)
            st.plotly_chart(figure, width="stretch", key="landscape", theme=None)
            point_minutes = "minuti giocati nella partita" if data_source == "events" else "minuti osservati"
            if reference_plotted:
                st.caption(f"La dimensione dei punti dei candidati indica i {point_minutes}.")
            else:
                st.caption(f"Dimensione del punto: {point_minutes}.")
                if target_id is not None:
                    st.caption("Il riferimento non ha entrambe le misure necessarie per essere posizionato in questo grafico.")
            st.caption("Il grafico mostra le due misure scelte nelle loro unità originali. Il punteggio usa tutte le caratteristiche attive e i pesi impostati: cambiare gli assi aggiorna solo la visualizzazione.")
            candidates_without_plot = int((~valid_coordinates).sum())
            if candidates_without_plot:
                st.caption(f"{candidates_without_plot} candidati della tabella non hanno entrambe le misure necessarie per questo grafico.")
    opportunity_column = ({"attack_opportunities": "Recuperi analizzabili"} if phase == "in_possession" else
                          {"defense_opportunities": "Perdite analizzabili"} if phase == "out_of_possession" else
                          {"attack_opportunities": "Recuperi analizzabili", "defense_opportunities": "Perdite analizzabili"})
    display_columns = {**({"ranking_position": "Posizione"} if general_mode else {}),
                       "player_name": "Giocatore", "team_name": "Squadra", "match_id": "Partita", "score": score_label,
                       **{key + "_score": FAMILIES[key] for key in visible_families},
                       "minutes_played_selected_matches": "Minuti giocati complessivi", "minutes_played_match": "Minuti giocati nella finale", "market_value_m_eur": "Valore di mercato (M€)"}
    if data_source != "events":
        display_columns.update({"observed_minutes": "Minuti osservati totali" if phase == "all" else "Minuti nella fase",
                                "role_assigned_minutes": "Minuti con contesto ricostruito", **opportunity_column})
    display_columns["scouting_role_detail"] = "Profilo di ruolo"
    shortlist = readable_table(eligible, display_columns).round(2)
    st.subheader("Classifica dei giocatori" if general_mode else "Shortlist dei giocatori")
    st.dataframe(shortlist, hide_index=True, width="stretch", row_height=40, column_config=shortlist_columns(score_label=score_label))
    st.download_button("Scarica la classifica CSV" if general_mode else "Scarica la shortlist CSV",
                       shortlist.to_csv(index=False).encode("utf-8"), "scouting_classifica.csv" if general_mode else "tracking_shortlist.csv", "text/csv")
    st.caption("I candidati in tabella, nei grafici e nel CSV rispettano gli stessi vincoli di minuti e budget. I profili senza una quotazione nota sono esclusi dai candidati." +
               (" A parità di punteggio, l'ordine considera i minuti da referto, poi nome e partita." if general_mode else " La stella identifica separatamente il riferimento del confronto."))

comparison_pool = eligible
candidate_id = None
if len(comparison_pool):
    lookup = comparison_pool.set_index("profile_id")
    candidate_ids = comparison_pool.profile_id.tolist()
    candidate_labels = lookup.display_name.to_dict()
    # Le etichette visibili possono coincidere fra fasi, mentre gli ID cambiano.
    # Uno stato esplicito impedisce di riusare il candidato di una fase precedente.
    if st.session_state.get("comparison_candidate") not in candidate_ids:
        st.session_state["comparison_candidate"] = candidate_ids[0]
    with tabs[1]:
        candidate_id = st.selectbox("Candidato da esaminare", candidate_ids,
                                    format_func=candidate_labels.get, key="comparison_candidate")
        candidate = lookup.loc[candidate_id]
        st.subheader(candidate.player_name)
        st.caption(f"{team_name(candidate.team_name)} · {match_name(candidate.match_id)} · {candidate.scouting_primary_role}")
        table = []
        detail_lookup = ranking_detail.loc[ranking_detail.profile_id.eq(candidate_id)].set_index("feature") if general_mode else None
        for key in keys:
            _, unit, family = FEATURES[key]
            title = FEATURE_TITLES[key]
            if general_mode:
                detail = detail_lookup.loc[key]
                table.append({"Caratteristica": title, "Unità": unit, "Valore": candidate[key],
                              "Percentile favorevole": detail.percentile, "Criterio": DIRECTION_LABELS[directions[key]],
                              "Contributo al punteggio": detail.weighted_points, "Dimensione": FAMILIES[family]})
            else:
                table.append({"Caratteristica": title, "Unità": unit, "Candidato": candidate[key], "Riferimento": target[key],
                              "Differenza": candidate[key] - target[key], "Dimensione": FAMILIES[family]})
        st.dataframe(pd.DataFrame(table).round(3), hide_index=True, width="stretch")

else:
    with tabs[1]:
        st.info("Nessun candidato confrontabile con le impostazioni attuali. Modifica partite, minuti giocati, budget o pesi per visualizzare un confronto.")

with tabs[2], st.container(key="scouting-method"):
    render_method(base, scaler, config, phase, selected_role, minimum_played, budget_eur, data_source, candidate_roles,
                  general_mode=general_mode, directions=directions, keys=keys)
    export_columns = {
        "player_name": "Giocatore", "team_name": "Squadra", "match_id": "Partita", "role_group": "Ruolo",
        "phase": "Fase di gioco", "period_id": "Tempo di gioco",
        "minutes_played_selected_matches": "Minuti giocati complessivi", "minutes_played_match": "Minuti giocati nella finale", "market_value_m_eur": "Valore di mercato (M€)",
        "market_value_origin": "Fonte del valore", "market_value_date": "Data della quotazione",
        "scouting_secondary_roles": "Altri ruoli",
    }
    if data_source != "events":
        export_columns.update({"observed_minutes": "Minuti osservati", "role_assigned_minutes": "Minuti con contesto ricostruito",
                               "attack_opportunities": "Opportunità offensive", "defense_opportunities": "Opportunità difensive"})
    export_columns.update({key: f"{FEATURE_TITLES[key]} ({FEATURES[key][1]})" for key in source_features})
    export_profiles = profiles.loc[profiles.phase.isin(DASHBOARD_PHASES)]
    export_profiles = export_profiles.loc[role_mask(export_profiles, candidate_roles)]
    if data_source == "events":
        export_profiles = export_profiles.loc[export_profiles.period_id.eq(0)]
    export_data = readable_table(filter_candidates(export_profiles, selected_matches, minimum_played, budget_eur, minutes_source=base), export_columns)
    st.caption(f"Il CSV comprende solo i {SOURCE_DESCRIPTIONS[data_source]} delle partite selezionate, rispettando minuti e budget. Include le due fasi e la vista totale: sono viste sovrapposte e non vanno sommate. Anche i minuti da referto sono ripetuti per fase e periodo: non vanno sommati fra righe. I profili delle due finali restano distinti.")
    if data_source != "tracking":
        st.caption("Le statistiche evento hanno granularità partita: sono disponibili solo per Intera partita e nella loro fase funzionale o in Tutte le fasi.")
    st.download_button("Scarica le misure dei giocatori CSV", export_data.to_csv(index=False).encode("utf-8"),
                       "misure_giocatori_tracking.csv", "text/csv", key="download_all")

with st.sidebar:
    meaning = "premia i percentili nei criteri scelti" if general_mode else "misura la somiglianza"
    st.info(f"**Come leggere i filtri**\n\nMinuti giocati e budget selezionano i candidati. Il punteggio {meaning} nei {SOURCE_DESCRIPTIONS[data_source]} selezionati.")
