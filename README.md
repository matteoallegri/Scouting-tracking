# Tracking metrics a supporto dello scouting

Copia della dashboard con dati preelaborati e controlli di integrità originali.
Comprende la rimozione del ruolo secondario di De Paul.

## Streamlit Community Cloud

1. Carica il contenuto di questa cartella in un repository GitHub nuovo, mantenendo le sottocartelle.
2. Collega il tuo account GitHub a Streamlit Community Cloud e scegli Create app.
3. Seleziona repository, branch pubblicato e Main file path: `apps/scouting.py`.
4. In Advanced settings scegli Python **3.11**, poi Deploy.

Sono sufficienti le dipendenze in requirements.txt. Non servono chiavi API.
Per avviare in locale: `python -m streamlit run apps/scouting.py`.

La copia serve a consultare la dashboard. Il ricalcolo delle metriche dai tracciati
rimane nel progetto completo. I JSON evento inclusi sono richiesti dai controlli
di integrità: non eliminarli. I grandi tracciati grezzi non servono all’avvio.

Ogni aggiornamento del progetto richiede una nuova esportazione e il caricamento
dei file aggiornati nel repository della dashboard. Lo ZIP va estratto prima del caricamento.
