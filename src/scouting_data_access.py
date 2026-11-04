"""Caricamento verificato dei risultati usati dalla dashboard locale."""

import hashlib
import json
from pathlib import Path

import pandas as pd


class ScoutingDataError(ValueError):
    """Risultati mancanti, modificati o non coerenti con il codice di calcolo."""


FILES = ("profiles.csv", "reference_scalers.json", "feature_correlations.csv", "baseline_compatibility.csv")


def bundle_version(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text())
    paths = [folder / "manifest.json", *(folder / name for name in FILES)]
    paths.extend(root / name for name in manifest["code_sha256"])
    paths.extend(folder / match / "manifest.json" for match in manifest["matches"])
    if "role_catalog" in manifest:
        paths.append(root / manifest["role_catalog"]["path"])
    paths.extend(root / name for name in manifest.get("selection_inputs_sha256", {}))
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def load_verified_bundle(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("validation_passed"):
        raise ScoutingDataError("Validazione del motore non superata")
    expected = manifest.get("outputs_sha256", {})
    if not set(FILES).issubset(expected) or not manifest.get("code_sha256"):
        raise ScoutingDataError("Provenienza dei risultati incompleta")
    checks = [(folder / name, expected[name]) for name in FILES]
    checks.extend((root / name, value) for name, value in manifest["code_sha256"].items())
    if set(manifest.get("match_manifests_sha256", {})) != set(manifest["matches"]):
        raise ScoutingDataError("Provenienza delle partite incompleta")
    checks.extend((folder / match / "manifest.json", value) for match, value in manifest["match_manifests_sha256"].items())
    if "role_catalog" in manifest:
        checks.append((root / manifest["role_catalog"]["path"], manifest["role_catalog"]["sha256"]))
    checks.extend((root / name, value) for name, value in manifest.get("selection_inputs_sha256", {}).items())
    for path, expected_hash in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ScoutingDataError(f"Risultati da aggiornare: {path.name}")
    profiles = pd.read_csv(folder / "profiles.csv")
    if set(profiles.match_id) != set(manifest["matches"]):
        raise ScoutingDataError("Partite dei profili non coerenti con i risultati dichiarati")
    return (manifest, profiles, json.loads((folder / "reference_scalers.json").read_text()),
            pd.read_csv(folder / "feature_correlations.csv"))
