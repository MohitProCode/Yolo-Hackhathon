from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_paths(cfg: dict, base_dir: str | None = None) -> dict:
    base = Path(base_dir or ".").resolve()
    paths = cfg.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        if isinstance(value, str):
            resolved[key] = str((base / value).resolve())
        else:
            resolved[key] = value
    cfg["paths"] = resolved
    return cfg
