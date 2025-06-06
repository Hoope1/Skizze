from pathlib import Path
try:
    import tomllib  # py>=3.11
    _LOADER = "tomllib"
except ImportError:
    try:
        import toml
        _LOADER = "toml"
    except ImportError:
        _LOADER = None

def load_config_from_pyproject() -> dict:
    cfg = {}
    if _LOADER is None:
        return cfg
    path = Path(__file__).parent.parent / "pyproject.toml"
    if not path.exists():
        return cfg
    if _LOADER == "tomllib":
        with open(path, "rb") as f:
            data = tomllib.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = toml.load(f)
    cfg.update(data.get("tool", {}).get("skizze", {}))
    return cfg
