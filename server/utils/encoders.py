def _to_py(obj):
    """Recursively convert SQLAlchemy and NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_py(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return _to_py(vars(obj))
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    else:
        return obj
