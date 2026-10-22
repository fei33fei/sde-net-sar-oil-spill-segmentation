import copy
from typing import Any, Dict

import yaml


def _parse_value(v: str):
    low = v.lower()
    if low in ['true', 'false']:
        return low == 'true'
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _set_nested(cfg: Dict[str, Any], key: str, value: Any):
    keys = key.split('.')
    cur = cfg
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_config(path: str, opts=None):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if opts:
        for item in opts:
            if '=' not in item:
                continue
            k, v = item.split('=', 1)
            _set_nested(cfg, k, _parse_value(v))
    return cfg
