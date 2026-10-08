"""Atomic local JSON publication for experiment ledgers and result caches."""
import json
import os
from pathlib import Path
import tempfile


def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,
                                     prefix='.'+path.name+'-',suffix='.tmp',delete=False) as f:
        json.dump(payload,f,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno());temporary=f.name
    os.replace(temporary,path)
