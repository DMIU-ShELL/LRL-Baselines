import os
from pathlib import Path
import ctypes
import sys

_ROOT = Path(__file__).resolve().parent

os.environ.setdefault('XDG_CACHE_HOME', str(_ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(_ROOT / '.cache' / 'matplotlib'))

_CONDA_PREFIX = Path(os.environ.get('CONDA_PREFIX', sys.prefix))
_CONDA_LIB = _CONDA_PREFIX / 'lib'
_CONDA_LIBSTDCXX = _CONDA_LIB / 'libstdc++.so.6'

if _CONDA_LIBSTDCXX.exists():
    _conda_lib = str(_CONDA_LIB)
    _ld_paths = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    if os.environ.get('_LRL_CONDA_LIB_REEXEC') != '1' and _ld_paths[:1] != [_conda_lib]:
        os.environ['_LRL_CONDA_LIB_REEXEC'] = '1'
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(
            [_conda_lib] + [path for path in _ld_paths if path and path != _conda_lib]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ctypes.CDLL(str(_CONDA_LIBSTDCXX), mode=ctypes.RTLD_GLOBAL)
