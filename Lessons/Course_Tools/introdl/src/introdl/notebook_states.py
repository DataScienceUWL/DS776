# introdl/notebook_states.py
"""Notebook state management for saving and restoring session state in unstable environments."""
from __future__ import annotations
import os, sys, json, time, tempfile, shutil, warnings, gzip, hashlib
from typing import Any, Dict, Iterable, Optional, List, Tuple
from dataclasses import dataclass
import random
import numpy as np

# Optional deps
try:
    import torch
    from torch.optim.lr_scheduler import _LRScheduler
except Exception:
    torch = None
    class _LRScheduler:  # type: ignore
        pass

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from IPython import get_ipython
except Exception:
    def get_ipython():
        return None

# -------------------------
# Config / helpers
# -------------------------

def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _default_root() -> str:
    """Return the notebook_states directory in current working directory."""
    return os.path.abspath("notebook_states")

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _get_notebook_name() -> Optional[str]:
    """Try to detect the current notebook filename."""
    try:
        # Try to get notebook name from IPython
        ipython = get_ipython()
        if ipython and hasattr(ipython, 'kernel'):
            # In Jupyter, we can sometimes get the notebook name from the kernel
            import re
            import json

            # Try to get from IPython's config
            if hasattr(ipython, 'config'):
                notebook_path = getattr(ipython.config, 'IPKernelApp', {}).get('connection_file', '')
                if notebook_path:
                    # Extract notebook name from kernel connection file
                    # Usually like: kernel-<notebook_name>-<id>.json
                    match = re.search(r'kernel-([^-]+)', os.path.basename(notebook_path))
                    if match:
                        return match.group(1)

            # Try alternative method using notebook filename from environment
            notebook_name = os.environ.get('NOTEBOOK_NAME', '')
            if notebook_name:
                return os.path.splitext(notebook_name)[0]

            # Try to get from Jupyter notebook server
            try:
                import requests
                from notebook import notebookapp
                import urllib
                import json

                # Get list of running notebook servers
                servers = list(notebookapp.list_running_servers())
                for server in servers:
                    try:
                        # Query the server for notebooks
                        response = requests.get(
                            urllib.parse.urljoin(server['url'], 'api/sessions'),
                            params={'token': server.get('token', '')}
                        )
                        for session in response.json():
                            if session['kernel']['id'] == ipython.kernel.id:
                                return os.path.splitext(os.path.basename(session['notebook']['path']))[0]
                    except:
                        pass
            except:
                pass

    except Exception:
        pass

    return None

def _notebook_name(user_name: Optional[str] = None) -> str:
    """Get notebook name with fallback logic."""
    if user_name:
        return user_name

    # Check environment variable override
    env = os.getenv("INTRODL_NOTEBOOK_NAME")
    if env:
        return env

    # Try auto-detection
    auto_name = _get_notebook_name()
    if auto_name:
        return auto_name

    # Fallback
    return "notebook"

def _ipython_user_ns() -> Dict[str, Any]:
    ip = get_ipython()
    if ip and hasattr(ip, "user_ns"):
        return ip.user_ns
    # Non-IPython fallback
    return globals()

def _atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(path)
    _ensure_dir(d)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def _atomic_copy(src: str, dst: str) -> None:
    d = os.path.dirname(dst)
    _ensure_dir(d)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".part")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try: total += os.path.getsize(fp)
            except Exception: pass
    return total

def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

# -------------------------
# DataLoader Resume Support
# -------------------------

from typing import Callable

class _LoaderRegistry:
    """Global registry mapping a logical loader name -> factory() that rebuilds it."""
    def __init__(self):
        self.factories: Dict[str, Callable[[], Any]] = {}

    def register(self, name: str, factory: Callable[[], Any]):
        self.factories[name] = factory

    def build(self, name: str):
        if name not in self.factories:
            raise KeyError(f"No dataloader factory registered for '{name}'.")
        return self.factories[name]()

_loader_registry = _LoaderRegistry()

class TrackedDataLoader:
    """
    Wraps a DataLoader to track which batch index has been yielded.
    On resume we rebuild the underlying DataLoader and fast-forward to that index.
    """
    def __init__(self, name: str, loader: Any):
        self.__name = name
        self._loader = loader
        self._last_yielded_idx = 0

    @property
    def name(self) -> str:
        return self.__name

    @property
    def last_index(self) -> int:
        return self._last_yielded_idx

    @property
    def dataset(self):
        """Pass through to underlying loader's dataset."""
        return self._loader.dataset

    @property
    def batch_size(self):
        """Pass through to underlying loader's batch_size."""
        return self._loader.batch_size

    def __iter__(self):
        # Iterate underlying loader while tracking progress
        for i, batch in enumerate(self._loader):
            self._last_yielded_idx = i + 1
            yield batch

    def __len__(self):
        """Pass through to underlying loader's length."""
        return len(self._loader)

    # Snapshot/restore hooks used by save_state/load_state
    def snapshot_state(self) -> dict:
        return {"name": self.__name, "index": int(self._last_yielded_idx)}

    def restore_state(self, index: int):
        # Rebuild the underlying loader via registry, then fast-forward to index
        self._loader = _loader_registry.build(self.__name)
        if index <= 0:
            self._last_yielded_idx = 0
            return
        it = iter(self._loader)
        # Consume 'index' batches quickly; StopIteration-safe
        for _ in range(index):
            try:
                next(it)
            except StopIteration:
                break
        self._last_yielded_idx = index

def register_dataloader(name: str, factory: Callable[[], Any]) -> TrackedDataLoader:
    """
    Register a DataLoader factory under 'name' and return a tracked instance
    students can use directly in their loops.

    Example:
        def make_train_loader():
            return DataLoader(train_dataset, batch_size=64, shuffle=True)

        train_loader = register_dataloader("train", make_train_loader)
    """
    _loader_registry.register(name, factory)
    return TrackedDataLoader(name, factory())

def rebuild_registered_dataloaders():
    """
    After load_state() (which restores RNG), rebuild any TrackedDataLoader variables
    from their saved progress and fast-forward them.
    This uses the metadata saved by save_state().
    """
    ns = _ipython_user_ns()
    meta = ns.get("_INTRODL_INTERNAL_LAST_SESSION_META__", {})
    dl_states = meta.get("dataloader_states", {})

    for varname, st in dl_states.items():
        obj = ns.get(varname, None)
        if isinstance(obj, TrackedDataLoader):
            try:
                obj.restore_state(int(st.get("index", 0)))
                print(f"[load_state] Restored dataloader '{varname}' to batch {st.get('index', 0)}")
            except Exception as e:
                warnings.warn(f"[load_state] Failed to restore dataloader '{varname}': {e}")

def _prune_keep_last(nb_dir: str, keep_last: int) -> None:
    if keep_last is None or keep_last <= 0:
        return
    stamps = sorted([d for d in os.listdir(nb_dir) if os.path.isdir(os.path.join(nb_dir, d)) and d != "_blobstore"])
    to_remove = stamps[:-keep_last] if len(stamps) > keep_last else []
    for s in to_remove:
        shutil.rmtree(os.path.join(nb_dir, s), ignore_errors=True)

def _prune_until_within_size(nb_dir: str, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    stamps = sorted([d for d in os.listdir(nb_dir) if os.path.isdir(os.path.join(nb_dir, d)) and d != "_blobstore"])
    while stamps and _dir_size_bytes(nb_dir) > max_bytes:
        victim = stamps.pop(0)
        shutil.rmtree(os.path.join(nb_dir, victim), ignore_errors=True)

def _is_small_plain(obj: Any) -> bool:
    SMALL = (int, float, bool, str, type(None))
    if isinstance(obj, SMALL):
        return True
    if isinstance(obj, (list, tuple, set)):
        return all(_is_small_plain(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, (str, int, float, bool)) and _is_small_plain(v)
                   for k, v in obj.items())
    return False

def _safe_len_bytes_of_tensor(x) -> int:
    try:
        return int(x.numel()) * int(x.element_size())
    except Exception:
        return 0

def _sizeof_ndarray(a: np.ndarray) -> int:
    try:
        return int(a.size) * int(a.itemsize)
    except Exception:
        return 0

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _blob_path(root: str, digest: str) -> str:
    return os.path.join(root, "_blobstore", "sha256", digest[:2], digest)

def _store_blob_from_tmp(root: str, tmpfile: str) -> str:
    digest = _sha256_file(tmpfile)
    dst = _blob_path(root, digest)
    _ensure_dir(os.path.dirname(dst))
    if not os.path.exists(dst):
        _atomic_copy(tmpfile, dst)
    return os.path.relpath(dst, root)

def _gzip_if_large(src: str, threshold_mb: int = 50) -> str:
    try:
        if os.path.getsize(src) < threshold_mb * 1024 * 1024:
            return src
        gz = src + ".gz"
        with open(src, "rb") as f_in, gzip.open(gz, "wb", compresslevel=5) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(src)
        return gz
    except Exception:
        return src

# -------------------------
# RNG state management
# -------------------------

@dataclass
class RNGState:
    python_state: Any
    numpy_state: Any
    torch_state: Optional[Any] = None
    torch_cuda_states: Optional[Dict[int, Any]] = None
    torch_deterministic: Optional[bool] = None

def _capture_rng() -> RNGState:
    py = random.getstate()
    np_state = np.random.get_state()
    t_state = None
    t_cuda = None
    t_det = None
    if torch is not None:
        try:
            t_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                t_cuda = {}
                for i in range(torch.cuda.device_count()):
                    t_cuda[i] = torch.cuda.get_rng_state(i).cpu()
            t_det = torch.are_deterministic_algorithms_enabled()
        except Exception:
            pass
    return RNGState(py, np_state, t_state, t_cuda, t_det)

def _restore_rng(s: RNGState) -> None:
    try: random.setstate(s.python_state)
    except Exception: pass
    try: np.random.set_state(s.numpy_state)
    except Exception: pass
    if torch is not None:
        try:
            if s.torch_state is not None:
                torch.random.set_rng_state(s.torch_state)
            if s.torch_cuda_states:
                for dev, state in s.torch_cuda_states.items():
                    torch.cuda.set_rng_state(state, device=dev)
            if s.torch_deterministic is not None:
                torch.use_deterministic_algorithms(s.torch_deterministic)
        except Exception:
            pass

# -------------------------
# Type checking helpers
# -------------------------

def _is_model(x):  return (torch is not None) and isinstance(x, torch.nn.Module)
def _is_optim(x):  return (torch is not None) and isinstance(x, torch.optim.Optimizer)
def _is_sched(x):  return (torch is not None) and isinstance(x, _LRScheduler)
def _is_tensor(x): return (torch is not None) and isinstance(x, torch.Tensor)
def _is_np(x):     return isinstance(x, np.ndarray)
def _is_df(x):     return (pd is not None) and isinstance(x, (pd.DataFrame, pd.Series))

# -------------------------
# Main save/load functions
# -------------------------

def save_state(
    notebook_name: Optional[str] = None,
    keep_last: int = 2,
    max_blob_mb: int = 200,
    max_total_gb: float = 2.0,
    note: Optional[str] = None,
    include_vars: Optional[Iterable[str]] = None,
    exclude_vars: Iterable[str] = ("_",),
) -> str:
    """
    Save the current notebook session state.

    Saves variables, models, optimizers, and RNG states to a timestamped directory
    under notebook_states/<notebook_name>/. Automatically manages storage with
    rotation and size limits.

    Args:
        notebook_name: Name for this notebook's states. Auto-detected if not provided.
        keep_last: Number of recent states to keep (older ones deleted).
        max_blob_mb: Skip individual objects larger than this (MB).
        max_total_gb: Maximum total storage for this notebook (GB).
        note: Optional note to save with the state.
        include_vars: If specified, only save these variable names.
        exclude_vars: Variable names to exclude (default: names starting with '_').

    Returns:
        Path to the saved state directory.

    Example:
        >>> save_state()  # Auto-detect notebook, use defaults
        [save_state] Saved state: notebook_states/Homework_01/20240315-143022
        [save_state] Counts: {'inline': 5, 'models': 1, 'blobs': 3, 'skipped': 2}
    """
    ns = _ipython_user_ns()
    root = _default_root()
    stamp = _now_stamp()
    nb = _notebook_name(notebook_name)
    nb_dir = _ensure_dir(os.path.join(root, nb))
    out_dir = _ensure_dir(os.path.join(nb_dir, stamp))

    # Metadata
    meta = {
        "created_at": stamp,
        "notebook_name": nb,
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": sys.platform,
        "note": note or "",
    }

    # Collect variables
    vars_inline: Dict[str, Any] = {}
    model_files: Dict[str, str] = {}
    optim_files: Dict[str, str] = {}
    sched_files: Dict[str, str] = {}
    blobs: Dict[str, Dict[str, str]] = {}
    skip_report: List[Tuple[str, str]] = []

    names = include_vars if include_vars is not None else list(ns.keys())
    single_blob_limit = max(1, int(max_blob_mb)) * 1024 * 1024
    tmp_dir = _ensure_dir(os.path.join(out_dir, ".tmp"))

    def _save_large(obj, kind: str, name: str) -> None:
        """Serialize, size-check, compress, dedupe to blobstore."""
        try:
            tmp_path = os.path.join(tmp_dir, f"{kind}_{name}")

            if kind == "tensor" and torch is not None:
                if _safe_len_bytes_of_tensor(obj) > single_blob_limit:
                    skip_report.append((name, f"too_big>{max_blob_mb}MB"))
                    return
                torch.save(obj, tmp_path)
            elif kind == "ndarray":
                if _sizeof_ndarray(obj) > single_blob_limit:
                    skip_report.append((name, f"too_big>{max_blob_mb}MB"))
                    return
                np.savez_compressed(tmp_path + ".npz", arr=obj)
                tmp_path = tmp_path + ".npz"
            elif kind == "pandas" and pd is not None:
                try:
                    import pyarrow
                    tmp_path = tmp_path + ".parquet"
                    obj.to_parquet(tmp_path, index=True)
                except Exception:
                    tmp_path = tmp_path + ".pkl"
                    obj.to_pickle(tmp_path)
            else:
                skip_report.append((name, "unknown_kind"))
                return

            tmp_path = _gzip_if_large(tmp_path, threshold_mb=50)
            rel_blob = _store_blob_from_tmp(root, tmp_path)
            blobs[name] = {
                "rel": rel_blob,
                "kind": kind,
                "gz": "1" if tmp_path.endswith(".gz") else "0",
            }
            try: os.remove(tmp_path)
            except Exception: pass
        except Exception as e:
            skip_report.append((name, f"error:{type(e).__name__}"))

    # Process each variable
    for name in names:
        if name in exclude_vars:
            continue
        if name.startswith("__"):
            continue
        if name in ("In", "Out"):
            continue

        try:
            obj = ns[name]
        except Exception:
            continue

        if callable(obj) or isinstance(obj, type):
            continue

        try:
            if _is_model(obj):
                p = os.path.join(out_dir, f"model_{name}.pth")
                torch.save(obj.state_dict(), p)
                model_files[name] = os.path.basename(p)
            elif _is_optim(obj):
                p = os.path.join(out_dir, f"optim_{name}.pth")
                torch.save(obj.state_dict(), p)
                optim_files[name] = os.path.basename(p)
            elif _is_sched(obj):
                p = os.path.join(out_dir, f"sched_{name}.pth")
                torch.save(obj.state_dict(), p)
                sched_files[name] = os.path.basename(p)
            elif _is_tensor(obj):
                _save_large(obj, "tensor", name)
            elif _is_np(obj):
                _save_large(obj, "ndarray", name)
            elif _is_df(obj):
                _save_large(obj, "pandas", name)
            elif _is_small_plain(obj):
                vars_inline[name] = obj
            else:
                # Try dill for other objects
                import dill
                try:
                    dill.dumps(obj)
                    vars_inline[name] = obj
                except Exception:
                    skip_report.append((name, "unserializable"))
        except Exception as e:
            skip_report.append((name, f"error:{type(e).__name__}"))

    # Gather dataloader progress (tiny data)
    dataloader_states = {}
    for name in names:
        if name not in ns:
            continue
        obj = ns[name]
        if isinstance(obj, TrackedDataLoader):
            try:
                st = obj.snapshot_state()  # {"name":..., "index":...}
                dataloader_states[name] = st
            except Exception:
                pass

    # Save RNG state
    rng = _capture_rng()

    # Prepare session data
    session = {
        "meta": meta,
        "vars_inline": vars_inline,
        "models": model_files,
        "optimizers": optim_files,
        "schedulers": sched_files,
        "blobs": blobs,
        "dataloader_states": dataloader_states,  # NEW
        "rng": {
            "python": rng.python_state,
            "numpy": rng.numpy_state,
            "torch": rng.torch_state.cpu().numpy().tolist() if (torch is not None and rng.torch_state is not None) else None,
            "torch_cuda": {k: v.cpu().numpy().tolist() for k, v in (rng.torch_cuda_states or {}).items()},
            "torch_deterministic": rng.torch_deterministic,
        },
        "skip_report": skip_report,
    }

    # Stash this tiny dict in the live namespace for rebuild_registered_dataloaders()
    ns["_INTRODL_INTERNAL_LAST_SESSION_META__"] = {"dataloader_states": dataloader_states}

    # Write session file
    import dill
    _atomic_write_bytes(os.path.join(out_dir, "session.pkl"), dill.dumps(session, byref=False, recurse=True))

    # Cleanup temp
    try: shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception: pass

    # Apply rotation and size limits
    _prune_keep_last(nb_dir, keep_last)
    if max_total_gb is not None and max_total_gb > 0:
        _prune_until_within_size(nb_dir, int(max_total_gb * (1024 ** 3)))

    # Report
    saved_counts = dict(
        inline=len(vars_inline),
        models=len(model_files),
        optimizers=len(optim_files),
        schedulers=len(sched_files),
        blobs=len(blobs),
        skipped=len(skip_report),
    )
    print(f"[save_state] Saved state: {out_dir}")
    print(f"[save_state] Counts: {saved_counts}")
    if skip_report:
        sk = ", ".join([f"{n}({why})" for n, why in skip_report[:10]])
        more = "" if len(skip_report) <= 10 else f" +{len(skip_report)-10} more"
        print(f"[save_state] Skipped: {sk}{more}")
    return out_dir

def load_state(
    path: Optional[str] = None,
    notebook_name: Optional[str] = None,
    only: Optional[Iterable[str]] = None,
    strict_models: bool = True,
) -> str:
    """
    Restore a saved notebook state.

    Loads the most recent state for the notebook (or from a specific path).
    Restores variables, model states, optimizer states, and RNG states.

    Args:
        path: Specific state directory to load. If None, loads most recent.
        notebook_name: Notebook name to load states from. Auto-detected if not provided.
        only: If specified, only load these variable names.
        strict_models: If False, allows partial loading of model states.

    Returns:
        Path to the loaded state directory.

    Example:
        >>> load_state()  # Load most recent state
        [load_state] Restored state from: notebook_states/Homework_01/20240315-143022
    """
    root = _default_root()
    nb = _notebook_name(notebook_name)
    nb_dir = os.path.join(root, nb)

    if path is None:
        if not os.path.isdir(nb_dir):
            raise FileNotFoundError(f"No states for notebook '{nb}' under {root}")
        stamps = sorted([d for d in os.listdir(nb_dir) if os.path.isdir(os.path.join(nb_dir, d)) and d != "_blobstore"])
        if not stamps:
            raise FileNotFoundError(f"No states found in {nb_dir}")
        path = os.path.join(nb_dir, stamps[-1])

    sess_pkl = os.path.join(path, "session.pkl")
    if not os.path.isfile(sess_pkl):
        raise FileNotFoundError(f"Missing session.pkl in {path}")

    import dill
    with open(sess_pkl, "rb") as f:
        sess = dill.loads(f.read())

    ns = _ipython_user_ns()
    want = set(only) if only else None

    # Restore inline variables
    for k, v in sess.get("vars_inline", {}).items():
        if (want is None) or (k in want):
            ns[k] = v

    # Restore PyTorch objects
    if torch is not None:
        # Models
        for k, rel in sess.get("models", {}).items():
            if want is not None and k not in want:
                continue
            obj = ns.get(k, None)
            if obj is None or not isinstance(obj, torch.nn.Module):
                warnings.warn(f"[load_state] '{k}' missing or not nn.Module; skipping model state")
                continue
            state = torch.load(os.path.join(path, rel), map_location="cpu")
            obj.load_state_dict(state, strict=strict_models)

        # Optimizers
        for k, rel in sess.get("optimizers", {}).items():
            if want is not None and k not in want:
                continue
            obj = ns.get(k, None)
            if obj is None or not isinstance(obj, torch.optim.Optimizer):
                warnings.warn(f"[load_state] '{k}' missing or not Optimizer; skipping optimizer state")
                continue
            state = torch.load(os.path.join(path, rel), map_location="cpu")
            obj.load_state_dict(state)

        # Schedulers
        for k, rel in sess.get("schedulers", {}).items():
            if want is not None and k not in want:
                continue
            obj = ns.get(k, None)
            if obj is None or not isinstance(obj, _LRScheduler):
                warnings.warn(f"[load_state] '{k}' missing or not LRScheduler; skipping scheduler state")
                continue
            state = torch.load(os.path.join(path, rel), map_location="cpu")
            obj.load_state_dict(state)

    # Restore blobs
    for k, info in sess.get("blobs", {}).items():
        if want is not None and k not in want:
            continue
        rel = info["rel"]
        kind = info.get("kind", "")
        gz = info.get("gz", "0") == "1"
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            warnings.warn(f"[load_state] Blob missing for '{k}': {full}")
            continue

        src = full
        if gz:
            # Handle gzipped files
            if kind == "tensor" and torch is not None:
                with gzip.open(full, "rb") as f:
                    ns[k] = torch.load(f, map_location="cpu")
                continue
            # For others, decompress to temp
            tmp = full + ".tmp_extract"
            with gzip.open(full, "rb") as f_in, open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            src = tmp

        try:
            if kind == "tensor" and torch is not None:
                ns[k] = torch.load(src, map_location="cpu")
            elif kind == "ndarray":
                if src.endswith(".npz"):
                    with np.load(src, allow_pickle=False) as z:
                        ns[k] = z["arr"]
                else:
                    ns[k] = np.load(src, allow_pickle=False)
            elif kind == "pandas" and pd is not None:
                if src.endswith(".parquet"):
                    ns[k] = pd.read_parquet(src)
                else:
                    ns[k] = pd.read_pickle(src)
            else:
                warnings.warn(f"[load_state] Unrecognized blob kind for '{k}': {kind}")
        finally:
            if gz and src.endswith(".tmp_extract"):
                try: os.remove(src)
                except Exception: pass

    # Restore RNG state
    rng = sess.get("rng", None)
    if rng:
        try:
            t_state = None
            if torch is not None and rng.get("torch") is not None:
                t_state = torch.tensor(rng["torch"])
            t_cuda = None
            if torch is not None and rng.get("torch_cuda"):
                t_cuda = {int(k): torch.tensor(v) for k, v in rng["torch_cuda"].items()}
            _restore_rng(RNGState(
                python_state=rng["python"],
                numpy_state=tuple(rng["numpy"]),
                torch_state=t_state,
                torch_cuda_states=t_cuda,
                torch_deterministic=rng.get("torch_deterministic", None),
            ))
        except Exception as e:
            warnings.warn(f"[load_state] RNG restore failed: {e}")

    # Publish dataloader states for rebuild helper
    dl_states = sess.get("dataloader_states", {})
    ns["_INTRODL_INTERNAL_LAST_SESSION_META__"] = {"dataloader_states": dl_states}

    print(f"[load_state] Restored state from: {path}")
    return path

# -------------------------
# Autosave functionality
# -------------------------

def enable_cell_autosave(every_n_cells: int = 3, notebook_name: Optional[str] = None):
    """
    Enable automatic state saving after every N cell executions.

    Args:
        every_n_cells: Save state after this many cells execute.
        notebook_name: Notebook name for states. Auto-detected if not provided.

    Example:
        >>> enable_cell_autosave(every_n_cells=5)
        [enable_cell_autosave] Enabled autosave every 5 cell(s).
    """
    from IPython import get_ipython as _get_ipython
    ip = _get_ipython()
    if not ip:
        print("[enable_cell_autosave] No IPython environment detected.")
        return

    counter = {"n": 0}
    def _hook(result):
        counter["n"] += 1
        if counter["n"] % max(1, int(every_n_cells)) == 0:
            try:
                save_state(notebook_name=notebook_name)
            except Exception as e:
                warnings.warn(f"[autosave] save_state failed: {e}")

    ip.events.register("post_run_cell", _hook)
    print(f"[enable_cell_autosave] Enabled autosave every {every_n_cells} cell(s).")

# -------------------------
# State management utilities
# -------------------------

def _state_dirs(notebook_name: Optional[str]) -> Tuple[str, List[str]]:
    """Get state directory and list of state timestamps."""
    root = _default_root()
    nb = _notebook_name(notebook_name)
    nb_dir = os.path.join(root, nb)
    stamps = []
    if os.path.isdir(nb_dir):
        stamps = sorted([d for d in os.listdir(nb_dir) if os.path.isdir(os.path.join(nb_dir, d)) and d != "_blobstore"])
    return nb_dir, stamps

def list_states(notebook_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all saved states for the notebook.

    Args:
        notebook_name: Notebook name to list states for. Auto-detected if not provided.

    Returns:
        List of state information dictionaries.

    Example:
        >>> list_states()
        timestamp           size       path
        20240315-143022     45.3 MB   notebook_states/Homework_01/20240315-143022
        20240315-151545     52.1 MB   notebook_states/Homework_01/20240315-151545
    """
    nb_dir, stamps = _state_dirs(notebook_name)
    out = []
    for s in stamps:
        p = os.path.join(nb_dir, s)
        size = _dir_size_bytes(p)
        out.append({"timestamp": s, "path": p, "size_bytes": size})

    # Print table
    if out:
        print("timestamp           size       path")
        print("-" * 60)
        for r in out:
            size_str = _format_size(r["size_bytes"])
            print(f"{r['timestamp']}   {size_str:>10}   {r['path']}")
    else:
        print(f"No states found in {nb_dir}")

    return out

def delete_states(
    notebook_name: Optional[str] = None,
    keep_last: Optional[int] = None,
    older_than_days: Optional[int] = None
) -> None:
    """
    Delete saved states based on criteria.

    Args:
        notebook_name: Notebook name to manage states for. Auto-detected if not provided.
        keep_last: Keep only this many recent states.
        older_than_days: Delete states older than this many days.

    Example:
        >>> delete_states(keep_last=1)  # Keep only most recent
        [delete_states] Deleted 3 state(s) under notebook_states/Homework_01
    """
    nb_dir, stamps = _state_dirs(notebook_name)
    victims = []

    if keep_last is not None and keep_last >= 0:
        victims.extend(stamps[:-keep_last] if len(stamps) > keep_last else [])

    if older_than_days is not None and older_than_days > 0:
        cutoff = time.time() - older_than_days * 24 * 3600
        for s in stamps:
            p = os.path.join(nb_dir, s)
            try:
                t = time.mktime(time.strptime(s, "%Y%m%d-%H%M%S"))
            except Exception:
                t = os.path.getmtime(p)
            if t < cutoff and s not in victims:
                victims.append(s)

    for s in sorted(set(victims)):
        shutil.rmtree(os.path.join(nb_dir, s), ignore_errors=True)

    print(f"[delete_states] Deleted {len(set(victims))} state(s) under {nb_dir}")

def get_states_size() -> int:
    """
    Get total size of all notebook states in bytes.

    Returns:
        Total size in bytes.
    """
    root = _default_root()
    if not os.path.exists(root):
        return 0
    return _dir_size_bytes(root)