import numpy as np
import h5py

def load_h5_to_dict(group):
    """Recursively loads an H5 group back into a dictionary."""
    data = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = load_h5_to_dict(item)
        else:
            data[key] = item[()] # [()] extracts the data as a numpy array/scalar
    return data

# ==========================================================================
def _to_h5_compatible(value):
    """
    Convert *value* to something h5py can write as a dataset.

    Resolution order
    ----------------
    1. None              → store as empty bytes
    2. dict              → signal caller to recurse (returns None sentinel)
    3. str/bytes         → np.bytes_
    4. bool              → int (must come before int check)
    5. int / float       → as-is
    6. np.ndarray
       a. 0-d object     → unwrap and recurse
       b. object dtype   → convert element-wise to str, store as bytes array
       c. Unicode dtype  → encode each element to bytes
       d. numeric/bool   → as-is
    7. list / tuple      → convert to np.ndarray; fall back to bytes on failure
    8. everything else   → str repr stored as bytes
    """
    if value is None:
        return np.bytes_(b'')
    if isinstance(value, dict):
        return None                          # sentinel: caller must recurse
    if isinstance(value, (bytes, np.bytes_)):
        return np.bytes_(value)
    if isinstance(value, str):
        return np.bytes_(value.encode('utf-8'))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_h5_compatible(value.item())
        if value.dtype == object:
            flat = [str(v).encode('utf-8') for v in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        if value.dtype.kind == 'U':
            flat = [s.encode('utf-8') for s in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        try:
            arr = np.array(value)
            if arr.dtype == object:
                raise ValueError('object array')
            if arr.dtype.kind == 'U':
                flat = [s.encode('utf-8') for s in arr.ravel()]
                return np.array(flat, dtype='S').reshape(arr.shape)
            return arr
        except (ValueError, TypeError):
            return np.bytes_(str(value).encode('utf-8'))
    return np.bytes_(str(value).encode('utf-8'))


def _save_dict_to_h5(h5file, data):
    """
    Recursively write a dict to an open h5py.File or h5py.Group.

    Tuple keys (used for nested bio-model fields) are serialised as
    their string representation so they round-trip safely.
    """
    for key, value in data.items():
        safe_key = str(key)          # h5py requires string keys; tuple → "('a','b')"
        if isinstance(value, dict):
            grp = h5file.require_group(safe_key)
            _save_dict_to_h5(grp, value)
        else:
            converted = _to_h5_compatible(value)
            if converted is None:
                grp = h5file.require_group(safe_key)
                _save_dict_to_h5(grp, value)
            else:
                h5file.create_dataset(safe_key, data=converted)