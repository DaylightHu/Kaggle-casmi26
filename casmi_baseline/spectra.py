import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(f):
            return f
        return decorate


def clean_spectrum(mzs, intensities, precursor, config):
    mz = np.asarray(mzs if mzs is not None else [], dtype=np.float64)
    intensity = np.asarray(intensities if intensities is not None else [], dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
        raise ValueError("Peak m/z and intensity arrays must be aligned 1D arrays")
    good = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0) & (mz <= precursor + 2.0)
    mz, intensity = mz[good], intensity[good]
    if len(mz):
        intensity /= intensity.max()
        good = intensity >= config["intensity_floor"]
        if config["remove_precursor"]:
            good &= mz < precursor - config["precursor_exclusion_da"]
        mz, intensity = mz[good], intensity[good]
        if len(mz) > config["top_n"]:
            keep = np.argpartition(intensity, -config["top_n"])[-config["top_n"]:]
            mz, intensity = mz[keep], intensity[keep]
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        # Merge exactly duplicated peak masses instead of scoring a peak twice.
        unique, starts = np.unique(mz, return_index=True)
        mz, intensity = unique, np.add.reduceat(intensity, starts)
        intensity = np.sqrt(intensity)
        intensity /= max(np.linalg.norm(intensity), 1e-12)
    return mz.astype(np.float32), intensity.astype(np.float32)


def mean_energy(value):
    if value is None:
        return np.nan
    try:
        x = np.atleast_1d(value).astype(float)
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def timstof(instrument, source=""):
    s = str(instrument or "").lower().replace(" ", "")
    return "timstof" in s or source in ("enveda-180", "enveda-np-examples")


def mode_code(mode):
    return {"positive": 1, "negative": -1}.get(str(mode).lower(), 0)


@njit(cache=True)
def cosine_scan(a, wa, b, wb, tolerance_da, tolerance_ppm):
    """One-to-one ordered peak matching, not a dense binned dot product."""
    i = 0
    j = 0
    score = 0.0
    while i < len(a) and j < len(b):
        tol = max(tolerance_da, min(a[i], b[j]) * tolerance_ppm * 1e-6)
        delta = a[i] - b[j]
        if abs(delta) <= tol:
            score += wa[i] * wb[j]
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return min(max(score, 0.0), 1.0)


@njit(cache=True)
def reference_scores(qmz, qw, qprecursor, qadduct, qmode, qce,
                     rmz, rw, rcount, rprecursor, radduct, rmode, rce,
                     ris_timstof, tolerance_da, tolerance_ppm):
    result = np.zeros(len(rmz), dtype=np.float32)
    qloss = qprecursor - qmz[::-1]
    qwl = qw[::-1]
    for i in range(len(rmz)):
        if qmode != 0 and rmode[i] != 0 and qmode != rmode[i]:
            continue
        n = rcount[i]
        if n == 0 or len(qmz) == 0:
            continue
        mz = rmz[i, :n]
        w = rw[i, :n]
        direct = cosine_scan(qmz, qw, mz, w, tolerance_da, tolerance_ppm)
        loss = cosine_scan(qloss, qwl, rprecursor[i] - mz[::-1], w[::-1], tolerance_da, tolerance_ppm)
        if qadduct == radduct[i] and qadduct != 0:
            score = 0.80 * direct + 0.20 * loss
        else:
            # A shifted precursor does not imply every fragment keeps the adduct.
            score = 0.90 * max(direct, loss)
        if np.isfinite(qce) and np.isfinite(rce[i]):
            score *= 0.90 + 0.10 * np.exp(-abs(qce - rce[i]) / 25.0)
        if not ris_timstof[i]:
            score *= 0.97
        result[i] = score
    return result


def query_quality(mz, weights, base_intensity=None):
    rich = min(1.0, len(mz) / 12.0)
    # Missing raw counts are not interpreted as a noisy measurement.
    try:
        raw = float(base_intensity)
        count_factor = min(1.0, max(0.2, np.log10(max(raw, 1.0)) / 5.0)) if np.isfinite(raw) else 1.0
    except (ValueError, TypeError):
        count_factor = 1.0
    return max(0.05, rich * count_factor)
