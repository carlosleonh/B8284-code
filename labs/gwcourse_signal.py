"""
gwcourse_signal.py
==================
The signal processing the course uses, in one place.

Every figure and every lab that touches real strain data goes through these
four functions. They are deliberately short and written to be read: a student
who wants to know what "whitened" means on the axis of a figure can open this
file and find out in a dozen lines.

    from gwcourse_signal import load_strain, whiten, bandpass, q_transform

    strain, t, fs = load_strain(path)
    w = whiten(strain, fs)
    ts, freqs, energy = q_transform(w, fs, t, centre=t_merger)

Nothing here fetches anything and nothing here knows where the data are. The
path of the file and the time of the merger are arguments: they come from the
configuration file ``labs/config.toml``, which a notebook reads in its first
code cell. The data lives in ``labs/data``, downloaded once and committed, so a
figure built on a train gives the same result as one built in the department.

It sits in ``labs/`` because that is where it is used: the laboratories call
these functions directly, and the figures in the lessons call the same ones, so
what a student runs is what produced the picture they were shown.

The conventions are the ones used throughout the course: strain is a plain
NumPy array, ``fs`` is in hertz, times are GPS seconds, and every function
returns rather than plots.
"""

from __future__ import annotations

import numpy as np

__all__ = ["load_strain", "whiten", "bandpass", "q_transform"]


def load_strain(path):
    """Read the strain of one detector from a GWOSC file.

    Args:
        path: the hdf5 file with the strain, in the layout of the GWOSC files.

    Returns:
        strain: the strain, a NumPy array (dimensionless).
        gps_times: the GPS time of each sample (s).
        sample_rate: the sampling frequency (Hz).
    """
    import h5py

    with h5py.File(path, "r") as f:
        strain = f["strain/Strain"][:]
        t0 = float(f["meta/GPSstart"][()])
        dt = float(f["strain/Strain"].attrs["Xspacing"])
    fs = 1.0 / dt
    t = t0 + np.arange(strain.size) * dt
    return strain, t, fs


def whiten(x, fs, seglen=4.0, alpha=0.125):
    """Divide every frequency by how noisy the instrument is there.

    Raw strain is dominated by seismic motion below 20 Hz and by the suspension
    resonances, so a plot of it shows nothing at all. Whitening rescales each
    frequency by the local noise amplitude, which is what makes a signal
    visible without inventing it: the result has a flat spectrum and, away from
    a signal, unit variance.

    A Tukey window is used rather than a Hann. A Hann over the whole record
    suppresses most of it, which would deflate any noise level measured
    afterwards.
    """
    from scipy.signal import welch
    from scipy.signal.windows import tukey

    n = x.size
    freqs, psd = welch(x, fs=fs, nperseg=int(seglen * fs), window="hann")
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # the fs/2 turns the one sided density into the variance per rfft bin, so
    # that the result has unit variance and not merely a flat spectrum
    amp = np.sqrt(np.interp(f, freqs, psd) * fs / 2)
    amp[amp <= 0] = np.inf
    whitened = np.fft.irfft(np.fft.rfft(x * tukey(n, alpha=alpha)) / amp, n=n)
    return whitened


def bandpass(x, fs, band=(35.0, 350.0), order=4):
    """Keep one band and drop the rest, forwards and backwards so no phase."""
    from scipy.signal import butter, filtfilt

    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, x)
    return filtered


def q_transform(x, fs, t, centre, half=0.5, q=12.0,
                frange=(20.0, 500.0), nf=300, pad=4.0):
    """A Q transform: energy against time and frequency, at constant Q.

    The plane is tiled with one filter per frequency whose **bandwidth grows
    with the frequency**, so that every tile holds the same number of cycles.
    That is the whole idea, and it is the right one for a chirp: a signal that
    sweeps upwards needs fine time resolution where it is fast and fine
    frequency resolution where it is slow, which is exactly what constant Q
    gives.

    Each filter is a bisquare window in the frequency domain, applied to the
    one sided spectrum. Taking the inverse transform of a one sided spectrum
    returns a complex time series whose modulus is the envelope, so the energy
    falls straight out with no rectifying or smoothing.

    Rows are divided by their own median, so the map shows **excess** energy:
    a value of 1 is an ordinary stretch of noise at that frequency and a loud
    signal stands at tens.

    Returns ``(times relative to centre, frequencies, energy)``.
    """
    m = np.abs(t - centre) < half + pad
    seg = np.asarray(x)[m]
    ts = t[m] - centre
    n = seg.size

    spectrum = np.fft.fft(seg)
    f = np.fft.fftfreq(n, 1.0 / fs)
    freqs = np.geomspace(*frange, nf)

    energy = np.empty((nf, n))
    for i, f0 in enumerate(freqs):
        # the bisquare window reaches zero at this distance from f0; the
        # sqrt(11) is what makes its width match a Q of q in the usual
        # definition, where Q is the centre frequency over the bandwidth
        halfwidth = f0 * np.sqrt(11.0) / q
        window = np.zeros(n)
        keep = (f > 0) & (np.abs(f - f0) < halfwidth)
        u = (f[keep] - f0) / halfwidth
        window[keep] = (1.0 - u ** 2) ** 2
        norm = np.sqrt(np.sum(window ** 2))
        if norm == 0:
            continue
        z = np.fft.ifft(spectrum * (window / norm)) * np.sqrt(n)
        row = np.abs(z) ** 2
        energy[i] = row / max(np.median(row), 1e-30)

    inner = np.abs(ts) < half
    return ts[inner], freqs, energy[:, inner]
