# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:48:02 2026

@author: Elian PC

Runs:
  python Te_ultrafast_analysis.py --file Te3_150K_Linear_Reflection_PumpProbe_trace_20260122_1048.h5 --channel CH1 --tfit-min 100 --tfit-max 6000
  python Te_ultrafast_analysis.py --file Te3_150K_overnight_PumpProbe_trace_20260121_1712.h5 --channel CH1 --nmodes 2
  python Te_ultrafast_analysis.py --file ...h5 --channel CH2 --do-shg
  
  Choose tfit-min and tfit-max according to where the data fits better

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


major = 6
minor = 3
width = 1

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc("axes", labelsize=18)
plt.rc("xtick", labelsize=16, top=True, direction="in")
plt.rc("ytick", labelsize=16, right=True, direction="in")
plt.rc("axes", titlesize=18)
plt.rc("legend", fontsize=14)
plt.rcParams['font.family'] = "serif"
plt.rcParams['axes.linewidth'] = width
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['xtick.major.width'] = width
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['ytick.major.width'] = width
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor



# HDF5 utilities

def list_scans(h5f):
    """Return only ScanXXXXXX groups, not ScanInfo."""
    scans = []
    for k in h5f.keys():
        if k.startswith("Scan") and k[4:].isdigit():  # e.g. Scan000000
            scans.append(k)
    scans.sort()
    return scans


@dataclass
class PumpProbeData:
    delay_fs: np.ndarray
    theta_deg: np.ndarray
    on: np.ndarray   # shape (n_steps, n_angles)
    off: np.ndarray  # shape (n_steps, n_angles)


def load_pumpprobe_h5(path: str, channel: str = "CH1") -> PumpProbeData:
    """
    Load pump–probe data from HDF5 file.

    Expected inside each ScanXXXXXX group:
        {channel}_on
        {channel}_off
        theta_on (or theta_off)
        delay_time
    """

    with h5.File(path, "r") as f:

        scans = list_scans(f)

        if len(scans) == 0:
            raise RuntimeError(
                "No ScanXXXXXX groups found. "
                "File only contains metadata like ScanInfo."
            )

        # Inspect first scan to get dimensions
        first = scans[0]
        g0 = f[first]

        if f"{channel}_on" not in g0 or f"{channel}_off" not in g0:
            raise KeyError(
                f"Datasets {channel}_on / {channel}_off not found in {first}. "
                f"Available keys: {list(g0.keys())}"
            )

        on0 = g0[f"{channel}_on"][:]
        off0 = g0[f"{channel}_off"][:]

        n_steps, n_angles = on0.shape

        on = np.empty((len(scans), n_steps, n_angles), dtype=float)
        off = np.empty_like(on)

        theta = None
        delay = None

        for i, sc in enumerate(scans):

            g = f[sc]

            # Load on/off
            on[i] = g[f"{channel}_on"][:]
            off[i] = g[f"{channel}_off"][:]

            # Load theta once
            if theta is None:
                if "theta_on" in g:
                    theta = g["theta_on"][0, :]
                elif "theta_off" in g:
                    theta = g["theta_off"][0, :]
                else:
                    raise KeyError(
                        f"No theta_on or theta_off found in {sc}. "
                        f"Available keys: {list(g.keys())}"
                    )

            # Load delay once
            if delay is None:
                if "delay_time" in g:
                    delay = g["delay_time"][:]
                else:
                    raise KeyError(
                        f"No delay_time found in {sc}. "
                        f"Available keys: {list(g.keys())}"
                    )

    return PumpProbeData(
        delay_fs=np.asarray(delay, float),
        theta_deg=np.asarray(theta, float),
        on=np.mean(on, axis=0),
        off=np.mean(off, axis=0),
    )



# Angle binning + harmonic extraction

def bin_theta(theta_deg: np.ndarray, sig: np.ndarray, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_deg, float)
    y = np.asarray(sig, float)

    _, edges = np.histogram(theta, bins=nbins)
    weighted = np.histogram(theta, bins=nbins, weights=y)[0]
    counts = np.histogram(theta, bins=nbins)[0].astype(float)
    counts[counts == 0] = np.nan
    binned = weighted / counts

    centers = edges[1:] - 0.5 * (edges[1] - edges[0])
    return centers, binned


def get_model_projection(sig: np.ndarray, theta_deg: np.ndarray, n: int) -> Tuple[float, float]:
    th = np.deg2rad(theta_deg)
    if n == 0:
        return float(np.trapezoid(sig, th) / np.trapezoid(np.ones_like(th), th)), 0.0

    cosn = np.cos(n * th)
    sinn = np.sin(n * th)
    C = np.trapezoid(sig * cosn, th) / np.trapezoid(cosn * cosn, th)
    S = np.trapezoid(sig * sinn, th) / np.trapezoid(sinn * sinn, th)
    return float(C), float(S)


def amplitude_phase(C: np.ndarray, S: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    A = np.sqrt(C**2 + S**2)
    phi = np.arctan2(S, C) / float(n) if n != 0 else np.zeros_like(A)
    return A, phi


def harmonic_reconstruction(theta_deg: np.ndarray, terms: List[Tuple[int, float, float]]) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    y = np.zeros_like(th, dtype=float)
    for n, C, S in terms:
        if n == 0:
            y += C
        else:
            y += C * np.cos(n * th) + S * np.sin(n * th)
    return y


def extract_harmonics(
    delay_fs: np.ndarray,
    theta_deg: np.ndarray,
    on: np.ndarray,
    off: np.ndarray,
    nbins: int = 100,
    ns: Tuple[int, ...] = (0, 4, 8, 12),
) -> Dict[str, np.ndarray]:
    n_steps = on.shape[0]
    theta_centers, _ = bin_theta(theta_deg, on[0], nbins=nbins)

    on_b = np.zeros((n_steps, len(theta_centers)), dtype=float)
    off_b = np.zeros_like(on_b)

    for i in range(n_steps):
        _, on_b[i] = bin_theta(theta_deg, on[i], nbins=nbins)
        _, off_b[i] = bin_theta(theta_deg, off[i], nbins=nbins)

    out: Dict[str, np.ndarray] = dict(
        delay_fs=np.asarray(delay_fs, float),
        theta_binned_deg=np.asarray(theta_centers, float),
        on_binned=on_b,
        off_binned=off_b,
    )

    for n in ns:
        C_on = np.zeros(n_steps, dtype=float)
        S_on = np.zeros(n_steps, dtype=float)
        C_off = np.zeros(n_steps, dtype=float)
        S_off = np.zeros(n_steps, dtype=float)

        for i in range(n_steps):
            C_on[i], S_on[i] = get_model_projection(on_b[i], theta_centers, n)
            C_off[i], S_off[i] = get_model_projection(off_b[i], theta_centers, n)

        out[f"C{n}_on"] = C_on
        out[f"S{n}_on"] = S_on
        out[f"C{n}_off"] = C_off
        out[f"S{n}_off"] = S_off

        if n > 0:
            A_on, phi_on = amplitude_phase(C_on, S_on, n)
            A_off, phi_off = amplitude_phase(C_off, S_off, n)
            out[f"I{n}_on"] = A_on
            out[f"I{n}_off"] = A_off
            out[f"phi{n}_on"] = phi_on
            out[f"phi{n}_off"] = phi_off

    out["I0_on"] = out["C0_on"]
    out["I0_off"] = out["C0_off"]
    return out



# Coherent phonon analysis

def damped_cosine(t_fs: np.ndarray, A: float, tau_fs: float, f_THz: float, phase: float) -> np.ndarray:
    # 1 THz = 1/ps = 1e-3 / fs
    return A * np.exp(-t_fs / tau_fs) * np.cos(2.0 * np.pi * f_THz * (t_fs * 1e-3) + phase)


def phonon_sum_model(t_fs: np.ndarray, *p: float) -> np.ndarray:
    """
    offset + Σ_k A_k exp(-t/tau_k) cos(2π f_k t + phase_k)
    p = [offset, A1, tau1, f1, ph1, A2, tau2, f2, ph2, ...]
    """
    offset = p[0]
    y = offset * np.ones_like(t_fs, dtype=float)
    n_modes = (len(p) - 1) // 4
    for k in range(n_modes):
        A, tau, f, ph = p[1 + 4*k : 1 + 4*k + 4]
        y += damped_cosine(t_fs, A, tau, f, ph)
    return y


def spline_background(t_fs: np.ndarray, y: np.ndarray, s: float) -> np.ndarray:
    idx = np.argsort(t_fs)
    sp = UnivariateSpline(t_fs[idx], y[idx], s=s)
    return sp(t_fs)


def fft_spectrum(t_fs: np.ndarray, y: np.ndarray, tmin_fs: float, tmax_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t_fs >= tmin_fs) & (t_fs <= tmax_fs)
    tt = t_fs[mask].astype(float)
    yy = y[mask].astype(float)

    if len(tt) < 16:
        return np.array([]), np.array([])

    dt = np.median(np.diff(tt))
    # if not uniform, interpolate to uniform
    if not np.allclose(np.diff(tt), dt, rtol=1e-2, atol=1e-2):
        t_uni = np.arange(tt.min(), tt.max(), dt)
        yy = np.interp(t_uni, tt, yy)
        tt = t_uni

    yy = yy - np.mean(yy)
    n = len(tt)
    freq = np.fft.rfftfreq(n, d=dt) * 1e3  # cycles/fs -> THz
    mag = np.abs(np.fft.rfft(yy))
    return freq, mag


def guess_modes_from_fft(freq_THz: np.ndarray, mag: np.ndarray, fmin: float = 0.2, fmax: float = 10.0, n_peaks: int = 3) -> List[float]:
    mask = (freq_THz >= fmin) & (freq_THz <= fmax)
    f = freq_THz[mask]
    m = mag[mask]
    if len(f) < 10:
        return []
    peaks, props = find_peaks(m, prominence=np.max(m) * 0.02)
    if len(peaks) == 0:
        return []
    prominences = props["prominences"]
    order = np.argsort(prominences)[::-1][:n_peaks]
    return [float(x) for x in f[peaks][order]]


def fit_phonons(
    t_fs: np.ndarray,
    y: np.ndarray,
    n_modes: int = 1,
    tfit_min_fs: float = 100.0,
    tfit_max_fs: float = 6000.0,
    bg_method: str = "spline",
    spline_s: float = 1e-2,
    initial_freqs_THz: Optional[List[float]] = None,
) -> Dict[str, object]:
    m = (t_fs >= tfit_min_fs) & (t_fs <= tfit_max_fs)
    t = t_fs[m].astype(float)
    yy = y[m].astype(float)

    # background
    if bg_method == "spline":
        bg = spline_background(t, yy, s=spline_s)
    elif bg_method == "exp":
        def bg_exp(t, c, a1, tau1, a2, tau2):
            return c + a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2)
        p0 = [np.median(yy[-50:]), yy[0]-yy[-1], 300.0, 0.2*(yy[0]-yy[-1]), 3000.0]
        try:
            popt, _ = curve_fit(bg_exp, t, yy, p0=p0, maxfev=20000)
            bg = bg_exp(t, *popt)
        except Exception:
            bg = spline_background(t, yy, s=spline_s)
    else:
        raise ValueError("bg_method must be 'spline' or 'exp'")

    resid = yy - bg

    # FFT guess
    freq, mag = fft_spectrum(t, resid, tmin_fs=t.min(), tmax_fs=t.max())
    if initial_freqs_THz is None or len(initial_freqs_THz) == 0:
        initial_freqs_THz = guess_modes_from_fft(freq, mag, n_peaks=max(n_modes, 1))

    while len(initial_freqs_THz) < n_modes:
        initial_freqs_THz.append(3.6)

    # nonlinear fit of residual
    p0 = [0.0]
    lo = [-np.inf]
    hi = [ np.inf]
    amp0 = np.std(resid) if np.std(resid) > 0 else 1e-3

    for k in range(n_modes):
        f0 = initial_freqs_THz[k]
        p0 += [amp0, 2000.0, f0, 0.0]
        lo += [-np.inf,  50.0, 0.1, -2*np.pi]
        hi += [ np.inf, 1e6,  20.0,  2*np.pi]

    try:
        popt, pcov = curve_fit(phonon_sum_model, t, resid, p0=p0, bounds=(lo, hi), maxfev=200000)
        fit = phonon_sum_model(t, *popt)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan, dtype=float)
        ok = True
    except Exception:
        popt = np.array(p0, dtype=float)
        perr = np.full_like(popt, np.nan, dtype=float)
        fit = phonon_sum_model(t, *popt)
        ok = False

    modes = []
    for k in range((len(popt)-1)//4):
        A, tau, f, ph = popt[1+4*k:1+4*k+4]
        dA, dtau, df, dph = perr[1+4*k:1+4*k+4]
        modes.append(dict(A=A, tau_fs=tau, f_THz=f, phase=ph, dA=dA, dtau_fs=dtau, df_THz=df, dphase=dph))

    return dict(
        ok=ok, t_fs=t, y=yy, bg=bg, resid=resid,
        fft_freq_THz=freq, fft_mag=mag,
        popt=popt, perr=perr, fit=fit, modes=modes,
        initial_freqs_THz=initial_freqs_THz,
        bg_method=bg_method, spline_s=spline_s,
        tfit_min_fs=tfit_min_fs, tfit_max_fs=tfit_max_fs,
    )



# SHG “orientation” models (phenomenological)

def shg_model_c_out(theta_deg: np.ndarray, C0: float, C6: float, theta0_deg: float, C12: float = 0.0) -> np.ndarray:
    th = np.deg2rad(theta_deg - theta0_deg)
    return C0 + C6*np.cos(6*th) + C12*np.cos(12*th)


def shg_model_c_in(theta_deg: np.ndarray, C0: float, C2: float, theta0_deg: float, C4: float = 0.0) -> np.ndarray:
    th = np.deg2rad(theta_deg - theta0_deg)
    return C0 + C2*np.cos(2*th) + C4*np.cos(4*th)


def fit_shg_orientation(theta_deg: np.ndarray, I: np.ndarray) -> Dict[str, object]:
    theta = np.asarray(theta_deg, float)
    y = np.asarray(I, float)
    y_shift = y - np.min(y) + 1e-12

    def rss(yhat): return float(np.mean((y_shift - yhat)**2))

    # 6-fold
    p0a = [np.median(y_shift), 0.3*(np.max(y_shift)-np.min(y_shift)), 0.0, 0.05*(np.max(y_shift)-np.min(y_shift))]
    bnda_lo = [0.0, -np.inf, -180.0, -np.inf]
    bnda_hi = [np.inf, np.inf, 180.0, np.inf]

    # 2/4-fold
    p0b = [np.median(y_shift), 0.3*(np.max(y_shift)-np.min(y_shift)), 0.0, 0.1*(np.max(y_shift)-np.min(y_shift))]
    bndb_lo = [0.0, -np.inf, -180.0, -np.inf]
    bndb_hi = [np.inf, np.inf, 180.0, np.inf]

    try:
        poptA, _ = curve_fit(lambda th, C0, C6, th0, C12: shg_model_c_out(th, C0, C6, th0, C12),
                             theta, y_shift, p0=p0a, bounds=(bnda_lo, bnda_hi), maxfev=50000)
        yA = shg_model_c_out(theta, *poptA)
        rssA = rss(yA)
        okA = True
    except Exception:
        poptA = np.array(p0a, float)
        yA = shg_model_c_out(theta, *poptA)
        rssA = rss(yA)
        okA = False

    try:
        poptB, _ = curve_fit(lambda th, C0, C2, th0, C4: shg_model_c_in(th, C0, C2, th0, C4),
                             theta, y_shift, p0=p0b, bounds=(bndb_lo, bndb_hi), maxfev=50000)
        yB = shg_model_c_in(theta, *poptB)
        rssB = rss(yB)
        okB = True
    except Exception:
        poptB = np.array(p0b, float)
        yB = shg_model_c_in(theta, *poptB)
        rssB = rss(yB)
        okB = False

    if rssA <= rssB:
        return dict(best="c_out_of_plane_6fold", ok=okA, popt=poptA, theta0_deg=float(poptA[2]), rss=rssA,
                    theta_deg=theta, I=y_shift, Ifit=yA)
    else:
        return dict(best="c_in_plane_2fold", ok=okB, popt=poptB, theta0_deg=float(poptB[2]), rss=rssB,
                    theta_deg=theta, I=y_shift, Ifit=yB)



# Plotting helpers

def plot_harmonics(h: Dict[str, np.ndarray], title: str = "") -> None:
    t = h["delay_fs"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

    ax[0, 0].plot(t, h["I0_on"], label=r"$I_0$ on")
    ax[0, 0].plot(t, h["I0_off"], label=r"$I_0$ off")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.25)

    ax[0, 1].plot(t, h.get("I4_on", np.nan*t), label=r"$I_4$ on")
    ax[0, 1].plot(t, h.get("I4_off", np.nan*t), label=r"$I_4$ off")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.25)

    ax[1, 0].plot(t, h.get("I8_on", np.nan*t), label=r"$I_8$ on")
    ax[1, 0].plot(t, h.get("I8_off", np.nan*t), label=r"$I_8$ off")
    ax[1, 0].legend(); ax[1, 0].grid(True, alpha=0.25)

    ax[1, 1].plot(t, h.get("I12_on", np.nan*t), label=r"$I_{12}$ on")
    ax[1, 1].plot(t, h.get("I12_off", np.nan*t), label=r"$I_{12}$ off")
    ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.25)

    for a in ax.flat:
        a.set_xlabel(r"Delay (fs)")

    fig.suptitle(title)
    plt.show()


def plot_polar_snapshot(h: Dict[str, np.ndarray], t0_fs: float = 0.0) -> None:
    delay = h["delay_fs"]
    idx = int(np.argmin(np.abs(delay - t0_fs)))
    th = h["theta_binned_deg"]

    terms_on = [(0, h["C0_on"][idx], 0.0),
                (4, h["C4_on"][idx], h["S4_on"][idx]),
                (8, h["C8_on"][idx], h["S8_on"][idx]),
                (12, h["C12_on"][idx], h["S12_on"][idx])]
    terms_off = [(0, h["C0_off"][idx], 0.0),
                 (4, h["C4_off"][idx], h["S4_off"][idx]),
                 (8, h["C8_off"][idx], h["S8_off"][idx]),
                 (12, h["C12_off"][idx], h["S12_off"][idx])]

    th_dense = np.linspace(0, 360, 400)
    y_on_fit = harmonic_reconstruction(th_dense, terms_on)
    y_off_fit = harmonic_reconstruction(th_dense, terms_off)

    fig = plt.figure(figsize=(9, 4), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    ax1.set_title(fr"On, $t={delay[idx]:.0f}$ fs")
    ax2.set_title(fr"Off, $t={delay[idx]:.0f}$ fs")

    ax1.plot(np.deg2rad(th), h["on_binned"][idx], "k", lw=1.0)
    ax1.plot(np.deg2rad(th_dense), y_on_fit, "b", lw=1.2)

    ax2.plot(np.deg2rad(th), h["off_binned"][idx], "k", lw=1.0)
    ax2.plot(np.deg2rad(th_dense), y_off_fit, "b", lw=1.2)

    plt.show()


def plot_phonon_fit(res: Dict[str, object], label: str = "") -> None:
    t = res["t_fs"]
    yy = res["y"]
    bg = res["bg"]
    resid = res["resid"]
    fit = res["fit"]
    freq = res["fft_freq_THz"]
    mag = res["fft_mag"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

    ax[0, 0].plot(t, yy, label="data")
    ax[0, 0].plot(t, bg, label="background", lw=2)
    ax[0, 0].set_title(label + r" : data + background")
    ax[0, 0].set_xlabel(r"$t$ (fs)")
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=0.25)

    ax[0, 1].plot(t, resid, label="residual")
    ax[0, 1].plot(t, fit, label="fit", lw=2)
    ax[0, 1].set_title("coherent phonon residual + fit")
    ax[0, 1].set_xlabel(r"$t$ (fs)")
    ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.25)

    if len(freq) > 0:
        ax[1, 0].plot(freq, mag)
        ax[1, 0].set_xlim(0, 10)
        ax[1, 0].set_title("FFT magnitude (residual)")
        ax[1, 0].set_xlabel(r"Frequency (THz)")
        ax[1, 0].grid(True, alpha=0.25)
    else:
        ax[1, 0].axis("off")

    txt = []
    for k, m in enumerate(res["modes"]):
        txt.append(fr"$f_{{{k+1}}}={m['f_THz']:.3f}\pm{m['df_THz']:.3f}$ THz")
        txt.append(fr"$\tau_{{{k+1}}}={m['tau_fs']:.0f}\pm{m['dtau_fs']:.0f}$ fs")
        txt.append(fr"$A_{{{k+1}}}={m['A']:.3g}$")
        txt.append("")

    ax[1, 1].axis("off")
    ax[1, 1].text(0.02, 0.95, "\n".join(txt) if txt else "No fit modes", va="top")

    plt.show()



# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="HDF5 file path")
    ap.add_argument("--channel", default="CH1", help="channel prefix, e.g., CH1 or CH2")
    ap.add_argument("--nbins", type=int, default=100, help="theta bin count")
    ap.add_argument("--tfit-min", type=float, default=100.0, help="fit window start (fs)")
    ap.add_argument("--tfit-max", type=float, default=6000.0, help="fit window end (fs)")
    ap.add_argument("--nmodes", type=int, default=1, help="number of coherent phonon modes to fit")
    ap.add_argument("--bg", choices=["spline", "exp"], default="spline", help="background method")
    ap.add_argument("--spline-s", type=float, default=1e-2, help="smoothing factor for spline background")
    ap.add_argument("--do-shg", action="store_true", help="also fit SHG polar orientation (uses pre-pump snapshot)")
    ap.add_argument("--shg-t0", type=float, default=-200.0, help="time (fs) used for SHG polar pattern (typically pre-pump)")
    ap.add_argument("--export-csv", default="", help="export extracted harmonic traces to CSV")
    args = ap.parse_args()

    pp = load_pumpprobe_h5(args.file, channel=args.channel)
    h = extract_harmonics(pp.delay_fs, pp.theta_deg, pp.on, pp.off, nbins=args.nbins)

    plot_harmonics(h, title=os.path.basename(args.file) + f" ({args.channel})")
    plot_polar_snapshot(h, t0_fs=0.0)

    t = h["delay_fs"]
    eps = 1e-12

    # On/Off ratios per symmetry channel
    y_iso = h["I0_on"] / (h["I0_off"] + eps)
    y4 = h.get("I4_on", np.nan*t) / (h.get("I4_off", np.nan*t) + eps)
    y8 = h.get("I8_on", np.nan*t) / (h.get("I8_off", np.nan*t) + eps)
    y12 = h.get("I12_on", np.nan*t) / (h.get("I12_off", np.nan*t) + eps)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    ax[0, 0].plot(t, y_iso, label=r"$I_0^{on}/I_0^{off}$")
    ax[0, 1].plot(t, y4, label=r"$I_4^{on}/I_4^{off}$")
    ax[1, 0].plot(t, y8, label=r"$I_8^{on}/I_8^{off}$")
    ax[1, 1].plot(t, y12, label=r"$I_{12}^{on}/I_{12}^{off}$")
    for a in ax.flat:
        a.set_xlabel(r"Delay (fs)")
        a.legend()
        a.grid(True, alpha=0.25)
    plt.show()

    # Coherent phonon fits
    res_iso = fit_phonons(t, y_iso, n_modes=args.nmodes,
                          tfit_min_fs=args.tfit_min, tfit_max_fs=args.tfit_max,
                          bg_method=args.bg, spline_s=args.spline_s)
    plot_phonon_fit(res_iso, label=r"Isotropic channel ($I_0$ ratio)")

    init_freqs = [m["f_THz"] for m in res_iso["modes"]] if res_iso["modes"] else None

    res_aniso4 = fit_phonons(t, y4, n_modes=args.nmodes,
                             tfit_min_fs=args.tfit_min, tfit_max_fs=args.tfit_max,
                             bg_method=args.bg, spline_s=args.spline_s,
                             initial_freqs_THz=init_freqs)
    plot_phonon_fit(res_aniso4, label=r"Anisotropy channel ($I_4$ ratio)")

    res_aniso8 = fit_phonons(t, y8, n_modes=args.nmodes,
                             tfit_min_fs=args.tfit_min, tfit_max_fs=args.tfit_max,
                             bg_method=args.bg, spline_s=args.spline_s,
                             initial_freqs_THz=init_freqs)
    plot_phonon_fit(res_aniso8, label=r"Anisotropy channel ($I_8$ ratio)")

    # Optional SHG orientation (run the script on --SHG)
    if args.do_shg:
        idx = int(np.argmin(np.abs(t - args.shg_t0)))
        theta = h["theta_binned_deg"]
        Ipol = h["off_binned"][idx]  # use equilibrium (off) pattern
        shg = fit_shg_orientation(theta, Ipol)

        print("\n=== SHG ORIENTATION FIT ===")
        print(f"Best model: {shg['best']} | theta0 = {shg['theta0_deg']:.2f} deg | RSS={shg['rss']:.3e}")

        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        ax.plot(np.deg2rad(shg["theta_deg"]), shg["I"], "k", label="data")
        ax.plot(np.deg2rad(shg["theta_deg"]), shg["Ifit"], "b", label="fit")
        ax.set_title(f"SHG orientation: {shg['best']} (theta0={shg['theta0_deg']:.1f} deg)")
        ax.legend()
        plt.show()

    # Export CSV for downstream work
    if args.export_csv:
        import pandas as pd
        df = pd.DataFrame({
            "delay_fs": t,
            "I0_ratio": y_iso,
            "I4_ratio": y4,
            "I8_ratio": y8,
            "I12_ratio": y12,
            "phi4_delta": h.get("phi4_on", np.nan*t) - h.get("phi4_off", np.nan*t),
            "phi8_delta": h.get("phi8_on", np.nan*t) - h.get("phi8_off", np.nan*t),
            "phi12_delta": h.get("phi12_on", np.nan*t) - h.get("phi12_off", np.nan*t),
        })
        df.to_csv(args.export_csv, index=False)
        print(f"Wrote {args.export_csv} ({len(df)} rows)")

    def print_modes(label: str, res: Dict[str, object]) -> None:
        print(f"\n=== {label} ===")
        print(f"fit ok: {res['ok']} | initial freqs: {res['initial_freqs_THz']}")
        for i, m in enumerate(res["modes"]):
            print(f"mode {i+1}: f={m['f_THz']:.4f}±{m['df_THz']:.4f} THz, "
                  f"tau={m['tau_fs']:.0f}±{m['dtau_fs']:.0f} fs, A={m['A']:.4g}")

    print_modes("Isotropic (I0 ratio)", res_iso)
    print_modes("Anisotropic (I4 ratio)", res_aniso4)
    print_modes("Anisotropic (I8 ratio)", res_aniso8)


if __name__ == "__main__":
    main()
