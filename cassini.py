#!/usr/bin/env python3
"""
Cassini I MGA trajectory animation.

Animates the evolutionary algorithm results for the Cassini I MGA problem
(GTOP benchmark).  Gene layout:
  gene_0          – launch date in days past J2000
  gene_1..gene_5  – time-of-flight [days] for each leg:
                    Earth → Venus → Venus → Earth → Jupiter → Saturn
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CartesianRepresentation

from poliastro.bodies import Sun, Earth, Venus, Jupiter, Saturn
from poliastro.ephem import Ephem
from poliastro.frames import Planes
from poliastro.iod import lambert
from poliastro.twobody import Orbit
from poliastro.twobody.mean_elements import get_mean_elements
from poliastro.twobody.sampling import EpochBounds
from poliastro.plotting.orbit.plotter import OrbitPlotter
from poliastro.plotting.util import BODY_COLORS
import poliastro.plotting.orbit.backends as orbit_backends
from poliastro.util import time_range

warnings.filterwarnings("ignore")
plt.style.use("default")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
J2000 = Time("2000-01-01 12:00:00", scale="tdb")
PLANE = Planes.EARTH_EQUATOR

SEQUENCE = [Earth, Venus, Venus, Earth, Jupiter, Saturn]
SEQ_LABELS = [
    "Earth (dep.)",
    "Venus (1st)",
    "Venus (2nd)",
    "Earth (swby)",
    "Jupiter (swby)",
    "Saturn (arr.)",
]
UNIQUE_BODIES = [Earth, Venus, Jupiter, Saturn]

LEG_COLORS = ["#ef5350", "#ab47bc", "#26a69a", "#ffa726", "#66bb6a"]
MARKER_SIZE = 9          # pt – flyby position markers
ORBIT_ALPHA = 0.35       # transparency of background orbits
N_TRAJ_PTS = 120         # sample points per Lambert arc

# Fixed axis limits (Saturn ~9.5 AU, keep 11 AU margin)
AXIS_LIMIT_AU = 11.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_epochs(gene_0, tofs):
    """Return the sequence of absolute epochs from the gene values."""
    t = J2000 + gene_0 * u.day
    epochs = [t]
    for tof in tofs:
        t = t + tof * u.day
        epochs.append(t)
    return epochs


def lambert_arc(r0, r1, tof_days, epoch_dep):
    """Solve Lambert problem and sample the transfer arc.

    Returns a CartesianRepresentation or None if the solver fails.
    """
    tof = tof_days * u.day
    v0, _ = lambert(Sun.k, r0, r1, tof)
    orb = Orbit.from_vectors(Sun, r0, v0, epoch=epoch_dep)
    epoch_arr = epoch_dep + tof
    ephem = orb.to_ephem(
        strategy=EpochBounds(
            min_epoch=epoch_dep, max_epoch=epoch_arr, num_values=N_TRAJ_PTS
        )
    )
    return ephem.sample()  # returns the pre-sampled coordinates


def body_orbit_coords(body, ref_epoch, n=200):
    """Sample one full revolution of *body* as a CartesianRepresentation."""
    period = get_mean_elements(body, ref_epoch).period
    epochs = time_range(ref_epoch, num_values=n, end=ref_epoch + period, scale="tdb")
    ephem = Ephem.from_body(body, epochs, attractor=Sun, plane=PLANE)
    return ephem.sample()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computation
# ─────────────────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv("cassini-I.csv")

# Reference epoch ≈ centre of the launch-date range in the CSV
ref_gene0 = float(df["gene_0"].median())
REF_EPOCH = J2000 + ref_gene0 * u.day

print("Pre-computing planetary orbital paths …")
orbit_coords = {body.name: body_orbit_coords(body, REF_EPOCH) for body in UNIQUE_BODIES}

# Reference orbit used to set the plotter's 2-D frame (fixed throughout)
r_ref, v_ref = Ephem.from_body(Earth, REF_EPOCH, attractor=Sun).rv(REF_EPOCH)
ref_orbit = Orbit.from_vectors(Sun, r_ref, v_ref, epoch=REF_EPOCH, plane=PLANE)

print(f"Pre-computing {len(df)} generations … (this may take a minute)")
gen_data = []
for idx, row in df.iterrows():
    genes = [row[f"gene_{i}"] for i in range(6)]
    tofs = genes[1:]
    epochs = get_epochs(genes[0], tofs)

    # Collect body positions at each flyby epoch
    positions = []
    for i in range(len(SEQUENCE)):
        r, _ = Ephem.from_body(SEQUENCE[i], epochs[i], attractor=Sun).rv(epochs[i])
        positions.append(r)

    # Solve Lambert arcs for each leg
    transfers = []
    for i in range(len(SEQUENCE) - 1):
        try:
            arc = lambert_arc(positions[i], positions[i + 1], tofs[i], epochs[i])
        except Exception:
            arc = None
        transfers.append(arc)

    gen_data.append(
        {
            "generation": int(row["generation"]),
            "best_fitness": float(row["best_fitness"]),
            "transfers": transfers,
            "positions": positions,   # List[Quantity shape (3,)]
        }
    )
    if (idx + 1) % 20 == 0:
        print(f"  … {idx + 1}/{len(df)} done")

print("Pre-computation complete.")

# ─────────────────────────────────────────────────────────────────────────────
# Figure & animation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

# Build an OrbitPlotter once — we reuse the same ax each frame.
# We need its _frame vectors so we can project body-position markers ourselves.
_backend0 = orbit_backends.Matplotlib2D(ax=ax)
_setup_plotter = OrbitPlotter(
    backend=_backend0, length_scale_units=u.AU, plane=PLANE
)
_setup_plotter.set_attractor(Sun)
_setup_plotter.set_orbit_frame(ref_orbit)
FRAME = _setup_plotter._frame       # (p_vec, q_vec, w_vec) – fixed
ax.cla()                            # clear the setup lines


def project_xy(r_km):
    """Project a heliocentric Cartesian position into the plotter's XY plane.

    Parameters
    ----------
    r_km : Quantity, shape (3,)
        Position in km.

    Returns
    -------
    x, y : float in AU
    """
    r_val = r_km.to_value(u.AU)                  # dimensionless (AU)
    r_arr = np.atleast_2d(r_val)                  # shape (1, 3)
    x = float(r_arr @ FRAME[0])
    y = float(r_arr @ FRAME[1])
    return x, y


def animate(frame_idx):
    ax.cla()

    data = gen_data[frame_idx]

    # ── Build a fresh OrbitPlotter for this frame ────────────────────────────
    backend = orbit_backends.Matplotlib2D(ax=ax)
    plotter = OrbitPlotter(backend=backend, length_scale_units=u.AU, plane=PLANE)
    plotter.set_attractor(Sun)
    plotter._frame = FRAME          # reuse the pre-computed frame (skips a set_orbit_frame call)

    # ── Planet orbital paths (faint background) ──────────────────────────────
    for body in UNIQUE_BODIES:
        coords = orbit_coords[body.name]
        color = BODY_COLORS.get(body.name, "#999999")
        plotter.plot_trajectory(
            coords,
            label=body.name,
            color=matplotlib.colors.to_rgba(color, ORBIT_ALPHA),
            dashed=True,
        )

    # ── Lambert transfer arcs (individual legs) ──────────────────────────────
    for i, (arc, color) in enumerate(zip(data["transfers"], LEG_COLORS)):
        if arc is not None:
            plotter.plot_trajectory(arc, label=f"Leg {i + 1}", color=color)

    # ── Spacecraft trajectory (all legs concatenated) ────────────────────────
    valid_arcs = [arc for arc in data["transfers"] if arc is not None]
    if valid_arcs:
        full_traj = CartesianRepresentation(
            x=np.concatenate([a.x for a in valid_arcs]),
            y=np.concatenate([a.y for a in valid_arcs]),
            z=np.concatenate([a.z for a in valid_arcs]),
        )
        plotter.plot_trajectory(
            full_traj, label="Spacecraft trajectory", color="#1a1a1a"
        )

    # ── Body-position markers ────────────────────────────────────────────────
    for i, (r, label) in enumerate(zip(data["positions"], SEQ_LABELS)):
        body = SEQUENCE[i]
        color = BODY_COLORS.get(body.name, "#999999")
        x, y = project_xy(r)
        ax.plot(
            x, y,
            "o",
            color=color,
            markersize=MARKER_SIZE,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=10,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + 0.25, y + 0.25),
            fontsize=7,
            color=color,
            zorder=11,
        )

    # ── Fix axis limits so Saturn is always visible ──────────────────────────
    ax.set_xlim(-AXIS_LIMIT_AU, AXIS_LIMIT_AU)
    ax.set_ylim(-AXIS_LIMIT_AU, AXIS_LIMIT_AU)
    ax.set_aspect("equal")

    # ── Annotations ─────────────────────────────────────────────────────────
    ax.set_title(
        f"Cassini I MGA  —  Generation {data['generation']}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    fitness = data["best_fitness"]
    fitness_str = f"∞" if not np.isfinite(fitness) else f"{fitness:.4f} km/s"
    ax.text(
        0.02, 0.97,
        f"Best Δv: {fitness_str}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#aaaaaa",
            alpha=0.9,
        ),
    )

    return []


anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(gen_data),
    interval=200,        # ms between frames
    blit=False,
    repeat=True,
)

plt.tight_layout()

if "--save" in sys.argv:
    out = "cassini_mga.mp4"
    writer = animation.FFMpegWriter(fps=5, bitrate=1800)
    print(f"Saving animation to {out} …")
    anim.save(out, writer=writer, dpi=150)
    print("Done.")
else:
    plt.show()
