import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

KM_TO_M = 1000.0
LEO_EARTH_GGC = 398600.7  # Earth gravitational parameter μ in km^3/s^2 (as in your C++ code)

# WGS84 equatorial radius in kilometers (a.k.a. semi-major axis a)
EARTH_RADIUS_KM = 6378.137


def generate_progress_vector(
    orbit_alt_km: float, time_step_sec: float, inc_deg: float = 28.0
) -> np.ndarray:
    """
    Generate an array of angular offsets ("progress vector") that advance around a circular orbit
    by a fixed time step.

    This mirrors your C++ logic:
      - Assume a circular orbit with radius r = (Earth radius + altitude).
      - Use circular-orbit speed v = sqrt(μ / r).
      - Convert time step into arc-length step: step_size = v * dt.
      - Convert arc-length to fraction of orbital circumference, then to an angle increment.

    Parameters
    ----------
    orbit_alt_km
        Orbit altitude above the reference Earth radius, in kilometers.
    time_step_sec
        Simulation sampling interval (dt), in seconds. One progress-vector entry is produced per dt.
    inc_deg
        Inclination in degrees, used only to replicate your sign convention (flip for retrograde
        if inclination > 90 deg).

    Returns
    -------
    np.ndarray
        Array of angles (radians) from 0 up to ~2π (one orbital revolution), sampled at dt.
        The length is computed so that the last step is approximately one full orbit.
    """
    # Orbit radius from Earth's equatorial radius (WGS84) + altitude
    orbit_radius_km = EARTH_RADIUS_KM + orbit_alt_km

    # Circular-orbit speed (km/s): v = sqrt(μ / r)
    node_speed_kms = np.sqrt(LEO_EARTH_GGC / orbit_radius_km)

    # Orbital circumference for a circular orbit: 2πr (km)
    orbit_perimeter_km = orbit_radius_km * 2.0 * np.pi

    # Distance traveled in one simulation step (km)
    step_size_km = node_speed_kms * time_step_sec

    # Fraction of full orbit per step (dimensionless)
    step_fraction = step_size_km / orbit_perimeter_km

    # Number of steps to cover ~one full orbit
    steps = int(round(1.0 / step_fraction))

    # Angle advanced per step (radians), before applying sign
    step_angle_base = 2.0 * np.pi * step_fraction

    # Match your C++ sign convention: flip direction for inclinations > 90° (retrograde)
    sign = 1 if np.deg2rad(inc_deg) <= (np.pi / 2.0) else -1
    step_angle = sign * step_angle_base

    # Angles from 0, step_angle, 2*step_angle, ...
    return np.array([k * step_angle for k in range(steps)], dtype=float)


def apply_raan_and_inclination(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, raan_deg: float, inclination_deg: float
):
    """
    Rotate orbit points from the initial XY plane (z=0) into a tilted/oriented orbital plane.

    The transform applied is:
        r_new = Rz(Ω) * Rx(i) * r
    where:
        Ω (RAAN) rotates the line of nodes around the +Z axis,
        i (inclination) tilts the plane by rotating around the +X axis.

    Parameters
    ----------
    x, y, z
        Arrays (N,) representing point coordinates (km).
    raan_deg
        Right ascension of the ascending node Ω, in degrees (rotation about +Z).
    inclination_deg
        Inclination i, in degrees (rotation about +X).

    Returns
    -------
    (x2, y2, z2)
        Rotated coordinate arrays (km).
    """
    Om = np.deg2rad(raan_deg)
    i = np.deg2rad(inclination_deg)

    # Rotation about Z axis (RAAN)
    Rz = np.array(
        [
            [np.cos(Om), -np.sin(Om), 0.0],
            [np.sin(Om), np.cos(Om), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Rotation about X axis (inclination tilt)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ]
    )

    pts = np.vstack([x, y, z])  # shape (3, N)
    pts_new = (Rz @ Rx) @ pts  # shape (3, N)
    return pts_new[0], pts_new[1], pts_new[2]


def rodrigues_rotate_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    inclination_deg: float,
    raan_deg: float,
    plane_rotation_deg: float,
):
    """
    Apply an additional rigid rotation to all points using Rodrigues' rotation formula.

    This rotates each 3D point about a user-defined axis vector `n` by `plane_rotation_deg`.
    Rodrigues’ formula (for unit axis n) is: v' = v*cos(a) + (n×v)*sin(a) + n*(n·v)*(1-cos(a)).

    Notes
    -----
    - The axis `n` you compute below is treated as the rotation axis in the same coordinate frame
      as (x,y,z). Ensure the axis definition matches the physical meaning you want.
    - `n` is normalized internally; it must not be the zero vector.

    Parameters
    ----------
    x, y, z
        Arrays (N,) of point coordinates (km).
    inclination_deg
        Inclination used in the axis construction below (degrees).
    raan_deg
        RAAN used in the axis construction below (degrees). (Your original code called this "latitude".)
    plane_rotation_deg
        Rotation angle about axis n, in degrees.

    Returns
    -------
    (x2, y2, z2)
        Rotated coordinate arrays (km).
    """
    P = np.column_stack([x, y, z]).astype(float)  # shape (N,3)

    inc = np.deg2rad(inclination_deg)
    Om = np.deg2rad(raan_deg)
    a = np.deg2rad(plane_rotation_deg)

    # Your chosen rotation axis (interpreted as "rotate plane axis").
    # Keep this as-is, but document what it is: a direction derived from (inc, RAAN).
    n = np.array([np.sin(Om) * np.sin(inc), -np.cos(Om) * np.sin(inc), np.cos(inc)], dtype=float)

    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("Rotation axis n is zero; check inclination/raan inputs.")
    n = n / n_norm  # must be unit axis for Rodrigues

    c = np.cos(a)
    s = np.sin(a)

    # Vectorized Rodrigues rotation for all points:
    # P2 = P*c + (n×P)*s + n*(P·n)*(1-c)
    P2 = P * c + np.cross(n, P) * s + (n * (P @ n)[:, None]) * (1.0 - c)

    return P2[:, 0], P2[:, 1], P2[:, 2]


# ----------------------------
# Example parameters
# ----------------------------

orbit_alt_km = 500.0
time_step_sec = 600.0

inclination_deg = 30.0
raan_deg = 0.0  # we call it "latitude" in ns-3, to infer it starts from prime meridian and rotate like time zones
plane_rotation_deg = 60.0  # extra rotation about chosen axis (dial/spin)

# 1) Progress angles over ~one orbit
angles = generate_progress_vector(orbit_alt_km, time_step_sec, inc_deg=inclination_deg)

# 2) Start with a circle in the XY plane (z=0), radius r = EarthRadius + altitude
r_km = EARTH_RADIUS_KM + orbit_alt_km
x = r_km * np.cos(angles)
y = r_km * np.sin(angles)
z = np.zeros_like(angles)

# 3) Tilt/orient the orbit plane using RAAN + inclination
x, y, z = apply_raan_and_inclination(x, y, z, raan_deg=raan_deg, inclination_deg=inclination_deg)

# 4) Optional: rotate the entire set again using Rodrigues (about axis derived from inc+RAAN)
x, y, z = rodrigues_rotate_points(
    x,
    y,
    z,
    inclination_deg=inclination_deg,
    raan_deg=raan_deg,
    plane_rotation_deg=plane_rotation_deg,
)

# ----------------------------
# Plot
# ----------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter points (orbit samples). If you want a connected line, use ax.plot instead.
ax.scatter(x, y, z, color="tab:blue", s=8, label="Progress vector samples")
ax.scatter(x[0], y[0], z[0], color="green", s=50, label="Start")
ax.scatter(x[-1], y[-1], z[-1], color="red", s=50, label="End")

# Earth reference circle in the XY plane
theta = np.linspace(0.0, 2.0 * np.pi, 200)
ax.plot(
    EARTH_RADIUS_KM * np.cos(theta),
    EARTH_RADIUS_KM * np.sin(theta),
    np.zeros_like(theta),
    "k--",
    linewidth=1.0,
    label="Earth radius (equatorial)",
)

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title(
    "LEO orbit progress samples\n"
    f"Alt={orbit_alt_km} km, dt={time_step_sec} s, N={len(angles)}\n"
    f"inc={inclination_deg}°, RAAN={raan_deg}°, extra rot={plane_rotation_deg}°"
)
ax.legend()
plt.show()
