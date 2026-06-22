"""
data_export.py
--------------
Functions for generating and exporting simulation data to CSV files.

Two export modes are provided:

1. generate_csv_data  — run a **single** simulation and write every trip record.
2. generate_sweep_csv — sweep over gamma values and write per-gamma statistics
                        (mean, std of headways and tour times).

Both functions return the absolute path of the written file.
"""

import csv
import os

import numpy as np

from simulation import simulate_bus_system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not yet exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _mean_std(arr) -> tuple:
    """Return (mean, std) of *arr*, or (0.0, 0.0) if *arr* is empty."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_csv_data(
    output_path: str = "simulation_output.csv",
    T1_initial: float = 1.0,
    T2_initial: float = 2.5,
    gamma: float = 1.0,
    S1: float = 0.0,
    S2: float = 0.0,
    num_trips: int = 1000,
    include_used_flag: bool = False,
) -> str:
    """
    Run a single simulation and export the arrival-time records to a CSV file.

    Each row in the output corresponds to one trip (the bias/sentinel entry
    with trip_number == -1 is excluded).

    Output columns
    --------------
    time, bus_id, trip_number [, used_flag]

    Parameters
    ----------
    output_path : str
        Destination file path.  Parent directories are created automatically.
    T1_initial, T2_initial : float
        Initial departure times of bus 1 and bus 2 from the origin.
    gamma : float
        Loading parameter Γ.
    S1, S2 : float
        Speedup parameters for bus 1 and bus 2.
    num_trips : int
        Number of trips to simulate for each bus.
    include_used_flag : bool
        When True, append a fourth column with the internal ``used_flag``
        boolean (useful for debugging the simulation logic).

    Returns
    -------
    str
        Absolute path of the written CSV file.

    Example
    -------
    >>> from data_export import generate_csv_data
    >>> path = generate_csv_data(
    ...     "data/run_gamma05.csv",
    ...     gamma=0.5, S1=0.5, S2=0.2, num_trips=1000,
    ... )
    >>> print(f"Saved to {path}")
    """
    arrival_time = simulate_bus_system(T1_initial, T2_initial, gamma, S1, S2, num_trips)

    _ensure_dir(output_path)

    header = ["time", "bus_id", "trip_number"]
    if include_used_flag:
        header.append("used_flag")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for time, bus_id, trip_number, used_flag in arrival_time:
            if trip_number == -1:       # skip sentinel entry
                continue
            row = [time, bus_id, trip_number]
            if include_used_flag:
                row.append(used_flag)
            writer.writerow(row)

    abs_path = os.path.abspath(output_path)
    print(f"[generate_csv_data] {num_trips * 2} trips  →  {abs_path}")
    return abs_path


def generate_sweep_csv(
    output_path: str = "simulation_sweep.csv",
    T1_initial: float = 1.0,
    T2_initial: float = 2.5,
    S1: float = 0.5,
    S2: float = 0.2,
    num_trips: int = 1000,
    gamma_range: tuple = (0.0, 2.0),
    num_gamma: int = 200,
    trip_min: int = 900,
    trip_max: int = 1000,
) -> str:
    """
    Sweep over gamma values and export per-gamma steady-state statistics
    (mean and standard deviation of headways and tour times) to a CSV file.

    Output columns
    --------------
    gamma,
    h1_mean, h1_std,
    h2_mean, h2_std,
    t1_mean, t1_std,
    t2_mean, t2_std,
    h1_count, h2_count

    Parameters
    ----------
    output_path : str
        Destination file path.  Parent directories are created automatically.
    T1_initial, T2_initial : float
        Initial departure times of bus 1 and bus 2.
    S1, S2 : float
        Speedup parameters for bus 1 and bus 2.
    num_trips : int
        Number of trips per simulation run.
    gamma_range : tuple of (float, float)
        (min_gamma, max_gamma) for the sweep.
    num_gamma : int
        Number of evenly-spaced gamma values to sweep.
    trip_min, trip_max : int
        Inclusive trip-index window used to extract steady-state data.

    Returns
    -------
    str
        Absolute path of the written CSV file.

    Example
    -------
    >>> from data_export import generate_sweep_csv
    >>> path = generate_sweep_csv(
    ...     "data/sweep_S05_S02.csv",
    ...     S1=0.5, S2=0.2, num_gamma=200,
    ... )
    >>> print(f"Saved to {path}")
    """
    gammas = np.linspace(*gamma_range, num_gamma)

    _ensure_dir(output_path)

    header = [
        "gamma",
        "h1_mean", "h1_std",
        "h2_mean", "h2_std",
        "t1_mean", "t1_std",
        "t2_mean", "t2_std",
        "h1_count", "h2_count",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for gamma in gammas:
            at = simulate_bus_system(T1_initial, T2_initial, gamma, S1, S2, num_trips)

            # --- Headways (gap in the interleaved chronological sequence) ---
            h1 = [
                at[i][0] - at[i - 1][0]
                for i in range(1, len(at))
                if at[i][1] == 1 and trip_min <= at[i][2] <= trip_max
            ]
            h2 = [
                at[i][0] - at[i - 1][0]
                for i in range(1, len(at))
                if at[i][1] == 2 and trip_min <= at[i][2] <= trip_max
            ]

            # --- Tour times (gap between successive arrivals of the same bus) ---
            at1 = np.array([r[0] for r in at if r[1] == 1 and trip_min <= r[2] <= trip_max])
            at2 = np.array([r[0] for r in at if r[1] == 2 and trip_min <= r[2] <= trip_max])
            t1 = np.abs(np.diff(at1)) if len(at1) > 1 else np.array([])
            t2 = np.abs(np.diff(at2)) if len(at2) > 1 else np.array([])

            h1m, h1s = _mean_std(h1)
            h2m, h2s = _mean_std(h2)
            t1m, t1s = _mean_std(t1)
            t2m, t2s = _mean_std(t2)

            writer.writerow([
                gamma,
                h1m, h1s,
                h2m, h2s,
                t1m, t1s,
                t2m, t2s,
                len(h1), len(h2),
            ])

    abs_path = os.path.abspath(output_path)
    print(
        f"[generate_sweep_csv] {num_gamma} gamma steps "
        f"({gamma_range[0]:.2f} → {gamma_range[1]:.2f})  →  {abs_path}"
    )
    return abs_path
