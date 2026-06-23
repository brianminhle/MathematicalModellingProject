"""
simulation.py
-------------
Core simulation engine for the two-bus transportation system.

Model equation (dimensionless arrival time):
    T_i(m+1) = T_i(m) + Γ * (T_i(m) - T_i'(m')) + 1 / (1 + S_i * (T_i(m) - T_i'(m')))

Parameters:
    Γ (gamma)  : Loading parameter
    S_i        : Speedup parameter for bus i
    T_i(m)     : Arrival time of bus i at trip m at the origin
    T_i'(m')   : Arrival time of the other bus at trip m' (the bus that left just before bus i)
"""


def simulate_bus_system(
    T1_initial: float = 1.0,
    T2_initial: float = 2.5,
    gamma: float = 1.0,
    S1: float = 0.0,
    S2: float = 0.0,
    num_trips: int = 1000,
) -> list:
    """
    Simulate a two-bus transportation system using the recurrence relation:
        T_i(m+1) = T_i(m) + Γ * (T_i(m) - T_i'(m')) + 1 / (1 + S_i * (T_i(m) - T_i'(m')))

    Parameters
    ----------
    T1_initial : float
        Initial departure time of bus 1 from the origin.
    T2_initial : float
        Initial departure time of bus 2 from the origin.
    gamma : float
        Loading sensitivity parameter (Γ) affecting the spacing between trips.
    S1 : float
        Speedup parameter for bus 1.
    S2 : float
        Speedup parameter for bus 2.
    num_trips : int
        Total number of trips to simulate for each bus.

    Returns
    -------
    list of [float, int, int, bool]
        Each entry is [time, bus_id, trip_number, used_flag]:
          - time        : Arrival time at the origin.
          - bus_id      : Bus identifier (1 or 2).
          - trip_number : Sequence index of the trip for that bus (-1 for the bias entry).
          - used_flag   : True if this entry has already been used to compute a successor trip.
    """
    # arrival_time stores all processed trips as [time, bus_id, trip_number, used_flag]
    arrival_time = [
        [0.0, 2, -1, False],          # Bias/sentinel entry (always marked as used later)
        [T1_initial, 1, 0, False],    # First trip of bus 1
    ]

    # Pending (computed but not yet committed) next trip for each bus
    # Index 0 → bus 1,  index 1 → bus 2
    temp_time = [None, T2_initial]
    temp_trip = [None, [T2_initial, 2, 0, False]]

    # How many trips have been committed for each bus so far
    count_trip = [0, 0]

    i = 1  # Pointer into arrival_time; index 0 is the bias entry
    while count_trip[0] < num_trips + 1 or count_trip[1] < num_trips + 1:
        if i >= len(arrival_time):
            break

        current_row = arrival_time[i]

        # Skip the bias entry and already-processed entries
        if current_row[2] == -1 or current_row[3]:
            i += 1
            continue

        bus_id = current_row[1]

        # Find the most recent uncommitted trip (used_flag = False) before index i
        for j in range(i - 1, -1, -1):
            if not arrival_time[j][3]:
                pre_row = arrival_time[j]
                break

        # Time gap between current trip and its predecessor
        delta = current_row[0] - pre_row[0]
        if delta < 0:
            print(
                f"Warning: delta < 0 at gamma={gamma}, S1={S1}, S2={S2} — "
                "check initial conditions."
            )

        S = S1 if bus_id == 1 else S2

        # Recurrence: next arrival time at the origin
        next_time = current_row[0] + gamma * delta + 1.0 / (1.0 + S * delta)

        # Mark predecessor as used
        pre_row[3] = True

        # Stage the newly computed trip
        temp_time[bus_id - 1] = next_time
        temp_trip[bus_id - 1] = [next_time, bus_id, current_row[2] + 1, False]

        # Commit whichever pending trip arrives first
        if temp_time[0] is not None and temp_time[1] is not None:
            if temp_time[0] < temp_time[1]:
                arrival_time.append(temp_trip[0])
                count_trip[0] += 1
                temp_time[0], temp_trip[0] = None, None
            else:
                arrival_time.append(temp_trip[1])
                count_trip[1] += 1
                temp_time[1], temp_trip[1] = None, None

        i += 1

    return arrival_time
