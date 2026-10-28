import numpy as np

BOUNDARY = 50.0
MIN_SPAN = 10.0
GRADES = [(20, "Good"), (50, "Mild wear"), (80, "Significant wear"), (101, "Severe")]

def degradation_pct(base_health, health):
    span = max(base_health - BOUNDARY, MIN_SPAN)
    return float(np.clip((base_health - health) / span * 100, 0, 100))


def grade(pct):
    return next(name for limit, name in GRADES if pct < limit)


def spectral_shift_pct(base_profile, profile):
    base = np.asarray(base_profile, float)
    now = np.asarray(profile, float)
    if base.shape != now.shape:
        return None
    return float(np.linalg.norm(now - base) / (np.linalg.norm(base) + 1e-9) * 100)


def days_to_boundary(times, health, min_points=5, min_days=2.0):
    times = np.asarray(times, float)
    if len(health) < min_points or times[-1] - times[0] < min_days:
        return None
    slope, intercept = np.polyfit(times, health, 1)
    if slope >= 0:
        return None
    return max(float((BOUNDARY - intercept) / slope - times[-1]), 0.0)


def plot_degradation(ax, prev_health, new_health, base_health):
    past = list(prev_health)
    x_past = list(range(1, len(past) + 1))
    x_new = len(past) + 1
    if past:
        ax.plot(x_past, past, marker="o", color="#1565c0", label="saved readings")
        ax.plot([x_past[-1], x_new], [past[-1], new_health], color="#1565c0", linestyle=":")
    ax.scatter([x_new], [new_health], marker="*", s=180, color="#c62828", zorder=5, label="this reading")
    ax.axhline(base_health, color="green", linestyle="--", label=f"machine's ideal ({base_health:.0f})")
    ax.axhline(BOUNDARY, color="red", linestyle="--", label="abnormal boundary (50)")
    ax.fill_between([0.5, x_new + 0.5], BOUNDARY, base_health, color="orange", alpha=0.12)
    pct = degradation_pct(base_health, new_health)
    ax.set_title(f"Degradation: {pct:.0f}% ({grade(pct)})")
    ax.set_xlabel("Reading number")
    ax.set_ylabel("Health index")
    ax.set_xlim(0.5, x_new + 0.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(1, x_new + 1))
    ax.legend(fontsize=8, loc="lower left")
    return pct