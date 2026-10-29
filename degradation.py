import numpy as np
from sklearn.metrics import r2_score

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

def r2_value(base_profile, profile):
    base = np.asarray(base_profile, float)
    now = np.asarray(profile, float)
    if base.shape != now.shape:
        return None
    return float(r2_score(base, now))


def r2_degradation_pct(base_profile, profile):
    value = r2_value(base_profile, profile)
    if value is None:
        return None
    return float(np.clip((1 - value) * 100, 0, 100))



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

R2_HEALTHY = 0.92
R2_WATCH = 0.88


def r2_condition(value):
    if value is None:
        return "No baseline"
    if value >= R2_HEALTHY:
        return "Healthy"
    if value >= R2_WATCH:
        return "Watch"
    return "Degraded"

def pct_condition(pct):
    return r2_condition(1 - pct / 100)


def r2_series(base_profile, profiles):
    values = []
    for profile in profiles:
        value = r2_degradation_pct(base_profile, profile)
        values.append(0.0 if value is None else value)
    return values


def plot_r2_trend(ax, prev_pct, new_pct):
    past = list(prev_pct)
    x_new = len(past) + 1
    if past:
        ax.plot(range(1, len(past) + 1), past, marker="o", color="#1565c0", label="saved readings")
        ax.plot([len(past), x_new], [past[-1], new_pct], color="#1565c0", linestyle=":")
    ax.scatter([x_new], [new_pct], marker="*", s=180, color="#c62828", zorder=5, label="this reading")
    ax.axhline(100 * (1 - R2_HEALTHY), color="orange", linestyle="--", label="watch limit")
    ax.axhline(100 * (1 - R2_WATCH), color="red", linestyle="--", label="degraded limit")
    ax.set_title(f"Degradation vs ideal (R2): {new_pct:.0f}% ({pct_condition(new_pct)})")
    ax.set_xlabel("Reading number since the ideal was set")
    ax.set_ylabel("Degradation %")
    ax.set_xlim(0.5, x_new + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, x_new + 1))
    ax.legend(fontsize=8, loc="upper left")