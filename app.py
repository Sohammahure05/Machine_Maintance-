import hashlib
import io
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analytics
import auth
import storage
from audio import read_audio, read_table
from config import CHANNELS, MODEL_PATH, SAVE_UPLOADS, SR, SUBF_SR
from inference import analyse
from report import (build_pdf, load_reference, plot_profile, plot_spectrogram, plot_timeline,
                    plot_trend, plot_waveform)

st.set_page_config(page_title="Machine Health Monitor", layout="wide")
storage.init()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def flash(message):
    st.session_state["flash"] = message


def login_screen():
    st.title("Machine Health Monitor")
    st.caption("Log in as a manufacturer owner to manage machines and keep a reading history.")
    left, right = st.columns(2)
    with left:
        st.subheader("Log in")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            user, message = auth.authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            st.error(message)
    with right:
        st.subheader("Create owner account")
        with st.form("register"):
            company = st.text_input("Company name")
            new_name = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm = st.text_input("Repeat the password", type="password")
            created = st.form_submit_button("Create account")
        if created:
            if new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = auth.register(new_name, new_pass, company)
                (st.success if ok else st.error)(message)


def machine_picker(user, bundle):
    machines = storage.list_machines(user["id"])
    with st.expander("Add a machine", expanded=machines.empty):
        with st.form("add_machine", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            code = c1.text_input("Machine number")
            name = c2.text_input("Machine name")
            mtype = c3.selectbox("Machine type", list(bundle["machines"]))
            location = st.text_input("Location")
            notes = st.text_area("Notes")
            added = st.form_submit_button("Add machine")
        if added:
            if not code.strip() or not name.strip():
                st.error("Machine number and name are required.")
            elif storage.add_machine(user["id"], code.strip(), name.strip(), mtype, location.strip(), notes.strip()):
                flash(f"Machine {code.strip()} added.")
                st.rerun()
            else:
                st.error("You already have a machine with that number.")
    if machines.empty:
        return None
    fleet = storage.fleet(user["id"])
    st.subheader("Your machines")
    st.dataframe(fleet, hide_index=True)
    labels = {int(m.id): f"{m.code}: {m.name} ({m.type})" for m in machines.itertuples()}
    chosen = st.selectbox("Select a machine to analyse", list(labels), format_func=labels.get)
    return machines[machines["id"] == chosen].iloc[0]


def remove_machine(user, machine):
    with st.expander("Remove this machine"):
        sure = st.checkbox("I understand this deletes the machine and all its saved readings",
                           key=f"sure_{machine['id']}")
        if st.button("Remove machine", disabled=not sure):
            storage.delete_machine(user["id"], int(machine["id"]))
            flash(f"Machine {machine['code']} removed.")
            st.rerun()


def read_upload(upload, rate):
    source = io.BytesIO(upload.getvalue())
    if upload.name.lower().endswith(".csv"):
        return [read_table(source, rate)]
    return read_audio(source, CHANNELS)


def analysis_section(user, machine, bundle):
    entry = bundle["machines"][machine["type"]]
    model_id = bundle["info"]["trained_at"]
    st.subheader(f"Analyse a recording for {machine['code']}")
    upload = st.file_uploader("Recording", type=["wav", "flac", "ogg", "mp3", "csv"],
                              key=f"upload_{machine['id']}")
    if upload is None:
        st.info("Upload a recording of this machine to get its condition report.")
        return
    is_table = upload.name.lower().endswith(".csv")
    rate = st.number_input("CSV sampling rate (Hz)", 1000, 200000, SUBF_SR) if is_table else SR
    digest = hashlib.sha256(upload.getvalue()).hexdigest()[:16]
    key = (int(machine["id"]), digest, int(rate))
    if st.session_state.get("analysis", {}).get("key") != key:
        try:
            signals = read_upload(upload, rate)
        except Exception as error:
            st.error(f"Could not read the file: {error}")
            return
        st.session_state["analysis"] = {"key": key, "signals": signals, "result": analyse(signals, entry)}
    current = st.session_state["analysis"]
    signals, result = current["signals"], current["result"]
    signal = signals[0]

    prev = storage.readings(user["id"], int(machine["id"]), model_id)
    trend_info = analytics.trend(list(prev["health"]) + [result["health"]])
    comparison = analytics.compare(result, prev)
    alert_list = analytics.alerts(prev, trend_info, comparison)

    banner = f"{result['status']}: {result['level']}"
    if result["level"] == "Healthy":
        st.success(banner)
    elif result["level"] == "Watch":
        st.warning(banner)
    else:
        st.error(banner)
    for alert in alert_list:
        st.warning(alert)

    cols = st.columns(5)
    cols[0].metric("Health index", f"{result['health']:.0f} / 100")
    cols[1].metric("File score", f"{result['score']:.3f}")
    cols[1].caption(f"Threshold {result['threshold']:.3f}, {result['pooling']} of segments")
    cols[2].metric("Segments flagged", f"{result['flagged']} / {result['segments']}")
    cols[3].metric("Novelty", "Unusual" if result["novel"] else "Familiar")
    cols[3].caption(f"{result['novelty']:.3f} (limit {result['novelty_limit']:.3f})")
    cols[4].metric("Input reliability", "Good" if result["reliable"] else "Check inputs")
    st.write(result["advice"])
    if not is_table:
        st.audio(upload.getvalue())

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6))
    plot_timeline(axes[0], result)
    plot_waveform(axes[1], signal)
    plot_spectrogram(axes[2], signal)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Comparison with this machine's history**")
    if comparison is None:
        st.info("No saved readings with the current model yet. Save this one to start a baseline.")
    else:
        left, right = st.columns(2)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        plot_trend(ax, prev, result)
        fig.tight_layout()
        left.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        plot_profile(ax, result, comparison)
        fig.tight_layout()
        right.pyplot(fig)
        plt.close(fig)
        st.write(
            f"Against {comparison['readings_used']} earlier reading(s): health index "
            f"{comparison['health_change']:+.0f} points from the baseline of {comparison['baseline_health']:.0f}. "
            f"Trend: {trend_info['direction'].lower()}. Largest spectral changes: "
            + "; ".join(analytics.describe_changes(comparison)) + "."
        )

    with st.expander("Segment scores and input checks"):
        st.dataframe(pd.DataFrame({
            "Segment start (s)": result["segment_times"],
            "Abnormal probability": result["segment_scores"].round(3),
            "Flagged": result["segment_flags"],
        }), hide_index=True)
        st.dataframe([{"Check": n, "Passed": ok, "Detail": d} for n, ok, d in result["checks"]], hide_index=True)

    saved = st.session_state.setdefault("saved", set())
    if key in saved:
        st.info("This recording is already saved to the machine's history.")
    elif st.button("Save reading to history"):
        path = storage.save_upload(user["id"], int(machine["id"]), upload.name, upload.getvalue()) if SAVE_UPLOADS else None
        storage.add_reading(user["id"], int(machine["id"]), upload.name, path, result, model_id)
        saved.add(key)
        flash("Reading saved to history.")
        st.rerun()

    meta = {"code": machine["code"], "name": machine["name"], "type": machine["type"],
            "location": machine["location"], "file": upload.name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    history = {"prev": prev, "comparison": comparison, "trend": trend_info, "alerts": alert_list}
    pdf = build_pdf(result, signal, meta, load_reference(machine["type"]), history)
    st.download_button("Download report (PDF)", pdf,
                       file_name=f"report_{machine['code']}_{upload.name.rsplit('.', 1)[0]}.pdf",
                       mime="application/pdf")


def history_section(user, machine, bundle):
    st.subheader(f"Reading history for {machine['code']}")
    history = storage.readings(user["id"], int(machine["id"]))
    if history.empty:
        st.info("No saved readings yet.")
        return
    current = bundle["info"]["trained_at"]
    table = pd.DataFrame({
        "ID": history["id"], "Date": history["recorded_at"].dt.strftime("%Y-%m-%d %H:%M"),
        "File": history["file_name"], "Status": history["status"], "Level": history["level"],
        "Health": history["health"].round(0), "Score": history["score"].round(3),
        "Flagged %": (history["flag_ratio"] * 100).round(0),
        "Reliable": history["reliable"].astype(bool),
        "Model": ["current" if m == current else "older" for m in history["model_id"]],
    })
    st.dataframe(table, hide_index=True)
    shown = history[history["model_id"] == current]
    if len(shown) >= 2:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(1, len(shown) + 1), shown["health"], marker="o", color="#1565c0")
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 105)
        ax.set_xticks(range(1, len(shown) + 1))
        ax.set_xlabel("Reading number")
        ax.set_ylabel("Health index")
        ax.set_title("Health index of saved readings (current model)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    if (history["model_id"] != current).any():
        st.caption("Readings marked older were made with a previous model and are left out of comparisons.")
    left, right = st.columns(2)
    left.download_button("Export history (CSV)", table.to_csv(index=False),
                         file_name=f"history_{machine['code']}.csv", mime="text/csv")
    with right.expander("Delete a saved reading"):
        target = st.selectbox("Reading ID", list(history["id"]), key=f"del_{machine['id']}")
        if st.button("Delete reading"):
            storage.delete_reading(user["id"], int(target))
            flash(f"Reading {target} deleted.")
            st.rerun()


def main():
    user = st.session_state.get("user")
    if user is None:
        login_screen()
        return
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    with st.sidebar:
        st.write(f"**{user['company']}**")
        st.caption(f"Signed in as {user['username']}")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
    st.title("Machine Health Monitor")
    if not MODEL_PATH.exists():
        st.error("Model not found. Run train.py first.")
        return
    bundle = load_bundle()
    info = bundle["info"]
    st.caption(f"Model trained {info['trained_at']} on {info['train_snr']} dB recordings. "
               + ("Trained on all machine IDs." if not info["holdout_ids"] else
                  f"Held-out machine IDs for testing: {', '.join(info['holdout_ids'])}."))
    machine = machine_picker(user, bundle)
    if machine is None:
        return
    remove_machine(user, machine)
    analysis_section(user, machine, bundle)
    history_section(user, machine, bundle)


main()
