
# This version plots all the QSOs for each band for each week.
# Gaps between the end of one year and the start of the next..

import re
from datetime import datetime
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


ADIF_RECORD_SPLIT = re.compile(r"<eor>", re.IGNORECASE)
ADIF_FIELD = re.compile(r"<([^:>]+):(\d+)>([^<]*)", re.IGNORECASE)


def parse_adif(filename):
    """
    Parses an ADIF file and returns a list of dicts, one per QSO.
    """
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Remove header
    if "<eoh>" in text.lower():
        text = text.lower().split("<eoh>", 1)[1]

    records = []

    for raw in ADIF_RECORD_SPLIT.split(text):
        fields = {}
        for name, length, value in ADIF_FIELD.findall(raw):
            fields[name.lower()] = value.strip()
        if fields:
            records.append(fields)

    return records


def effective_mode(qso):
    """
    Determine effective mode:
    - FT8 is its own mode
    - FT4 is submode of MFSK
    """
    mode = qso.get("mode", "").upper()
    submode = qso.get("submode", "").upper()

    if mode == "MFSK" and submode == "FT4":
        return "FT4"
    return mode


def qso_week(qso):
    """
    Convert qso_date + time_on to ISO year/week.
    """
    date = qso.get("qso_date")
    time = qso.get("time_on", "000000")

    if not date:
        return None

    dt = datetime.strptime(date + time[:6], "%Y%m%d%H%M%S")
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}", iso_year * 100 + iso_week


def build_dataframe(qsos):
    """
    Build a Pandas DataFrame with columns:
    week_label, week_index, type, count
    """
    counts = defaultdict(int)

    for qso in qsos:
        band = qso.get("band")
        if not band:
            continue

        mode = effective_mode(qso)
        week = qso_week(qso)
        if not week:
            continue

        week_label, week_index = week
        qso_type = f"{band} {mode}"

        counts[(week_label, week_index, qso_type)] += 1

    rows = [
        {
            "week_label": wl,
            "week_index": wi,
            "type": t,
            "count": c,
        }
        for (wl, wi, t), c in counts.items()
    ]

    return pd.DataFrame(rows)


def plot_contacts(df):
    """
    Plot contacts per week, grouped by band/mode.
    """
    plt.figure(figsize=(14, 8))

    for qso_type in sorted(df["type"].unique()):
        subset = df[df["type"] == qso_type].sort_values("week_index")
        plt.plot(
            subset["week_index"],
            subset["count"],
            marker="o",
            linewidth=1.8,
            label=qso_type,
        )

    # X-axis formatting
    week_ticks = df.sort_values("week_index").drop_duplicates("week_index")
    plt.xticks(
        week_ticks["week_index"],
        week_ticks["week_label"],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    plt.xlabel("ISO Week")
    plt.ylabel("Number of Contacts")
    plt.title("WSJT-X Contacts per Week by Band / Mode")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Band / Mode", fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    adif_file = "wsjtx_log.adi"   # ← change this to your filename

    qsos = parse_adif(adif_file)
    print(f"Loaded {len(qsos)} QSOs")

    df = build_dataframe(qsos)
    plot_contacts(df)


if __name__ == "__main__":
    main()
