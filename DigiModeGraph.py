
# This version plots all the QSOs for all bands cumulative for each week.
# No gaps between the end of one year and the start of the next.
# pip install tqdm

import re
from datetime import datetime
from collections import defaultdict, OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


ADIF_RECORD_SPLIT = re.compile(r"<eor>", re.IGNORECASE)
ADIF_FIELD = re.compile(r"<([^:>]+):(\d+)>([^<]*)", re.IGNORECASE)


def parse_adif(filename):
    """
    Parses an ADIF file and returns a list of dicts, one per QSO,
    with a progress bar.
    """
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Strip header
    if "<eoh>" in text.lower():
        text = text.lower().split("<eoh>", 1)[1]

    raw_records = ADIF_RECORD_SPLIT.split(text)
    records = []

    for raw in tqdm(raw_records, desc="Parsing QSOs"):
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


def qso_iso_week(qso):
    """
    Return ISO year/week tuple and label.
    """
    date = qso.get("qso_date")
    time = qso.get("time_on", "000000")

    if not date:
        return None

    dt = datetime.strptime(date + time[:6], "%Y%m%d%H%M%S")
    iso_year, iso_week, _ = dt.isocalendar()
    return iso_year, iso_week, f"{iso_year}-W{iso_week:02d}"


def build_dataframe(qsos):
    """
    Build a DataFrame with one row per (week, type) count.
    Weeks are mapped to a linear sequential index.
    """
    counts = defaultdict(int)
    week_set = set()

    for qso in qsos:
        band = qso.get("band")
        if not band:
            continue

        mode = effective_mode(qso)
        week_info = qso_iso_week(qso)
        if not week_info:
            continue

        year, week, label = week_info
        qso_type = f"{band} {mode}"

        counts[(year, week, label, qso_type)] += 1
        week_set.add((year, week, label))

    # Sort weeks chronologically
    sorted_weeks = sorted(week_set)
    week_index_map = {
        wk: idx for idx, wk in enumerate(sorted_weeks)
    }

    rows = []
    for (year, week, label, qso_type), count in counts.items():
        idx = week_index_map[(year, week, label)]
        rows.append({
            "week_index": idx,
            "week_label": label,
            "type": qso_type,
            "count": count,
        })

    return pd.DataFrame(rows), sorted_weeks


def plot_stacked_contacts(df, weeks):
    """
    Plot stacked area chart of QSOs per week by band/mode.
    """
    # Pivot into wide format: week_index x type
    pivot = df.pivot_table(
        index="week_index",
        columns="type",
        values="count",
        fill_value=0
    ).sort_index()

    x = pivot.index.values
    y = [pivot[col].values for col in pivot.columns]

    plt.figure(figsize=(15, 9))
    plt.stackplot(x, y, labels=pivot.columns, alpha=0.85)

    # X-axis labels (sparse for readability)
    week_labels = [label for (_, _, label) in weeks]
    step = max(1, len(week_labels) // 20)

    plt.xticks(
        ticks=x[::step],
        labels=week_labels[::step],
        rotation=45,
        ha="right",
        fontsize=9,
    )

    plt.xlabel("ISO Week")
    plt.ylabel("Total QSOs")
    plt.title("WSJT-X QSOs per Week (Stacked by Band / Mode)")
    plt.legend(
        title="Band / Mode",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    adif_file = "wsjtx_log.adi"   # ← change as needed

    qsos = parse_adif(adif_file)
    print(f"Loaded {len(qsos)} QSOs")

    df, weeks = build_dataframe(qsos)
    plot_stacked_contacts(df, weeks)


if __name__ == "__main__":
    main()
