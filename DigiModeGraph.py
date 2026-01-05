
# This version plots all the QSOs for all bands cumulative for all time week.
# Tyring to figure out a more workable colour mode.
# Haven't quite got the hang of the interaction thing yet.
# Now toggles teh graphs on and off by clicking teh key.
# Need to redraw the graph without the "off" data, though.

# pip install tqdm
# pip install mplcursors

import re
from datetime import datetime
from collections import defaultdict, OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mplcursors
import numpy as np


BAND_ORDER = [
    "2200m", "630m", "160m", "80m", "60m", "40m", "30m",
    "20m", "17m", "15m", "12m", "10m", "6m", "4m",
    "2m", "70cm"
]


def colour_for_type(qso_type):
    band, mode = qso_type.split()

    if band in BAND_ORDER:
        frac = BAND_ORDER.index(band) / (len(BAND_ORDER) - 1)
    else:
        frac = 0.5

    # Red → Blue gradient by band
    red = 0.9 * (1 - frac) + 0.2 * frac
    blue = 0.2 * (1 - frac) + 0.9 * frac

    # Strong green separation by mode
    green = 0.85 if mode == "FT8" else 0.30

    return (red, green, blue)


def plot_cumulative_stacked_interactive(df, weeks):
    """
    Interactive cumulative stacked area chart of QSOs per week by band/mode.
    """
    # Pivot weekly counts into wide format
    pivot = df.pivot_table(
        index="week_index",
        columns="type",
        values="count",
        fill_value=0
    ).sort_index()

    # Cumulative sum over time
    pivot = pivot.cumsum()

    # Sort bands logically
    def sort_key(t):
        band, mode = t.split()
        return (BAND_ORDER.index(band) if band in BAND_ORDER else 99, mode)

    types = sorted(pivot.columns, key=sort_key)
    pivot = pivot[types]

    x = pivot.index.values
    y = np.row_stack([pivot[t].values for t in types])
    colours = [colour_for_type(t) for t in types]

    fig, ax = plt.subplots(figsize=(15, 9))

    stacks = ax.stackplot(
        x,
        y,
        colors=colours,
        alpha=0.85
    )

    # X-axis labels
    week_labels = [label for (_, _, label) in weeks]
    step = max(1, len(week_labels) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        week_labels[::step],
        rotation=45,
        ha="right",
        fontsize=9
    )

    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Cumulative QSOs")
    ax.set_title("WSJT-X Cumulative QSOs (Stacked by Band / Mode)")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend (reverse order to match stack)
    handles = stacks[::-1]
    labels = types[::-1]

    legend = ax.legend(
        handles,
        labels,
        title="Band / Mode",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9
    )

    # Click-to-toggle visibility (robust)
    visibility = dict(zip(labels, [True] * len(labels)))


    # Map legend artists to stack artists
    legend_artist_to_stack = {
        leg: stack for leg, stack in zip(legend.legend_handles, handles)
    }

    visibility = {stack: True for stack in handles}

    def on_pick(event):
        leg_artist = event.artist
        if leg_artist not in legend_artist_to_stack:
            return

        stack = legend_artist_to_stack[leg_artist]
        visibility[stack] = not visibility[stack]
        stack.set_visible(visibility[stack])

        fig.canvas.draw_idle()

    # Enable picking on legend handles
    for leg in legend.legend_handles:
        leg.set_picker(True)

    fig.canvas.mpl_connect("pick_event", on_pick)


    # Hover tooltips (no PolyCollection warning)
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        xidx = int(round(sel.target[0]))
        if xidx < 0 or xidx >= len(x):
            return

        for t in types:
            val = pivot[t].iloc[xidx]
            if sel.target[1] <= val:
                sel.annotation.set_text(
                    f"{t}\n"
                    f"{week_labels[xidx]}\n"
                    f"QSOs: {int(val)}"
                )
                return

    plt.tight_layout()
    plt.show()


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
    # Pivot and cumulative sum
    pivot = df.pivot_table(
        index="week_index",
        columns="type",
        values="count",
        fill_value=0
    ).sort_index()

    pivot = pivot.cumsum()

    # Sort columns by band order, then mode
    def sort_key(t):
        band, mode = t.split()
        return (BAND_ORDER.get(band, 99), mode)

    types = sorted(pivot.columns, key=sort_key)
    pivot = pivot[types]

    x = pivot.index.values
    y = [pivot[t].values for t in types]
    colours = [colour_for_type(t) for t in types]

    fig, ax = plt.subplots(figsize=(15, 9))

    stacks = ax.stackplot(
        x,
        y,
        labels=types,
        colors=colours,
        alpha=0.85,
        picker=True
    )

    # X-axis labels
    week_labels = [label for (_, _, label) in weeks]
    step = max(1, len(week_labels) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        week_labels[::step],
        rotation=45,
        ha="right",
        fontsize=9
    )

    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Cumulative QSOs")
    ax.set_title("WSJT-X Cumulative QSOs (Stacked by Band / Mode)")
    ax.grid(True, axis="y", alpha=0.3)

    # Legend — reverse to match stack order
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]

    legend = ax.legend(
        handles,
        labels,
        title="Band / Mode",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9
    )

    # Enable legend click-to-toggle
    visibility = {label: True for label in labels}

    def on_pick(event):
        label = event.artist.get_label()
        visibility[label] = not visibility[label]
        idx = types.index(label)
        stacks[idx].set_visible(visibility[label])
        fig.canvas.draw_idle()

    for legpatch in legend.legendHandles:
        legpatch.set_picker(True)

    fig.canvas.mpl_connect("pick_event", on_pick)

    # Hover tooltips
    cursor = mplcursors.cursor(stacks, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        idx = stacks.index(sel.artist)
        week = int(sel.target[0])
        value = int(sel.target[1])
        sel.annotation.set_text(
            f"{types[idx]}\n"
            f"Week: {week_labels[week]}\n"
            f"QSOs: {value}"
        )

    plt.tight_layout()
    plt.show()

def main():
    adif_file = "wsjtx_log.adi"   # ← change as needed

    qsos = parse_adif(adif_file)
    print(f"Loaded {len(qsos)} QSOs")

    df, weeks = build_dataframe(qsos)
    plot_cumulative_stacked_interactive(df, weeks)


if __name__ == "__main__":
    main()
