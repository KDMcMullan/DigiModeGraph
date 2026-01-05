
# This version plots all the QSOs for all bands cumulative for all time week.
# Happier with the coours.
# Toggling data now works.
# Added checkboxes for showing / hiding FT8 and FT4.
# Some duplicate code remoevd.

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

from matplotlib.widgets import CheckButtons


BAND_ORDER = [
    "2200m", "630m", "160m", "80m", "60m", "40m", "30m",
    "20m", "17m", "15m", "12m", "10m", "6m", "4m",
    "2m", "70cm"
]

import warnings
warnings.filterwarnings("ignore", message="Pick support for PolyCollection")


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
    import numpy as np
    import matplotlib.pyplot as plt
    import mplcursors

    # Pivot + cumulative sum
    pivot = df.pivot_table(
        index="week_index",
        columns="type",
        values="count",
        fill_value=0
    ).sort_index().cumsum()

    # Sort types by band order then mode
    def sort_key(t):
        band, mode = t.split()
        return (BAND_ORDER.index(band) if band in BAND_ORDER else 99, mode)

    all_types = sorted(pivot.columns, key=sort_key)
    enabled_types = set(all_types)

    x = pivot.index.values
    week_labels = [label for (_, _, label) in weeks]

    fig, ax = plt.subplots(figsize=(15, 9))

    # --- Draw initial stacks ---
    types_to_draw = [t for t in all_types if t in enabled_types]
    y = np.row_stack([pivot[t].values for t in types_to_draw])
    colours = [colour_for_type(t) for t in types_to_draw]
    stack_artists = ax.stackplot(x, y, colors=colours, alpha=0.85)

    # Axes formatting
    step = max(1, len(week_labels) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(week_labels[::step], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Cumulative QSOs")
    ax.set_title("WSJT-X Cumulative QSOs (Stacked by Band / Mode)")
    ax.grid(True, axis="y", alpha=0.3)

    # --- Persistent legend ---
    handles = [plt.Line2D([0], [0], color=colour_for_type(t), lw=6) for t in all_types[::-1]]
    labels = all_types[::-1]
    legend = ax.legend(handles, labels, title="Band / Mode",
                       loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    legend_map = dict(zip(legend.legend_handles, all_types[::-1]))

    for leg in legend.legend_handles:
        t = legend_map[leg]
        leg.set_alpha(1.0 if t in enabled_types else 0.3)
        leg.set_picker(True)

    # --- Draw/redraw function (only updates stacks) ---
    def draw_stack():
        # Remove old stack artists
        for s in stack_artists:
            s.remove()
        stack_artists.clear()

        # Draw only enabled types
        types_to_draw = [t for t in all_types if t in enabled_types]
        if types_to_draw:
            y = np.row_stack([pivot[t].values for t in types_to_draw])
            colours = [colour_for_type(t) for t in types_to_draw]
            new_stacks = ax.stackplot(x, y, colors=colours, alpha=0.85)
            stack_artists.extend(new_stacks)

        # Rescale Y-axis to max of visible stacks
        if types_to_draw:
            visible_sum = pivot[types_to_draw].sum(axis=1)
            max_y = visible_sum.max()
            ax.set_ylim(0, max_y * 1.1)
        else:
            ax.set_ylim(0, 1)

        fig.canvas.draw_idle()


    # --- Legend toggle handler ---
    def on_pick(event):
        leg = event.artist
        if leg not in legend_map:
            return

        t = legend_map[leg]
        if t in enabled_types:
            enabled_types.remove(t)
            leg.set_alpha(0.3)
        else:
            enabled_types.add(t)
            leg.set_alpha(1.0)

        draw_stack()  # redraw only visible stacks

    fig.canvas.mpl_connect("pick_event", on_pick)


    # --- Create checkbox axes ---
    rax = fig.add_axes([0.5, 0.7, 0.1, 0.15])  # x, y, width, height
    modes = ["FT4", "FT8"]
    visibility = [True, True]  # initially visible
    check = CheckButtons(rax, modes, visibility)



    # --- Checkbox callback ---
    def checkbox_func(label):
        mode = label
        # Toggle all types of this mode
        for t in all_types:
            if t.endswith(mode):
                if mode == "FT4":
                    if visibility[0]:
                        enabled_types.discard(t)
                    else:
                        enabled_types.add(t)
                else:
                    if visibility[1]:
                        enabled_types.discard(t)
                    else:
                        enabled_types.add(t)
        # Update the visibility list
        if mode == "FT4":
            visibility[0] = not visibility[0]
        else:
            visibility[1] = not visibility[1]

        draw_stack()

    check.on_clicked(checkbox_func)



    # --- Hover cursor ---
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        idx = int(round(sel.target[0]))
        if idx < 0 or idx >= len(x):
            return

        cumulative = 0
        # Only enabled types
        for t in all_types:
            if t not in enabled_types:
                continue
            val = pivot[t].iloc[idx]
            if sel.target[1] <= val:
                sel.annotation.set_text(
                    f"{t}\n{week_labels[idx]}\nQSOs: {int(val)}"
                )
                return
            cumulative = val

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


def main():
    adif_file = "wsjtx_log.adi"   # ← change as needed

    qsos = parse_adif(adif_file)
    print(f"Loaded {len(qsos)} QSOs")

    df, weeks = build_dataframe(qsos)
    plot_cumulative_stacked_interactive(df, weeks)


if __name__ == "__main__":
    main()
