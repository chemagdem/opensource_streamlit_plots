import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# CONFIG

st.set_page_config(
    page_title="Player Explorer",
    layout="wide"
)

# Base directory = folder where this app.py lives
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "raw_data.csv"
LOGO_PATH = BASE_DIR / "cch.png"          # o BASE_DIR / "assets" / "cch.png"
SIGNATURE_PATH = BASE_DIR / "cch.png"

DARK_BG = "#0E1117"


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df



df = load_data()


# FRIENDLY COLUMN NAMES

def friendly_name(col: str) -> str:
    """Convert raw column name into a more readable label."""
    name = col.replace("_", " ")
    name = name.replace(" pct", " %")
    return name.title()



# HEADER WITH LOGO

header_cols = st.columns([1, 6])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
with header_cols[1]:
    st.title("Player Data Explorer")
    st.caption("Free-to-use tool for newcomers to football analytics. Explore players using open-source metrics.")


# SIDEBAR

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    st.title("Filters")


# FILTERS

competitions = sorted(df["competition_name"].dropna().unique().tolist())
competition_select = st.sidebar.multiselect(
    "Competition",
    competitions,
    default=competitions
)

positions = sorted(df["position_group"].dropna().unique().tolist())
position_select = st.sidebar.multiselect(
    "Position Group",
    positions,
    default=positions
)

player_query = st.sidebar.text_input(
    "Search player name",
    placeholder="Type partial name..."
).strip().lower()


# APPLY FILTERS

filtered = df.copy()

if competition_select:
    filtered = filtered[filtered["competition_name"].isin(competition_select)]

if position_select:
    filtered = filtered[filtered["position_group"].isin(position_select)]

if player_query:
    filtered = filtered[filtered["player_name"].str.lower().str.contains(player_query)]


# METRIC SELECTION

numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["player_id", "team_id"]]

if not numeric_cols:
    st.error("No numeric metrics available after filtering.")
    st.stop()

metric_options = {friendly_name(c): c for c in numeric_cols}

x_label = st.sidebar.selectbox("X metric", list(metric_options.keys()))
y_label = st.sidebar.selectbox("Y metric", list(metric_options.keys()))

x_metric = metric_options[x_label]
y_metric = metric_options[y_label]


# SCATTER PLOT

st.subheader("Scatter Plot")

if filtered.empty:
    st.warning("No players match the filters.")
else:
    fig = px.scatter(
        filtered,
        x=x_metric,
        y=y_metric,
        hover_data=["player_name", "team_name", "position_group"],
        color="position_group",
        opacity=0.78,
        height=650
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)


# RADAR CHART HEADER

radar_header = st.columns([6, 1])
with radar_header[0]:
    st.subheader("Radar Comparison")
with radar_header[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)


# RADAR CHART

if filtered.empty:
    st.info("Radar chart is not available because there are no players after filtering.")
else:
    filtered = filtered.copy()
    filtered["player_label"] = filtered["player_name"] + " (" + filtered["team_name"] + ")"

    radar_metric_labels = st.multiselect(
        "Radar metrics (normalized per metric)",
        list(metric_options.keys()),
        default=list(metric_options.keys())[:5]
    )

    if len(radar_metric_labels) < 2:
        st.info("Select at least two metrics.")
    else:
        radar_metrics = [metric_options[l] for l in radar_metric_labels]
        players = filtered["player_label"].unique().tolist()

        selected_players = st.multiselect(
            "Players to compare (max 4)",
            players,
            default=players[:4]
        )

        if not selected_players:
            st.info("Select at least one player.")
        else:
            if len(selected_players) > 4:
                st.warning("More than 4 players selected. Showing first 4.")
                selected_players = selected_players[:4]

            radar_df = filtered[filtered["player_label"].isin(selected_players)]
            radar_df = radar_df.drop_duplicates(subset=["player_label"])

            # Normalize metrics (0–1) based on current filtered dataset
            norm_df = radar_df.copy()
            for col in radar_metrics:
                col_min = filtered[col].min()
                col_max = filtered[col].max()
                if col_max == col_min:
                    norm_df[col] = 0.5
                else:
                    norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)

            categories = radar_metric_labels
            fig_radar = go.Figure()

            for _, row in norm_df.iterrows():
                values = [row[metric_options[label]] for label in categories]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill="toself",
                    name=row["player_label"],
                    opacity=0.7
                ))

            fig_radar.update_layout(
                polar=dict(
                    bgcolor=DARK_BG,
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor="gray",
                        linecolor="white",
                        tickfont=dict(color="white")
                    ),
                    angularaxis=dict(
                        tickfont=dict(color="white")
                    )
                ),
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_BG,
                font_color="white",
                showlegend=True,
                height=600
            )

            st.plotly_chart(fig_radar, use_container_width=True)


# TOP PLAYERS BAR CHART

st.subheader(f"Top Players by {y_label}")

if not filtered.empty:
    df_top = filtered.sort_values(by=y_metric, ascending=False).copy()
    top_n = st.slider("Number of players to display", min_value=5, max_value=30, value=15)
    df_top = df_top.head(top_n)

    fig_bar = px.bar(
        df_top,
        x=y_metric,
        y="player_name",
        color="position_group",
        orientation="h",
        hover_data=["team_name", "competition_name"],
        height=600
    )
    fig_bar.update_layout(
        xaxis_title=y_label,
        yaxis_title="Player",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font_color="white",
        legend_title="Position Group"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# DISTRIBUTION BY POSITION (VIOLIN + BOX)

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("Metric Distribution by Position Group")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty:
    dist_metric_label = st.selectbox(
        "Metric for distribution plot",
        list(metric_options.keys()),
        index=list(metric_options.keys()).index(y_label) if y_label in metric_options else 0
    )
    dist_metric = metric_options[dist_metric_label]

    df_dist = filtered.dropna(subset=[dist_metric]).copy()

    if df_dist.empty:
        st.info("No valid values for this metric after filtering.")
    else:
        fig_violin = px.violin(
            df_dist,
            x="position_group",
            y=dist_metric,
            color="position_group",
            box=True,
            points="all",
            hover_data=["player_name", "team_name", "competition_name"],
            height=650
        )
        fig_violin.update_layout(
            xaxis_title="Position Group",
            yaxis_title=dist_metric_label,
            plot_bgcolor=DARK_BG,
            paper_bgcolor=DARK_BG,
            font_color="white",
            legend_title="Position Group"
        )
        st.plotly_chart(fig_violin, use_container_width=True)


# 3D METRIC SPACE (WOW PLOT)

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("3D Metric Space")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty:
    metric_labels_list = list(metric_options.keys())

    # Default: first three metrics (fallback-safe)
    default_x3 = metric_labels_list[0] if len(metric_labels_list) > 0 else x_label
    default_y3 = metric_labels_list[1] if len(metric_labels_list) > 1 else y_label
    default_z3 = metric_labels_list[2] if len(metric_labels_list) > 2 else metric_labels_list[0]

    col_3d_1, col_3d_2, col_3d_3 = st.columns(3)
    with col_3d_1:
        x3_label = st.selectbox("3D X metric", metric_labels_list, index=metric_labels_list.index(default_x3))
    with col_3d_2:
        y3_label = st.selectbox("3D Y metric", metric_labels_list, index=metric_labels_list.index(default_y3))
    with col_3d_3:
        z3_label = st.selectbox("3D Z metric", metric_labels_list, index=metric_labels_list.index(default_z3))

    x3_metric = metric_options[x3_label]
    y3_metric = metric_options[y3_label]
    z3_metric = metric_options[z3_label]

    df_3d = filtered.dropna(subset=[x3_metric, y3_metric, z3_metric]).copy()

    if df_3d.empty:
        st.info("No valid values for the selected 3D metrics after filtering.")
    else:
        fig_3d = px.scatter_3d(
            df_3d,
            x=x3_metric,
            y=y3_metric,
            z=z3_metric,
            color="position_group",
            hover_name="player_name",
            hover_data=["team_name", "competition_name"],
            opacity=0.8,
            height=750
        )

        fig_3d.update_layout(
            scene=dict(
                xaxis_title=x3_label,
                yaxis_title=y3_label,
                zaxis_title=z3_label,
                bgcolor=DARK_BG,
                xaxis=dict(
                    gridcolor="gray",
                    zerolinecolor="gray",
                    title_font=dict(color="white"),
                    tickfont=dict(color="white")
                ),
                yaxis=dict(
                    gridcolor="gray",
                    zerolinecolor="gray",
                    title_font=dict(color="white"),
                    tickfont=dict(color="white")
                ),
                zaxis=dict(
                    gridcolor="gray",
                    zerolinecolor="gray",
                    title_font=dict(color="white"),
                    tickfont=dict(color="white")
                ),
            ),
            paper_bgcolor=DARK_BG,
            font_color="white",
            legend_title="Position Group"
        )

        st.plotly_chart(fig_3d, use_container_width=True)


# PLAYER SIMILARITY MAP (PCA-BASED)

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("PCA Similarity Map")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty:
    sim_metric_labels = st.multiselect(
        "Metrics to define similarity",
        list(metric_options.keys()),
        default=list(metric_options.keys())[:8]
    )

    if len(sim_metric_labels) < 2:
        st.info("Select at least two metrics to build the similarity map.")
    else:
        sim_cols = [metric_options[l] for l in sim_metric_labels]
        df_sim = filtered.dropna(subset=sim_cols).copy()

        if df_sim.empty:
            st.info("No valid values for the selected similarity metrics after filtering.")
        else:
            # Standardize data
            X = df_sim[sim_cols].values.astype(float)
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0, ddof=0)
            X_std[X_std == 0] = 1.0
            X_norm = (X - X_mean) / X_std

            # PCA via eigen-decomposition
            cov = np.cov(X_norm, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            W = eigvecs[:, :2]  # first two PCs
            Z = X_norm @ W      # projections

            explained_var = eigvals / eigvals.sum()
            pc1_var = explained_var[0] * 100
            pc2_var = explained_var[1] * 100

            df_sim["PC1"] = Z[:, 0]
            df_sim["PC2"] = Z[:, 1]
            df_sim["player_label"] = df_sim["player_name"] + " (" + df_sim["team_name"] + ")"

            # Focus player + k-nearest neighbours
            focus_options = ["None"] + df_sim["player_label"].unique().tolist()
            focus_player = st.selectbox(
                "Focus player (highlight nearest neighbours)",
                focus_options
            )

            base_size = 8
            focus_size = 22
            neighbor_size = 14
            k_neighbors = 5

            sizes = np.full(len(df_sim), base_size, dtype=float)

            Z_points = df_sim[["PC1", "PC2"]].values

            if focus_player != "None":
                mask_focus = df_sim["player_label"] == focus_player
                if mask_focus.any():
                    sizes[mask_focus] = focus_size
                    focus_point = Z_points[mask_focus][0]

                    dists = np.linalg.norm(Z_points - focus_point, axis=1)
                    neighbor_idx = np.argsort(dists)
                    neighbor_idx = [i for i in neighbor_idx if not mask_focus.iloc[i]][:k_neighbors]
                    for i in neighbor_idx:
                        sizes[i] = neighbor_size

            df_sim["marker_size"] = sizes

            fig_pca = px.scatter(
                df_sim,
                x="PC1",
                y="PC2",
                color="position_group",
                size="marker_size",
                size_max=focus_size,
                hover_name="player_name",
                hover_data=["team_name", "competition_name"],
                height=750
            )

            fig_pca.update_layout(
                xaxis_title=f"PC1 ({pc1_var:.1f}% variance)",
                yaxis_title=f"PC2 ({pc2_var:.1f}% variance)",
                plot_bgcolor=DARK_BG,
                paper_bgcolor=DARK_BG,
                font_color="white",
                legend_title="Position Group"
            )

            st.plotly_chart(fig_pca, use_container_width=True)


# TABLE OF RESULTS

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("Filtered Results")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty:
    cols_to_show = [
        "player_name", "team_name", "competition_name",
        "position_group", x_metric, y_metric
    ]
    # Remove duplicates in case x_metric == y_metric
    cols_to_show = list(dict.fromkeys(cols_to_show))

    st.dataframe(filtered[cols_to_show], use_container_width=True)


# PLAYER PERCENTILE TORNADO (PROFILE WOW PLOT)

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("Tornado View")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty and numeric_cols:
    profile_df = filtered.copy()
    profile_df["player_label"] = profile_df["player_name"] + " (" + profile_df["team_name"] + ")"
    player_labels_profile = sorted(profile_df["player_label"].unique().tolist())

    col_prof_1, col_prof_2 = st.columns(2)
    with col_prof_1:
        ref_player_label = st.selectbox(
            "Reference player",
            player_labels_profile
        )
    with col_prof_2:
        comp_options = ["None"] + player_labels_profile
        comp_player_label = st.selectbox(
            "Comparison player (optional)",
            comp_options,
            index=0
        )

    max_metrics = min(40, len(numeric_cols))
    min_metrics = 5 if max_metrics >= 5 else max_metrics
    default_metrics = min(20, max_metrics)

    top_k = st.slider(
        "Number of metrics to display in profile",
        min_value=min_metrics,
        max_value=max_metrics,
        value=default_metrics
    )

    # Take the first occurrence for each player label
    ref_row = profile_df[profile_df["player_label"] == ref_player_label].iloc[0]
    comp_row = None
    if comp_player_label != "None":
        comp_row = profile_df[profile_df["player_label"] == comp_player_label].iloc[0]

    records = []
    for col in numeric_cols:
        series = filtered[col].dropna()
        # Skip metrics with very low variation or very few values
        if series.size < 5 or series.nunique() < 3:
            continue

        ref_val = ref_row[col]
        if pd.isna(ref_val):
            continue

        # Percentile of the reference player in this metric
        ref_pct = float(np.round((series <= ref_val).mean() * 100, 1))

        comp_pct = None
        comp_val = None
        if comp_row is not None:
            comp_val = comp_row[col]
            if not pd.isna(comp_val):
                comp_pct = float(np.round((series <= comp_val).mean() * 100, 1))

        records.append({
            "metric_raw": col,
            "metric": friendly_name(col),
            "ref_pct": ref_pct,
            "ref_val": ref_val,
            "comp_pct": comp_pct,
            "comp_val": comp_val
        })

    if not records:
        st.info("Not enough valid metrics to build percentile profile with current filters.")
    else:
        prof_df = pd.DataFrame(records)
        prof_df = prof_df.sort_values("ref_pct", ascending=False).head(top_k)

        # Build long-form data for tornado plot
        plot_rows = []
        for _, r in prof_df.iterrows():
            # Reference player -> positive side
            plot_rows.append({
                "metric": r["metric"],
                "player": ref_player_label,
                "percentile": r["ref_pct"],
                "x": r["ref_pct"],
                "value": r["ref_val"]
            })
            # Comparison player -> negative side
            if r["comp_pct"] is not None and comp_player_label != "None":
                plot_rows.append({
                    "metric": r["metric"],
                    "player": comp_player_label,
                    "percentile": r["comp_pct"],
                    "x": -r["comp_pct"],
                    "value": r["comp_val"]
                })

        plot_df = pd.DataFrame(plot_rows)

        if plot_df.empty:
            st.info("Could not compute percentile profile for selected players.")
        else:
            fig_profile = go.Figure()

            # Add one bar trace per player
            for player_label in plot_df["player"].unique():
                df_p = plot_df[plot_df["player"] == player_label]
                # customdata will carry [percentile, value] for hover
                customdata = np.stack(
                    [df_p["percentile"].values, df_p["value"].values],
                    axis=-1
                )

                fig_profile.add_trace(
                    go.Bar(
                        x=df_p["x"],
                        y=df_p["metric"],
                        name=player_label,
                        orientation="h",
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Player: %{text}<br>"
                            "Percentile: %{customdata[0]:.1f}<br>"
                            "Value: %{customdata[1]:.3f}<extra></extra>"
                        ),
                        text=[player_label] * len(df_p),
                        customdata=customdata
                    )
                )

            # Symmetric x-axis: negative side = comparison, positive = reference
            fig_profile.update_layout(
                barmode="relative",
                xaxis=dict(
                    title="Percentile vs current selection",
                    tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
                    ticktext=["100", "75", "50", "25", "0", "25", "50", "75", "100"],
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    title="Metric",
                    automargin=True
                ),
                plot_bgcolor=DARK_BG,
                paper_bgcolor=DARK_BG,
                font_color="white",
                legend_title="Player",
                height=max(500, 22 * len(prof_df))
            )

            # Vertical line at 0 (center)
            fig_profile.add_vline(
                x=0,
                line_width=1,
                line_dash="dash",
                line_color="white"
            )

            st.plotly_chart(fig_profile, use_container_width=True)

            # Optional detail table
            with st.expander("Show raw values and percentiles"):
                table = prof_df.copy()
                rename_map = {
                    "metric": "Metric",
                    "ref_val": f"{ref_player_label} value",
                    "ref_pct": f"{ref_player_label} percentile"
                }
                if comp_player_label != "None":
                    rename_map["comp_val"] = f"{comp_player_label} value"
                    rename_map["comp_pct"] = f"{comp_player_label} percentile"
                else:
                    rename_map["comp_val"] = "Comparison value"
                    rename_map["comp_pct"] = "Comparison percentile"

                table = table[["metric", "ref_val", "ref_pct", "comp_val", "comp_pct"]].rename(columns=rename_map)
                st.dataframe(table, use_container_width=True)


# PLAYER ARCHETYPES CLUSTER MAP (UNSUPERVISED WOW PLOT)

sec_cols = st.columns([6, 1])
with sec_cols[0]:
    st.subheader("Player Archetypes Cluster Map")
with sec_cols[1]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)

if not filtered.empty and numeric_cols:
    # Allow the user to choose which metrics define archetypes
    archetype_metric_labels = st.multiselect(
        "Metrics to define archetypes (clustering space)",
        list(metric_options.keys()),
        default=list(metric_options.keys())[:10]
    )

    if len(archetype_metric_labels) < 2:
        st.info("Select at least two metrics to build archetypes.")
    else:
        archetype_cols = [metric_options[l] for l in archetype_metric_labels]
        df_cluster = filtered.dropna(subset=archetype_cols).copy()

        if df_cluster.empty or len(df_cluster) < 3:
            st.info("Not enough players with valid values to build archetypes.")
        else:
            # Small helper for simple K-Means (no external dependencies)
            def simple_kmeans(X, k, n_init=5, max_iter=100):
                """Very small K-Means implementation using NumPy only."""
                n_samples, n_features = X.shape
                best_inertia = None
                best_labels = None
                best_centers = None

                rng = np.random.RandomState(42)

                for _ in range(n_init):
                    # Random unique initial centers
                    if k > n_samples:
                        k_eff = n_samples
                    else:
                        k_eff = k
                    init_idx = rng.choice(n_samples, size=k_eff, replace=False)
                    centers = X[init_idx].copy()

                    # If k_eff < k (rare), we will reuse some centers later
                    if k_eff < k:
                        pad_idx = rng.choice(n_samples, size=k - k_eff, replace=False)
                        pad_centers = X[pad_idx].copy()
                        centers = np.vstack([centers, pad_centers])

                    for _ in range(max_iter):
                        # Compute squared distances to each center
                        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                        labels = np.argmin(distances, axis=1)

                        new_centers = centers.copy()
                        for j in range(k):
                            mask = labels == j
                            if np.any(mask):
                                new_centers[j] = X[mask].mean(axis=0)
                            else:
                                # Handle empty cluster: reinitialize with random point
                                rand_idx = rng.randint(0, n_samples)
                                new_centers[j] = X[rand_idx]

                        if np.allclose(new_centers, centers):
                            centers = new_centers
                            break
                        centers = new_centers

                    # Compute inertia (sum of squared distances to assigned centers)
                    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                    min_dist = distances[np.arange(n_samples), labels]
                    inertia = float((min_dist ** 2).sum())

                    if best_inertia is None or inertia < best_inertia:
                        best_inertia = inertia
                        best_labels = labels.copy()
                        best_centers = centers.copy()

                return best_labels, best_centers

            # Standardize metrics
            X_raw = df_cluster[archetype_cols].values.astype(float)
            X_mean = X_raw.mean(axis=0)
            X_std = X_raw.std(axis=0, ddof=0)
            X_std[X_std == 0] = 1.0
            X_norm = (X_raw - X_mean) / X_std

            # Choose number of clusters
            max_clusters = min(8, len(df_cluster))
            if max_clusters < 2:
                st.info("Not enough players to form multiple archetypes.")
            else:
                k_clusters = st.slider(
                    "Number of archetypes (clusters)",
                    min_value=2,
                    max_value=max_clusters,
                    value=min(4, max_clusters)
                )

                # Run K-Means on normalized space
                labels, centers = simple_kmeans(X_norm, k=k_clusters)

                # Project to 2D via PCA (again, only on the archetype space)
                cov_arch = np.cov(X_norm, rowvar=False)
                eigvals_arch, eigvecs_arch = np.linalg.eigh(cov_arch)
                idx_arch = np.argsort(eigvals_arch)[::-1]
                eigvals_arch = eigvals_arch[idx_arch]
                eigvecs_arch = eigvecs_arch[:, idx_arch]

                W_arch = eigvecs_arch[:, :2]
                Z_arch = X_norm @ W_arch

                explained_var_arch = eigvals_arch / eigvals_arch.sum()
                pc1_var_arch = explained_var_arch[0] * 100
                pc2_var_arch = explained_var_arch[1] * 100

                # Build DataFrame for plotting
                df_cluster["PC1"] = Z_arch[:, 0]
                df_cluster["PC2"] = Z_arch[:, 1]
                df_cluster["cluster_id"] = labels
                df_cluster["cluster_label"] = "Archetype " + (df_cluster["cluster_id"] + 1).astype(str)
                df_cluster["player_label"] = df_cluster["player_name"] + " (" + df_cluster["team_name"] + ")"

                # Compute 2D cluster centers in PCA space
                centers_2d = centers @ W_arch
                centers_df = pd.DataFrame({
                    "PC1": centers_2d[:, 0],
                    "PC2": centers_2d[:, 1],
                    "cluster_id": np.arange(k_clusters),
                    "cluster_label": ["Archetype " + str(i + 1) for i in range(k_clusters)]
                })

                # Optional focus player to highlight inside the cluster map
                focus_labels = ["None"] + df_cluster["player_label"].unique().tolist()
                focus_player_cluster = st.selectbox(
                    "Focus player inside archetype map (optional)",
                    focus_labels
                )

                # Base sizes for points
                base_size = 9
                focus_size = 24

                size_array = np.full(len(df_cluster), base_size, dtype=float)
                if focus_player_cluster != "None":
                    mask_focus_cluster = df_cluster["player_label"] == focus_player_cluster
                    size_array[mask_focus_cluster] = focus_size

                df_cluster["marker_size"] = size_array

                # Main scatter: players colored by archetype cluster
                fig_cluster = px.scatter(
                    df_cluster,
                    x="PC1",
                    y="PC2",
                    color="cluster_label",
                    size="marker_size",
                    size_max=focus_size,
                    hover_name="player_name",
                    hover_data=["team_name", "competition_name", "position_group"],
                    height=800
                )

                # Add cluster centers as separate scatter trace
                fig_cluster.add_trace(
                    go.Scatter(
                        x=centers_df["PC1"],
                        y=centers_df["PC2"],
                        mode="markers+text",
                        name="Archetype centers",
                        marker=dict(
                            size=26,
                            symbol="x",
                            line=dict(width=2, color="white")
                        ),
                        text=centers_df["cluster_label"],
                        textposition="top center",
                        hoverinfo="text"
                    )
                )

                fig_cluster.update_layout(
                    xaxis_title=f"Archetype PC1 ({pc1_var_arch:.1f}% variance)",
                    yaxis_title=f"Archetype PC2 ({pc2_var_arch:.1f}% variance)",
                    plot_bgcolor=DARK_BG,
                    paper_bgcolor=DARK_BG,
                    font_color="white",
                    legend_title="Cluster Archetype",
                )

                # Slightly stronger grid lines for reading the space
                fig_cluster.update_xaxes(
                    showgrid=True,
                    gridcolor="gray",
                    zeroline=True,
                    zerolinecolor="white"
                )
                fig_cluster.update_yaxes(
                    showgrid=True,
                    gridcolor="gray",
                    zeroline=True,
                    zerolinecolor="white"
                )

                st.plotly_chart(fig_cluster, use_container_width=True)

                # NOTE UNDER THE PLOT EXPLAINING IT
                st.markdown(
                    """
                    **How to read this map**  
                    - Each dot is a player, positioned so that players with similar metric profiles appear close to each other.  
                    - Colors represent automatically discovered *archetypes* (playing style profiles) built from the selected metrics.  
                    - The `×` markers are the centers of each archetype: the most typical profile inside that group.  
                    - Use the metric selector and the number of archetypes slider to redefine which aspects of the game drive these profiles.
                    """
                )

                # Optional: cluster composition table
                with st.expander("Show archetype composition table"):
                    comp_table = df_cluster[[
                        "player_name",
                        "team_name",
                        "competition_name",
                        "position_group",
                        "cluster_label"
                    ]].sort_values(["cluster_label", "position_group", "team_name", "player_name"])
                    st.dataframe(comp_table, use_container_width=True)


# FINAL SIGNATURE / FOOTER IMAGE


if SIGNATURE_PATH.exists():
    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])
    with center:
        # Cast a str por si acaso
        st.image(str(SIGNATURE_PATH), width=280)
