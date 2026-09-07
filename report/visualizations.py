"""
Plotly Visualizations for Data Quality Analyzer

Generates interactive charts for the Streamlit dashboard and reports.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_readiness_gauge(score: int, status: str) -> go.Figure:
    """
    Animated gauge chart showing readiness score 0-100.
    Clean modern color palette with distinct threshold zones.
    """
    if score >= 80:
        bar_color = "#16a34a"  # Green
    elif score >= 60:
        bar_color = "#d97706"  # Amber
    else:
        bar_color = "#dc2626"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "/100", 'font': {'size': 44, 'color': '#1f2937'}},
        title={'text': f"Readiness: {status}", 'font': {'size': 20, 'color': '#374151'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9ca3af"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [60, 80], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.15)"},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=25, r=25, t=50, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_missing_values_chart(missing_report: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing missing % per column with gradient color."""
    if missing_report.empty:
        return go.Figure()

    df_sorted = missing_report.sort_values(by="missing_pct", ascending=True)
    fig = px.bar(
        df_sorted,
        x="missing_pct",
        y=df_sorted.index,
        orientation='h',
        color="missing_pct",
        color_continuous_scale="Reds",
        labels={'missing_pct': 'Missing %', 'index': 'Column'},
        title="Missing Values by Column (%)",
    )
    fig.update_layout(
        height=max(280, len(df_sorted) * 30),
        margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Interactive heatmap of feature correlation matrix."""
    fig = px.imshow(
        corr_matrix.round(2),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix"
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_class_distribution_chart(distribution: pd.DataFrame) -> go.Figure:
    """Bar chart of target class counts with percentage callouts."""
    dist = distribution.copy()
    labels = [str(idx) for idx in dist.index]
    counts = dist["count"].tolist()
    pcts = dist["percentage"].tolist()

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        text=[f"{p:.1f}% ({c:,})" for p, c in zip(pcts, counts)],
        textposition='outside',
        marker_color="#6366f1"
    ))
    fig.update_layout(
        title="Target Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=350,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_leakage_chart(all_scores: pd.DataFrame, threshold: float) -> go.Figure:
    """Horizontal bar chart of feature solo accuracy, highlighting suspicious leakage."""
    if all_scores.empty:
        return go.Figure()

    df_sorted = all_scores.sort_values(by="solo_accuracy", ascending=True)
    features = df_sorted["feature"].tolist()
    scores = (df_sorted["solo_accuracy"] * 100).tolist()
    thresh_pct = threshold * 100

    colors = ['#ef4444' if s >= thresh_pct else '#3b82f6' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
    ))
    fig.add_vline(
        x=thresh_pct,
        line_width=2,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Threshold ({thresh_pct:.0f}%)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Feature Solo Predictive Power (Data Leakage Check)",
        xaxis_title="Solo Accuracy (%)",
        yaxis_title="Feature",
        xaxis=dict(range=[0, 105]),
        height=max(320, len(features) * 32),
        margin=dict(l=20, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_outlier_box_plots(df: pd.DataFrame, outlier_columns: list) -> go.Figure:
    """Box plots for columns containing outliers."""
    import plotly.subplots as sp

    cols = outlier_columns[:8]
    n_cols = len(cols)
    if n_cols == 0:
        return go.Figure()

    rows = int(np.ceil(n_cols / 2))
    fig = sp.make_subplots(rows=rows, cols=min(2, n_cols), subplot_titles=cols)

    for i, col in enumerate(cols):
        r = (i // 2) + 1
        c = (i % 2) + 1
        fig.add_trace(
            go.Box(y=df[col].dropna(), name=col, boxpoints='outliers', marker_color='#8b5cf6'),
            row=r, col=c
        )

    fig.update_layout(
        height=max(350, 180 * rows),
        showlegend=False,
        title="Outlier Distributions",
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_quality_radar_chart(category_scores: dict) -> go.Figure:
    """
    Radar chart across 5 core quality dimensions:
    Completeness, Consistency, Accuracy, Uniqueness, Validity.
    """
    categories = list(category_scores.keys())
    values = [float(v) for v in category_scores.values()]

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=6, color='#4338ca'),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            )
        ),
        showlegend=False,
        height=380,
        title="Quality Dimensions Breakdown",
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_data_types_chart(dtypes_input) -> go.Figure:
    """Donut chart showing distribution of data types."""
    if isinstance(dtypes_input, pd.Series):
        if pd.api.types.is_numeric_dtype(dtypes_input):
            # Already counts
            labels = dtypes_input.index.astype(str).tolist()
            counts = dtypes_input.values.tolist()
        else:
            vc = dtypes_input.astype(str).value_counts()
            labels = vc.index.tolist()
            counts = vc.values.tolist()
    elif isinstance(dtypes_input, dict):
        labels = list(dtypes_input.keys())
        counts = list(dtypes_input.values())
    else:
        return go.Figure()

    fig = px.pie(
        values=counts,
        names=labels,
        hole=0.55,
        title="Column Data Types",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
