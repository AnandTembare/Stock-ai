from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def price_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="OHLC",
        )
    )
    for column, color in [("SMA_50", "#2563eb"), ("SMA_200", "#dc2626")]:
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=dict(color=color, width=1.6)))
    fig.update_layout(
        title=title,
        height=480,
        margin=dict(l=10, r=10, t=46, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def probability_chart(probabilities: pd.Series, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=probabilities.index,
            y=probabilities,
            mode="lines",
            name="Buy probability",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626", annotation_text="Entry threshold")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=24, b=10), yaxis_tickformat=".0%")
    return fig


def equity_curve_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["Strategy"], mode="lines", name="Strategy", line=dict(color="#16a34a", width=2.4)))
    fig.add_trace(go.Scatter(x=equity.index, y=equity["Buy & Hold"], mode="lines", name="Buy & Hold", line=dict(color="#64748b", width=1.8)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=24, b=10), yaxis_title="Portfolio value")
    return fig


def drawdown_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["Drawdown"],
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="#dc2626", width=1.8),
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=24, b=10), yaxis_tickformat=".0%")
    return fig


def roc_chart(roc_frame: pd.DataFrame, auc: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_frame["False Positive Rate"],
            y=roc_frame["True Positive Rate"],
            mode="lines",
            name=f"ROC AUC {auc:.3f}" if auc is not None else "ROC",
            line=dict(color="#2563eb", width=2.4),
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94a3b8", dash="dash")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=24, b=10), xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def confusion_matrix_chart(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="Blues",
            text=matrix.values,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=24, b=10))
    return fig

