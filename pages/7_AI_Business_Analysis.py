import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Business Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --analysis-navy: #14213d;
        --analysis-blue: #1f5a94;
        --analysis-teal: #168a8a;
        --analysis-muted: #526173;
        --analysis-border: #dce3ed;
        --analysis-surface: #f7f9fc;
    }
    [data-testid="stAppViewContainer"] h1 {
        color: var(--analysis-navy);
        font-size: clamp(1.7rem, 2.4vw, 2.45rem);
    }
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {
        color: var(--analysis-navy);
    }
    [data-testid="stColumn"] {
        min-width: 0;
    }
    .analysis-question,
    .analysis-section,
    .chart-placeholder {
        border: 1px solid var(--analysis-border);
        border-radius: 8px;
        background: #ffffff;
        padding: 1.1rem 1.25rem;
    }
    .analysis-question {
        border-left: 4px solid var(--analysis-blue);
        color: var(--analysis-navy);
        font-size: 1.05rem;
    }
    .analysis-label {
        color: var(--analysis-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .answer-card {
        background: var(--analysis-navy);
        border-radius: 8px;
        color: #ffffff;
        padding: 1.35rem 1.5rem;
    }
    .answer-placeholder {
        color: #ffffff;
        font-size: clamp(1.45rem, 2.6vw, 2.25rem);
        font-weight: 700;
        margin-top: 0.45rem;
    }
    .kpi-card {
        background: var(--analysis-surface);
        border: 1px solid var(--analysis-border);
        border-radius: 8px;
        min-height: 108px;
        padding: 0.9rem 1rem;
    }
    .kpi-label {
        color: var(--analysis-muted);
        font-size: 0.8rem;
        margin-bottom: 0.55rem;
    }
    .kpi-placeholder {
        color: var(--analysis-navy);
        font-size: 1.35rem;
        font-weight: 700;
    }
    .chart-placeholder {
        align-items: center;
        background: repeating-linear-gradient(
            -45deg,
            #ffffff,
            #ffffff 12px,
            #f7f9fc 12px,
            #f7f9fc 24px
        );
        color: var(--analysis-muted);
        display: flex;
        justify-content: center;
        min-height: 280px;
        text-align: center;
    }
    .chart-placeholder strong {
        color: var(--analysis-navy);
        display: block;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .placeholder-note {
        color: var(--analysis-muted);
        font-size: 0.85rem;
    }
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: min(100%, 18rem);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


analysis_result = st.session_state.get("genai_analysis_result")
has_analysis_result = isinstance(analysis_result, dict)
analysis_question = st.session_state.get(
    "genai_question",
    analysis_result.get("question", "") if has_analysis_result else "",
)


def render_chart(result):
    chart_data = result.get("chart_data")
    if not chart_data or not chart_data.get("labels"):
        return

    chart_frame = pd.DataFrame(
        {"label": chart_data["labels"], "value": chart_data["values"]}
    )
    chart_type = result.get("chart_type")
    if chart_type == "line":
        chart = alt.Chart(chart_frame).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=55, color="#1f5a94"),
            color="#1f5a94",
        ).encode(
            x=alt.X("label:N", title="Period", sort=None),
            y=alt.Y("value:Q", title="Value", scale=alt.Scale(zero=False)),
            tooltip=["label:N", alt.Tooltip("value:Q", format=",.2f")],
        )
    elif chart_type == "donut":
        chart = alt.Chart(chart_frame).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("label:N", title=None),
            tooltip=["label:N", alt.Tooltip("value:Q", format=",.2f")],
        )
    else:
        chart = alt.Chart(chart_frame).mark_bar(color="#168a8a").encode(
            x=alt.X("value:Q", title="Value"),
            y=alt.Y("label:N", title=None, sort="-x"),
            tooltip=["label:N", alt.Tooltip("value:Q", format=",.2f")],
        )
    st.altair_chart(chart.properties(height=320), use_container_width=True)


with st.sidebar:
    st.header("Olist Business Intelligence")
    st.subheader("AI Business Analysis")
    st.caption("Deterministic analysis result" if has_analysis_result else "Static layout prototype")


st.title("AI Business Analysis")
st.caption(
    "Displaying a structured analyst result."
    if has_analysis_result
    else "Layout prototype: this page is not connected to the assistant or live calculations."
)

st.markdown("### Your Question")
if has_analysis_result:
    st.markdown(
        f'<div class="analysis-question">{analysis_question or "[Question not provided]"}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="analysis-question">[Placeholder question will appear here after the AI assistant is connected.]</div>',
        unsafe_allow_html=True,
    )

st.markdown("### Answer")
if has_analysis_result:
    st.markdown(
        f'<div class="answer-card"><div class="analysis-label" style="color: #cbd7e5;">Answer</div>'
        f'<div class="answer-placeholder">{analysis_result.get("answer", "[Answer not provided]")}</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="answer-card">
            <div class="analysis-label" style="color: #cbd7e5;">Placeholder executive answer</div>
            <div class="answer-placeholder">[Exact result will appear here]</div>
            <div class="placeholder-note" style="color: #dce3ed; margin-top: 0.55rem;">Static prototype content only. No business result is being calculated.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if has_analysis_result and analysis_result.get("supporting_metrics"):
    st.markdown("### Supporting Metrics")
    kpis = [
        (metric.get("label", "Metric"), metric.get("value", ""))
        for metric in analysis_result["supporting_metrics"]
    ]
    kpi_columns = st.columns(len(kpis))
    for column, kpi in zip(kpi_columns, kpis):
        with column:
            label, value = kpi
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-placeholder">{value}</div></div>',
                unsafe_allow_html=True,
            )
elif not has_analysis_result:
    st.markdown("### Supporting Metrics")
    kpis = [(label, "[Placeholder]") for label in [
        "Primary Metric",
        "Comparison Metric",
        "Change / Rate",
        "Orders or Items",
        "Average Value",
    ]]
    kpi_columns = st.columns(len(kpis))
    for column, kpi in zip(kpi_columns, kpis):
        with column:
            label, value = kpi
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-placeholder">{value}</div></div>',
                unsafe_allow_html=True,
            )

if has_analysis_result and analysis_result.get("chart_data"):
    st.markdown("### Visual Analysis")
    render_chart(analysis_result)
elif not has_analysis_result:
    st.markdown("### Visual Analysis")
    st.markdown(
        """
        <div class="chart-placeholder">
            <div>
                <strong>[Relevant chart placeholder]</strong>
                <span class="placeholder-note">A question-specific Altair visualization will be rendered here later.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if has_analysis_result and analysis_result.get("interpretation"):
    st.markdown("### Business Interpretation")
    st.markdown(
        f'<div class="analysis-section"><span class="placeholder-note">{analysis_result["interpretation"]}</span></div>',
        unsafe_allow_html=True,
    )
elif not has_analysis_result:
    st.markdown("### Business Interpretation")
    st.markdown(
        '<div class="analysis-section"><span class="placeholder-note">[Placeholder interpretation based on the eventual deterministic result.]</span></div>',
        unsafe_allow_html=True,
    )

if has_analysis_result:
    st.markdown("### Methodology")
    st.markdown(
        f'<div class="analysis-section"><span class="placeholder-note">{analysis_result.get("methodology", "[Methodology not provided]")}</span>'
        f'<div class="placeholder-note" style="margin-top: 0.55rem;">Analysis type: {analysis_result.get("intent", "unspecified")} · Visualization: {analysis_result.get("chart_type", "unspecified")}</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown("### Methodology")
    st.markdown(
        '<div class="analysis-section"><span class="placeholder-note">[Placeholder calculation methodology and applied filters.]</span></div>',
        unsafe_allow_html=True,
    )

if has_analysis_result and analysis_result.get("limitations"):
    st.markdown("### Considerations & Limitations")
    st.markdown(
        f'<div class="analysis-section"><span class="placeholder-note">{analysis_result["limitations"]}</span></div>',
        unsafe_allow_html=True,
    )
elif not has_analysis_result:
    st.markdown("### Considerations & Limitations")
    st.markdown(
        """
        <div class="analysis-section">
            <ul class="placeholder-note">
                <li>[Placeholder business consideration]</li>
                <li>[Placeholder data limitation]</li>
                <li>[Placeholder interpretation boundary]</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
