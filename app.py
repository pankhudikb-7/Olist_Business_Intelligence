from pathlib import Path

import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from analytics import (
    get_average_order_value,
    get_category_sales,
    get_delivery_metrics,
    get_items_sold,
    get_repeat_customer_rate,
    get_review_summary,
    get_total_customers,
    get_total_orders,
    get_total_sales,
)

from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master

from genai_assistant import (
    apply_dashboard_filters,
    dashboard_layout,
)


st.set_page_config(
    page_title="Olist Business Intelligence Dashboard",
    page_icon="O",
    layout="wide",
)


@st.cache_data
def load_dashboard_data():
    """Load, clean, and model the source data for the executive dashboard."""
    datasets = clean_datasets(load_datasets(PROJECT_ROOT / "data"))
    return datasets, build_sales_master(datasets)


def format_currency(value):
    return f"R$ {value:,.2f}"


def format_compact_currency(value):
    if abs(value) >= 1_000_000:
        return f"R${value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"R${value / 1_000:.2f}K"

    return format_currency(value)


datasets, sales_master = load_dashboard_data()


with dashboard_layout(datasets, sales_master):

    # ---------------------------------------------------------
    # APPLY COMMON DASHBOARD FILTERS
    # ---------------------------------------------------------
    filtered_datasets, filtered_sales_master = apply_dashboard_filters(
        datasets,
        sales_master,
    )

    # ---------------------------------------------------------
    # EXECUTIVE KPIs
    # ---------------------------------------------------------
    total_customers = get_total_customers(
        filtered_datasets["customers"]
    )

    total_orders = get_total_orders(
        filtered_sales_master
    )

    total_sales = get_total_sales(
        filtered_sales_master
    )

    average_order_value = get_average_order_value(
        filtered_sales_master
    )

    items_sold = get_items_sold(
        filtered_sales_master
    )

    total_sellers = int(
        filtered_sales_master["seller_id"].nunique()
    )

    review_summary = get_review_summary(
        filtered_datasets
    )

    delivery_metrics = get_delivery_metrics(
        filtered_datasets
    )

    # ---------------------------------------------------------
    # EXISTING EXECUTIVE DASHBOARD STYLING
    # ---------------------------------------------------------
    st.markdown(
        """
        <style>
        .kpi-card {
            background: #f7f9fc;
            border: 1px solid #dce3ed;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 96px;
        }

        .kpi-label {
            color: #526173;
            font-size: 0.85rem;
            margin-bottom: 8px;
        }

        .kpi-value {
            color: #14213d;
            font-size: 1.4rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .question-card {
            background: #f7f9fc;
            border-left: 3px solid #1f5a94;
            padding: 10px 12px;
            min-height: 54px;
            color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("Olist Business Intelligence")
        st.subheader("Executive Overview")
        st.caption("Data Source: Olist Brazilian E-Commerce Dataset")

    # ---------------------------------------------------------
    # PAGE TITLE
    # ---------------------------------------------------------
    st.title("Olist Business Intelligence Dashboard")
    st.subheader("Executive Overview")

    # ---------------------------------------------------------
    # KPI CARDS
    # ---------------------------------------------------------
    kpis = [
        ("Total Customers", f"{total_customers:,}"),
        ("Total Orders", f"{total_orders:,}"),
        ("Total Sales", format_compact_currency(total_sales)),
        ("Average Order Value", format_currency(average_order_value)),
        ("Items Sold", f"{items_sold:,}"),
        ("Total Sellers", f"{total_sellers:,}"),
        (
            "Average Review Score",
            f"{review_summary['average_review_score']:.2f} / 5",
        ),
        (
            "Late Delivery Rate",
            f"{delivery_metrics['late_delivery_rate']:.2f}%",
        ),
    ]

    for row_start in range(0, len(kpis), 4):

        columns = st.columns(4)

        for column, (label, value) in zip(
            columns,
            kpis[row_start : row_start + 4],
        ):
            with column:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ---------------------------------------------------------
    # BUSINESS QUESTIONS
    # ---------------------------------------------------------
    st.markdown("### Business Questions")

    questions = [
        "How is overall sales performance?",
        "Which categories drive sales?",
        "How are customers responding to their purchases?",
        "How efficiently are orders being delivered?",
    ]

    question_columns = st.columns(4)

    for question_column, question in zip(
        question_columns,
        questions,
    ):
        with question_column:
            st.markdown(
                f'<div class="question-card">{question}</div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # MONTHLY SALES TREND
    # ---------------------------------------------------------
    sales_by_month = (
        filtered_sales_master.assign(
            month=filtered_sales_master[
                "order_purchase_timestamp"
            ].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("month", as_index=False)["price"]
        .sum()
        .rename(columns={"price": "Sales"})
    )

    # ---------------------------------------------------------
    # MONTHLY ORDER TREND
    # ---------------------------------------------------------
    orders_by_month = (
        filtered_sales_master.assign(
            month=filtered_sales_master[
                "order_purchase_timestamp"
            ].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("month")["order_id"]
        .nunique()
        .rename("Orders")
        .to_frame()
        .reset_index()
    )

    CHART_HEIGHT = 320

    PRIMARY_BLUE = "#1f5a94"
    TEAL = "#168a8a"
    ON_TIME_GREEN = "#2e8b57"
    LATE_ORANGE = "#d97706"

    # ---------------------------------------------------------
    # SALES TREND
    # ---------------------------------------------------------
    st.markdown("### Monthly Sales Trend")
    st.caption("Monthly sales based on order item price")

    sales_chart = (
        alt.Chart(sales_by_month)
        .mark_line(
            point=alt.OverlayMarkDef(
                filled=True,
                size=55,
                color=PRIMARY_BLUE,
            ),
            color=PRIMARY_BLUE,
        )
        .encode(
            x=alt.X(
                "month:T",
                title="Month",
                axis=alt.Axis(format="%b %Y"),
            ),
            y=alt.Y(
                "Sales:Q",
                title="Sales (R$)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format="~s"),
            ),
            tooltip=[
                alt.Tooltip(
                    "month:T",
                    title="Month",
                    format="%b %Y",
                ),
                alt.Tooltip(
                    "Sales:Q",
                    title="Sales",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )

    st.altair_chart(
        sales_chart,
        width="stretch",
    )

    # ---------------------------------------------------------
    # PARTIAL SEPTEMBER DATA NOTE
    # ---------------------------------------------------------
    if not filtered_sales_master.empty:

        max_purchase_timestamp = filtered_sales_master[
            "order_purchase_timestamp"
        ].max()

        september_dates = filtered_sales_master.loc[
            filtered_sales_master[
                "order_purchase_timestamp"
            ].dt.to_period("M")
            == "2018-09",
            "order_purchase_timestamp",
        ]

        september_max_timestamp = september_dates.max()

        if (
            pd.notna(september_max_timestamp)
            and max_purchase_timestamp
            > pd.Timestamp("2018-09-30 23:59:59")
            and september_max_timestamp
            < pd.Timestamp("2018-09-30 23:59:59")
        ):
            st.caption(
                "Note: September 2018 contains partial-period data "
                "in the dataset."
            )

    # ---------------------------------------------------------
    # ORDER TREND + TOP CATEGORIES
    # ---------------------------------------------------------
    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )

    with left_chart:

        st.markdown("### Monthly Order Trend")

        order_chart = (
            alt.Chart(orders_by_month)
            .mark_line(
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=55,
                    color=PRIMARY_BLUE,
                ),
                color=PRIMARY_BLUE,
            )
            .encode(
                x=alt.X(
                    "month:T",
                    title="Month",
                    axis=alt.Axis(format="%b %Y"),
                ),
                y=alt.Y(
                    "Orders:Q",
                    title="Unique orders",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(
                        "month:T",
                        title="Month",
                        format="%b %Y",
                    ),
                    alt.Tooltip(
                        "Orders:Q",
                        title="Orders",
                        format=",d",
                    ),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )

        st.altair_chart(
            order_chart,
            width="stretch",
        )

    with right_chart:

        st.markdown("### Top 5 Categories by Sales")

        top_categories = (
            get_category_sales(
                filtered_sales_master
            )
            .sort_values(
                "sales",
                ascending=False,
            )
            .head(5)
        )

        category_chart = (
            alt.Chart(top_categories)
            .mark_bar(color=TEAL)
            .encode(
                x=alt.X(
                    "sales:Q",
                    title="Sales (R$)",
                    axis=alt.Axis(format="~s"),
                ),
                y=alt.Y(
                    "product_category_name_english:N",
                    title="English product category",
                    sort="-x",
                ),
                tooltip=[
                    alt.Tooltip(
                        "product_category_name_english:N",
                        title="Category",
                    ),
                    alt.Tooltip(
                        "sales:Q",
                        title="Sales",
                        format=",.2f",
                    ),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )

        st.altair_chart(
            category_chart,
            width="stretch",
        )

    # ---------------------------------------------------------
    # DELIVERY + CUSTOMER SATISFACTION
    # ---------------------------------------------------------
    delivery_chart_data = pd.DataFrame(
        {
            "Status": [
                "On-Time",
                "Late",
            ],
            "Rate": [
                delivery_metrics["on_time_delivery_rate"],
                delivery_metrics["late_delivery_rate"],
            ],
        }
    )

    delivery_colors = alt.Scale(
        domain=[
            "On-Time",
            "Late",
        ],
        range=[
            ON_TIME_GREEN,
            LATE_ORANGE,
        ],
    )

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )

    with left_chart:

        st.markdown("### Delivery Performance")

        delivery_bars = (
            alt.Chart(delivery_chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Status:N",
                    title="Delivery status",
                    sort=[
                        "On-Time",
                        "Late",
                    ],
                ),
                y=alt.Y(
                    "Rate:Q",
                    title="Rate (%)",
                    scale=alt.Scale(
                        domain=[0, 100]
                    ),
                ),
                color=alt.Color(
                    "Status:N",
                    scale=delivery_colors,
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip(
                        "Status:N",
                        title="Status",
                    ),
                    alt.Tooltip(
                        "Rate:Q",
                        title="Rate",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )

        delivery_labels = delivery_bars.mark_text(
            dy=-8,
            color="#334155",
        ).encode(
            text=alt.Text(
                "Rate:Q",
                format=".2f",
            )
        )

        st.altair_chart(
            delivery_bars + delivery_labels,
            width="stretch",
        )

    with right_chart:

        st.markdown("### Customer Satisfaction")

        review_distribution = (
            filtered_datasets["order_reviews"][
                "review_score"
            ]
            .value_counts()
            .reindex(
                range(1, 6),
                fill_value=0,
            )
            .rename("Reviews")
            .rename_axis("Review Score")
            .reset_index()
        )

        review_chart = (
            alt.Chart(review_distribution)
            .mark_bar(color=TEAL)
            .encode(
                x=alt.X(
                    "Review Score:O",
                    title="Review Score",
                    sort=[1, 2, 3, 4, 5],
                ),
                y=alt.Y(
                    "Reviews:Q",
                    title="Number of Reviews",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Review Score:O",
                        title="Review Score",
                    ),
                    alt.Tooltip(
                        "Reviews:Q",
                        title="Reviews",
                        format=",d",
                    ),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )

        st.altair_chart(
            review_chart,
            width="stretch",
        )

    # ---------------------------------------------------------
    # KEY BUSINESS INSIGHTS
    # ---------------------------------------------------------
    st.markdown("### Key Business Insights")

    repeat_rate = get_repeat_customer_rate(
        filtered_datasets
    )

    st.markdown(
        f"- Total sales were {format_currency(total_sales)} "
        f"across {total_orders:,} unique orders.\n"
        f"- Average order value was "
        f"{format_currency(average_order_value)}.\n"
        f"- The repeat customer rate was "
        f"{repeat_rate:.2f}%.\n"
        f"- The average review score was "
        f"{review_summary['average_review_score']:.2f} out of 5, "
        f"and the late delivery rate was "
        f"{delivery_metrics['late_delivery_rate']:.2f}%."
    )