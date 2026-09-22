from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =========================================================
# IMPORTS
# =========================================================

from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master
from genai_assistant import (dashboard_layout,apply_dashboard_filters,)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Seller Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_data
def load_seller_data():
    datasets = clean_datasets(
        load_datasets(PROJECT_ROOT / "data")
    )

    sales_master = build_sales_master(datasets)

    return datasets, sales_master


def format_currency(value):
    return f"R$ {value:,.2f}"


def format_compact_currency(value):
    if abs(value) >= 1_000_000:
        return f"R${value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"R${value / 1_000:.2f}K"

    return format_currency(value)


def shorten_seller_id(seller_id):
    seller_id = str(seller_id)
    return f"{seller_id[:8]}...{seller_id[-4:]}"


# =========================================================
# SELLER RATING DATA
# =========================================================

def get_seller_rating_data(datasets):

    order_sellers = (
        datasets["order_items"][
            [
                "order_id",
                "seller_id",
            ]
        ]
        .drop_duplicates(
            subset=[
                "order_id",
                "seller_id",
            ]
        )
    )

    reviews = (
        datasets["order_reviews"][
            [
                "review_id",
                "order_id",
                "review_score",
            ]
        ]
        .drop_duplicates(
            subset=[
                "review_id",
            ]
        )
    )

    review_sellers = (
        reviews
        .merge(
            order_sellers,
            how="inner",
            on="order_id",
        )
        .drop_duplicates(
            subset=[
                "order_id",
                "seller_id",
                "review_id",
            ]
        )
    )

    return (
        review_sellers
        .groupby(
            "seller_id",
            as_index=False,
        )
        .agg(
            average_rating=(
                "review_score",
                "mean",
            ),
            review_count=(
                "review_id",
                "nunique",
            ),
        )
    )


# =========================================================
# SELLER METRICS
# =========================================================

def get_seller_metrics(
    datasets,
    sales_master,
):

    if sales_master.empty:
        return pd.DataFrame()

    seller_metrics = (
        sales_master
        .groupby(
            "seller_id",
            as_index=False,
        )
        .agg(
            Revenue=(
                "price",
                "sum",
            ),
            Orders=(
                "order_id",
                "nunique",
            ),
            **{
                "Items Sold": (
                    "price",
                    "size",
                )
            },
            **{
                "Average Selling Price": (
                    "price",
                    "mean",
                )
            },
            City=(
                "seller_city",
                "first",
            ),
            State=(
                "seller_state",
                "first",
            ),
        )
        .rename(
            columns={
                "seller_id": "Seller ID"
            }
        )
    )

    seller_ratings = (
        get_seller_rating_data(
            datasets
        )
        .rename(
            columns={
                "seller_id": "Seller ID"
            }
        )
    )

    return seller_metrics.merge(
        seller_ratings,
        how="left",
        on="Seller ID",
    )


# =========================================================
# LOAD DATA
# =========================================================

datasets, sales_master = load_seller_data()


# =========================================================
# DASHBOARD
# =========================================================

with dashboard_layout(
    datasets,
    sales_master,
):

    # =====================================================
    # STYLING
    # =====================================================

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
            font-size: 1.35rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .seller-detail-card {
            background: #f7f9fc;
            border: 1px solid #dce3ed;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 96px;
        }

        .seller-detail-label {
            color: #526173;
            font-size: 0.82rem;
            margin-bottom: 8px;
        }

        .seller-detail-value {
            color: #14213d;
            font-size: 1.15rem;
            font-weight: 700;
            white-space: nowrap;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # SIDEBAR INFORMATION
    # =====================================================

    with st.sidebar:

        st.header(
            "Olist Business Intelligence"
        )

        st.subheader(
            "Seller Analysis"
        )

        st.caption(
            "Data Source: Olist Brazilian E-Commerce Dataset"
        )


    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title(
        "Seller Analysis"
    )

    st.caption(
        "How concentrated is seller revenue, and how does "
        "seller activity relate to satisfaction?"
    )


    # =====================================================
    # COMMON FILTER VALUES
    # =====================================================

    selected_customer_state = (
        st.session_state.get(
            "customer_state_filter",
            "All States",
        )
    )

    selected_date_range = (
        st.session_state.get(
            "order_date_range"
        )
        or st.session_state.get(
            "date_range"
        )
        or st.session_state.get(
            "dashboard_date_range"
        )
    )

    # =====================================================
    # APPLY COMMON FILTERS
    # =====================================================

    filtered_datasets, filtered_sales_master = apply_dashboard_filters(
        datasets,
        sales_master,
    )

    # =====================================================
    # SELLER METRICS
    # =====================================================

    seller_metrics = (
        get_seller_metrics(
            filtered_datasets,
            filtered_sales_master,
        )
    )


    if seller_metrics.empty:

        st.warning(
            "No seller records are available "
            "for the selected filters."
        )

        st.stop()


    seller_metrics[
        "Seller Label"
    ] = (
        seller_metrics[
            "Seller ID"
        ]
        .map(
            shorten_seller_id
        )
    )

    # =====================================================
    # BASIC SELLER KPIs
    # =====================================================

    total_sellers = int(
        seller_metrics[
            "Seller ID"
        ]
        .nunique()
    )

    seller_revenue = float(
        seller_metrics[
            "Revenue"
        ]
        .sum()
    )

    seller_orders = int(
        seller_metrics[
            "Orders"
        ]
        .sum()
    )

    items_sold = int(
        seller_metrics[
            "Items Sold"
        ]
        .sum()
    )

    average_seller_rating = float(
        seller_metrics[
            "average_rating"
        ]
        .mean()
    )

    top_revenue_seller = (
        seller_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .iloc[0]
    )


    # =====================================================
    # FILTER DESCRIPTION
    # =====================================================

    if selected_customer_state == "All States":

        state_text = "all states"

    else:

        state_text = (
            selected_customer_state
        )


    if (
        isinstance(
            selected_date_range,
            (list, tuple),
        )
        and len(
            selected_date_range
        ) == 2
    ):

        date_text = (
            f"{selected_date_range[0]} "
            f"to "
            f"{selected_date_range[1]}"
        )

    else:

        date_text = (
            "the full available period"
        )


    st.caption(
        f"Showing seller analysis for "
        f"{state_text} and {date_text}."
    )


    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [

        (
            "Total Sellers",
            f"{total_sellers:,}",
        ),

        (
            "Seller Revenue",
            format_compact_currency(
                seller_revenue
            ),
        ),

        (
            "Seller Orders",
            f"{seller_orders:,}",
        ),

        (
            "Items Sold",
            f"{items_sold:,}",
        ),

        (
            "Average Seller Rating",
            f"{average_seller_rating:.2f} / 5",
        ),

        (
            "Top Revenue Seller",
            shorten_seller_id(
                top_revenue_seller[
                    "Seller ID"
                ]
            ),
        ),
    ]


    kpi_columns = st.columns(6)


    for column, (label, value) in zip(
        kpi_columns,
        kpis,
    ):

        with column:

            st.markdown(
                f"""
                <div class="kpi-card">
                    <span class="kpi-label">{label}</span>
                    <br>
                    <span class="kpi-value">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


    # =====================================================
    # CHART CONSTANTS
    # =====================================================

    CHART_HEIGHT = 310

    PRIMARY_BLUE = "#1f5a94"

    TEAL = "#168a8a"

    CORAL = "#e07a5f"


    # =====================================================
    # SELLER REVENUE CONCENTRATION
    # =====================================================

    pareto = (
        seller_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    pareto[
        "Seller Rank"
    ] = (
        pareto.index + 1
    )

    pareto[
        "Revenue Share %"
    ] = (
        pareto[
            "Revenue"
        ]
        / seller_revenue
        * 100
    )

    pareto[
        "Cumulative Revenue %"
    ] = (
        pareto[
            "Revenue Share %"
        ]
        .cumsum()
    )


    st.markdown(
        "### Seller Revenue Concentration"
    )


    pareto_chart = (
        alt.Chart(
            pareto
        )
        .mark_line(
            color=PRIMARY_BLUE,
            point=alt.OverlayMarkDef(
                filled=True,
                size=28,
                color=PRIMARY_BLUE,
            ),
        )
        .encode(

            x=alt.X(
                "Seller Rank:Q",
                title="Seller rank by revenue",
                axis=alt.Axis(
                    format="d"
                ),
            ),

            y=alt.Y(
                "Cumulative Revenue %:Q",
                title="Cumulative seller revenue (%)",
                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            tooltip=[

                alt.Tooltip(
                    "Seller Rank:Q",
                    title="Seller Rank",
                    format=",d",
                ),

                alt.Tooltip(
                    "Seller ID:N",
                    title="Seller ID",
                ),

                alt.Tooltip(
                    "Revenue:Q",
                    title="Revenue",
                    format=",.2f",
                ),

                alt.Tooltip(
                    "Revenue Share %:Q",
                    title="Revenue Share %",
                    format=".2f",
                ),

                alt.Tooltip(
                    "Cumulative Revenue %:Q",
                    title="Cumulative Revenue %",
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=350
        )
    )


    pareto_rule = (
        alt.Chart(
            pd.DataFrame(
                {
                    "threshold": [
                        80
                    ]
                }
            )
        )
        .mark_rule(
            color=CORAL,
            strokeDash=[
                6,
                4,
            ],
        )
        .encode(
            y=alt.Y(
                "threshold:Q",
                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            )
        )
    )


    st.altair_chart(
        pareto_chart + pareto_rule,
        width="stretch",
    )


    # =====================================================
    # SELLER SCALE MEDIANS
    # =====================================================

    median_orders = float(
        seller_metrics[
            "Orders"
        ]
        .median()
    )

    median_items = float(
        seller_metrics[
            "Items Sold"
        ]
        .median()
    )


    # =====================================================
    # SELLER SCALE + RATING
    # =====================================================

    left_chart, right_chart = (
        st.columns(
            2,
            gap="large",
        )
    )


    # =====================================================
    # SELLER SCALE
    # =====================================================

    with left_chart:

        st.markdown(
            "### Seller Scale: Orders vs Items Sold"
        )


        scale_chart = (
            alt.Chart(
                seller_metrics
            )
            .mark_circle(
                color=TEAL,
                opacity=0.7,
            )
            .encode(

                x=alt.X(
                    "Orders:Q",
                    title="Unique Orders",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                y=alt.Y(
                    "Items Sold:Q",
                    title="Items Sold",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                size=alt.Size(
                    "Revenue:Q",
                    title="Revenue",
                    scale=alt.Scale(
                        range=[
                            30,
                            850,
                        ]
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Seller ID:N",
                        title="Seller ID",
                    ),

                    alt.Tooltip(
                        "Revenue:Q",
                        title="Revenue",
                        format=",.2f",
                    ),

                    alt.Tooltip(
                        "Orders:Q",
                        title="Orders",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Items Sold:Q",
                        title="Items Sold",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "average_rating:Q",
                        title="Average Rating",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        scale_x_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Orders": [
                            median_orders
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[
                    5,
                    5,
                ],
            )
            .encode(
                x="Orders:Q"
            )
        )


        scale_y_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Items Sold": [
                            median_items
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[
                    5,
                    5,
                ],
            )
            .encode(
                y="Items Sold:Q"
            )
        )


        st.altair_chart(
            scale_chart
            + scale_x_rule
            + scale_y_rule,
            width="stretch",
        )


    # =====================================================
    # SELLER REVENUE VS CUSTOMER RATING
    # =====================================================

    with right_chart:

        st.markdown(
            "### Seller Revenue vs Customer Rating"
        )


        # Analytical threshold only.
        SELLER_REVIEW_THRESHOLD = 20


        rated_sellers = (
            seller_metrics[
                seller_metrics[
                    "review_count"
                ]
                .fillna(0)
                >= SELLER_REVIEW_THRESHOLD
            ]
            .copy()
        )


        if rated_sellers.empty:

            st.info(
                "There are not enough rated sellers "
                "for the selected filters."
            )

        else:

            rated_median_revenue = float(
                rated_sellers[
                    "Revenue"
                ]
                .median()
            )

            rated_median_rating = float(
                rated_sellers[
                    "average_rating"
                ]
                .median()
            )


            revenue_rating_chart = (
                alt.Chart(
                    rated_sellers
                )
                .mark_circle(
                    color=TEAL,
                    opacity=0.75,
                )
                .encode(

                    x=alt.X(
                        "Revenue:Q",
                        title="Seller Revenue (R$)",
                        axis=alt.Axis(
                            format="~s"
                        ),
                    ),

                    y=alt.Y(
                        "average_rating:Q",
                        title="Average Seller Rating",
                        scale=alt.Scale(
                            domain=[
                                0,
                                5,
                            ]
                        ),
                    ),

                    size=alt.Size(
                        "review_count:Q",
                        title="Review Count",
                        scale=alt.Scale(
                            range=[
                                35,
                                850,
                            ]
                        ),
                    ),

                    tooltip=[

                        alt.Tooltip(
                            "Seller ID:N",
                            title="Seller ID",
                        ),

                        alt.Tooltip(
                            "Revenue:Q",
                            title="Revenue",
                            format=",.2f",
                        ),

                        alt.Tooltip(
                            "average_rating:Q",
                            title="Average Rating",
                            format=".2f",
                        ),

                        alt.Tooltip(
                            "review_count:Q",
                            title="Review Count",
                            format=",d",
                        ),

                        alt.Tooltip(
                            "Orders:Q",
                            title="Orders",
                            format=",d",
                        ),

                        alt.Tooltip(
                            "Items Sold:Q",
                            title="Items Sold",
                            format=",d",
                        ),
                    ],
                )
                .properties(
                    height=CHART_HEIGHT
                )
            )


            rating_x_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Revenue": [
                                rated_median_revenue
                            ]
                        }
                    )
                )
                .mark_rule(
                    color="#526173",
                    strokeDash=[
                        5,
                        5,
                    ],
                )
                .encode(
                    x="Revenue:Q"
                )
            )


            rating_y_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "average_rating": [
                                rated_median_rating
                            ]
                        }
                    )
                )
                .mark_rule(
                    color="#526173",
                    strokeDash=[
                        5,
                        5,
                    ],
                )
                .encode(
                    y="average_rating:Q"
                )
            )


            st.altair_chart(
                revenue_rating_chart
                + rating_x_rule
                + rating_y_rule,
                width="stretch",
            )


    # =====================================================
    # RATING DISTRIBUTION
    # =====================================================

    rating_bands = [
        "<3.5",
        "3.5-3.9",
        "4.0-4.4",
        "4.5-5.0",
    ]


    if not rated_sellers.empty:

        rating_distribution = (
            pd.cut(

                rated_sellers[
                    "average_rating"
                ],

                bins=[
                    0,
                    3.5,
                    4.0,
                    4.5,
                    5.01,
                ],

                labels=rating_bands,

                right=False,

                include_lowest=True,
            )
            .value_counts()
            .reindex(
                rating_bands,
                fill_value=0,
            )
            .rename(
                "Seller Count"
            )
            .rename_axis(
                "Rating Band"
            )
            .reset_index()
        )


        rating_distribution[
            "Percentage"
        ] = (
            rating_distribution[
                "Seller Count"
            ]
            /
            rating_distribution[
                "Seller Count"
            ].sum()
            * 100
        )

    else:

        rating_distribution = pd.DataFrame(
            {
                "Rating Band": [],
                "Seller Count": [],
                "Percentage": [],
            }
        )


    # =====================================================
    # TOP SELLERS VS OTHERS
    # =====================================================

    seller_top5 = (
        seller_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(5)[
            [
                "Seller ID",
                "Revenue",
            ]
        ]
        .copy()
    )


    other_seller_revenue = float(
        seller_metrics.iloc[5:][
            "Revenue"
        ]
        .sum()
    )


    seller_share = pd.concat(
        [

            seller_top5.rename(
                columns={
                    "Seller ID":
                        "Seller Group"
                }
            ),

            pd.DataFrame(
                [
                    {
                        "Seller Group":
                            "Other Sellers",

                        "Revenue":
                            other_seller_revenue,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


    seller_share[
        "Revenue Share %"
    ] = (
        seller_share[
            "Revenue"
        ]
        / seller_revenue
        * 100
    )


    # =====================================================
    # RATING + REVENUE SHARE
    # =====================================================

    left_chart, right_chart = (
        st.columns(
            2,
            gap="large",
        )
    )


    # =====================================================
    # SELLER RATING DISTRIBUTION
    # =====================================================

    with left_chart:

        st.markdown(
            "### Seller Rating Distribution"
        )


        distribution_chart = (
            alt.Chart(
                rating_distribution
            )
            .mark_bar(
                color=TEAL
            )
            .encode(

                x=alt.X(
                    "Rating Band:N",
                    title="Average Rating Band",
                    sort=rating_bands,
                ),

                y=alt.Y(
                    "Seller Count:Q",
                    title="Number of Sellers",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Rating Band:N",
                        title="Rating Band",
                    ),

                    alt.Tooltip(
                        "Seller Count:Q",
                        title="Seller Count",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Percentage:Q",
                        title="Percentage",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        st.altair_chart(
            distribution_chart,
            width="stretch",
        )


    # =====================================================
    # REVENUE SHARE
    # =====================================================

    with right_chart:

        st.markdown(
            "### Revenue Share: Top Sellers vs Others"
        )


        share_chart = (
            alt.Chart(
                seller_share
            )
            .mark_arc(
                innerRadius=55
            )
            .encode(

                theta=alt.Theta(
                    "Revenue:Q",
                    title="Revenue",
                ),

                color=alt.Color(
                    "Seller Group:N",
                    title="Seller Group",
                ),

                tooltip=[

                    alt.Tooltip(
                        "Seller Group:N",
                        title="Seller Group",
                    ),

                    alt.Tooltip(
                        "Revenue:Q",
                        title="Revenue",
                        format=",.2f",
                    ),

                    alt.Tooltip(
                        "Revenue Share %:Q",
                        title="Revenue Share %",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        st.altair_chart(
            share_chart,
            width="stretch",
        )


    # =====================================================
    # KEY SELLER INSIGHTS
    # =====================================================

    highest_revenue_seller = (
        seller_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .iloc[0]
    )


    highest_volume_seller = (
        seller_metrics
        .sort_values(
            "Items Sold",
            ascending=False,
        )
        .iloc[0]
    )


    top5_share = (
        seller_share.iloc[:5][
            "Revenue Share %"
        ]
        .sum()
    )


    st.markdown(
        "### Key Seller Insights"
    )


    insights = [

        (
            f"{shorten_seller_id(highest_revenue_seller['Seller ID'])} "
            f"generated the highest recorded seller revenue at "
            f"{format_currency(highest_revenue_seller['Revenue'])}."
        ),

        (
            f"{shorten_seller_id(highest_volume_seller['Seller ID'])} "
            f"had the highest item volume with "
            f"{int(highest_volume_seller['Items Sold']):,} "
            f"items sold."
        ),

        (
            f"The top five sellers contributed "
            f"{top5_share:.2f}% of total seller revenue."
        ),
    ]


    if not rated_sellers.empty:

        highest_rated_seller = (
            rated_sellers
            .sort_values(
                "average_rating",
                ascending=False,
            )
            .iloc[0]
        )


        revenue_rating_correlation = (
            rated_sellers[
                [
                    "Revenue",
                    "average_rating",
                ]
            ]
            .corr()
            .iloc[0, 1]
        )


        insights.append(
            (
                f"Among sellers with at least "
                f"{SELLER_REVIEW_THRESHOLD} mapped reviews, "
                f"{shorten_seller_id(highest_rated_seller['Seller ID'])} "
                f"had the highest average rating at "
                f"{highest_rated_seller['average_rating']:.2f}."
            )
        )


        if not pd.isna(
            revenue_rating_correlation
        ):

            insights.append(
                (
                    f"Among sellers meeting the "
                    f"{SELLER_REVIEW_THRESHOLD}-review analytical threshold, "
                    f"the revenue-rating correlation was "
                    f"{revenue_rating_correlation:.2f}."
                )
            )


    st.markdown(
        "\n".join(
            f"- {insight}"
            for insight in insights
        )
    )


    # =====================================================
    # SELLER DRILL-DOWN
    # =====================================================

    st.markdown(
        "---"
    )


    st.markdown(
        "### Seller Drill-Down"
    )


    st.caption(
        "Select a seller to view detailed performance metrics. "
        "The comparison charts above remain unchanged."
    )


    seller_ids = (
        seller_metrics[
            "Seller ID"
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )


    selected_seller = st.selectbox(
        "Search / Select Seller",
        ["Select a Seller"] + seller_ids,
        key="seller_drilldown_filter",
        format_func=lambda seller_id: (
            seller_id
            if seller_id == "Select a Seller"
            else shorten_seller_id(
                seller_id
            )
        ),
    )


    # =====================================================
    # SELECTED SELLER DETAILS
    # =====================================================

    if selected_seller != "Select a Seller":

        selected_seller_data = (
            seller_metrics[
                seller_metrics[
                    "Seller ID"
                ]
                .astype(str)
                == selected_seller
            ]
            .iloc[0]
        )


        st.markdown(
            f"**Seller ID:** `{selected_seller}`"
        )


        detail_columns = st.columns(6)


        detail_metrics = [

            (
                "Revenue",
                format_currency(
                    selected_seller_data[
                        "Revenue"
                    ]
                ),
            ),

            (
                "Orders",
                f"{int(selected_seller_data['Orders']):,}",
            ),

            (
                "Items Sold",
                f"{int(selected_seller_data['Items Sold']):,}",
            ),

            (
                "Average Selling Price",
                format_currency(
                    selected_seller_data[
                        "Average Selling Price"
                    ]
                ),
            ),

            (
                "Average Rating",
                (
                    f"{selected_seller_data['average_rating']:.2f} / 5"
                    if pd.notna(
                        selected_seller_data[
                            "average_rating"
                        ]
                    )
                    else "No rating"
                ),
            ),

            (
                "Review Count",
                (
                    f"{int(selected_seller_data['review_count']):,}"
                    if pd.notna(
                        selected_seller_data[
                            "review_count"
                        ]
                    )
                    else "0"
                ),
            ),
        ]


        # =================================================
        # DRILL-DOWN KPI CARDS
        # =================================================

        for column, (label, value) in zip(
            detail_columns,
            detail_metrics,
        ):

            with column:

                st.markdown(
                    f"""
                    <div class="seller-detail-card">
                        <span class="seller-detail-label">{label}</span>
                        <br>
                        <span class="seller-detail-value">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # =================================================
        # SELLER LOCATION
        # =================================================

        location_column_1, location_column_2 = (
            st.columns(2)
        )


        with location_column_1:

            st.info(
                f"Seller City: "
                f"{selected_seller_data['City']}"
            )


        with location_column_2:

            st.info(
                f"Seller State: "
                f"{selected_seller_data['State']}"
            )


    # =====================================================
    # METHODOLOGY
    # =====================================================

    st.caption(
        "Seller ratings are based on distinct "
        "review_id + order_id + seller_id relationships "
        "and are not weighted by the number of order-item rows. "
        "The 20-review threshold is used only for "
        "rating-comparison charts to reduce the influence "
        "of sellers with very few reviews."
    )