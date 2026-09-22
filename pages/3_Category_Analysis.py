from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

from src.genai_assistant import (
    dashboard_layout,
    apply_dashboard_filters,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from analytics import (
    get_category_items,
    get_category_sales,
    get_items_sold,
)
from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master
from genai_assistant import dashboard_layout


st.set_page_config(
    page_title="Category Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_category_data():
    datasets = clean_datasets(
        load_datasets(PROJECT_ROOT / "data")
    )

    sales_master = build_sales_master(datasets)

    return datasets, sales_master


# =========================================================
# FORMATTING
# =========================================================

def format_currency(value):
    return f"R$ {value:,.2f}"


def format_compact_currency(value):
    if abs(value) >= 1_000_000:
        return f"R${value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"R${value / 1_000:.2f}K"

    return format_currency(value)


# =========================================================
# CATEGORY RATING DATA
# =========================================================

def get_category_rating_data(datasets, sales_master):
    """
    Map reviews to unique order-category combinations
    so that category ratings are not inflated by multiple
    products from the same category within an order.
    """

    order_categories = (
        sales_master[
            [
                "order_id",
                "product_category_name_english",
            ]
        ]
        .assign(
            product_category_name_english=lambda dataframe:
            dataframe["product_category_name_english"].fillna("Unknown")
        )
        .drop_duplicates(
            subset=[
                "order_id",
                "product_category_name_english",
            ]
        )
    )

    reviews = (
        datasets["order_reviews"][
            [
                "order_id",
                "review_score",
            ]
        ]
        .drop_duplicates(
            subset=[
                "order_id",
                "review_score",
            ]
        )
    )

    review_categories = (
        reviews.merge(
            order_categories,
            how="inner",
            on="order_id",
            validate="many_to_many",
        )
        .drop_duplicates(
            subset=[
                "order_id",
                "product_category_name_english",
                "review_score",
            ]
        )
    )

    rating_summary = (
        review_categories
        .groupby(
            "product_category_name_english",
            as_index=False,
        )
        .agg(
            average_rating=("review_score", "mean"),
            review_count=("review_score", "size"),
        )
        .rename(
            columns={
                "product_category_name_english": "Category"
            }
        )
    )

    return rating_summary


# =========================================================
# CATEGORY METRICS
# =========================================================

def get_category_metrics(datasets, sales_master):

    category_sales = (
        get_category_sales(sales_master)
        .rename(
            columns={
                "product_category_name_english": "Category",
                "sales": "Revenue",
            }
        )
    )

    category_items = (
        get_category_items(sales_master)
        .rename(
            columns={
                "product_category_name_english": "Category",
                "items_sold": "Items Sold",
            }
        )
    )

    ratings = get_category_rating_data(
        datasets,
        sales_master,
    )

    return (
        category_sales,
        category_items,
        ratings,
    )


# =========================================================
# LOAD
# =========================================================

datasets, sales_master = load_category_data()


# =========================================================
# CONSTANTS
# =========================================================

MIN_REVIEWS = 100

CHART_HEIGHT = 310

PRIMARY_BLUE = "#1f5a94"
TEAL = "#168a8a"
CORAL = "#e07a5f"
NAVY = "#14213d"


# =========================================================
# DASHBOARD
# =========================================================

with dashboard_layout(datasets, sales_master):

    # -----------------------------------------------------
    # KPI CARD STYLE
    # -----------------------------------------------------

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

        </style>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # PAGE HEADER
    # =====================================================

    st.title("Category Analysis")

    st.caption(
        "Understand category revenue, volume, ratings, "
        "and sales performance."
    )


    # =====================================================
    # COMMON DASHBOARD FILTERS
    # =====================================================

    selected_customer_state = st.session_state.get(
        "customer_state_filter",
        "All States",
    )

    selected_date_range = (
        st.session_state.get("order_date_range")
        or st.session_state.get("date_range")
        or st.session_state.get("dashboard_date_range")
    )


    # =====================================================
    # FILTER DATA
    # =====================================================

    filtered_datasets, filtered_sales_master = apply_dashboard_filters(
        datasets,
        sales_master,
    )

    # =====================================================
    # CATEGORY METRICS
    # =====================================================
    (
        category_sales,
        category_items,
        ratings,
    ) = get_category_metrics(
        filtered_datasets,
        filtered_sales_master,
    )


    category_sales = category_sales.sort_values(
        "Revenue",
        ascending=False,
    )


    # -----------------------------------------------------
    # EMPTY DATA CHECK
    # -----------------------------------------------------

    if (
        category_sales.empty
        or category_items.empty
    ):

        st.warning(
            "No category sales data is available "
            "for the selected filters."
        )

        st.stop()


    # =====================================================
    # CATEGORY-SPECIFIC FILTER
    # =====================================================

    available_categories = (
        category_sales["Category"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


    selected_category = st.selectbox(
        "Search / Select Category",
        ["All Categories"] + available_categories,
        key="category_drilldown_filter",
    )


    # =====================================================
    # RATING CALCULATIONS
    # =====================================================

    ratings_with_enough_reviews = ratings[
        ratings["review_count"] >= MIN_REVIEWS
    ].copy()


    category_rating_comparison = (
        ratings_with_enough_reviews
        .sort_values(
            "average_rating",
            ascending=False,
        )
    )


    top_rated = (
        ratings_with_enough_reviews
        .nlargest(
            5,
            "average_rating",
        )
        .assign(
            rating_group="Top Rated"
        )
    )


    bottom_rated = (
        ratings_with_enough_reviews
        .nsmallest(
            5,
            "average_rating",
        )
        .assign(
            rating_group="Bottom Rated"
        )
    )


    rating_extremes = (
        pd.concat(
            [
                bottom_rated,
                top_rated,
            ]
        )
        .drop_duplicates(
            subset=["Category"]
        )
        .sort_values(
            "average_rating",
            ascending=True,
        )
    )


    # =====================================================
    # CATEGORY VOLUME TABLE
    # =====================================================

    category_volume = (
        category_sales
        .merge(
            category_items,
            how="inner",
            on="Category",
        )
    )


    category_volume["Average Selling Price"] = (
        category_volume["Revenue"]
        / category_volume["Items Sold"]
    )


    category_volume = (
        category_volume
        .merge(
            ratings[
                [
                    "Category",
                    "average_rating",
                    "review_count",
                ]
            ],
            how="left",
            on="Category",
        )
    )


    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_categories = int(
        category_sales["Category"].nunique()
    )


    category_revenue = float(
        category_sales["Revenue"].sum()
    )


    items_sold = get_items_sold(
        filtered_sales_master
    )


    average_category_rating = (
        float(
            ratings["average_rating"].mean()
        )
        if len(ratings)
        else 0.0
    )


    top_revenue_category = (
        category_sales.iloc[0]
    )


    top_items_category = (
        category_items
        .sort_values(
            "Items Sold",
            ascending=False,
        )
        .iloc[0]
    )


    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [
        (
            "Total Categories",
            f"{total_categories:,}",
        ),
        (
            "Category Revenue",
            format_compact_currency(
                category_revenue
            ),
        ),
        (
            "Items Sold",
            f"{items_sold:,}",
        ),
        (
            "Average Category Rating",
            f"{average_category_rating:.2f} / 5",
        ),
        (
            "Top Revenue Category",
            str(
                top_revenue_category[
                    "Category"
                ]
            ),
        ),
        (
            "Top Volume Category",
            str(
                top_items_category[
                    "Category"
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
                    <div class="kpi-label">
                        {label}
                    </div>
                    <div class="kpi-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    # =====================================================
    # CATEGORY DRILL-DOWN
    # =====================================================

    if selected_category != "All Categories":

        selected_sales = category_sales[
            category_sales["Category"]
            == selected_category
        ]

        selected_items = category_items[
            category_items["Category"]
            == selected_category
        ]

        selected_rating = ratings[
            ratings["Category"]
            == selected_category
        ]


        if not selected_sales.empty:

            selected_revenue = float(
                selected_sales.iloc[0][
                    "Revenue"
                ]
            )

            selected_items_sold = int(
                selected_items.iloc[0][
                    "Items Sold"
                ]
            )


            selected_avg_price = (
                selected_revenue
                / selected_items_sold
                if selected_items_sold > 0
                else 0
            )


            if not selected_rating.empty:

                selected_rating_value = float(
                    selected_rating.iloc[0][
                        "average_rating"
                    ]
                )

                selected_review_count = int(
                    selected_rating.iloc[0][
                        "review_count"
                    ]
                )

            else:

                selected_rating_value = 0.0
                selected_review_count = 0


            revenue_share = (
                selected_revenue
                / category_revenue
                * 100
                if category_revenue > 0
                else 0
            )


            st.markdown(
                f"### Category Drill-Down: {selected_category}"
            )


            detail_columns = st.columns(5)


            detail_values = [
                (
                    "Revenue",
                    format_compact_currency(
                        selected_revenue
                    ),
                ),
                (
                    "Items Sold",
                    f"{selected_items_sold:,}",
                ),
                (
                    "Average Selling Price",
                    format_currency(
                        selected_avg_price
                    ),
                ),
                (
                    "Average Rating",
                    (
                        f"{selected_rating_value:.2f} / 5"
                        if selected_review_count > 0
                        else "N/A"
                    ),
                ),
                (
                    "Revenue Share",
                    f"{revenue_share:.2f}%",
                ),
            ]


            for column, (
                label,
                value,
            ) in zip(
                detail_columns,
                detail_values,
            ):

                with column:

                    st.metric(
                        label,
                        value,
                    )


            st.caption(
                f"Mapped review count: "
                f"{selected_review_count:,}"
            )


    # =====================================================
    # TOP 10 CATEGORY REVENUE
    # =====================================================

    revenue_top10 = (
        category_sales
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(10)
        .copy()
    )


    revenue_top10["Revenue Share"] = (
        revenue_top10["Revenue"]
        / category_revenue
    )


    revenue_top10["Revenue Share %"] = (
        revenue_top10["Revenue Share"]
        * 100
    )


    revenue_top10["Revenue Label"] = (
        revenue_top10["Revenue"]
        .map(format_compact_currency)
    )


    # =====================================================
    # REVENUE SHARE
    # =====================================================

    revenue_share_top5 = (
        category_sales
        .head(5)[
            [
                "Category",
                "Revenue",
            ]
        ]
        .copy()
    )


    other_revenue = float(
        category_sales.iloc[5:][
            "Revenue"
        ].sum()
    )


    revenue_share_top5 = pd.concat(
        [
            revenue_share_top5,
            pd.DataFrame(
                [
                    {
                        "Category": "Other Categories",
                        "Revenue": other_revenue,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


    revenue_share_top5[
        "Revenue Share"
    ] = (
        revenue_share_top5["Revenue"]
        / category_revenue
    )


    revenue_share_top5[
        "Revenue Share %"
    ] = (
        revenue_share_top5[
            "Revenue Share"
        ]
        * 100
    )


    revenue_share_top5[
        "Share Start"
    ] = (
        revenue_share_top5[
            "Revenue Share"
        ].cumsum()
        - revenue_share_top5[
            "Revenue Share"
        ]
    )


    revenue_share_top5[
        "Share End"
    ] = (
        revenue_share_top5[
            "Revenue Share"
        ].cumsum()
    )


    revenue_share_top5[
        "Composition"
    ] = "Total category revenue"


    # =====================================================
    # HIGHLIGHT CATEGORIES
    # =====================================================

    highlight_categories = set(
        revenue_top10
        .head(3)["Category"]
    ).union(
        {
            top_items_category[
                "Category"
            ]
        }
    )


    category_volume_highlights = (
        category_volume[
            category_volume["Category"].isin(
                highlight_categories
            )
        ]
    )


    # =====================================================
    # ROW 1
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    # -----------------------------------------------------
    # TOP 10 REVENUE
    # -----------------------------------------------------

    with left_chart:

        st.markdown(
            "### Top 10 Categories by Revenue"
        )


        revenue_bars = (
            alt.Chart(revenue_top10)
            .mark_bar(
                color=PRIMARY_BLUE
            )
            .encode(
                x=alt.X(
                    "Category:N",
                    title=None,
                    sort="-y",
                    axis=alt.Axis(
                        labelAngle=-35
                    ),
                ),
                y=alt.Y(
                    "Revenue:Q",
                    title="Revenue (R$)",
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Category:N",
                        title="Category",
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


        revenue_labels = (
            alt.Chart(revenue_top10)
            .mark_text(
                dy=-8,
                color=NAVY,
                fontSize=11,
            )
            .encode(
                x=alt.X(
                    "Category:N",
                    sort="-y",
                ),
                y=alt.Y(
                    "Revenue:Q"
                ),
                text=alt.Text(
                    "Revenue Label:N"
                ),
            )
        )


        st.altair_chart(
            revenue_bars + revenue_labels,
            width="stretch",
        )


    # -----------------------------------------------------
    # REVENUE SHARE
    # -----------------------------------------------------

    with right_chart:

        st.markdown(
            "### Category Revenue Share"
        )


        share_rects = (
            alt.Chart(revenue_share_top5)
            .mark_rect(
                stroke="white",
                strokeWidth=1,
            )
            .encode(
                x=alt.X(
                    "Share Start:Q",
                    title=(
                        "Share of Total "
                        "Category Revenue"
                    ),
                    scale=alt.Scale(
                        domain=[0, 1]
                    ),
                    axis=alt.Axis(
                        format=".0%"
                    ),
                ),
                x2="Share End:Q",
                y=alt.Y(
                    "Composition:N",
                    title=None,
                    axis=None,
                ),
                color=alt.Color(
                    "Category:N",
                    title="Category",
                    scale=alt.Scale(
                        range=[
                            PRIMARY_BLUE,
                            "#3978ac",
                            "#5d96bd",
                            TEAL,
                            "#56a8a8",
                            "#a7b3bf",
                        ]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Category:N",
                        title="Category",
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
            share_rects,
            width="stretch",
        )


    # =====================================================
    # ROW 2
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    # -----------------------------------------------------
    # REVENUE VS ITEMS SOLD
    # -----------------------------------------------------

    with left_chart:

        st.markdown(
            "### Revenue vs Items Sold by Category"
        )


        volume_chart = (
            alt.Chart(category_volume)
            .mark_circle(
                opacity=0.78
            )
            .encode(
                x=alt.X(
                    "Items Sold:Q",
                    title="Items Sold",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),
                y=alt.Y(
                    "Revenue:Q",
                    title="Revenue (R$)",
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),
                color=alt.Color(
                    "Revenue:Q",
                    title="Revenue",
                    scale=alt.Scale(
                        range=[
                            "#8fc9ca",
                            PRIMARY_BLUE,
                        ]
                    ),
                    legend=None,
                ),
                size=alt.Size(
                    "Revenue:Q",
                    title="Revenue",
                    scale=alt.Scale(
                        range=[40, 900]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Category:N",
                        title="Category",
                    ),
                    alt.Tooltip(
                        "Revenue:Q",
                        title="Revenue",
                        format=",.2f",
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
                    alt.Tooltip(
                        "review_count:Q",
                        title="Review Count",
                        format=",d",
                    ),
                    alt.Tooltip(
                        "Average Selling Price:Q",
                        title="Average Selling Price",
                        format=",.2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        volume_labels = (
            alt.Chart(
                category_volume_highlights
            )
            .mark_text(
                dx=7,
                dy=-7,
                align="left",
                color=NAVY,
                fontSize=10,
            )
            .encode(
                x=alt.X(
                    "Items Sold:Q"
                ),
                y=alt.Y(
                    "Revenue:Q"
                ),
                text=alt.Text(
                    "Category:N"
                ),
            )
        )


        st.altair_chart(
            volume_chart + volume_labels,
            width="stretch",
        )


    # -----------------------------------------------------
    # TOP & BOTTOM RATED
    # -----------------------------------------------------

    with right_chart:

        st.markdown(
            "### Top & Bottom Rated Categories"
        )


        if not rating_extremes.empty:

            rating_chart = (
                alt.Chart(
                    rating_extremes
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "average_rating:Q",
                        title="Average Rating",
                        scale=alt.Scale(
                            domain=[0, 5]
                        ),
                    ),
                    y=alt.Y(
                        "Category:N",
                        title=None,
                        sort=alt.SortField(
                            "average_rating"
                        ),
                    ),
                    color=alt.Color(
                        "rating_group:N",
                        title=None,
                        scale=alt.Scale(
                            domain=[
                                "Top Rated",
                                "Bottom Rated",
                            ],
                            range=[
                                TEAL,
                                CORAL,
                            ],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Category:N",
                            title="Category",
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
                    ],
                )
                .properties(
                    height=CHART_HEIGHT
                )
            )


            st.altair_chart(
                rating_chart,
                width="stretch",
            )

        else:

            st.info(
                "Not enough review data "
                "for the selected filters."
            )


    # =====================================================
    # SALES VS RATING
    # =====================================================

    st.markdown(
        "### Sales vs Rating by Category"
    )


    if len(category_scatter := (
        category_sales
        .merge(
            ratings,
            how="inner",
            on="Category",
        )
    )):

        category_scatter = (
            category_scatter[
                category_scatter[
                    "review_count"
                ] >= MIN_REVIEWS
            ]
            .merge(
                category_items,
                how="left",
                on="Category",
            )
        )


    if category_scatter.empty:

        st.info(
            f"No categories have at least "
            f"{MIN_REVIEWS} mapped reviews "
            "under the selected filters."
        )

    else:

        min_revenue = max(
            float(
                category_scatter[
                    "Revenue"
                ].min()
            ),
            1.0,
        )


        max_revenue = float(
            category_scatter[
                "Revenue"
            ].max()
        )


        median_revenue = float(
            category_scatter[
                "Revenue"
            ].median()
        )


        median_rating = float(
            category_scatter[
                "average_rating"
            ].median()
        )


        revenue_domain = [
            min_revenue * 0.8,
            max_revenue * 1.2,
        ]


        quadrant_data = pd.DataFrame(
            [
                {
                    "x_start": revenue_domain[0],
                    "x_end": median_revenue,
                    "y_start": median_rating,
                    "y_end": 5,
                    "x_label": (
                        revenue_domain[0]
                        * median_revenue
                    ) ** 0.5,
                    "y_label": 4.65,
                    "quadrant":
                        "Niche Performers",
                },
                {
                    "x_start": median_revenue,
                    "x_end": revenue_domain[1],
                    "y_start": median_rating,
                    "y_end": 5,
                    "x_label": (
                        median_revenue
                        * revenue_domain[1]
                    ) ** 0.5,
                    "y_label": 4.65,
                    "quadrant":
                        "Strong Performers",
                },
                {
                    "x_start": revenue_domain[0],
                    "x_end": median_revenue,
                    "y_start": 0,
                    "y_end": median_rating,
                    "x_label": (
                        revenue_domain[0]
                        * median_revenue
                    ) ** 0.5,
                    "y_label": 0.35,
                    "quadrant":
                        "Low Priority",
                },
                {
                    "x_start": median_revenue,
                    "x_end": revenue_domain[1],
                    "y_start": 0,
                    "y_end": median_rating,
                    "x_label": (
                        median_revenue
                        * revenue_domain[1]
                    ) ** 0.5,
                    "y_label": 0.35,
                    "quadrant":
                        "Attention Needed",
                },
            ]
        )


        sales_x = alt.X(
            "Revenue:Q",
            title=(
                "Category Revenue "
                "(R$; logarithmic scale)"
            ),
            scale=alt.Scale(
                type="log",
                domain=revenue_domain,
            ),
            axis=alt.Axis(
                format="~s"
            ),
        )


        sales_y = alt.Y(
            "average_rating:Q",
            title="Average Rating",
            scale=alt.Scale(
                domain=[0, 5]
            ),
        )


        quadrant_background = (
            alt.Chart(quadrant_data)
            .mark_rect(
                opacity=0.045
            )
            .encode(
                x=alt.X(
                    "x_start:Q",
                    scale=alt.Scale(
                        type="log",
                        domain=revenue_domain,
                    ),
                ),
                x2="x_end:Q",
                y=alt.Y(
                    "y_start:Q",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),
                y2="y_end:Q",
                color=alt.Color(
                    "quadrant:N",
                    scale=alt.Scale(
                        domain=[
                            "Niche Performers",
                            "Strong Performers",
                            "Low Priority",
                            "Attention Needed",
                        ],
                        range=[
                            "#4f86c6",
                            TEAL,
                            "#94a3b8",
                            CORAL,
                        ],
                    ),
                    legend=None,
                ),
            )
        )


        quadrant_labels = (
            alt.Chart(quadrant_data)
            .mark_text(
                color="#526173",
                fontSize=12,
                fontWeight="bold",
                opacity=0.8,
            )
            .encode(
                x=alt.X(
                    "x_label:Q",
                    scale=alt.Scale(
                        type="log",
                        domain=revenue_domain,
                    ),
                ),
                y=alt.Y(
                    "y_label:Q",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),
                text=alt.Text(
                    "quadrant:N"
                ),
            )
        )


        revenue_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Revenue": [
                            median_revenue
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[5, 5],
            )
            .encode(
                x=alt.X(
                    "Revenue:Q",
                    scale=alt.Scale(
                        type="log",
                        domain=revenue_domain,
                    ),
                )
            )
        )


        rating_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "average_rating": [
                            median_rating
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[5, 5],
            )
            .encode(
                y=alt.Y(
                    "average_rating:Q",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                )
            )
        )


        scatter_chart = (
            alt.Chart(category_scatter)
            .mark_circle(
                color=TEAL,
                opacity=0.82,
            )
            .encode(
                x=sales_x,
                y=sales_y,
                size=alt.Size(
                    "review_count:Q",
                    title="Review Count",
                    scale=alt.Scale(
                        range=[40, 900]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Category:N",
                        title="Category",
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
                        "Items Sold:Q",
                        title="Items Sold",
                        format=",d",
                    ),
                ],
            )
            .properties(
                height=420
            )
        )


        st.altair_chart(
            quadrant_background
            + scatter_chart
            + revenue_rule
            + rating_rule
            + quadrant_labels,
            width="stretch",
        )


    # =====================================================
    # KEY INSIGHTS
    # =====================================================

    highest_rated = (
        category_rating_comparison.iloc[0]
        if len(category_rating_comparison)
        else None
    )


    lowest_rated = (
        ratings_with_enough_reviews
        .nsmallest(
            1,
            "average_rating",
        )
        .iloc[0]
        if len(ratings_with_enough_reviews)
        else None
    )


    if not category_scatter.empty:

        low_rating_threshold = float(
            category_scatter[
                "average_rating"
            ].quantile(0.25)
        )

        high_revenue_cutoff = float(
            category_scatter[
                "Revenue"
            ].quantile(0.75)
        )

        high_revenue_low_rating = (
            category_scatter[
                (
                    category_scatter[
                        "Revenue"
                    ]
                    >= high_revenue_cutoff
                )
                &
                (
                    category_scatter[
                        "average_rating"
                    ]
                    <= low_rating_threshold
                )
            ]
            .sort_values(
                "Revenue",
                ascending=False,
            )
        )

    else:

        high_revenue_low_rating = pd.DataFrame()


    st.markdown(
        "### Key Category Insights"
    )


    insights = [
        (
            f"{top_revenue_category['Category']} "
            f"had the highest category revenue at "
            f"{format_currency(top_revenue_category['Revenue'])}."
        ),

        (
            f"{top_items_category['Category']} "
            f"had the highest item volume with "
            f"{int(top_items_category['Items Sold']):,} "
            f"items sold. "
            f"The revenue and volume leaders "
            f"{'are the same category' if top_revenue_category['Category'] == top_items_category['Category'] else 'are different categories'}."
        ),
    ]


    if highest_rated is not None:

        insights.append(
            (
                f"Among categories with at least "
                f"{MIN_REVIEWS} mapped reviews, "
                f"{highest_rated['Category']} had "
                f"the highest average rating at "
                f"{highest_rated['average_rating']:.2f} "
                f"out of 5."
            )
        )


    if lowest_rated is not None:

        insights.append(
            (
                f"Among categories with at least "
                f"{MIN_REVIEWS} mapped reviews, "
                f"{lowest_rated['Category']} had "
                f"the lowest average rating at "
                f"{lowest_rated['average_rating']:.2f} "
                f"out of 5."
            )
        )


    if len(high_revenue_low_rating):

        low_category = (
            high_revenue_low_rating.iloc[0]
        )

        insights.append(
            (
                f"{low_category['Category']} "
                f"is in the upper quartile of "
                f"category revenue and has an "
                f"average rating of "
                f"{low_category['average_rating']:.2f} "
                f"out of 5."
            )
        )

    else:

        insights.append(
            "No category in the upper revenue "
            "quartile falls at or below the lower "
            "rating quartile."
        )


    st.markdown(
        "\n".join(
            f"- {insight}"
            for insight in insights[:5]
        )
    )


    # =====================================================
    # METHODOLOGY NOTE
    # =====================================================

    st.caption(
        "Category ratings are based on deduplicated "
        "review-to-order-category mapping and are "
        "not weighted by the number of products in "
        "an order. The Top/Bottom Rated comparison "
        f"uses a minimum threshold of {MIN_REVIEWS} "
        "mapped reviews."
    )