from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from analytics import get_average_selling_price, get_items_sold
from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master
from genai_assistant import (
    dashboard_layout,
    apply_dashboard_filters,
)

st.set_page_config(
    page_title="Product Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


@st.cache_data
def load_product_data():
    """Load cleaned datasets and build the existing transaction-level master."""
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


def shorten_product_id(product_id):
    product_id = str(product_id)
    return f"{product_id[:8]}...{product_id[-4:]}"


def get_product_rating_data(datasets):
    """Map reviews to deduplicated order-product combinations for ratings."""

    order_products = datasets["order_items"][
        ["order_id", "product_id"]
    ].drop_duplicates(
        subset=["order_id", "product_id"]
    )

    reviews = datasets["order_reviews"][
        ["order_id", "review_score"]
    ].drop_duplicates(
        subset=["order_id", "review_score"]
    )

    review_products = reviews.merge(
        order_products,
        how="inner",
        on="order_id",
        validate="many_to_many",
    ).drop_duplicates(
        subset=["order_id", "product_id", "review_score"]
    )

    return (
        review_products
        .groupby(
            "product_id",
            as_index=False,
        )
        .agg(
            average_rating=("review_score", "mean"),
            review_count=("review_score", "size"),
        )
    )


def get_product_metrics(datasets, sales_master):
    """Return sold-product metrics and separately mapped product ratings."""

    product_metrics = (
        sales_master.assign(
            category=sales_master[
                "product_category_name_english"
            ].fillna("Unknown")
        )
        .groupby(
            "product_id",
            as_index=False,
        )
        .agg(
            Revenue=("price", "sum"),
            **{
                "Items Sold": ("price", "size"),
                "Average Selling Price": ("price", "mean"),
            },
            Category=("category", "first"),
        )
        .rename(
            columns={
                "product_id": "Product ID"
            }
        )
    )

    product_ratings = (
        get_product_rating_data(datasets)
        .rename(
            columns={
                "product_id": "Product ID"
            }
        )
    )

    return product_metrics.merge(
        product_ratings,
        how="left",
        on="Product ID",
    )


# =========================================================
# LOAD DATA
# =========================================================

datasets, sales_master = load_product_data()


# =========================================================
# DASHBOARD
# =========================================================

with dashboard_layout(
    datasets,
    sales_master,
):

    # =====================================================
    # PAGE FILTERS
    # =====================================================

    # Keep the Product page's searchable Customer State selector.

    customer_states = (
        datasets["customers"]["customer_state"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_customer_state = st.selectbox(
        "Search / Select Customer State",
        ["All States"] + customer_states,
        key="product_customer_state_filter",
    )


    # Read the common Dashboard Filters date range.

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
        customer_state=selected_customer_state,
        date_range=selected_date_range,
    )

    # =====================================================
    # PRODUCT METRICS
    # =====================================================

    product_metrics = get_product_metrics(
        filtered_datasets,
        filtered_sales_master,
    )


    product_metrics["Product Label"] = (
        product_metrics["Product ID"].map(
            shorten_product_id
        )
    )


    total_products = int(
        product_metrics[
            "Product ID"
        ].nunique()
    )


    items_sold = get_items_sold(
        filtered_sales_master
    )


    average_selling_price = (
        get_average_selling_price(
            filtered_sales_master
        )
        if len(filtered_sales_master)
        else 0.0
    )


    average_product_rating = (
        float(
            product_metrics[
                "average_rating"
            ].mean()
        )
        if len(product_metrics)
        else 0.0
    )


    top_revenue_product = (
        product_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .iloc[0]
        if len(product_metrics)
        else {
            "Product ID": "N/A",
            "Revenue": 0,
        }
    )


    top_volume_product = (
        product_metrics
        .sort_values(
            "Items Sold",
            ascending=False,
        )
        .iloc[0]
        if len(product_metrics)
        else {
            "Product ID": "N/A",
            "Items Sold": 0,
        }
    )


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
        </style>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.header(
            "Olist Business Intelligence"
        )

        st.subheader(
            "Product Analysis"
        )

        st.caption(
            "Data Source: Olist Brazilian E-Commerce Dataset"
        )


    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title("Product Analysis")

    st.caption(
        "Which products drive demand, revenue, pricing, and customer satisfaction?"
    )


    # =====================================================
    # FILTER STATUS
    # =====================================================

    if selected_customer_state == "All States":

        st.caption(
            "Showing product analysis for all states."
        )

    else:

        st.caption(
            f"Showing product analysis for "
            f"**{selected_customer_state}**."
        )


    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [
        (
            "Total Products",
            f"{total_products:,}",
        ),

        (
            "Items Sold",
            f"{items_sold:,}",
        ),

        (
            "Most Revenue Product",
            shorten_product_id(
                top_revenue_product[
                    "Product ID"
                ]
            ),
        ),

        (
            "Most Sold Product",
            shorten_product_id(
                top_volume_product[
                    "Product ID"
                ]
            ),
        ),

        (
            "Average Selling Price",
            format_currency(
                average_selling_price
            ),
        ),

        (
            "Average Product Rating",
            f"{average_product_rating:.2f} / 5",
        ),
    ]


    kpi_columns = st.columns(6)


    for column, (label, value) in zip(
        kpi_columns,
        kpis,
    ):

        with column:

            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


    # =====================================================
    # PARTIAL MONTH NOTE
    # =====================================================

    if len(filtered_sales_master):

        max_purchase_timestamp = (
            filtered_sales_master[
                "order_purchase_timestamp"
            ].max()
        )

        final_month_end = (
            max_purchase_timestamp
            .to_period("M")
            .end_time
        )

        if max_purchase_timestamp < final_month_end:

            st.caption(
                "Note: The final month contains "
                "partial-period data in the dataset."
            )


    CHART_HEIGHT = 310

    PRIMARY_BLUE = "#1f5a94"
    TEAL = "#168a8a"
    CORAL = "#e07a5f"
    NAVY = "#14213d"


    # =====================================================
    # TOP PRODUCTS
    # =====================================================

    revenue_top10 = (
        product_metrics
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(10)
        .copy()
    )


    volume_top10 = (
        product_metrics
        .sort_values(
            "Items Sold",
            ascending=False,
        )
        .head(10)
        .copy()
    )


    revenue_top10["Revenue Label"] = (
        revenue_top10[
            "Revenue"
        ].map(
            format_compact_currency
        )
    )


    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    with left_chart:

        st.markdown(
            "### Top 10 Products by Revenue"
        )

        revenue_chart = (
            alt.Chart(
                revenue_top10
            )
            .mark_bar(
                color=PRIMARY_BLUE
            )
            .encode(

                x=alt.X(
                    "Revenue:Q",
                    title="Revenue (R$)",
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),

                y=alt.Y(
                    "Product Label:N",
                    title="Product ID",
                    sort="-x",
                ),

                tooltip=[

                    alt.Tooltip(
                        "Product ID:N",
                        title="Product ID",
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
                        "Average Selling Price:Q",
                        title="Average Selling Price",
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
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        revenue_labels = (
            alt.Chart(
                revenue_top10
            )
            .mark_text(
                dx=6,
                align="left",
                color=NAVY,
                fontSize=10,
            )
            .encode(

                x=alt.X(
                    "Revenue:Q"
                ),

                y=alt.Y(
                    "Product Label:N",
                    sort="-x",
                ),

                text=alt.Text(
                    "Revenue Label:N"
                ),
            )
        )


        st.altair_chart(
            revenue_chart
            + revenue_labels,
            width="stretch",
        )


    with right_chart:

        st.markdown(
            "### Top 10 Products by Items Sold"
        )


        volume_chart = (
            alt.Chart(
                volume_top10
            )
            .mark_bar(
                color=TEAL
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
                    "Product Label:N",
                    title="Product ID",
                    sort="-x",
                ),

                tooltip=[

                    alt.Tooltip(
                        "Product ID:N",
                        title="Product ID",
                    ),

                    alt.Tooltip(
                        "Items Sold:Q",
                        title="Items Sold",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Revenue:Q",
                        title="Revenue",
                        format=",.2f",
                    ),

                    alt.Tooltip(
                        "Average Selling Price:Q",
                        title="Average Selling Price",
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
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        st.altair_chart(
            volume_chart,
            width="stretch",
        )


    # =====================================================
    # PRICE DISTRIBUTION
    # =====================================================

    price_bands = [
        "R$0-50",
        "R$50-100",
        "R$100-250",
        "R$250-500",
        "R$500-1,000",
        "R$1,000+",
    ]


    price_distribution = (
        pd.cut(
            product_metrics[
                "Average Selling Price"
            ],

            bins=[
                0,
                50,
                100,
                250,
                500,
                1_000,
                float("inf"),
            ],

            labels=price_bands,

            right=False,

            include_lowest=True,
        )
        .value_counts()
        .reindex(
            price_bands,
            fill_value=0,
        )
        .rename(
            "Number of Products"
        )
        .rename_axis(
            "Price Band"
        )
        .reset_index()
    )


    if (
        price_distribution[
            "Number of Products"
        ].sum()
        > 0
    ):

        price_distribution[
            "Percentage"
        ] = (
            price_distribution[
                "Number of Products"
            ]
            / price_distribution[
                "Number of Products"
            ].sum()
            * 100
        )

    else:

        price_distribution[
            "Percentage"
        ] = 0.0


    # =====================================================
    # PRODUCT BUBBLE CHART
    # =====================================================

    PRODUCT_REVIEW_THRESHOLD = 5


    bubble_products = product_metrics[
        product_metrics[
            "review_count"
        ].fillna(0)
        >= PRODUCT_REVIEW_THRESHOLD
    ].copy()


    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    with left_chart:

        st.markdown(
            "### Revenue vs Items Sold by Product"
        )


        if len(bubble_products):

            bubble_min_items = float(
                bubble_products[
                    "Items Sold"
                ].min()
            )

            bubble_max_items = float(
                bubble_products[
                    "Items Sold"
                ].max()
            )

            bubble_min_revenue = float(
                bubble_products[
                    "Revenue"
                ].min()
            )

            bubble_max_revenue = float(
                bubble_products[
                    "Revenue"
                ].max()
            )

            bubble_median_items = float(
                bubble_products[
                    "Items Sold"
                ].median()
            )

            bubble_median_revenue = float(
                bubble_products[
                    "Revenue"
                ].median()
            )

            bubble_high_label = (
                bubble_min_revenue
                + (
                    bubble_max_revenue
                    - bubble_min_revenue
                )
                * 0.92
            )

            bubble_low_label = (
                bubble_min_revenue
                + (
                    bubble_max_revenue
                    - bubble_min_revenue
                )
                * 0.08
            )


            bubble_quadrants = pd.DataFrame(
                [
                    {
                        "x_start": bubble_min_items,
                        "x_end": bubble_median_items,
                        "y_start": bubble_median_revenue,
                        "y_end": bubble_max_revenue,
                        "x_label": (
                            bubble_min_items
                            + bubble_median_items
                        )
                        / 2,
                        "y_label": bubble_high_label,
                        "quadrant": "Low Volume / High Revenue",
                    },

                    {
                        "x_start": bubble_median_items,
                        "x_end": bubble_max_items,
                        "y_start": bubble_median_revenue,
                        "y_end": bubble_max_revenue,
                        "x_label": (
                            bubble_median_items
                            + bubble_max_items
                        )
                        / 2,
                        "y_label": bubble_high_label,
                        "quadrant": "High Volume / High Revenue",
                    },

                    {
                        "x_start": bubble_min_items,
                        "x_end": bubble_median_items,
                        "y_start": bubble_min_revenue,
                        "y_end": bubble_median_revenue,
                        "x_label": (
                            bubble_min_items
                            + bubble_median_items
                        )
                        / 2,
                        "y_label": bubble_low_label,
                        "quadrant": "Low Volume / Low Revenue",
                    },

                    {
                        "x_start": bubble_median_items,
                        "x_end": bubble_max_items,
                        "y_start": bubble_min_revenue,
                        "y_end": bubble_median_revenue,
                        "x_label": (
                            bubble_median_items
                            + bubble_max_items
                        )
                        / 2,
                        "y_label": bubble_low_label,
                        "quadrant": "High Volume / Low Revenue",
                    },
                ]
            )


            bubble_chart = (
                alt.Chart(
                    bubble_products
                )
                .mark_circle(
                    opacity=0.72
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
                        "Category:N",
                        title="Product Category",
                        legend=None,
                    ),

                    size=alt.Size(
                        "Revenue:Q",
                        title="Revenue",
                        scale=alt.Scale(
                            range=[30, 850]
                        ),
                    ),

                    tooltip=[

                        alt.Tooltip(
                            "Product ID:N",
                            title="Product ID",
                        ),

                        alt.Tooltip(
                            "Category:N",
                            title="Product Category",
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
                            "Average Selling Price:Q",
                            title="Average Selling Price",
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
                    ],
                )
                .properties(
                    height=CHART_HEIGHT
                )
            )


            bubble_background = (
                alt.Chart(
                    bubble_quadrants
                )
                .mark_rect(
                    opacity=0.04
                )
                .encode(

                    x=alt.X(
                        "x_start:Q"
                    ),

                    x2="x_end:Q",

                    y=alt.Y(
                        "y_start:Q"
                    ),

                    y2="y_end:Q",

                    color=alt.Color(
                        "quadrant:N",
                        legend=None,
                    ),
                )
            )


            bubble_labels = (
                alt.Chart(
                    bubble_quadrants
                )
                .mark_text(
                    color="#526173",
                    fontSize=10,
                    fontWeight="bold",
                    opacity=0.8,
                )
                .encode(

                    x=alt.X(
                        "x_label:Q"
                    ),

                    y=alt.Y(
                        "y_label:Q"
                    ),

                    text=alt.Text(
                        "quadrant:N"
                    ),
                )
            )


            bubble_items_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Items Sold": [
                                bubble_median_items
                            ]
                        }
                    )
                )
                .mark_rule(
                    color="#526173",
                    strokeDash=[5, 5],
                )
                .encode(
                    x="Items Sold:Q"
                )
            )


            bubble_revenue_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Revenue": [
                                bubble_median_revenue
                            ]
                        }
                    )
                )
                .mark_rule(
                    color="#526173",
                    strokeDash=[5, 5],
                )
                .encode(
                    y="Revenue:Q"
                )
            )


            st.altair_chart(
                bubble_background
                + bubble_chart
                + bubble_items_rule
                + bubble_revenue_rule
                + bubble_labels,
                width="stretch",
            )

        else:

            st.info(
                "Not enough reviewed products are available "
                "for this state to display the bubble chart."
            )


    with right_chart:

        st.markdown(
            "### Product Price Distribution"
        )


        price_chart = (
            alt.Chart(
                price_distribution
            )
            .mark_bar(
                color=TEAL
            )
            .encode(

                x=alt.X(
                    "Price Band:N",
                    title="Price Band",
                    sort=price_bands,
                ),

                y=alt.Y(
                    "Number of Products:Q",
                    title="Number of Products",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Price Band:N",
                        title="Price Band",
                    ),

                    alt.Tooltip(
                        "Number of Products:Q",
                        title="Number of Products",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Percentage:Q",
                        title="Percentage of Products",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        st.altair_chart(
            price_chart,
            width="stretch",
        )


    # =====================================================
    # RATED PRODUCTS
    # =====================================================

    rated_products = product_metrics[
        product_metrics[
            "review_count"
        ].fillna(0)
        >= 20
    ].copy()


    rated_products["Product Label"] = (
        rated_products[
            "Product ID"
        ].map(
            shorten_product_id
        )
    )


    top_rated_products = (
        rated_products
        .nlargest(
            10,
            "average_rating",
        )
        .copy()
    )


    bottom_rated_products = (
        rated_products
        .nsmallest(
            10,
            "average_rating",
        )
        .copy()
    )


    selected_products = (
        rated_products
        .nlargest(
            20,
            "Revenue",
        )
        .copy()
    )


    rating_tooltips = [

        alt.Tooltip(
            "Product ID:N",
            title="Product ID",
        ),

        alt.Tooltip(
            "Category:N",
            title="Product Category",
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
            "Average Selling Price:Q",
            title="Average Selling Price",
            format=",.2f",
        ),
    ]


    def rating_lollipop(
        dataframe,
        color,
        sort_order,
    ):

        y_encoding = alt.Y(
            "Product Label:N",
            title="Product ID",
            sort=alt.SortField(
                field="average_rating",
                order=sort_order,
            ),
        )


        stems = (
            alt.Chart(dataframe)
            .mark_rule(
                color=color,
                strokeWidth=2,
            )
            .encode(

                x=alt.X(
                    "average_rating:Q",
                    title="Average Rating",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),

                x2=alt.value(0),

                y=y_encoding,
            )
        )


        dots = (
            alt.Chart(dataframe)
            .mark_circle(
                color=color,
                size=90,
            )
            .encode(

                x=alt.X(
                    "average_rating:Q",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),

                y=alt.Y(
                    "Product Label:N",
                    sort=alt.SortField(
                        field="average_rating",
                        order=sort_order,
                    ),
                ),

                tooltip=rating_tooltips,
            )
        )


        return (
            stems + dots
        ).properties(
            height=CHART_HEIGHT
        )


    st.markdown(
        "### Top & Bottom Rated Products"
    )


    left_rating_chart, right_rating_chart = (
        st.columns(
            2,
            gap="large",
        )
    )


    with left_rating_chart:

        st.markdown(
            "#### Top 10 Rated Products"
        )

        if len(top_rated_products):

            st.altair_chart(
                rating_lollipop(
                    top_rated_products,
                    TEAL,
                    "descending",
                ),
                width="stretch",
            )

        else:

            st.info(
                "Not enough reviewed products are "
                "available for this state."
            )


    with right_rating_chart:

        st.markdown(
            "#### Bottom 10 Rated Products"
        )

        if len(bottom_rated_products):

            st.altair_chart(
                rating_lollipop(
                    bottom_rated_products,
                    CORAL,
                    "ascending",
                ),
                width="stretch",
            )

        else:

            st.info(
                "Not enough reviewed products are "
                "available for this state."
            )


    st.caption(
        "Ratings shown for products with at least "
        "20 reviews. Product IDs are used because "
        "the Olist dataset does not provide product names."
    )


    # =====================================================
    # REVENUE VS RATING — SELECTED PRODUCTS
    # =====================================================

    st.markdown(
        "### Revenue vs Rating - Selected Products"
    )


    if len(selected_products):

        selected_median_revenue = float(
            selected_products[
                "Revenue"
            ].median()
        )


        selected_median_rating = float(
            selected_products[
                "average_rating"
            ].median()
        )


        selected_chart = (
            alt.Chart(
                selected_products
            )
            .mark_circle(
                color=TEAL,
                opacity=0.8,
            )
            .encode(

                x=alt.X(
                    "average_rating:Q",
                    title="Average Rating",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),

                y=alt.Y(
                    "Revenue:Q",
                    title="Revenue (R$)",
                    scale=alt.Scale(
                        zero=True
                    ),
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),

                size=alt.Size(
                    "review_count:Q",
                    title="Review Count",
                    scale=alt.Scale(
                        range=[40, 850]
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Product ID:N",
                        title="Product ID",
                    ),

                    alt.Tooltip(
                        "Category:N",
                        title="Product Category",
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
                        "Average Selling Price:Q",
                        title="Average Selling Price",
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
                ],
            )
            .properties(
                height=360
            )
        )


        selected_rating_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "average_rating": [
                            selected_median_rating
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[5, 5],
            )
            .encode(
                x="average_rating:Q"
            )
        )


        selected_revenue_rule = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Revenue": [
                            selected_median_revenue
                        ]
                    }
                )
            )
            .mark_rule(
                color="#526173",
                strokeDash=[5, 5],
            )
            .encode(
                y="Revenue:Q"
            )
        )


        selected_labels = (
            alt.Chart(
                selected_products
            )
            .mark_text(
                dy=-8,
                color=NAVY,
                fontSize=9,
            )
            .encode(

                x=alt.X(
                    "average_rating:Q",
                    scale=alt.Scale(
                        domain=[0, 5]
                    ),
                ),

                y=alt.Y(
                    "Revenue:Q",
                    scale=alt.Scale(
                        zero=True
                    ),
                ),

                text=alt.Text(
                    "Product Label:N"
                ),
            )
        )


        st.altair_chart(
            selected_chart
            + selected_rating_rule
            + selected_revenue_rule
            + selected_labels,
            width="stretch",
        )

    else:

        st.info(
            "No products with sufficient review data "
            "are available for this state."
        )


    # =====================================================
    # KEY INSIGHTS
    # =====================================================

    if len(product_metrics):

        highest_revenue_product = (
            product_metrics
            .sort_values(
                "Revenue",
                ascending=False,
            )
            .iloc[0]
        )


        highest_volume_product = (
            product_metrics
            .sort_values(
                "Items Sold",
                ascending=False,
            )
            .iloc[0]
        )

    else:

        highest_revenue_product = {
            "Product ID": "N/A",
            "Revenue": 0,
        }

        highest_volume_product = {
            "Product ID": "N/A",
            "Items Sold": 0,
        }


    if len(rated_products):

        highest_rated_product = (
            rated_products
            .sort_values(
                "average_rating",
                ascending=False,
            )
            .iloc[0]
        )


        lowest_rated_product = (
            rated_products
            .sort_values(
                "average_rating"
            )
            .iloc[0]
        )


        overall_median_rating = float(
            rated_products[
                "average_rating"
            ].median()
        )

    else:

        highest_rated_product = {
            "Product ID": "N/A",
            "average_rating": 0,
        }

        lowest_rated_product = {
            "Product ID": "N/A",
            "average_rating": 0,
        }

        overall_median_rating = 0.0


    if len(price_distribution):

        dominant_price_band = (
            price_distribution
            .sort_values(
                "Number of Products",
                ascending=False,
            )
            .iloc[0]
        )

    else:

        dominant_price_band = {
            "Price Band": "N/A",
            "Percentage": 0,
        }


    if len(selected_products):

        selected_above_median = int(
            selected_products[
                "average_rating"
            ]
            .gt(
                overall_median_rating
            )
            .sum()
        )

        selected_below_or_equal_median = (
            len(selected_products)
            - selected_above_median
        )

    else:

        selected_above_median = 0
        selected_below_or_equal_median = 0


    st.markdown(
        "### Key Product Insights"
    )


    insights = [

        (
            f"{shorten_product_id(highest_revenue_product['Product ID'])} "
            f"generated the highest product revenue at "
            f"{format_currency(highest_revenue_product['Revenue'])}."
        ),

        (
            f"{shorten_product_id(highest_volume_product['Product ID'])} "
            f"had the highest item volume with "
            f"{int(highest_volume_product['Items Sold']):,} "
            f"items sold."
        ),

        (
            f"Among products with at least 20 reviews, "
            f"{shorten_product_id(highest_rated_product['Product ID'])} "
            f"was highest rated at "
            f"{highest_rated_product['average_rating']:.2f}, "
            f"while "
            f"{shorten_product_id(lowest_rated_product['Product ID'])} "
            f"was lowest rated at "
            f"{lowest_rated_product['average_rating']:.2f}."
        ),

        (
            f"The {dominant_price_band['Price Band']} "
            f"band contained the most products, representing "
            f"{dominant_price_band['Percentage']:.2f}% "
            f"of sold products."
        ),

        (
            f"Of the 20 highest-revenue products with at "
            f"least 20 reviews, {selected_above_median} "
            f"had ratings above the overall median of "
            f"{overall_median_rating:.2f}, while "
            f"{selected_below_or_equal_median} were at "
            f"or below it."
        ),
    ]


    st.markdown(
        "\n".join(
            f"- {insight}"
            for insight in insights[:5]
        )
    )


    st.caption(
        "Product ratings are based on deduplicated "
        "review-to-order-product mapping and are "
        "not weighted by the number of order-item rows."
    )