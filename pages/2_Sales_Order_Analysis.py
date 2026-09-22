from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from analytics import (
    get_average_order_value,
    get_average_selling_price,
    get_items_sold,
    get_total_order_value,
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
    page_title="Sales & Order Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


# =========================================================
# LOAD DATA
# =========================================================


def load_sales_data():
    """Load cleaned datasets and build the existing sales master."""

    datasets = clean_datasets(
        load_datasets(PROJECT_ROOT / "data")
    )

    return datasets, build_sales_master(datasets)


# =========================================================
# FORMATTING FUNCTIONS
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
# MONTHLY SALES / ORDER CALCULATION
# =========================================================

@st.cache_data
def get_monthly_sales_orders(master_df):
    """Return monthly sales, orders, order value, and average order value."""

    if master_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "Sales",
                "Orders",
                "Order_Value",
                "Average Order Value",
            ]
        )

    monthly = (
        master_df.assign(
            month=master_df[
                "order_purchase_timestamp"
            ]
            .dt.to_period("M")
            .dt.to_timestamp(),

            order_value=(
                master_df["price"]
                + master_df["freight_value"]
            ),
        )
        .groupby(
            "month",
            as_index=False,
        )
        .agg(
            Sales=("price", "sum"),
            Orders=("order_id", "nunique"),
            Order_Value=("order_value", "sum"),
        )
    )

    monthly["Average Order Value"] = (
        monthly["Order_Value"]
        / monthly["Orders"].replace(0, pd.NA)
    )

    monthly["Average Order Value"] = (
        monthly["Average Order Value"]
        .fillna(0)
    )

    return monthly.sort_values("month")


# =========================================================
# STATE SALES
# =========================================================


def get_state_sales(datasets, master_df):
    """Return item-price sales by customer state."""

    if master_df.empty:
        return pd.DataFrame(
            columns=[
                "State",
                "Sales",
            ]
        )

    order_customers = datasets["orders"][
        [
            "order_id",
            "customer_id",
        ]
    ].merge(
        datasets["customers"][
            [
                "customer_id",
                "customer_state",
            ]
        ],
        how="left",
        on="customer_id",
        validate="many_to_one",
    )

    item_states = master_df[
        [
            "order_id",
            "price",
        ]
    ].merge(
        order_customers[
            [
                "order_id",
                "customer_state",
            ]
        ],
        how="left",
        on="order_id",
        validate="many_to_one",
    )

    return (
        item_states
        .dropna(
            subset=[
                "customer_state"
            ]
        )
        .groupby(
            "customer_state",
            as_index=False,
        )["price"]
        .sum()
        .rename(
            columns={
                "customer_state": "State",
                "price": "Sales",
            }
        )
        .sort_values(
            "Sales",
            ascending=False,
        )
    )


# =========================================================
# ORDER STATUS
# =========================================================


def get_order_status_distribution(datasets):
    """Return unique order counts and percentages by order status."""

    if datasets["orders"].empty:
        return pd.DataFrame(
            columns=[
                "Status",
                "Orders",
                "Percentage",
            ]
        )

    result = (
        datasets["orders"]
        .groupby(
            "order_status",
            as_index=False,
        )["order_id"]
        .nunique()
        .rename(
            columns={
                "order_status": "Status",
                "order_id": "Orders",
            }
        )
        .sort_values(
            "Orders",
            ascending=False,
        )
    )

    total_orders = result["Orders"].sum()

    if total_orders > 0:
        result["Percentage"] = (
            result["Orders"]
            / total_orders
            * 100
        )
    else:
        result["Percentage"] = 0

    return result


# =========================================================
# ORDER VALUES
# =========================================================


def get_order_values(master_df):
    """Aggregate item price plus freight to one total value per order."""

    if master_df.empty:
        return pd.DataFrame(
            columns=[
                "order_id",
                "Order_Value",
            ]
        )

    return (
        master_df
        .assign(
            Order_Value=(
                master_df["price"]
                + master_df["freight_value"]
            )
        )
        .groupby(
            "order_id",
            as_index=False,
        )["Order_Value"]
        .sum()
    )


# =========================================================
# LOAD DATA
# =========================================================

datasets, sales_master = load_sales_data()


# =========================================================
# APPLY COMMON DASHBOARD FILTERS
# =========================================================
#
# IMPORTANT:
# Customer State and Order Date Range are now controlled
# from the common sidebar in genai_assistant.py.
#
# No separate Customer State filter is created on this page.
# =========================================================

filtered_datasets, filtered_sales_master = (
    apply_dashboard_filters(
        datasets,
        sales_master,
    )
)


# =========================================================
# RECALCULATE SALES & ORDER METRICS
# =========================================================

monthly = get_monthly_sales_orders(
    filtered_sales_master
)


# =========================================================
# TOP PRODUCTS
# =========================================================

if filtered_sales_master.empty:

    top_products = pd.DataFrame(
        columns=[
            "Product ID",
            "Items Sold",
        ]
    )

else:

    top_products = (
        filtered_sales_master
        .groupby(
            "product_id",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "Items Sold",
                "product_id": "Product ID",
            }
        )
        .sort_values(
            "Items Sold",
            ascending=False,
        )
        .head(10)
    )


# =========================================================
# STATE SALES
# =========================================================

state_sales = (
    get_state_sales(
        filtered_datasets,
        filtered_sales_master,
    )
    .head(10)
)


# =========================================================
# ORDER STATUS
# =========================================================

status_distribution = (
    get_order_status_distribution(
        filtered_datasets
    )
)


# =========================================================
# ORDER VALUES
# =========================================================

order_values = get_order_values(
    filtered_sales_master
)


# =========================================================
# KPIs
# =========================================================

total_orders = get_total_orders(
    filtered_sales_master
)

total_sales = get_total_sales(
    filtered_sales_master
)

total_order_value = get_total_order_value(
    filtered_sales_master
)

average_order_value = get_average_order_value(
    filtered_sales_master
)

items_sold = get_items_sold(
    filtered_sales_master
)

average_selling_price = get_average_selling_price(
    filtered_sales_master
)


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
            "Sales & Order Analysis"
        )

        st.caption(
            "Data Source: Olist Brazilian E-Commerce Dataset"
        )


    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title(
        "Sales & Order Analysis"
    )

    st.caption(
        "Track sales performance, order volume, products, geography, and order status."
    )


    # =====================================================
    # FILTER STATUS
    # =====================================================

    selected_customer_state = st.session_state.get(
        "customer_state_filter",
        "All States",
    )

    if selected_customer_state == "All States":

        st.caption(
            "Showing sales and order analysis for all states."
        )

    else:

        st.caption(
            f"Showing sales and order analysis for "
            f"**{selected_customer_state}**."
        )


    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [

        (
            "Total Orders",
            f"{total_orders:,}",
        ),

        (
            "Total Sales",
            format_compact_currency(
                total_sales
            ),
        ),

        (
            "Total Order Value",
            format_compact_currency(
                total_order_value
            ),
        ),

        (
            "Average Order Value",
            format_currency(
                average_order_value
            ),
        ),

        (
            "Items Sold",
            f"{items_sold:,}",
        ),

        (
            "Average Selling Price",
            format_currency(
                average_selling_price
            ),
        ),

    ]


    kpi_columns = st.columns(6)


    for column, (
        label,
        value,
    ) in zip(
        kpi_columns,
        kpis,
    ):

        with column:

            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


    # =====================================================
    # CHART SETTINGS
    # =====================================================

    CHART_HEIGHT = 300

    PRIMARY_BLUE = "#1f5a94"

    TEAL = "#168a8a"


    # =====================================================
    # MONTHLY SALES TREND
    # =====================================================

    st.markdown(
        "### Monthly Sales Trend"
    )

    st.caption(
        "Monthly sales based on order item price"
    )


    sales_chart = (

        alt.Chart(monthly)

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
                axis=alt.Axis(
                    format="%b %Y"
                ),
            ),

            y=alt.Y(
                "Sales:Q",
                title="Sales (R$)",
                scale=alt.Scale(
                    zero=False
                ),
                axis=alt.Axis(
                    format="~s"
                ),
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

        .properties(
            height=CHART_HEIGHT
        )

    )


    st.altair_chart(
        sales_chart,
        width="stretch",
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


    # =====================================================
    # MONTHLY ORDER TREND + SALES VS ORDER VOLUME
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    # =====================================================
    # MONTHLY ORDER TREND
    # =====================================================

    with left_chart:

        st.markdown(
            "### Monthly Order Trend"
        )


        order_chart = (

            alt.Chart(monthly)

            .mark_line(
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=55,
                    color=TEAL,
                ),
                color=TEAL,
            )

            .encode(

                x=alt.X(
                    "month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y"
                    ),
                ),

                y=alt.Y(
                    "Orders:Q",
                    title="Unique orders",
                    scale=alt.Scale(
                        zero=False
                    ),
                    axis=alt.Axis(
                        format=",d"
                    ),
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

            .properties(
                height=CHART_HEIGHT
            )

        )


        st.altair_chart(
            order_chart,
            width="stretch",
        )


    # =====================================================
    # SALES VS ORDER VOLUME
    # =====================================================

    with right_chart:

        st.markdown(
            "### Sales vs Order Volume"
        )


        sales_bars = (

            alt.Chart(monthly)

            .mark_bar(
                color=PRIMARY_BLUE,
                opacity=0.75,
            )

            .encode(

                x=alt.X(
                    "month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y"
                    ),
                ),

                y=alt.Y(
                    "Sales:Q",
                    title="Sales (R$)",
                    scale=alt.Scale(
                        zero=True
                    ),
                    axis=alt.Axis(
                        format="~s"
                    ),
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

                    alt.Tooltip(
                        "Orders:Q",
                        title="Orders",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Average Order Value:Q",
                        title="Average Order Value",
                        format=",.2f",
                    ),

                ],

            )

        )


        order_line = (

            alt.Chart(monthly)

            .mark_line(
                color=TEAL,
                point=True,
            )

            .encode(

                x=alt.X(
                    "month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y"
                    ),
                ),

                y=alt.Y(
                    "Orders:Q",
                    title="Orders",
                    scale=alt.Scale(
                        zero=True
                    ),
                    axis=alt.Axis(
                        orient="right",
                        format=",d"
                    ),
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

                    alt.Tooltip(
                        "Orders:Q",
                        title="Orders",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Average Order Value:Q",
                        title="Average Order Value",
                        format=",.2f",
                    ),

                ],

            )

        )


        sales_volume_chart = (

            alt.layer(
                sales_bars,
                order_line,
            )

            .resolve_scale(
                y="independent"
            )

            .properties(
                height=CHART_HEIGHT
            )

        )


        st.altair_chart(
            sales_volume_chart,
            width="stretch",
        )


    # =====================================================
    # TOP PRODUCTS + STATE SALES
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    # =====================================================
    # TOP PRODUCTS
    # =====================================================

    with left_chart:

        st.markdown(
            "### Top 10 Most Ordered Products"
        )


        product_chart = (

            alt.Chart(top_products)

            .mark_bar(
                color=PRIMARY_BLUE
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
                    "Product ID:N",
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

                ],

            )

            .properties(
                height=CHART_HEIGHT
            )

        )


        st.altair_chart(
            product_chart,
            width="stretch",
        )


    # =====================================================
    # SALES BY CUSTOMER STATE
    # =====================================================

    with right_chart:

        st.markdown(
            "### Sales by Customer State"
        )


        state_chart = (

            alt.Chart(state_sales)

            .mark_bar(
                color=TEAL
            )

            .encode(

                x=alt.X(
                    "Sales:Q",
                    title="Sales (R$)",
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),

                y=alt.Y(
                    "State:N",
                    title="State",
                    sort="-x",
                ),

                tooltip=[

                    alt.Tooltip(
                        "State:N",
                        title="State",
                    ),

                    alt.Tooltip(
                        "Sales:Q",
                        title="Sales",
                        format=",.2f",
                    ),

                ],

            )

            .properties(
                height=CHART_HEIGHT
            )

        )


        st.altair_chart(
            state_chart,
            width="stretch",
        )


    # =====================================================
    # ORDER STATUS + ORDER VALUE
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )


    # =====================================================
    # ORDER STATUS
    # =====================================================

    with left_chart:

        st.markdown(
            "### Order Status Distribution"
        )


        status_chart = (

            alt.Chart(
                status_distribution
            )

            .mark_bar(
                color=PRIMARY_BLUE
            )

            .encode(

                x=alt.X(
                    "Orders:Q",
                    title="Unique orders",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                y=alt.Y(
                    "Status:N",
                    title="Order status",
                    sort="-x",
                ),

                tooltip=[

                    alt.Tooltip(
                        "Status:N",
                        title="Status",
                    ),

                    alt.Tooltip(
                        "Orders:Q",
                        title="Unique orders",
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
            status_chart,
            width="stretch",
        )


    # =====================================================
    # ORDER VALUE DISTRIBUTION
    # =====================================================

    with right_chart:

        st.markdown(
            "### Order Value Distribution"
        )


        order_value_ranges = [

            "R$0-100",
            "R$100-250",
            "R$250-500",
            "R$500-1,000",
            "R$1,000-2,500",
            "R$2,500+",

        ]


        if order_values.empty:

            order_value_bands = pd.DataFrame(
                {
                    "Order Value Range":
                        order_value_ranges,
                    "Orders": [0] * 6,
                    "Percentage": [0.0] * 6,
                }
            )

        else:

            order_value_bands = (

                pd.cut(

                    order_values[
                        "Order_Value"
                    ],

                    bins=[
                        0,
                        100,
                        250,
                        500,
                        1_000,
                        2_500,
                        float("inf"),
                    ],

                    labels=order_value_ranges,

                    right=False,

                    include_lowest=True,

                )

                .value_counts()

                .reindex(
                    order_value_ranges,
                    fill_value=0,
                )

                .rename("Orders")

                .rename_axis(
                    "Order Value Range"
                )

                .reset_index()

            )


            total_value_orders = (
                order_value_bands["Orders"].sum()
            )

            if total_value_orders > 0:

                order_value_bands[
                    "Percentage"
                ] = (
                    order_value_bands[
                        "Orders"
                    ]
                    / total_value_orders
                    * 100
                )

            else:

                order_value_bands[
                    "Percentage"
                ] = 0.0


        order_value_chart = (

            alt.Chart(
                order_value_bands
            )

            .mark_bar(
                color=TEAL
            )

            .encode(

                x=alt.X(
                    "Order Value Range:N",
                    title="Order Value Range",
                    sort=order_value_ranges,
                ),

                y=alt.Y(
                    "Orders:Q",
                    title="Number of Orders",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Order Value Range:N",
                        title="Order Value Range",
                    ),

                    alt.Tooltip(
                        "Orders:Q",
                        title="Number of Orders",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Percentage:Q",
                        title="Percentage of Total Orders",
                        format=".2f",
                    ),

                ],

            )

            .properties(
                height=CHART_HEIGHT
            )

        )


        st.altair_chart(
            order_value_chart,
            width="stretch",
        )


    # =====================================================
    # KEY INSIGHTS
    # =====================================================

    if len(monthly):

        highest_sales_month = monthly.loc[
            monthly["Sales"].idxmax()
        ]

    else:

        highest_sales_month = None


    if len(state_sales):

        highest_state = state_sales.iloc[0]

    else:

        highest_state = {
            "State": "N/A",
            "Sales": 0,
        }


    if len(status_distribution):

        dominant_status = (
            status_distribution.iloc[0]
        )

    else:

        dominant_status = {
            "Status": "N/A",
            "Orders": 0,
        }


    if len(top_products):

        most_ordered_product = (
            top_products.iloc[0]
        )

    else:

        most_ordered_product = {
            "Product ID": "N/A",
            "Items Sold": 0,
        }


    st.markdown(
        "### Key Sales & Order Insights"
    )


    if highest_sales_month is not None:

        highest_month_text = (
            f"{highest_sales_month['month']:%B %Y}"
        )

        highest_month_sales = format_currency(
            highest_sales_month["Sales"]
        )

    else:

        highest_month_text = "N/A"

        highest_month_sales = "R$ 0.00"


    st.markdown(

        f"- Total sales were "
        f"{format_currency(total_sales)} "
        f"across {total_orders:,} unique orders.\n"

        f"- Average order value was "
        f"{format_currency(average_order_value)}, "
        f"with {items_sold:,} items sold.\n"

        f"- The highest-sales month was "
        f"{highest_month_text} with "
        f"{highest_month_sales}.\n"

        f"- {highest_state['State']} had the highest "
        f"customer-state sales at "
        f"{format_currency(highest_state['Sales'])}.\n"

        f"- {dominant_status['Status']} was the most common "
        f"order status, with "
        f"{int(dominant_status['Orders']):,} unique orders; "

        f"product {most_ordered_product['Product ID']} "
        f"had the most order-item records "
        f"({int(most_ordered_product['Items Sold']):,})."

    )