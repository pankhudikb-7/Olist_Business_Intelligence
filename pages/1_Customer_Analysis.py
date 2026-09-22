from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from analytics import (
    get_active_customers,
    get_repeat_customer_rate,
    get_total_customers,
)
from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master
from genai_assistant import (
    apply_dashboard_filters,
    dashboard_layout,
)


st.set_page_config(
    page_title="Customer Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


@st.cache_data
def load_customer_data():
    """Load cleaned datasets and build the existing sales master."""
    datasets = clean_datasets(
        load_datasets(PROJECT_ROOT / "data")
    )
    sales_master = build_sales_master(datasets)
    return datasets, sales_master


def get_customer_orders(datasets):
    """Return orders with customer_unique_id at one row per order."""

    customers = datasets["customers"][
        ["customer_id", "customer_unique_id"]
    ]

    return datasets["orders"].merge(
        customers,
        how="left",
        on="customer_id",
        validate="many_to_one",
    )


def get_monthly_customers(customer_orders):
    """Return unique customers by purchase month."""

    monthly_customers = (
        customer_orders.dropna(
            subset=[
                "customer_unique_id",
                "order_purchase_timestamp",
            ]
        )
        .assign(
            month=lambda dataframe: dataframe[
                "order_purchase_timestamp"
            ]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby(
            "month",
            as_index=False,
        )["customer_unique_id"]
        .nunique()
        .rename(
            columns={
                "customer_unique_id": "Customers"
            }
        )
        .sort_values("month")
    )

    return monthly_customers


def get_customer_frequency(customer_orders):
    """Return one-time and repeat customer counts."""

    order_counts = (
        customer_orders.dropna(
            subset=["customer_unique_id"]
        )
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    frequency = pd.DataFrame(
        {
            "Customer Type": [
                "One-Time Customers",
                "Repeat Customers",
            ],
            "Customers": [
                int(order_counts.eq(1).sum()),
                int(order_counts.gt(1).sum()),
            ],
        }
    )

    total = frequency["Customers"].sum()

    frequency["Percentage"] = (
        0.0
        if total == 0
        else frequency["Customers"] / total * 100
    )

    return frequency


def get_monthly_new_returning_customers(customer_orders):
    """Return monthly first-time and returning customer counts."""

    order_history = (
        customer_orders.dropna(
            subset=[
                "customer_unique_id",
                "order_purchase_timestamp",
            ]
        )[
            [
                "customer_unique_id",
                "order_purchase_timestamp",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "order_purchase_timestamp"
        )
    )

    order_history["month"] = (
        order_history[
            "order_purchase_timestamp"
        ]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    first_order_month = (
        order_history
        .groupby("customer_unique_id")["month"]
        .transform("min")
    )

    order_history["Customer Type"] = (
        "Returning Customers"
    )

    order_history.loc[
        order_history["month"] == first_order_month,
        "Customer Type",
    ] = "New Customers"

    monthly_customers = (
        order_history
        .groupby(
            [
                "month",
                "Customer Type",
            ]
        )["customer_unique_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            columns=[
                "New Customers",
                "Returning Customers",
            ],
            fill_value=0,
        )
        .reset_index()
        .sort_values("month")
    )

    return monthly_customers


def get_purchase_frequency(customer_orders):
    """Return customer counts by number of orders."""

    order_counts = (
        customer_orders.dropna(
            subset=["customer_unique_id"]
        )
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    frequency = (
        pd.cut(
            order_counts,
            bins=[
                0,
                1,
                2,
                3,
                float("inf"),
            ],
            labels=[
                "1 order",
                "2 orders",
                "3 orders",
                "4+ orders",
            ],
        )
        .value_counts()
        .reindex(
            [
                "1 order",
                "2 orders",
                "3 orders",
                "4+ orders",
            ],
            fill_value=0,
        )
    )

    result = (
        frequency
        .rename("Customers")
        .rename_axis("Order Frequency")
        .reset_index()
    )

    total_customers = result["Customers"].sum()

    result["Percentage"] = (
        0.0
        if total_customers == 0
        else result["Customers"]
        / total_customers
        * 100
    )

    return result


def get_customer_rating_distribution(datasets):
    """Group customer average review scores."""

    customer_orders = get_customer_orders(
        datasets
    )

    reviews = datasets["order_reviews"][
        [
            "order_id",
            "review_score",
        ]
    ]

    customer_reviews = reviews.merge(
        customer_orders[
            [
                "order_id",
                "customer_unique_id",
            ]
        ],
        how="left",
        on="order_id",
        validate="many_to_one",
    ).dropna(
        subset=[
            "customer_unique_id",
            "review_score",
        ]
    )

    customer_averages = (
        customer_reviews
        .groupby("customer_unique_id")[
            "review_score"
        ]
        .mean()
    )

    bins = [
        1.0,
        2.0,
        3.0,
        4.0,
        4.5,
        5.01,
    ]

    labels = [
        "1.0-1.9",
        "2.0-2.9",
        "3.0-3.9",
        "4.0-4.4",
        "4.5-5.0",
    ]

    distribution = (
        pd.cut(
            customer_averages,
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        .value_counts()
        .reindex(
            labels,
            fill_value=0,
        )
        .rename("Customers")
        .rename_axis("Average Rating")
        .reset_index()
    )

    return distribution, customer_averages


def format_percentage(value):
    return f"{value:.2f}%"


def format_growth(
    latest_value,
    previous_value,
):

    if previous_value is None:
        return "N/A"

    if previous_value == 0:
        return "N/A"

    change = (
        (latest_value - previous_value)
        / previous_value
        * 100
    )

    return f"{change:+.2f}% vs prior month"


# =========================================================
# LOAD DATA
# =========================================================

datasets, sales_master = load_customer_data()


# =========================================================
# DASHBOARD
# =========================================================

with dashboard_layout(
    datasets,
    sales_master,
):

    # =====================================================
    # APPLY COMMON DASHBOARD FILTERS
    # =====================================================

    filtered_datasets, filtered_sales_master = (
        apply_dashboard_filters(
            datasets,
            sales_master,
        )
    )

    # =====================================================
    # CUSTOMER ANALYSIS CALCULATIONS
    # =====================================================

    customer_orders = get_customer_orders(
        filtered_datasets
    )

    monthly_customers = get_monthly_customers(
        customer_orders
    )

    frequency = get_customer_frequency(
        customer_orders
    )

    monthly_new_returning = (
        get_monthly_new_returning_customers(
            customer_orders
        )
    )

    purchase_frequency = get_purchase_frequency(
        customer_orders
    )

    rating_distribution, customer_averages = (
        get_customer_rating_distribution(
            filtered_datasets
        )
    )

    total_customers = get_total_customers(
        filtered_datasets["customers"]
    )

    active_customers = get_active_customers(
        filtered_datasets
    )

    repeat_customer_rate = get_repeat_customer_rate(
        filtered_datasets
    )

    if not frequency.empty:

        repeat_customers = int(
            frequency.loc[
                frequency["Customer Type"]
                == "Repeat Customers",
                "Customers",
            ].iloc[0]
        )

    else:

        repeat_customers = 0

    average_orders_per_customer = (
        float(
            customer_orders["order_id"].nunique()
            / total_customers
        )
        if total_customers
        else 0.0
    )

    average_customer_rating = (
        float(
            customer_averages.mean()
        )
        if len(customer_averages)
        else 0.0
    )

    latest_customers = (
        int(
            monthly_customers.iloc[-1][
                "Customers"
            ]
        )
        if len(monthly_customers)
        else 0
    )

    previous_customers = (
        int(
            monthly_customers.iloc[-2][
                "Customers"
            ]
        )
        if len(monthly_customers) > 1
        else None
    )

    customer_growth_value = format_growth(
        latest_customers,
        previous_customers,
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
            font-size: 1.4rem;
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
            "Customer Analysis"
        )

        st.caption(
            "Data Source: Olist Brazilian E-Commerce Dataset"
        )

    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title(
        "Customer Analysis"
    )

    st.caption(
        "Understanding customer activity, repeat purchasing, and satisfaction."
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [
        (
            "Total Customers",
            f"{total_customers:,}",
        ),
        (
            "Active Customers",
            f"{active_customers:,}",
        ),
        (
            "Repeat Customer Rate",
            format_percentage(
                repeat_customer_rate
            ),
        ),
        (
            "Repeat Customers",
            f"{repeat_customers:,}",
        ),
        (
            "Avg Orders per Customer",
            f"{average_orders_per_customer:.2f}",
        ),
        (
            "Average Customer Rating",
            f"{average_customer_rating:.2f} / 5",
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
    # MONTHLY CUSTOMER GROWTH
    # =====================================================

    st.markdown(
        "### Monthly Customer Growth"
    )

    customer_growth_chart = (
        alt.Chart(
            monthly_customers
        )
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
                "Customers:Q",
                title="Unique customers",
                scale=alt.Scale(
                    zero=False
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "month:T",
                    title="Month",
                    format="%b %Y",
                ),
                alt.Tooltip(
                    "Customers:Q",
                    title="Customers",
                    format=",d",
                ),
            ],
        )
        .properties(
            height=CHART_HEIGHT
        )
    )

    st.altair_chart(
        customer_growth_chart,
        width="stretch",
    )

    # =====================================================
    # NEW VS RETURNING + REPEAT VS ONE-TIME
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )

    with left_chart:

        st.markdown(
            "### New vs Returning Customers by Month"
        )

        monthly_customer_mix = (
            monthly_new_returning.copy()
        )

        monthly_customer_mix[
            "Total Customers"
        ] = (
            monthly_customer_mix[
                "New Customers"
            ]
            + monthly_customer_mix[
                "Returning Customers"
            ]
        )

        monthly_customer_mix[
            "New Customers %"
        ] = (
            monthly_customer_mix[
                "New Customers"
            ]
            / monthly_customer_mix[
                "Total Customers"
            ]
        )

        monthly_customer_mix[
            "Returning Customers %"
        ] = (
            monthly_customer_mix[
                "Returning Customers"
            ]
            / monthly_customer_mix[
                "Total Customers"
            ]
        )

        chart_tooltips = [
            alt.Tooltip(
                "month:T",
                title="Month",
                format="%b %Y",
            ),
            alt.Tooltip(
                "New Customers:Q",
                title="New Customers",
                format=",d",
            ),
            alt.Tooltip(
                "Returning Customers:Q",
                title="Returning Customers",
                format=",d",
            ),
            alt.Tooltip(
                "Total Customers:Q",
                title="Total Customers",
                format=",d",
            ),
            alt.Tooltip(
                "Returning Customers %:Q",
                title="Returning Customers %",
                format=".1%",
            ),
        ]

        shared_x = alt.X(
            "month:T",
            title="Month",
            axis=alt.Axis(
                format="%b %Y",
                tickCount={
                    "interval": "month",
                    "step": 3,
                },
            ),
        )

        new_customer_bars = (
            alt.Chart(
                monthly_customer_mix
            )
            .mark_bar(
                color=PRIMARY_BLUE
            )
            .encode(
                x=shared_x,
                y=alt.Y(
                    "New Customers:Q",
                    title="New Customers",
                    scale=alt.Scale(
                        zero=True
                    ),
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),
                tooltip=chart_tooltips,
            )
        )

        returning_customer_line = (
            alt.Chart(
                monthly_customer_mix
            )
            .mark_line(
                color=TEAL,
                point=alt.OverlayMarkDef(
                    filled=True,
                    color=TEAL,
                    size=55,
                ),
            )
            .encode(
                x=shared_x,
                y=alt.Y(
                    "Returning Customers:Q",
                    title="Returning Customers",
                    scale=alt.Scale(
                        zero=True
                    ),
                    axis=alt.Axis(
                        orient="right",
                        format=",d",
                    ),
                ),
                tooltip=chart_tooltips,
            )
        )

        monthly_customer_chart = (
            alt.layer(
                new_customer_bars,
                returning_customer_line,
            )
            .resolve_scale(
                y="independent"
            )
            .properties(
                height=CHART_HEIGHT
            )
        )

        st.altair_chart(
            monthly_customer_chart,
            width="stretch",
        )

    with right_chart:

        st.markdown(
            "### Repeat vs One-Time Customers"
        )

        repeat_chart = (
            alt.Chart(
                frequency
            )
            .mark_bar(
                color=PRIMARY_BLUE
            )
            .encode(
                x=alt.X(
                    "Customers:Q",
                    title="Customers",
                ),
                y=alt.Y(
                    "Customer Type:N",
                    title="Customer type",
                    sort=[
                        "One-Time Customers",
                        "Repeat Customers",
                    ],
                ),
                tooltip=[
                    alt.Tooltip(
                        "Customer Type:N",
                        title="Customer type",
                    ),
                    alt.Tooltip(
                        "Customers:Q",
                        title="Customers",
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
            repeat_chart,
            width="stretch",
        )

    # =====================================================
    # PURCHASE FREQUENCY + SATISFACTION
    # =====================================================

    left_chart, right_chart = st.columns(
        2,
        gap="large",
    )

    with left_chart:

        st.markdown(
            "### Customer Purchase Frequency"
        )

        purchase_frequency_chart = (
            alt.Chart(
                purchase_frequency
            )
            .mark_bar(
                color=PRIMARY_BLUE
            )
            .encode(
                x=alt.X(
                    "Order Frequency:N",
                    title="Unique orders per customer",
                    sort=[
                        "1 order",
                        "2 orders",
                        "3 orders",
                        "4+ orders",
                    ],
                ),
                y=alt.Y(
                    "Customers:Q",
                    title="Customers",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Order Frequency:N",
                        title="Order frequency",
                    ),
                    alt.Tooltip(
                        "Customers:Q",
                        title="Customers",
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
            purchase_frequency_chart,
            width="stretch",
        )

    with right_chart:

        st.markdown(
            "### Customer Satisfaction Distribution"
        )

        rating_chart = (
            alt.Chart(
                rating_distribution
            )
            .mark_bar(
                color=TEAL
            )
            .encode(
                x=alt.X(
                    "Average Rating:O",
                    title="Average Rating",
                    sort=[
                        "1.0-1.9",
                        "2.0-2.9",
                        "3.0-3.9",
                        "4.0-4.4",
                        "4.5-5.0",
                    ],
                ),
                y=alt.Y(
                    "Customers:Q",
                    title="Customers",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Average Rating:O",
                        title="Average rating",
                    ),
                    alt.Tooltip(
                        "Customers:Q",
                        title="Customers",
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

    # =====================================================
    # KEY INSIGHTS
    # =====================================================

    st.markdown(
        "### Key Customer Insights"
    )

    st.markdown(
        f"- The dataset contains {total_customers:,} "
        f"unique customers, with "
        f"{active_customers:,} active customers.\n"
        f"- The repeat customer rate is "
        f"{repeat_customer_rate:.2f}%.\n"
        f"- There are {repeat_customers:,} repeat customers, "
        f"and average orders per customer is "
        f"{average_orders_per_customer:.2f}.\n"
        f"- The latest month contains "
        f"{latest_customers:,} unique customers; "
        f"month-over-month change is "
        f"{customer_growth_value}.\n"
        f"- The average customer review score among "
        f"customers with reviews is "
        f"{average_customer_rating:.2f} out of 5."
    )