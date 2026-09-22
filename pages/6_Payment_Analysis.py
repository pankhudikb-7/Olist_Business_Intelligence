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

from analytics import get_payment_summary
from data_cleaning import clean_datasets
from data_loader import load_datasets
from data_model import build_sales_master
from genai_assistant import (dashboard_layout,apply_dashboard_filters,)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Payment Analysis | Olist Business Intelligence",
    page_icon="O",
    layout="wide",
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_payment_data():
    """
    Load cleaned Olist datasets.

    Payment analysis remains at payment-record grain.
    """
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


datasets, sales_master = load_payment_data()


# =========================================================
# DASHBOARD LAYOUT
# =========================================================

with dashboard_layout(datasets, sales_master):

    # =====================================================
    # PAGE STYLING
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
    # SIDEBAR PAGE INFORMATION
    # =====================================================

    with st.sidebar:

        st.header("Olist Business Intelligence")

        st.subheader("Payment Analysis")

        st.caption(
            "Data Source: Olist Brazilian E-Commerce Dataset"
        )


    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title("Payment Analysis")

    st.caption(
        "Understand payment value, method mix, transaction behavior, "
        "and installment usage."
    )
        # =====================================================
    # COMMON FILTER VALUES
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
    # APPLY COMMON DASHBOARD FILTERS
    # =====================================================

    filtered_datasets, filtered_sales_master = apply_dashboard_filters(
        datasets,
        sales_master,
    )

    # =====================================================
    # FILTER PAYMENT RECORDS BY COMMON FILTERS
    # =====================================================

    payments = filtered_datasets[
        "order_payments"
    ].copy()

    # =====================================================
    # PAYMENT TYPE FILTER
    # =====================================================
    #
    # This is the INDIVIDUAL filter for Payment Analysis.
    #
    # It is intentionally separate from the common
    # Customer State and Order Date Range filters.
    #


    payment_types = (
        payments["payment_type"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


    selected_payment_type = st.selectbox(
        "Payment Type",
        ["All Payment Types"] + payment_types,
        key="payment_type_filter",
    )


    # =====================================================
    # APPLY PAYMENT TYPE FILTER
    # =====================================================

    if selected_payment_type != "All Payment Types":

        payments = payments[
            payments["payment_type"].astype(str)
            == selected_payment_type
        ].copy()


    # =====================================================
    # EMPTY RESULT HANDLING
    # =====================================================

    if payments.empty:

        st.warning(
            "No payment records are available for "
            "the selected filters."
        )

        st.stop()


    # =====================================================
    # PAYMENT SUMMARY
    # =====================================================

    payment_summary = (
        payments
        .groupby(
            "payment_type",
            as_index=False,
        )
        .agg(
            **{
                "Payment Transactions": (
                    "payment_type",
                    "size",
                ),
                "Payment Value": (
                    "payment_value",
                    "sum",
                ),
                "Average Payment Value": (
                    "payment_value",
                    "mean",
                ),
            }
        )
        .rename(
            columns={
                "payment_type": "Payment Type"
            }
        )
    )


    # =====================================================
    # PAYMENT KPIs
    # =====================================================

    total_payment_value = float(
        payments["payment_value"].sum()
    )


    payment_transactions = int(
        len(payments)
    )


    average_payment_value = float(
        payments["payment_value"].mean()
    )


    dominant_payment_method = (
        payment_summary
        .sort_values(
            "Payment Value",
            ascending=False,
        )
        .iloc[0]
    )


    # =====================================================
    # FILTER SUMMARY TEXT
    # =====================================================

    if selected_customer_state == "All States":

        state_text = "all states"

    else:

        state_text = selected_customer_state


    if selected_date_range is not None:

        try:

            if (
                isinstance(
                    selected_date_range,
                    (list, tuple),
                )
                and len(selected_date_range) == 2
            ):

                start_date = selected_date_range[0]
                end_date = selected_date_range[1]

                date_text = (
                    f"{start_date} to {end_date}"
                )

            else:

                date_text = (
                    "the selected period"
                )

        except Exception:

            date_text = (
                "the selected period"
            )

    else:

        date_text = (
            "the full available period"
        )


    if (
        selected_payment_type
        == "All Payment Types"
    ):

        payment_text = (
            "all payment types"
        )

    else:

        payment_text = (
            selected_payment_type
        )


    st.caption(
        f"Showing payment analysis for "
        f"{state_text} and {date_text}, "
        f"filtered to {payment_text}."
    )


    # =====================================================
    # KPI CARDS
    # =====================================================

    kpis = [

        (
            "Total Payment Value",
            format_compact_currency(
                total_payment_value
            ),
        ),

        (
            "Payment Transactions",
            f"{payment_transactions:,}",
        ),

        (
            "Average Payment Value",
            format_currency(
                average_payment_value
            ),
        ),

        (
            "Dominant Payment Method",
            str(
                dominant_payment_method[
                    "Payment Type"
                ]
            ),
        ),
    ]


    kpi_columns = st.columns(4)


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

    PRIMARY_BLUE = "#1f5a94"

    TEAL = "#168a8a"

    CHART_HEIGHT = 310


    # =====================================================
    # PAYMENT METHOD MIX
    # =====================================================

    payment_mix = payment_summary.copy()


    payment_mix[
        "Payment Value Share %"
    ] = (
        payment_mix["Payment Value"]
        / total_payment_value
        * 100
    )


    st.markdown(
        "### Payment Method Mix"
    )


    mix_chart = (
        alt.Chart(payment_mix)
        .mark_arc(innerRadius=70)
        .encode(

            theta=alt.Theta(
                "Payment Value:Q",
                title="Payment Value",
            ),

            color=alt.Color(
                "Payment Type:N",
                title="Payment Type",
            ),

            tooltip=[

                alt.Tooltip(
                    "Payment Type:N",
                    title="Payment Type",
                ),

                alt.Tooltip(
                    "Payment Value:Q",
                    title="Payment Value",
                    format=",.2f",
                ),

                alt.Tooltip(
                    "Payment Value Share %:Q",
                    title="% of Total Payment Value",
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=360
        )
    )


    st.altair_chart(
        mix_chart,
        width="stretch",
    )


    st.caption(
        "Total payment value processed: "
        f"{format_currency(total_payment_value)}"
    )


    # =====================================================
    # PAYMENT ACTIVITY TREND
    # =====================================================
    #
    # Payment records do not contain their own timestamp.
    # Order purchase timestamp is used to place each
    # payment record into a month.


    payment_dates = (
        payments[
            [
                "order_id",
                "payment_value",
            ]
        ]
        .merge(
            filtered_datasets["orders"][
                [
                    "order_id",
                    "order_purchase_timestamp",
                ]
            ],
            how="left",
            on="order_id",
            validate="many_to_one",
        )
    )


    payment_dates["Month"] = (
        payment_dates[
            "order_purchase_timestamp"
        ]
        .dt.to_period("M")
        .dt.to_timestamp()
    )


    monthly_payment_activity = (
        payment_dates
        .groupby(
            "Month",
            as_index=False,
        )
        .agg(
            **{
                "Payment Value": (
                    "payment_value",
                    "sum",
                ),

                "Payment Transactions": (
                    "payment_value",
                    "size",
                ),
            }
        )
    )


    monthly_payment_activity[
        "Average Payment Value"
    ] = (
        monthly_payment_activity[
            "Payment Value"
        ]
        /
        monthly_payment_activity[
            "Payment Transactions"
        ]
    )


    # =====================================================
    # PAYMENT VALUE DISTRIBUTION
    # =====================================================

    payment_bands = [

        "R$0-50",

        "R$50-100",

        "R$100-250",

        "R$250-500",

        "R$500-1,000",

        "R$1,000-2,500",

        "R$2,500+",
    ]


    payment_distribution = (
        pd.cut(

            payments[
                "payment_value"
            ],

            bins=[
                0,
                50,
                100,
                250,
                500,
                1_000,
                2_500,
                float("inf"),
            ],

            labels=payment_bands,

            right=False,

            include_lowest=True,
        )
        .value_counts()
        .reindex(
            payment_bands,
            fill_value=0,
        )
        .rename(
            "Payment Transactions"
        )
        .rename_axis(
            "Payment Band"
        )
        .reset_index()
    )


    payment_distribution[
        "Percentage"
    ] = (
        payment_distribution[
            "Payment Transactions"
        ]
        / payment_transactions
        * 100
    )


    # =====================================================
    # PAYMENT ACTIVITY + DISTRIBUTION
    # =====================================================

    left_chart, right_chart = (
        st.columns(
            2,
            gap="large",
        )
    )


    # =====================================================
    # PAYMENT ACTIVITY TREND CHART
    # =====================================================

    with left_chart:

        st.markdown(
            "### Payment Activity Trend"
        )


        payment_bars = (
            alt.Chart(
                monthly_payment_activity
            )
            .mark_bar(
                color=PRIMARY_BLUE,
                opacity=0.7,
            )
            .encode(

                x=alt.X(
                    "Month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y"
                    ),
                ),

                y=alt.Y(
                    "Payment Value:Q",
                    title="Payment Value (R$)",
                    axis=alt.Axis(
                        format="~s"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Month:T",
                        title="Month",
                        format="%b %Y",
                    ),

                    alt.Tooltip(
                        "Payment Value:Q",
                        title="Payment Value",
                        format=",.2f",
                    ),

                    alt.Tooltip(
                        "Payment Transactions:Q",
                        title="Payment Transactions",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Average Payment Value:Q",
                        title="Average Payment Value",
                        format=",.2f",
                    ),
                ],
            )
        )


        payment_line = (
            alt.Chart(
                monthly_payment_activity
            )
            .mark_line(
                color=TEAL,
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=45,
                    color=TEAL,
                ),
            )
            .encode(

                x=alt.X(
                    "Month:T",
                    title="Month",
                    axis=alt.Axis(
                        format="%b %Y"
                    ),
                ),

                y=alt.Y(
                    "Payment Transactions:Q",
                    title="Payment Transactions",
                    axis=alt.Axis(
                        orient="right",
                        format=",d",
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Month:T",
                        title="Month",
                        format="%b %Y",
                    ),

                    alt.Tooltip(
                        "Payment Value:Q",
                        title="Payment Value",
                        format=",.2f",
                    ),

                    alt.Tooltip(
                        "Payment Transactions:Q",
                        title="Payment Transactions",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Average Payment Value:Q",
                        title="Average Payment Value",
                        format=",.2f",
                    ),
                ],
            )
        )


        st.altair_chart(

            alt.layer(
                payment_bars,
                payment_line,
            )
            .resolve_scale(
                y="independent"
            )
            .properties(
                height=CHART_HEIGHT
            ),

            width="stretch",
        )


    # =====================================================
    # PAYMENT VALUE DISTRIBUTION CHART
    # =====================================================

    with right_chart:

        st.markdown(
            "### Payment Value Distribution"
        )


        distribution_chart = (
            alt.Chart(
                payment_distribution
            )
            .mark_bar(
                color=TEAL
            )
            .encode(

                x=alt.X(
                    "Payment Band:N",
                    title="Payment Band",
                    sort=payment_bands,
                ),

                y=alt.Y(
                    "Payment Transactions:Q",
                    title="Payment Transactions",
                    axis=alt.Axis(
                        format=",d"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Payment Band:N",
                        title="Payment Band",
                    ),

                    alt.Tooltip(
                        "Payment Transactions:Q",
                        title="Payment Transactions",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "Percentage:Q",
                        title="% of Payment Transactions",
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
    # INSTALLMENT BEHAVIOR
    # =====================================================

    installment_bands = [

        "0-1 installment",

        "2-3",

        "4-6",

        "7-12",

        "13-24",

        "25+",
    ]


    payments["Installment Band"] = pd.cut(

        payments[
            "payment_installments"
        ],

        bins=[
            -float("inf"),
            2,
            4,
            7,
            13,
            25,
            float("inf"),
        ],

        labels=installment_bands,

        right=False,
    )


    installment_distribution = (
        payments
        .groupby(
            [
                "payment_type",
                "Installment Band",
            ],
            observed=False,
        )
        .size()
        .rename(
            "Transaction Count"
        )
        .reset_index()
        .rename(
            columns={
                "payment_type":
                    "Payment Type"
            }
        )
    )


    method_totals = (
        installment_distribution
        .groupby(
            "Payment Type"
        )[
            "Transaction Count"
        ]
        .transform("sum")
    )


    installment_distribution[
        "% of Method Transactions"
    ] = (
        installment_distribution[
            "Transaction Count"
        ]
        / method_totals
        * 100
    )


    method_comparison = (
        payment_summary
        .sort_values(
            "Average Payment Value",
            ascending=True,
        )
        .copy()
    )


    # =====================================================
    # INSTALLMENT + AVERAGE PAYMENT CHARTS
    # =====================================================

    left_chart, right_chart = (
        st.columns(
            2,
            gap="large",
        )
    )


    # =====================================================
    # INSTALLMENT BEHAVIOR CHART
    # =====================================================

    with left_chart:

        st.markdown(
            "### Installment Behavior"
        )


        installment_chart = (
            alt.Chart(
                installment_distribution
            )
            .mark_rect()
            .encode(

                x=alt.X(
                    "Installment Band:N",
                    title="Installment Band",
                    sort=installment_bands,
                ),

                y=alt.Y(
                    "Payment Type:N",
                    title="Payment Type",
                ),

                color=alt.Color(
                    "Transaction Count:Q",
                    title="Transaction Count",
                    scale=alt.Scale(
                        scheme="tealblues"
                    ),
                ),

                tooltip=[

                    alt.Tooltip(
                        "Payment Type:N",
                        title="Payment Type",
                    ),

                    alt.Tooltip(
                        "Installment Band:N",
                        title="Installment Band",
                    ),

                    alt.Tooltip(
                        "Transaction Count:Q",
                        title="Transaction Count",
                        format=",d",
                    ),

                    alt.Tooltip(
                        "% of Method Transactions:Q",
                        title="% of Method Transactions",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=CHART_HEIGHT
            )
        )


        st.altair_chart(
            installment_chart,
            width="stretch",
        )


    # =====================================================
    # AVERAGE PAYMENT VALUE BY METHOD
    # =====================================================

    with right_chart:

        st.markdown(
            "### Average Payment Value by Method"
        )


        method_tooltip = [

            alt.Tooltip(
                "Payment Type:N",
                title="Payment Type",
            ),

            alt.Tooltip(
                "Average Payment Value:Q",
                title="Average Payment Value",
                format=",.2f",
            ),

            alt.Tooltip(
                "Payment Transactions:Q",
                title="Transaction Count",
                format=",d",
            ),

            alt.Tooltip(
                "Payment Value:Q",
                title="Total Payment Value",
                format=",.2f",
            ),
        ]


        method_y = alt.Y(

            "Payment Type:N",

            title="Payment Type",

            sort=alt.SortField(
                field="Average Payment Value",
                order="ascending",
            ),
        )


        method_stems = (
            alt.Chart(
                method_comparison
            )
            .mark_rule(
                color=TEAL,
                strokeWidth=2,
            )
            .encode(

                x=alt.X(
                    "Average Payment Value:Q",
                    title="Average Payment Value (R$)",
                ),

                x2=alt.value(0),

                y=method_y,
            )
        )


        method_dots = (
            alt.Chart(
                method_comparison
            )
            .mark_point(
                color=PRIMARY_BLUE,
                filled=True,
                size=85,
            )
            .encode(

                x=alt.X(
                    "Average Payment Value:Q"
                ),

                y=alt.Y(
                    "Payment Type:N",

                    sort=alt.SortField(
                        field="Average Payment Value",
                        order="ascending",
                    ),
                ),

                tooltip=method_tooltip,
            )
        )


        st.altair_chart(

            (
                method_stems
                + method_dots
            )
            .properties(
                height=CHART_HEIGHT
            ),

            width="stretch",
        )


    # =====================================================
    # KEY PAYMENT INSIGHTS
    # =====================================================

    highest_value_method = (
        payment_summary
        .sort_values(
            "Payment Value",
            ascending=False,
        )
        .iloc[0]
    )


    highest_volume_method = (
        payment_summary
        .sort_values(
            "Payment Transactions",
            ascending=False,
        )
        .iloc[0]
    )


    highest_average_method = (
        payment_summary
        .sort_values(
            "Average Payment Value",
            ascending=False,
        )
        .iloc[0]
    )


    installment_transaction_count = int(
        payments[
            "payment_installments"
        ]
        .gt(1)
        .sum()
    )


    installment_percentage = (
        installment_transaction_count
        / payment_transactions
        * 100
    )


    if not monthly_payment_activity.empty:

        highest_value_month = (
            monthly_payment_activity.loc[
                monthly_payment_activity[
                    "Payment Value"
                ].idxmax()
            ]
        )

    else:

        highest_value_month = None


    # =====================================================
    # KEY INSIGHTS DISPLAY
    # =====================================================

    st.markdown(
        "### Key Payment Insights"
    )


    insights = [

        (
            f"{highest_value_method['Payment Type']} "
            f"processed the highest payment value at "
            f"{format_currency(highest_value_method['Payment Value'])}, "
            f"representing "
            f"{highest_value_method['Payment Value'] / total_payment_value:.2%} "
            f"of total payment value."
        ),

        (
            f"{highest_volume_method['Payment Type']} "
            f"had the highest payment-record volume with "
            f"{int(highest_volume_method['Payment Transactions']):,} "
            f"transactions."
        ),

        (
            f"{highest_average_method['Payment Type']} "
            f"had the highest average payment value at "
            f"{format_currency(highest_average_method['Average Payment Value'])}."
        ),

        (
            f"{installment_percentage:.2f}% "
            f"of payment transactions used more than one installment."
        ),
    ]


    if highest_value_month is not None:

        insights.append(
            (
                f"{highest_value_month['Month']:%B %Y} "
                f"had the highest monthly payment value at "
                f"{format_currency(highest_value_month['Payment Value'])}."
            )
        )


    st.markdown(
        "\n".join(
            f"- {insight}"
            for insight in insights
        )
    )


    # =====================================================
    # METHODOLOGY NOTE
    # =====================================================

    st.caption(
        "Payment metrics are calculated from payment records "
        "at their natural grain; payment records are not joined "
        "to the transaction-level sales master."
    )