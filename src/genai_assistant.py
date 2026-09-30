"""
Reusable responsive dashboard shell with deterministic Olist AI panel.
"""

from contextlib import contextmanager

import pandas as pd
import streamlit as st

from analyst import analyze_question


# =========================================================
# SUGGESTED AI QUESTIONS
# =========================================================

SUGGESTED_QUESTIONS = [
    "What was the highest sales quarter?",
    "What were sales in Q2 2018?",
    "Which category generated the highest revenue?",
    "Which product sold the most items?",
    "Which seller generated the highest revenue?",
    "Which payment method had the highest payment value?",
    "What is the repeat customer rate?",
    "What is the late delivery rate?",
    "What is the average delivery time?",
    "What is the average order value?",
    "How many customers are there?",
    "What is the average review score?",
    "What is the profit margin of each seller?",
]


# =========================================================
# AI RESULT HANDLER
# =========================================================

def _store_result(question, datasets, sales_master):

    result = analyze_question(
        question,
        datasets,
        sales_master,
    )

    st.session_state["genai_analysis_result"] = result
    st.session_state["genai_question"] = question

    st.session_state.setdefault(
        "genai_history",
        [],
    ).append(
        {
            "question": question,
            "result": result,
        }
    )

    st.switch_page(
        "pages/7_AI_Business_Analysis.py"
    )

    return result


# =========================================================
# AI PANEL
# =========================================================

def _render_panel(
    datasets,
    sales_master,
):

    st.markdown(
        "### 🤖 Olist AI Analyst"
    )

    st.caption(
        "Ask questions about the Olist business data"
    )

    question = st.text_input(
        "Business question",
        placeholder="Ask a business question...",
        key="genai_question_input",
        label_visibility="collapsed",
    )

    if st.button(
        "Ask",
        key="genai_ask_button",
        type="primary",
        width="stretch",
    ):

        if question.strip():

            _store_result(
                question,
                datasets,
                sales_master,
            )

        else:

            st.warning(
                "Please enter a business question."
            )


    st.markdown(
        "**Suggested Business Questions**"
    )

    selected_question = st.selectbox(
        "Select a question",
        SUGGESTED_QUESTIONS,
        key="genai_suggestion_select",
        label_visibility="collapsed",
    )

    if st.button(
        "Ask Selected Question",
        key="genai_selected_question_button",
        use_container_width=True,
    ):

        _store_result(
            selected_question,
            datasets,
            sales_master,
        )


    history = st.session_state.get(
        "genai_history",
        [],
    )

    st.markdown(
        "**Answer**"
    )

    if history:

        result = history[-1]["result"]

        if result.get("status") == "ok":

            st.success(
                result["answer"]
            )

            st.markdown(
                "**How it was calculated**"
            )

            st.caption(
                result.get(
                    "methodology",
                    "Calculated from the Olist dataset.",
                )
            )

        else:

            st.info(
                result["answer"]
            )

    else:

        st.caption(
            "Your answer will appear here."
        )


# =========================================================
# FILTER INITIALIZATION
# =========================================================

def _initialize_dashboard_filters(
    datasets,
    sales_master,
):

    customer_states = (
        datasets["customers"]
        ["customer_state"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if (
        "customer_state_filter"
        not in st.session_state
    ):

        st.session_state[
            "customer_state_filter"
        ] = "All States"


    purchase_dates = pd.to_datetime(
        sales_master[
            "order_purchase_timestamp"
        ]
    )

    min_date = (
        purchase_dates
        .min()
        .normalize()
        .date()
    )

    max_date = (
        purchase_dates
        .max()
        .normalize()
        .date()
    )

    if (
        "order_date_range"
        not in st.session_state
    ):

        st.session_state[
            "order_date_range"
        ] = (
            min_date,
            max_date,
        )

    return (
        customer_states,
        min_date,
        max_date,
    )


# =========================================================
# SHARED DASHBOARD FILTERS
# =========================================================

def render_dashboard_filters(
    datasets,
    sales_master,
):

    (
        customer_states,
        min_date,
        max_date,
    ) = _initialize_dashboard_filters(
        datasets,
        sales_master,
    )

    with st.sidebar:

        st.markdown(
            "### Dashboard Filters"
        )

        st.selectbox(
            "Customer State",
            [
                "All States"
            ] + customer_states,
            key="customer_state_filter",
        )

        st.date_input(
        "Order Date Range",
        value=st.session_state[
        "order_date_range"
        ],
        min_value=min_date,
        max_value=max_date,
        key="order_date_range",
    )

        selected_state = st.session_state.get(
            "customer_state_filter",
            "All States",
        )

        selected_dates = st.session_state.get(
            "order_date_range"
        )

        st.caption(
            f"State: {selected_state}"
        )

        if (
            isinstance(
                selected_dates,
                (tuple, list),
            )
            and len(selected_dates) == 2
        ):

            st.caption(
                f"Dates: "
                f"{selected_dates[0]} "
                f"to "
                f"{selected_dates[1]}"
            )

# =========================================================
# COMPATIBILITY FILTER FUNCTION
# =========================================================

def apply_dashboard_filters(
    datasets,
    sales_master,
    customer_state=None,
    date_range=None,
):
    """
    Apply the shared Customer State and Order Date Range
    filters and return filtered datasets and sales master.

    This function keeps compatibility with dashboard pages
    that still call apply_dashboard_filters().
    """

    selected_customer_state = (
        customer_state
        if customer_state is not None
        else st.session_state.get(
            "customer_state_filter",
            "All States",
        )
    )

    selected_date_range = (
        date_range
        if date_range is not None
        else (
            st.session_state.get("order_date_range")
            or st.session_state.get("date_range")
            or st.session_state.get("dashboard_date_range")
        )
    )

    # -----------------------------------------------------
    # CUSTOMER STATE FILTER
    # -----------------------------------------------------

    customers = datasets["customers"]

    if selected_customer_state != "All States":

        filtered_customers = customers[
            customers["customer_state"]
            == selected_customer_state
        ].copy()

    else:

        filtered_customers = customers.copy()

    customer_ids = (
        filtered_customers["customer_id"]
        .unique()
    )

    # -----------------------------------------------------
    # ORDER FILTER
    # -----------------------------------------------------

    orders = datasets["orders"]

    filtered_orders = orders[
        orders["customer_id"].isin(customer_ids)
    ].copy()

    # -----------------------------------------------------
    # ORDER DATE FILTER
    # -----------------------------------------------------

    if selected_date_range is not None:

        if (
            isinstance(
                selected_date_range,
                (list, tuple),
            )
            and len(selected_date_range) == 2
        ):

            start_date = pd.to_datetime(
                selected_date_range[0]
            )

            end_date = pd.to_datetime(
                selected_date_range[1]
            )

            purchase_dates = pd.to_datetime(
                filtered_orders[
                    "order_purchase_timestamp"
                ]
            )

            date_mask = (
                purchase_dates.dt.normalize()
                .between(
                    start_date.normalize(),
                    end_date.normalize(),
                )
            )

            filtered_orders = (
                filtered_orders[
                    date_mask
                ].copy()
            )

    # -----------------------------------------------------
    # ORDER IDS
    # -----------------------------------------------------

    order_ids = (
        filtered_orders["order_id"]
        .unique()
    )

    # -----------------------------------------------------
    # BUILD FILTERED DATASETS
    # -----------------------------------------------------

    filtered_datasets = datasets.copy()

    filtered_datasets["customers"] = (
        filtered_customers
    )

    filtered_datasets["orders"] = (
        filtered_orders
    )

    if "order_reviews" in datasets:

        filtered_datasets["order_reviews"] = (
            datasets["order_reviews"][
                datasets["order_reviews"]["order_id"]
                .isin(order_ids)
            ].copy()
        )

    if "order_items" in datasets:

        filtered_datasets["order_items"] = (
            datasets["order_items"][
                datasets["order_items"]["order_id"]
                .isin(order_ids)
            ].copy()
        )

    if "order_payments" in datasets:

        filtered_datasets["order_payments"] = (
            datasets["order_payments"][
                datasets["order_payments"]["order_id"]
                .isin(order_ids)
            ].copy()
        )

    # -----------------------------------------------------
    # FILTER SALES MASTER
    # -----------------------------------------------------

    filtered_sales_master = (
        sales_master[
            sales_master["order_id"].isin(order_ids)
        ].copy()
    )

    return (
        filtered_datasets,
        filtered_sales_master,
    )


# =========================================================
# DASHBOARD LAYOUT
# =========================================================
# =========================================================
# DASHBOARD LAYOUT
# =========================================================

@contextmanager
def dashboard_layout(
    datasets,
    sales_master,
    show_dashboard_filters=True,
):

    """
    Shared dashboard layout.

    show_dashboard_filters=True:
        Show common filters in sidebar.

    show_dashboard_filters=False:
        Hide common filters from sidebar.
    """

    # -----------------------------------------------------
    # INITIALIZE FILTERS
    # -----------------------------------------------------

    _initialize_dashboard_filters(
        datasets,
        sales_master,
    )


    # =====================================================
    # RESPONSIVE CSS
    # =====================================================

    st.markdown(
        """
        <style>

        :root {

            --dashboard-title-size:
                clamp(1.55rem, 1.9vw, 2.25rem);

            --dashboard-section-size:
                clamp(1.05rem, 1.25vw, 1.35rem);

            --dashboard-body-size:
                clamp(0.82rem, 0.9vw, 0.98rem);

            --dashboard-kpi-size:
                clamp(1rem, 1.35vw, 1.4rem);
        }


        [data-testid="stAppViewContainer"] h1 {

            font-size:
                var(--dashboard-title-size);

        }


        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3 {

            font-size:
                var(--dashboard-section-size);

        }


        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label {

            font-size:
                var(--dashboard-body-size);

        }


        [data-testid="stColumn"] {

            min-width:
                0;

        }


        /* =================================================
           RESPONSIVE SIDEBAR
           ================================================= */

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {

            container-type:
                inline-size;

            container-name:
                olist-sidebar;

        }


        [data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a p,
        [data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a span {

            font-size:
                clamp(
                    0.75rem,
                    calc(
                        0.42rem
                        + 2.2cqw
                    ),
                    1rem
                ) !important;

            line-height:
                1.25 !important;

        }


        [data-testid="stSidebar"]
        [data-testid="stSidebarNav"] a {

            padding-block:
                clamp(
                    0.15rem,
                    calc(
                        0.08rem
                        + 0.8cqw
                    ),
                    0.4rem
                ) !important;

        }


        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {

            font-size:
                clamp(
                    0.9rem,
                    calc(
                        0.72rem
                        + 2.4cqw
                    ),
                    1.25rem
                ) !important;

            line-height:
                1.2 !important;

        }


        [data-testid="stSidebar"] p {

            font-size:
                clamp(
                    0.625rem,
                    calc(
                        0.4rem
                        + 1.7cqw
                    ),
                    0.85rem
                ) !important;

            line-height:
                1.3 !important;

        }


        /* =================================================
           FLOATING AI BUTTON
           ================================================= */

        [data-testid="stElementContainer"]:
        has(#genai-trigger-marker)
        +
        [data-testid="stElementContainer"]
        .st-key-genai_toggle_button {

            position:
                fixed !important;

            right:
                clamp(
                    0.75rem,
                    2vw,
                    1.5rem
                ) !important;

            bottom:
                clamp(
                    0.75rem,
                    2vw,
                    1.5rem
                ) !important;

            z-index:
                1000 !important;

        }


        [data-testid="stElementContainer"]:
        has(#genai-trigger-marker)
        +
        [data-testid="stElementContainer"]
        .st-key-genai_toggle_button
        [data-testid="stButton"] button {

            width:
                clamp(
                    2.75rem,
                    5vw,
                    3.5rem
                ) !important;

            height:
                clamp(
                    2.75rem,
                    5vw,
                    3.5rem
                ) !important;

            padding:
                0 !important;

            border-radius:
                50% !important;

            background:
                #1f5a94 !important;

            border:
                2px solid #ffffff !important;

            box-shadow:
                0 6px 18px
                rgba(
                    20,
                    33,
                    61,
                    0.24
                ) !important;

            font-size:
                clamp(
                    1.1rem,
                    2vw,
                    1.45rem
                ) !important;

        }


        /* =================================================
           AI DRAWER
           ================================================= */

        .st-key-genai_panel {

            position:
                fixed !important;

            right:
                clamp(
                    0.75rem,
                    2vw,
                    1.5rem
                ) !important;

            top:
                calc(50% + 3rem)
                !important;

            transform:
                translateY(-50%)
                !important;

            z-index:
                1001 !important;

            width:
                min(
                    24rem,
                    calc(100vw - 2rem)
                ) !important;

            max-height:
                min(
                    42rem,
                    calc(100vh - 2rem)
                ) !important;

            overflow-y:
                auto !important;

            background:
                #ffffff !important;

            border:
                1px solid #cbd7e5
                !important;

            border-top:
                4px solid #1f5a94
                !important;

            border-radius:
                8px !important;

            box-shadow:
                0 10px 28px
                rgba(
                    20,
                    33,
                    61,
                    0.2
                ) !important;

            padding:
                0.9rem !important;

        }


        .st-key-genai_panel
        [data-testid="stTextInput"] input {

            min-width:
                0 !important;

        }


        @media (max-width: 900px) {

            [data-testid="stHorizontalBlock"] {

                flex-wrap:
                    wrap;

            }

            [data-testid="stHorizontalBlock"]
            > [data-testid="stColumn"] {

                min-width:
                    min(
                        100%,
                        18rem
                    );

            }

        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================

    if show_dashboard_filters:

        render_dashboard_filters(
            datasets,
            sales_master,
        )


    # =====================================================
    # AI BUTTON
    # =====================================================

    if (
        "genai_open"
        not in st.session_state
    ):

        st.session_state[
            "genai_open"
        ] = False


    with st.container():

        st.markdown(
            '<span id="genai-trigger-marker"></span>',
            unsafe_allow_html=True,
        )

        if (
            not st.session_state.genai_open
            and st.button(
                "🤖",
                key="genai_toggle_button",
                help="Open Olist AI Analyst",
            )
        ):

            st.session_state.genai_open = True

            st.rerun()


    # =====================================================
    # AI DRAWER
    # =====================================================

    if st.session_state.genai_open:

        with st.container(
            key="genai_panel"
        ):

            close = st.button(
                "× Close",
                key="genai_close_button",
            )

            if close:

                st.session_state.genai_open = False

                st.rerun()


            _render_panel(
                datasets,
                sales_master,
            )


    yield