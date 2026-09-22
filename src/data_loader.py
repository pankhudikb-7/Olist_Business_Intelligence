from pathlib import Path

import pandas as pd
import streamlit as st


DATASET_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


@st.cache_data
def load_datasets(data_dir="data"):
    """Load all Olist CSV files once and reuse them from Streamlit cache."""
    data_path = Path(data_dir)

    return {
        name: pd.read_csv(data_path / filename)
        for name, filename in DATASET_FILES.items()
    }