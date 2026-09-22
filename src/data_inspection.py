from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_datasets


def likely_key_columns(dataframe):
    """Return columns that look like identifiers or lookup fields by name."""
    key_names = []
    for column in dataframe.columns:
        if column.endswith("_id") or column == "id":
            key_names.append(column)
        elif "zip_code_prefix" in column:
            key_names.append(column)
        elif column == "product_category_name":
            key_names.append(column)
    return key_names


def inspect_dataset(name, dataframe):
    print(f"\n=== {name} ===")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Column names: {list(dataframe.columns)}")
    print("Data types:")
    print(dataframe.dtypes.to_string())
    print("Missing-value counts:")
    print(dataframe.isna().sum().to_string())
    print(f"Likely key/ID columns: {likely_key_columns(dataframe)}")


def check_relationship(label, parent, parent_key, child, child_key):
    """Report nulls, duplicate parent keys, and unmatched child references."""
    parent_values = parent[parent_key].dropna()
    child_values = child[child_key].dropna()
    unmatched = child_values[~child_values.isin(parent_values)]
    duplicate_parent_keys = parent_values[parent_values.duplicated()].nunique()

    print(f"\n{label}")
    print(f"  Parent: {parent_key}; child: {child_key}")
    print(f"  Parent missing keys: {parent[parent_key].isna().sum()}")
    print(f"  Child missing keys: {child[child_key].isna().sum()}")
    print(f"  Unmatched child values: {unmatched.nunique()} distinct ({len(unmatched)} rows)")
    print(f"  Duplicate parent key values: {duplicate_parent_keys} distinct")


def main():
    datasets = load_datasets(PROJECT_ROOT / "data")

    for name, dataframe in datasets.items():
        inspect_dataset(name, dataframe)

    print("\n=== Relationship checks ===")
    check_relationship(
        "Customers -> orders through customer_id",
        datasets["customers"],
        "customer_id",
        datasets["orders"],
        "customer_id",
    )
    check_relationship(
        "Orders -> payments through order_id",
        datasets["orders"],
        "order_id",
        datasets["order_payments"],
        "order_id",
    )
    check_relationship(
        "Orders -> reviews through order_id",
        datasets["orders"],
        "order_id",
        datasets["order_reviews"],
        "order_id",
    )
    check_relationship(
        "Orders -> order_items through order_id",
        datasets["orders"],
        "order_id",
        datasets["order_items"],
        "order_id",
    )
    check_relationship(
        "Order_items -> products through product_id",
        datasets["products"],
        "product_id",
        datasets["order_items"],
        "product_id",
    )
    check_relationship(
        "Order_items -> sellers through seller_id",
        datasets["sellers"],
        "seller_id",
        datasets["order_items"],
        "seller_id",
    )
    check_relationship(
        "Products -> category translation through product_category_name",
        datasets["category_translation"],
        "product_category_name",
        datasets["products"],
        "product_category_name",
    )

    print("\n=== Geolocation note ===")
    geolocation = datasets["geolocation"]
    zip_counts = geolocation["geolocation_zip_code_prefix"].value_counts()
    repeated_prefixes = (zip_counts > 1).sum()
    print(
        "Geolocation ZIP-code prefixes are not a simple one-row-per-ZIP lookup: "
        f"{repeated_prefixes} prefixes repeat."
    )


if __name__ == "__main__":
    main()
