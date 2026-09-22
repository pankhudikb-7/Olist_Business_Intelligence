import math

import pandas as pd


def get_total_customers(customers):
    """Return the number of unique customers."""
    return int(customers["customer_unique_id"].nunique())


def _orders_with_unique_customers(datasets):
    """Add customer_unique_id to orders without changing the order grain."""
    customers = datasets["customers"][["customer_id", "customer_unique_id"]]
    return datasets["orders"].merge(
        customers,
        how="left",
        on="customer_id",
        validate="many_to_one",
    )


def get_active_customers(datasets):
    """Return unique customers with at least one order."""
    orders = _orders_with_unique_customers(datasets)
    return int(orders["customer_unique_id"].dropna().nunique())


def get_repeat_customer_rate(datasets):
    """Return the percentage of unique customers with more than one order."""
    orders = _orders_with_unique_customers(datasets)
    order_counts = orders.dropna(subset=["customer_unique_id"]).groupby(
        "customer_unique_id"
    )["order_id"].nunique()
    if order_counts.empty:
        return 0.0
    return float((order_counts.gt(1).mean()) * 100)


def get_total_orders(master_df):
    """Return the number of unique orders in the sales master."""
    return int(master_df["order_id"].nunique())


def get_total_sales(master_df):
    """Return item sales, excluding freight value."""
    return float(master_df["price"].sum())


def get_total_order_value(master_df):
    """Return item price plus freight value."""
    return float((master_df["price"] + master_df["freight_value"]).sum())


def get_average_order_value(master_df):
    """Return total order value divided by unique orders."""
    total_orders = get_total_orders(master_df)
    if total_orders == 0:
        return 0.0
    return get_total_order_value(master_df) / total_orders


def get_items_sold(master_df):
    """Return the number of order-item records."""
    return int(len(master_df))


def _category_column(master_df):
    """Return English categories, using Unknown for untranslated categories."""
    return master_df["product_category_name_english"].fillna("Unknown")


def get_category_sales(master_df):
    """Return sales grouped by English product category."""
    result = master_df.assign(category=_category_column(master_df)).groupby(
        "category", as_index=False
    )["price"].sum()
    return result.rename(columns={"category": "product_category_name_english", "price": "sales"})


def get_category_items(master_df):
    """Return item counts grouped by English product category."""
    result = master_df.assign(category=_category_column(master_df)).groupby(
        "category", as_index=False
    ).size()
    return result.rename(
        columns={"category": "product_category_name_english", "size": "items_sold"}
    )


def get_category_sales_share(master_df):
    """Return each category's percentage share of total sales."""
    result = get_category_sales(master_df)
    total_sales = result["sales"].sum()
    result["sales_share"] = 0.0 if total_sales == 0 else result["sales"] / total_sales
    return result


def get_product_sales(master_df):
    """Return sales grouped by product ID."""
    return master_df.groupby("product_id", as_index=False)["price"].sum().rename(
        columns={"price": "sales"}
    )


def get_product_items(master_df):
    """Return item counts grouped by product ID."""
    return master_df.groupby("product_id", as_index=False).size().rename(
        columns={"size": "items_sold"}
    )


def get_average_selling_price(master_df):
    """Return the average price per order-item record."""
    return float(master_df["price"].mean())


def get_seller_sales(master_df):
    """Return sales grouped by seller ID."""
    return master_df.groupby("seller_id", as_index=False)["price"].sum().rename(
        columns={"price": "sales"}
    )


def get_seller_orders(master_df):
    """Return unique orders grouped by seller ID."""
    return master_df.groupby("seller_id", as_index=False)["order_id"].nunique().rename(
        columns={"order_id": "orders"}
    )


def get_seller_items(master_df):
    """Return item counts grouped by seller ID."""
    return master_df.groupby("seller_id", as_index=False).size().rename(
        columns={"size": "items_sold"}
    )


def get_payment_summary(datasets):
    """Return payment transaction counts, totals, and averages by payment type."""
    return datasets["order_payments"].groupby("payment_type", as_index=False).agg(
        transaction_count=("payment_type", "size"),
        total_payment_value=("payment_value", "sum"),
        average_payment_value=("payment_value", "mean"),
    )


def get_payment_method_share(datasets):
    """Return the share of payment transactions by payment type."""
    result = get_payment_summary(datasets)[["payment_type", "transaction_count"]]
    total_transactions = result["transaction_count"].sum()
    result["transaction_share"] = (
        0.0
        if total_transactions == 0
        else result["transaction_count"] / total_transactions
    )
    return result


def get_installment_summary(datasets):
    """Return descriptive statistics for payment installments."""
    return datasets["order_payments"]["payment_installments"].describe().to_frame(
        name="payment_installments"
    )


def get_review_summary(datasets):
    """Return review count, average score, and 1-star and 5-star rates."""
    reviews = datasets["order_reviews"]
    total_reviews = len(reviews)
    scores = reviews["review_score"]
    return {
        "total_reviews": int(total_reviews),
        "average_review_score": float(scores.mean()),
        "five_star_rate": float(scores.eq(5).mean() * 100),
        "one_star_rate": float(scores.eq(1).mean() * 100),
    }


def get_category_rating(datasets):
    """Return average review score by category without multiplying review records."""
    reviews = datasets["order_reviews"][["order_id", "review_score"]]
    order_categories = datasets["order_items"][["order_id", "product_id"]].merge(
        datasets["products"][["product_id", "product_category_name"]],
        how="left",
        on="product_id",
        validate="many_to_one",
    )
    order_categories = order_categories[["order_id", "product_category_name"]].drop_duplicates(
        subset=["order_id", "product_category_name"]
    )
    review_categories = reviews.merge(
        order_categories,
        how="left",
        on="order_id",
        validate="many_to_many",
    )
    review_categories["product_category_name"] = review_categories[
        "product_category_name"
    ].fillna("Unknown")
    return review_categories.groupby("product_category_name", as_index=False)[
        "review_score"
    ].mean().rename(columns={"review_score": "average_review_score"})


def get_delivery_metrics(datasets):
    """Return delivery duration, delay, late rate, and on-time rate."""
    orders = datasets["orders"].copy()
    delivery_days = (
        orders["order_delivered_customer_date"]
        - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)
    delay_days = (
        orders["order_delivered_customer_date"]
        - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 60 * 60)
    valid_delay = delay_days.dropna()
    late_rate = float(valid_delay.gt(0).mean() * 100) if len(valid_delay) else 0.0
    return {
        "average_delivery_days": float(delivery_days.mean()),
        "average_delay_days": float(delay_days.mean()),
        "late_delivery_rate": late_rate,
        "on_time_delivery_rate": 100 - late_rate if len(valid_delay) else 0.0,
    }


def validate_analytics_metrics(datasets, master_df):
    """Run basic sanity checks for the analytical metrics."""
    total_customers = get_total_customers(datasets["customers"])
    total_orders = get_total_orders(master_df)
    total_sales = get_total_sales(master_df)
    items_sold = get_items_sold(master_df)
    repeat_rate = get_repeat_customer_rate(datasets)
    late_rate = get_delivery_metrics(datasets)["late_delivery_rate"]
    category_total = get_category_sales(master_df)["sales"].sum()
    product_total = get_product_sales(master_df)["sales"].sum()
    seller_total = get_seller_sales(master_df)["sales"].sum()

    checks = {
        "total_customers_positive": total_customers > 0,
        "total_orders_positive": total_orders > 0,
        "total_sales_nonnegative": total_sales >= 0,
        "items_sold_positive": items_sold > 0,
        "repeat_customer_rate_valid": 0 <= repeat_rate <= 100,
        "late_delivery_rate_valid": 0 <= late_rate <= 100,
        "category_sales_match": math.isclose(category_total, total_sales),
        "product_sales_match": math.isclose(product_total, total_sales),
        "seller_sales_match": math.isclose(seller_total, total_sales),
    }
    print("=== Analytics validation ===")
    for check_name, passed in checks.items():
        print(f"{check_name}: {passed}")
    return checks
