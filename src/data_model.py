
import pandas as pd
import streamlit as st

def _merge_with_validation(master_df, right_df, right_columns, join_key, label):
	"""Left merge one lookup table and report whether rows were preserved."""
	right_df = right_df[right_columns].copy()
	duplicate_keys = right_df[join_key].dropna().duplicated().sum()
	if duplicate_keys:
		raise ValueError(
			f"{label} has {duplicate_keys} duplicate non-missing "
			f"values in its join key: {join_key}"
		)

	original_row_count = len(master_df)
	merged_df = master_df.merge(
		right_df,
		how="left",
		on=join_key,
		validate="many_to_one",
	)
	print(f"After merging {label}: {len(merged_df)} rows")

	if len(merged_df) != original_row_count:
		raise ValueError(
			f"{label} changed the transaction row count from "
			f"{original_row_count} to {len(merged_df)}"
		)

	return merged_df


@st.cache_data
def build_sales_master(datasets):
	"""Build a transaction-level sales master from cleaned datasets."""
	master_df = datasets["order_items"].copy()
	print(f"Starting with order_items: {len(master_df)} rows")

	master_df = _merge_with_validation(
		master_df,
		datasets["orders"],
		[
			"order_id",
			"customer_id",
			"order_status",
			"order_purchase_timestamp",
			"order_approved_at",
			"order_delivered_carrier_date",
			"order_delivered_customer_date",
			"order_estimated_delivery_date",
		],
		"order_id",
		"orders",
	)
	master_df = _merge_with_validation(
		master_df,
		datasets["products"],
		[
			"product_id",
			"product_category_name",
			"product_weight_g",
			"product_length_cm",
			"product_height_cm",
			"product_width_cm",
		],
		"product_id",
		"products",
	)
	master_df = _merge_with_validation(
		master_df,
		datasets["sellers"],
		["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
		"seller_id",
		"sellers",
	)
	master_df = _merge_with_validation(
		master_df,
		datasets["category_translation"],
		["product_category_name", "product_category_name_english"],
		"product_category_name",
		"category translation",
	)

	print(f"Final sales master: {len(master_df)} rows")
	return master_df


def validate_sales_master(master_df, original_order_items):
	"""Report row preservation and important missing values in the master."""
	original_row_count = len(original_order_items)
	final_row_count = len(master_df)
	validation_report = {
		"original_order_items_rows": original_row_count,
		"final_master_rows": final_row_count,
		"row_counts_match": original_row_count == final_row_count,
		"missing_product_categories": int(
			master_df["product_category_name"].isna().sum()
		),
		"missing_english_categories": int(
			master_df["product_category_name_english"].isna().sum()
		),
		"missing_seller_ids": int(master_df["seller_id"].isna().sum()),
		"missing_customer_ids": int(master_df["customer_id"].isna().sum()),
		"missing_order_ids": int(master_df["order_id"].isna().sum()),
		"columns": list(master_df.columns),
	}

	print("=== Sales master validation ===")
	print(f"Original order_items rows: {original_row_count}")
	print(f"Final master rows: {final_row_count}")
	print(f"Row counts match: {validation_report['row_counts_match']}")
	print(
		"Missing product categories: "
		f"{validation_report['missing_product_categories']}"
	)
	print(
		"Missing English categories: "
		f"{validation_report['missing_english_categories']}"
	)
	print(f"Missing seller IDs: {validation_report['missing_seller_ids']}")
	print(f"Missing customer IDs: {validation_report['missing_customer_ids']}")
	print(f"Missing order IDs: {validation_report['missing_order_ids']}")
	print(f"Final columns: {validation_report['columns']}")

	return validation_report
    