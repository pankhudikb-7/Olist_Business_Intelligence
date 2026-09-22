import pandas as pd


DATE_COLUMNS = {
	"orders": [
		"order_purchase_timestamp",
		"order_approved_at",
		"order_delivered_carrier_date",
		"order_delivered_customer_date",
		"order_estimated_delivery_date",
	],
	"order_items": ["shipping_limit_date"],
	"order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

ID_COLUMNS = {
	"customers": ["customer_id", "customer_unique_id"],
	"orders": ["order_id", "customer_id"],
	"order_items": ["order_id", "product_id", "seller_id"],
	"order_payments": ["order_id"],
	"order_reviews": ["review_id", "order_id"],
	"products": ["product_id"],
	"sellers": ["seller_id"],
}


def clean_datasets(datasets):
	"""Return cleaned copies of the loaded Olist datasets."""
	cleaned_datasets = {
		name: dataframe.copy()
		for name, dataframe in datasets.items()
	}

	for dataset_name, columns in DATE_COLUMNS.items():
		dataframe = cleaned_datasets[dataset_name]
		for column in columns:
			dataframe[column] = pd.to_datetime(
				dataframe[column],
				errors="coerce",
			)

	for dataset_name, columns in ID_COLUMNS.items():
		dataframe = cleaned_datasets[dataset_name]
		for column in columns:
			dataframe[column] = dataframe[column].astype("string")

	cleaned_datasets["products"]["product_category_name"] = (
		cleaned_datasets["products"]["product_category_name"].fillna("Unknown")
	)

	return cleaned_datasets


def validate_cleaned_datasets(datasets):
	"""Report date conversions, missing converted dates, and Unknown categories."""
	validation_report = {}

	print("=== Date column validation ===")
	for dataset_name, columns in DATE_COLUMNS.items():
		dataframe = datasets[dataset_name]
		for column in columns:
			result = {
				"dtype": str(dataframe[column].dtype),
				"missing_values": int(dataframe[column].isna().sum()),
			}
			validation_report[f"{dataset_name}.{column}"] = result
			print(
				f"{dataset_name}.{column}: "
				f"dtype={result['dtype']}, "
				f"missing_values={result['missing_values']}"
			)

	unknown_categories = int(
		(datasets["products"]["product_category_name"] == "Unknown").sum()
	)
	validation_report["unknown_product_categories"] = unknown_categories
	print(f"Unknown product categories: {unknown_categories}")

	return validation_report
