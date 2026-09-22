"""Deterministic natural-language analysis for the GenAI Analyst page."""

import re

import pandas as pd

from analytics import (
	get_delivery_metrics,
	get_payment_summary,
	get_repeat_customer_rate,
	get_review_summary,
	get_total_customers,
	get_total_orders,
	get_total_sales,
)


def _currency(value):
	return f"R$ {float(value):,.2f}"


def _number(value):
	return f"{int(value):,}"


def _percentage(value):
	return f"{float(value):.2f}%"


def _period_column(sales_master, period):
	return sales_master["order_purchase_timestamp"].dt.to_period(period).dt.to_timestamp()


def _quarter_label(period):
	return f"Q{period.quarter} {period.year}"


def _sales_by_period(sales_master, period):
	result = (
		sales_master.assign(period=_period_column(sales_master, period))
		.groupby("period", as_index=False)["price"]
		.sum()
		.rename(columns={"price": "value"})
	)
	return result.sort_values("period")


def _orders_by_period(sales_master, period):
	return (
		sales_master.assign(period=_period_column(sales_master, period))
		.groupby("period", as_index=False)["order_id"]
		.nunique()
		.rename(columns={"order_id": "value"})
		.sort_values("period")
	)


def _period_from_question(question, period):
	quarter_match = re.search(r"\bq([1-4])\s*(20\d{2})\b", question)
	if period == "Q" and quarter_match:
		return pd.Period(f"{quarter_match.group(2)}Q{quarter_match.group(1)}")

	year_match = re.search(r"\b(20\d{2})\b", question)
	if period == "Y" and year_match:
		return pd.Period(year_match.group(1), freq="Y")

	month_match = re.search(
		r"\b(january|february|march|april|may|june|july|august|"
		r"september|october|november|december)\s+(20\d{2})\b",
		question,
	)
	if period == "M" and month_match:
		month_number = pd.Timestamp(month_match.group(1), format="%B").month
		return pd.Period(f"{month_match.group(2)}-{month_number:02d}", freq="M")
	return None


def _sales_answer(question, sales_master):
	if "quarter" in question or re.search(r"\bq[1-4]\b", question):
		values = _sales_by_period(sales_master, "Q")
		requested = _period_from_question(question, "Q")
		if requested is not None:
			selected = values[values["period"].dt.to_period("Q") == requested]
			if len(selected) == 0:
				return None
			value = selected.iloc[0]["value"]
			label = _quarter_label(requested)
			return {"answer": f"Sales in {label} were {_currency(value)}.", "metric": "Sales", "result": f"{label}: {_currency(value)}", "methodology": "Sales were summed from order-item prices by calendar quarter using order purchase dates."}
		if requested is None and re.search(r"\bq[1-4]\b", question):
			quarter_number = int(re.search(r"\bq([1-4])\b", question).group(1))
			quarter_values = values[values["period"].dt.quarter == quarter_number]
			if len(quarter_values) == 0:
				return None
			row = quarter_values.loc[quarter_values["value"].idxmax()]
			label = _quarter_label(row["period"].to_period("Q"))
			return {"answer": f"{label} had the highest sales among Q{quarter_number} periods, with {_currency(row['value'])}.", "metric": "Quarterly sales", "result": f"{label}: {_currency(row['value'])}", "methodology": "Sales were summed from order-item prices by calendar quarter using order purchase dates."}
		if "compare" in question or " and " in question:
			periods = re.findall(r"\bq([1-4])\s*(20\d{2})\b", question)
			if len(periods) >= 2:
				labels = []
				for quarter, year in periods[:2]:
					period_key = pd.Period(f"{year}Q{quarter}")
					row = values[values["period"].dt.to_period("Q") == period_key]
					if len(row) == 0:
						return None
					labels.append(f"Q{quarter} {year}: {_currency(row.iloc[0]['value'])}")
				return {"answer": "Compared sales: " + "; ".join(labels) + ".", "metric": "Quarterly sales comparison", "result": "; ".join(labels), "methodology": "Sales were summed from order-item prices by calendar quarter using order purchase dates."}
		row = values.loc[values["value"].idxmin() if "lowest" in question else values["value"].idxmax()]
		label = _quarter_label(row["period"].to_period("Q"))
		return {"answer": f"{label} had the {'lowest' if 'lowest' in question else 'highest'} sales, with {_currency(row['value'])}.", "metric": "Quarterly sales", "result": f"{label}: {_currency(row['value'])}", "methodology": "Sales were summed from order-item prices by calendar quarter using order purchase dates."}

	if "month" in question:
		values = _sales_by_period(sales_master, "M")
		requested = _period_from_question(question, "M")
		if requested is not None:
			selected = values[values["period"].dt.to_period("M") == requested]
			if len(selected) == 0:
				return None
			value = selected.iloc[0]["value"]
			label = requested.strftime("%B %Y")
			return {"answer": f"Sales in {label} were {_currency(value)}.", "metric": "Monthly sales", "result": f"{label}: {_currency(value)}", "methodology": "Sales were summed from order-item prices by month using order purchase dates."}
		row = values.loc[values["value"].idxmin() if "lowest" in question else values["value"].idxmax()]
		label = row["period"].strftime("%B %Y")
		return {"answer": f"{label} had the {'lowest' if 'lowest' in question else 'highest'} sales, with {_currency(row['value'])}.", "metric": "Monthly sales", "result": f"{label}: {_currency(row['value'])}", "methodology": "Sales were summed from order-item prices by month using order purchase dates."}

	if "total sales" in question or "how much did we sell" in question:
		value = get_total_sales(sales_master)
		return {"answer": f"Total sales were {_currency(value)}.", "metric": "Total sales", "result": _currency(value), "methodology": "Total sales are the sum of order-item prices in the sales master."}
	return None


def _order_answer(question, sales_master):
	period = "Q" if "quarter" in question or re.search(r"\bq[1-4]\s*20\d{2}\b", question) else "M" if "month" in question else "Y" if "year" in question else None
	if period is None:
		if "average order value" in question:
			order_values = sales_master.assign(value=sales_master["price"] + sales_master["freight_value"]).groupby("order_id")["value"].sum()
			value = order_values.mean() if len(order_values) else 0.0
			return {"answer": f"The average order value was {_currency(value)}.", "metric": "Average order value", "result": _currency(value), "methodology": "Order values were summed per unique order from item price plus freight, then averaged."}
		if "total orders" in question:
			value = get_total_orders(sales_master)
			return {"answer": f"There were {_number(value)} total orders.", "metric": "Total orders", "result": _number(value), "methodology": "Orders were counted as unique order IDs in the sales master."}
		return None
	values = _orders_by_period(sales_master, period)
	requested = _period_from_question(question, period)
	if requested is not None:
		selected = values[values["period"].dt.to_period(period) == requested]
		if len(selected) == 0:
			return None
		value = selected.iloc[0]["value"]
		label = _quarter_label(requested) if period == "Q" else requested.strftime("%B %Y") if period == "M" else str(requested.year)
		return {"answer": f"There were {_number(value)} orders in {label}.", "metric": "Orders", "result": f"{label}: {_number(value)}", "methodology": "Unique order IDs were counted by period using order purchase dates."}
	row = values.loc[values["value"].idxmax()]
	label = _quarter_label(row["period"].to_period("Q")) if period == "Q" else row["period"].strftime("%B %Y") if period == "M" else str(row["period"].year)
	return {"answer": f"{label} had the highest order count, with {_number(row['value'])} orders.", "metric": "Orders by period", "result": f"{label}: {_number(row['value'])}", "methodology": "Unique order IDs were counted by period using order purchase dates."}


def _category_answer(question, sales_master):
	grouped = sales_master.assign(category=sales_master["product_category_name_english"].fillna("Unknown")).groupby("category").agg(revenue=("price", "sum"), items=("price", "size"))
	requested_name = next((category for category in grouped.index if category.lower() in question), None)
	if requested_name is not None and ("revenue" in question or "sales" in question or "item" in question):
		metric = "items" if "item" in question else "revenue"
		value = grouped.loc[requested_name, metric]
		formatted = _number(value) if metric == "items" else _currency(value)
		return {"answer": f"{requested_name} had {formatted} {'items sold' if metric == 'items' else 'in revenue'}.", "metric": f"Category {metric}", "result": f"{requested_name}: {formatted}", "methodology": "Category metrics were calculated from order-item records using translated product categories."}
	if "share" in question or "percentage" in question or "percent" in question:
		name = next((category for category in grouped.index if category.lower() in question), None)
		if name is None and "top" in question:
			name = grouped["revenue"].idxmax()
		if name is None:
			return None
		value = grouped.loc[name, "revenue"] / grouped["revenue"].sum() * 100
		return {"answer": f"{name} represented {_percentage(value)} of sales.", "metric": "Top category sales share" if "top" in question else "Category sales share", "result": f"{name}: {_percentage(value)}", "methodology": "Category share is category item-price revenue divided by total item-price sales."}
	name = grouped["revenue"].idxmin() if "lowest" in question else grouped["revenue"].idxmax()
	if "item" in question:
		return {"answer": f"{name} sold {_number(grouped.loc[name, 'items'])} items.", "metric": "Category items sold", "result": f"{name}: {_number(grouped.loc[name, 'items'])}", "methodology": "Order-item records were counted by translated product category."}
	return {"answer": f"{name} generated {_currency(grouped.loc[name, 'revenue'])} in revenue.", "metric": "Category revenue", "result": f"{name}: {_currency(grouped.loc[name, 'revenue'])}", "methodology": "Order-item prices were summed by translated product category."}


def _product_answer(question, sales_master):
	grouped = sales_master.groupby("product_id").agg(revenue=("price", "sum"), items=("price", "size"), average_price=("price", "mean"))
	requested_id = next((product_id for product_id in grouped.index if str(product_id).lower() in question), None)
	if requested_id is not None:
		metric = "items" if "item" in question or "volume" in question else "revenue"
		value = grouped.loc[requested_id, metric]
		formatted = _number(value) if metric == "items" else _currency(value)
		return {"answer": f"Product {requested_id} had {formatted} {'items sold' if metric == 'items' else 'in revenue'}.", "metric": f"Product {metric}", "result": f"{requested_id}: {formatted}", "methodology": "Product metrics were calculated at product_id grain from order-item records."}
	if "average selling price" in question:
		value = float(sales_master["price"].mean())
		return {"answer": f"The average selling price was {_currency(value)} per item.", "metric": "Average selling price", "result": _currency(value), "methodology": "The mean order-item price was calculated from the sales master."}
	metric = "items" if "most-sold" in question or "most sold" in question or "item" in question else "revenue"
	name = grouped[metric].idxmax()
	value = grouped.loc[name, metric]
	return {"answer": f"Product {name} had the highest {'item volume' if metric == 'items' else 'revenue'}, at {_number(value) if metric == 'items' else _currency(value)}.", "metric": f"Product {metric}", "result": f"{name}: {_number(value) if metric == 'items' else _currency(value)}", "methodology": "Product metrics were calculated at product_id grain from order-item records. The source data does not include product names."}


def _seller_answer(question, datasets, sales_master):
	grouped = sales_master.groupby("seller_id").agg(revenue=("price", "sum"), orders=("order_id", "nunique"), items=("price", "size"))
	requested_id = next((seller_id for seller_id in grouped.index if str(seller_id).lower() in question), None)
	if requested_id is not None:
		metric = "orders" if "order" in question else "items" if "item" in question else "revenue"
		value = grouped.loc[requested_id, metric]
		formatted = _number(value) if metric != "revenue" else _currency(value)
		return {"answer": f"Seller {requested_id} had {formatted} {'orders' if metric == 'orders' else 'items sold' if metric == 'items' else 'in revenue'}.", "metric": f"Seller {metric}", "result": f"{requested_id}: {formatted}", "methodology": "Seller metrics were calculated from the transaction-level sales master; orders are unique order IDs."}
	if "rating" in question:
		order_sellers = datasets["order_items"][["order_id", "seller_id"]].drop_duplicates()
		reviews = datasets["order_reviews"][["review_id", "order_id", "review_score"]].drop_duplicates("review_id")
		ratings = reviews.merge(order_sellers, on="order_id", how="inner").drop_duplicates(["order_id", "seller_id", "review_id"]).groupby("seller_id")["review_score"].mean()
		value = ratings.mean() if len(ratings) else 0.0
		return {"answer": f"The average seller rating was {value:.2f} / 5.", "metric": "Average seller rating", "result": f"{value:.2f} / 5", "methodology": "Review scores were mapped to distinct seller-order-review relationships before averaging seller ratings."}
	metric = "orders" if "order" in question else "items" if "item" in question else "revenue"
	name = grouped[metric].idxmax()
	value = grouped.loc[name, metric]
	formatted = _number(value) if metric != "revenue" else _currency(value)
	return {"answer": f"Seller {name} had the highest {metric}, at {formatted}.", "metric": f"Seller {metric}", "result": f"{name}: {formatted}", "methodology": "Seller metrics were calculated from the transaction-level sales master; orders are unique order IDs."}


def _payment_answer(question, datasets):
	summary = get_payment_summary(datasets)
	if "average" in question:
		value = datasets["order_payments"]["payment_value"].mean()
		return {"answer": f"The average payment value was {_currency(value)}.", "metric": "Average payment value", "result": _currency(value), "methodology": "Payment values were averaged at payment-record grain without joining to order items."}
	if "count" in question or "transaction" in question:
		value = len(datasets["order_payments"])
		return {"answer": f"There were {_number(value)} payment transactions.", "metric": "Payment transaction count", "result": _number(value), "methodology": "Payment records were counted directly at payment-record grain."}
	if "common" in question or "most used" in question:
		row = summary.loc[summary["transaction_count"].idxmax()]
		return {"answer": f"{row['payment_type']} was the most common payment method, with {_number(row['transaction_count'])} transactions.", "metric": "Most common payment method", "result": f"{row['payment_type']}: {_number(row['transaction_count'])} transactions", "methodology": "Payment records were counted by payment type at payment-record grain."}
	row = summary.loc[summary["total_payment_value"].idxmax()]
	return {"answer": f"{row['payment_type']} had the highest payment value, at {_currency(row['total_payment_value'])}.", "metric": "Payment value by method", "result": f"{row['payment_type']}: {_currency(row['total_payment_value'])}", "methodology": "Payment values were summed by payment type at payment-record grain."}


def _structured_metadata(question, result):
	"""Return stable metadata for a successful analyst result."""
	metric = result["metric"].lower()

	if metric == "sales" or "quarterly sales" in metric:
		return {"intent": "sales_by_quarter", "chart_type": "line"}
	if "monthly sales" in metric:
		return {"intent": "sales_by_month", "chart_type": "line"}
	if "total sales" in metric:
		return {"intent": "total_sales", "chart_type": "metric"}
	if metric == "orders" or "orders by period" in metric:
		return {"intent": "orders_by_period", "chart_type": "line"}
	if "total orders" in metric:
		return {"intent": "total_orders", "chart_type": "metric"}
	if "average order value" in metric:
		return {"intent": "average_order_value", "chart_type": "metric"}
	if metric.startswith("category"):
		return {"intent": "category_analysis", "chart_type": "bar"}
	if metric.startswith("product") or "average selling price" in metric:
		return {"intent": "product_analysis", "chart_type": "bar"}
	if metric.startswith("seller"):
		return {"intent": "seller_analysis", "chart_type": "bar"}
	if "payment" in metric:
		return {"intent": "payment_analysis", "chart_type": "bar"}
	if "customer" in metric:
		return {"intent": "customer_analysis", "chart_type": "donut"}
	if "review" in metric or "star" in metric:
		return {"intent": "review_analysis", "chart_type": "bar"}
	if "deliver" in metric or "delay" in metric or "on-time" in metric:
		return {"intent": "delivery_analysis", "chart_type": "bar"}

	return {"intent": "business_analysis", "chart_type": "table"}


def _structured_analysis(question, datasets, sales_master, result):
	"""Add deterministic supporting values for the analysis result."""
	intent = result["intent"]
	metric = result["metric"].lower()
	structured = {"supporting_metrics": []}

	if intent == "sales_by_quarter":
		periods = _sales_by_period(sales_master, "Q")
		periods["label"] = periods["period"].map(
			lambda value: _quarter_label(value.to_period("Q"))
		)
		structured["chart_data"] = {
			"labels": periods["label"].tolist(),
			"values": periods["value"].astype(float).tolist(),
		}
		structured["supporting_metrics"] = [
			{"label": "Selected period", "value": result["result"]},
			{"label": "Total sales", "value": _currency(periods["value"].sum())},
			{"label": "Quarters available", "value": _number(len(periods))},
		]
	elif intent == "sales_by_month":
		periods = _sales_by_period(sales_master, "M")
		periods["label"] = periods["period"].dt.strftime("%b %Y")
		structured["chart_data"] = {
			"labels": periods["label"].tolist(),
			"values": periods["value"].astype(float).tolist(),
		}
		structured["supporting_metrics"] = [
			{"label": "Selected period", "value": result["result"]},
			{"label": "Total sales", "value": _currency(periods["value"].sum())},
			{"label": "Months available", "value": _number(len(periods))},
		]
	elif intent == "total_sales":
		structured["supporting_metrics"] = [
			{"label": "Total sales", "value": result["result"]},
			{"label": "Items sold", "value": _number(len(sales_master))},
		]
	elif intent == "orders_by_period":
		period = "Q" if "quarter" in question or re.search(r"\bq[1-4]\b", question) else "M" if "month" in question else "Y"
		periods = _orders_by_period(sales_master, period)
		periods["label"] = periods["period"].map(
			lambda value: _quarter_label(value.to_period("Q")) if period == "Q" else value.strftime("%b %Y") if period == "M" else str(value.year)
		)
		structured["chart_data"] = {
			"labels": periods["label"].tolist(),
			"values": periods["value"].astype(int).tolist(),
		}
		structured["supporting_metrics"] = [
			{"label": "Total orders", "value": _number(periods["value"].sum())},
			{"label": "Periods available", "value": _number(len(periods))},
		]
	elif intent == "total_orders":
		structured["supporting_metrics"] = [
			{"label": "Total orders", "value": result["result"]},
		]
	elif intent == "average_order_value":
		structured["supporting_metrics"] = [
			{"label": "Average order value", "value": result["result"]},
			{"label": "Orders", "value": _number(sales_master["order_id"].nunique())},
		]
	elif intent == "category_analysis":
		categories = sales_master.assign(
			category=sales_master["product_category_name_english"].fillna("Unknown")
		).groupby("category", as_index=False).agg(
			revenue=("price", "sum"), items=("price", "size")
		).sort_values("revenue", ascending=False)
		category_name = next(
			(category for category in categories["category"] if category.lower() in question),
			categories.iloc[0]["category"],
		)
		if "items" in metric and not any(category.lower() in question for category in categories["category"]):
			category_name = categories.loc[categories["items"].idxmax(), "category"]
		selected = categories[categories["category"] == category_name].iloc[0]
		total_revenue = categories["revenue"].sum()
		structured["supporting_metrics"] = [
			{"label": "Category revenue", "value": _currency(selected["revenue"])},
			{"label": "Revenue share", "value": _percentage(selected["revenue"] / total_revenue * 100)},
			{"label": "Items sold", "value": _number(selected["items"])},
		]
		top_categories = categories.head(10)
		structured["chart_data"] = {
			"labels": top_categories["category"].tolist(),
			"values": top_categories["revenue"].astype(float).tolist(),
		}
	elif intent == "product_analysis":
		products = sales_master.groupby("product_id", as_index=False).agg(
			revenue=("price", "sum"), items=("price", "size"), average_price=("price", "mean")
		)
		requested_id = next(
			(product_id for product_id in products["product_id"] if str(product_id).lower() in question),
			None,
		)
		selected = products[products["product_id"] == requested_id].iloc[0] if requested_id else products.loc[products["items"].idxmax() if "item" in metric else products["revenue"].idxmax()]
		chart_metric = "items" if "item" in metric else "revenue"
		structured["supporting_metrics"] = [
			{"label": "Product ID", "value": str(selected["product_id"])},
			{"label": "Revenue", "value": _currency(selected["revenue"])},
			{"label": "Items sold", "value": _number(selected["items"])},
			{"label": "Average selling price", "value": _currency(selected["average_price"])},
		]
		top_products = products.nlargest(10, chart_metric)
		structured["chart_data"] = {
			"labels": top_products["product_id"].astype(str).tolist(),
			"values": top_products[chart_metric].astype(float).tolist(),
		}
		structured["limitations"] = "The Olist dataset provides product IDs but does not contain actual product names."
	elif intent == "seller_analysis":
		sellers = sales_master.groupby("seller_id", as_index=False).agg(
			revenue=("price", "sum"), orders=("order_id", "nunique"), items=("price", "size")
		).sort_values("revenue", ascending=False)
		seller_id = next(
			(seller for seller in sellers["seller_id"] if str(seller).lower() in question),
			sellers.iloc[0]["seller_id"],
		)
		selected = sellers[sellers["seller_id"] == seller_id].iloc[0]
		structured["supporting_metrics"] = [
			{"label": "Seller revenue", "value": _currency(selected["revenue"])},
			{"label": "Orders", "value": _number(selected["orders"])},
			{"label": "Items sold", "value": _number(selected["items"])},
		]
		top_sellers = sellers.head(10)
		structured["chart_data"] = {
			"labels": top_sellers["seller_id"].astype(str).tolist(),
			"values": top_sellers["revenue"].astype(float).tolist(),
		}
	elif intent == "payment_analysis":
		payments = get_payment_summary(datasets).sort_values("total_payment_value", ascending=False)
		structured["supporting_metrics"] = [
			{"label": "Payment value", "value": result["result"]},
			{"label": "Transactions", "value": _number(payments["transaction_count"].sum())},
			{"label": "Average payment", "value": _currency(datasets["order_payments"]["payment_value"].mean())},
		]
		structured["chart_data"] = {
			"labels": payments["payment_type"].tolist(),
			"values": payments["total_payment_value"].astype(float).tolist(),
		}
	elif intent == "customer_analysis":
		orders = datasets["orders"][["customer_id", "order_id"]].merge(
			datasets["customers"][["customer_id", "customer_unique_id"]],
			how="left", on="customer_id", validate="many_to_one",
		)
		counts = orders.groupby("customer_unique_id")["order_id"].nunique()
		repeat_count = int(counts.gt(1).sum())
		total_customers = int(counts.size)
		structured["supporting_metrics"] = [
			{"label": "Repeat customer rate", "value": _percentage(repeat_count / total_customers * 100 if total_customers else 0)},
			{"label": "Repeat customers", "value": _number(repeat_count)},
			{"label": "Total customers", "value": _number(total_customers)},
		]
		structured["chart_data"] = {
			"labels": ["Repeat customers", "One-order customers"],
			"values": [repeat_count, total_customers - repeat_count],
		}
	elif intent == "review_analysis":
		summary = get_review_summary(datasets)
		structured["supporting_metrics"] = [
			{"label": "Average review score", "value": f"{summary['average_review_score']:.2f} / 5"},
			{"label": "Five-star rate", "value": _percentage(summary["five_star_rate"])},
			{"label": "One-star rate", "value": _percentage(summary["one_star_rate"])},
		]
		structured["chart_data"] = {
			"labels": ["Five-star rate", "One-star rate"],
			"values": [summary["five_star_rate"], summary["one_star_rate"]],
		}
	elif intent == "delivery_analysis":
		metrics = get_delivery_metrics(datasets)
		structured["supporting_metrics"] = [
			{"label": "Late rate", "value": _percentage(metrics["late_delivery_rate"])},
			{"label": "On-time rate", "value": _percentage(metrics["on_time_delivery_rate"])},
			{"label": "Average delivery time", "value": f"{metrics['average_delivery_days']:.2f} days"},
			{"label": "Average delay", "value": f"{metrics['average_delay_days']:.2f} days"},
		]
		structured["chart_data"] = {
			"labels": ["On-time", "Late"],
			"values": [metrics["on_time_delivery_rate"], metrics["late_delivery_rate"]],
		}

	if intent in {"sales_by_quarter", "sales_by_month", "category_analysis", "product_analysis", "seller_analysis", "payment_analysis", "customer_analysis", "review_analysis", "delivery_analysis"}:
		structured["interpretation"] = result["answer"]

	return structured


def analyze_question(question, datasets, sales_master):
	"""Return a user-facing deterministic answer, or a clarification/error."""
	question = " ".join(question.lower().strip().split())
	if not question:
		return {"status": "clarification", "answer": "Please enter a business question."}
	if "highest value" in question and not any(word in question for word in ("sales", "payment", "order")):
		return {"status": "clarification", "answer": "Do you mean highest sales value, highest payment value, or highest order value?"}

	result = None
	profitability_terms = (
		"profit",
		"profit margin",
		"gross margin",
		"net margin",
		"commission",
		"cost",
	)
	if any(term in question for term in profitability_terms):
		return {
			"status": "unsupported",
			"answer": "Profit or profit margin cannot be calculated from the available Olist data because product costs, seller costs, commissions, and other expense data are not provided. I can analyze seller revenue, orders, items sold, and ratings instead.",
		}
	if "categor" in question:
		result = _category_answer(question, sales_master)
	elif "product" in question:
		result = _product_answer(question, sales_master)
	elif "seller" in question:
		result = _seller_answer(question, datasets, sales_master)
	elif "payment" in question:
		result = _payment_answer(question, datasets)
	elif any(word in question for word in ("sales", "sell", "revenue")):
		result = _sales_answer(question, sales_master)
	elif "order" in question:
		result = _order_answer(question, sales_master)
	elif "customer" in question or "repeat" in question:
		if "repeat" in question:
			orders = datasets["orders"][["customer_id", "order_id"]].merge(
				datasets["customers"][["customer_id", "customer_unique_id"]],
				how="left",
				on="customer_id",
				validate="many_to_one",
			).groupby("customer_unique_id")["order_id"].nunique()
			if "how many" in question or "count" in question:
				value = orders.gt(1).sum()
				result = {"answer": f"There were {_number(value)} repeat customers.", "metric": "Repeat customers", "result": _number(value), "methodology": "Customers with more than one unique order were counted by customer_id."}
			else:
				value = get_repeat_customer_rate(datasets)
				result = {"answer": f"The repeat customer rate was {_percentage(value)}.", "metric": "Repeat customer rate", "result": _percentage(value), "methodology": "Customers with more than one unique order were divided by all unique customers with an order."}
		elif "average" in question:
			orders = datasets["orders"].groupby("customer_id")["order_id"].nunique()
			value = orders.mean() if len(orders) else 0.0
			result = {"answer": f"Customers placed an average of {value:.2f} orders.", "metric": "Average orders per customer", "result": f"{value:.2f}", "methodology": "Unique orders were grouped by customer_id and averaged."}
		else:
			value = get_total_customers(datasets["customers"])
			result = {"answer": f"There were {_number(value)} customers.", "metric": "Total customers", "result": _number(value), "methodology": "Customers were counted as unique customer_unique_id values."}
	elif "review" in question or "star" in question:
		summary = get_review_summary(datasets)
		key, label, formatter = ("five_star_rate", "Five-star review rate", _percentage) if "five" in question else ("one_star_rate", "One-star review rate", _percentage) if "one" in question else ("average_review_score", "Average review score", lambda value: f"{value:.2f} / 5") if "average" in question else ("total_reviews", "Review count", _number)
		value = summary[key]
		result = {"answer": f"{label}: {formatter(value)}.", "metric": label, "result": formatter(value), "methodology": "Review metrics were calculated directly from the deduplicated review dataset."}
	elif "deliver" in question or "late" in question or "on-time" in question or "delay" in question:
		if "state" in question:
			orders = datasets["orders"][["order_id", "order_delivered_customer_date", "order_estimated_delivery_date", "customer_id"]].merge(
				datasets["customers"][["customer_id", "customer_state"]], how="left", on="customer_id", validate="many_to_one"
			)
			delay = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.total_seconds() / (24 * 60 * 60)
			orders = orders.assign(delay=delay).dropna(subset=["delay", "customer_state"])
			state_rates = orders.groupby("customer_state")["delay"].apply(lambda values: values.gt(0).mean() * 100)
			state = state_rates.idxmax()
			value = state_rates.loc[state]
			result = {"answer": f"{state} had the highest late-delivery rate, at {_percentage(value)}.", "metric": "Late-delivery rate by state", "result": f"{state}: {_percentage(value)}", "methodology": "Delivered orders were grouped by customer state; late means delivered after the estimated delivery date."}
		else:
			metrics = get_delivery_metrics(datasets)
			key, label = ("average_delivery_days", "Average delivery time") if "average delivery" in question else ("average_delay_days", "Average delay") if "average delay" in question or "delay" in question else ("on_time_delivery_rate", "On-time delivery rate") if "on-time" in question else ("late_delivery_rate", "Late delivery rate")
			value = metrics[key]
			formatted = _percentage(value) if "rate" in label.lower() else f"{value:.2f} days"
			result = {"answer": f"The {label.lower()} was {formatted}.", "metric": label, "result": formatted, "methodology": "Delivery metrics use order purchase, delivered-customer, and estimated-delivery dates at order grain."}

	if result is None:
		return {"status": "unsupported", "answer": "I can currently answer questions about sales, orders, customers, categories, products, sellers, payments, reviews, and delivery. Try asking: Which category had the highest sales?"}
	result.update(_structured_metadata(question, result))
	result.update(_structured_analysis(question, datasets, sales_master, result))
	result["status"] = "ok"
	return result