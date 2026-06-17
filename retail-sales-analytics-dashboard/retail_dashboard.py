"""Retail Sales Analytics Dashboard.

This script cleans a Superstore-style retail CSV, calculates business KPIs,
builds a portfolio-ready dashboard image, and writes an executive summary.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

_RUNTIME_ROOT = Path(__file__).resolve().parent
_MPL_CONFIG_DIR = _RUNTIME_ROOT / ".matplotlib-cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = _RUNTIME_ROOT
DATA_PATH = PROJECT_ROOT / "data" / "superstore.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.png"
CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Customer Name",
    "Product Name",
    "Category",
    "Sales",
]

BUSINESS_NUMERIC_COLUMNS = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]


def configure_logging() -> None:
    """Configure concise console logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def format_currency(value: float) -> str:
    """Format large currency values for labels and KPI cards."""
    value = float(value)
    absolute_value = abs(value)

    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if absolute_value >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


def shorten_label(value: Any, width: int = 42) -> str:
    """Shorten long product or customer labels without breaking the chart."""
    return textwrap.shorten(str(value), width=width, placeholder="...")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the Superstore CSV with a fallback encoding."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {csv_path}. "
            "Download the Kaggle Superstore Sales Dataset and place it here."
        )

    logging.info("Loading dataset from %s", csv_path)
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        logging.warning("UTF-8 read failed. Retrying with latin-1 encoding.")
        return pd.read_csv(csv_path, encoding="latin-1")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim accidental whitespace from CSV column names."""
    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_df.columns.astype(str).str.strip()
    return cleaned_df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure all columns needed for the portfolio analysis are present."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def parse_order_dates(order_dates: pd.Series) -> pd.Series:
    """Parse order dates and retry day-first parsing if needed."""
    parsed_dates = pd.to_datetime(order_dates, errors="coerce")

    if parsed_dates.isna().mean() > 0.5:
        logging.info("Retrying Order Date parsing with day-first format.")
        parsed_dates = pd.to_datetime(
            order_dates,
            errors="coerce",
            dayfirst=True,
        )

    return parsed_dates


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert business numeric columns to numeric dtype."""
    cleaned_df = df.copy()
    for column in BUSINESS_NUMERIC_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = (
                cleaned_df[column]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            )
    return cleaned_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using business-safe defaults.

    Missing-value treatment:
    - Categorical columns are filled with the mode, preserving the most common
      business label without creating a new category unless the whole column is
      blank.
    - Numerical columns are filled with the median, which is robust to sales
      outliers commonly found in retail datasets.
    - Order Date is a datetime field, so missing dates are filled with the
      median order date to keep monthly trend aggregation available.
    """
    cleaned_df = df.copy()

    categorical_columns = [
        column
        for column in cleaned_df.columns
        if (
            pd.api.types.is_object_dtype(cleaned_df[column].dtype)
            or pd.api.types.is_string_dtype(cleaned_df[column].dtype)
            or isinstance(cleaned_df[column].dtype, pd.CategoricalDtype)
        )
    ]
    for column in categorical_columns:
        mode_values = cleaned_df[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        median_value = cleaned_df[column].median()
        fill_value = 0 if pd.isna(median_value) else median_value
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    if "Order Date" in cleaned_df.columns and cleaned_df["Order Date"].isna().any():
        median_date = cleaned_df["Order Date"].dropna().median()
        if pd.isna(median_date):
            raise ValueError("Order Date column does not contain valid dates.")
        cleaned_df["Order Date"] = cleaned_df["Order Date"].fillna(median_date)

    return cleaned_df


def validate_data_types(df: pd.DataFrame) -> None:
    """Validate key data types after cleaning."""
    if not pd.api.types.is_datetime64_any_dtype(df["Order Date"]):
        raise TypeError("Order Date must be converted to datetime dtype.")
    if not pd.api.types.is_numeric_dtype(df["Sales"]):
        raise TypeError("Sales must be converted to numeric dtype.")
    if df.empty:
        raise ValueError("Cleaned dataset has no rows to analyze.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps and save the cleaned dataset."""
    cleaned_df = standardize_columns(df)
    validate_required_columns(cleaned_df)

    duplicate_count = int(cleaned_df.duplicated().sum())
    logging.info("Duplicate rows detected: %s", duplicate_count)
    cleaned_df = cleaned_df.drop_duplicates().copy()

    missing_before = cleaned_df.isna().sum()
    logging.info("Missing values before treatment:\n%s", missing_before)

    cleaned_df["Order Date"] = parse_order_dates(cleaned_df["Order Date"])
    cleaned_df = coerce_numeric_columns(cleaned_df)
    cleaned_df = fill_missing_values(cleaned_df)
    validate_data_types(cleaned_df)

    cleaned_df = cleaned_df.sort_values("Order Date").reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    logging.info("Cleaned dataset saved to %s", CLEANED_DATA_PATH)

    print("\nDataFrame Info:")
    cleaned_df.info()
    print("\nDataFrame Summary Statistics:")
    print(cleaned_df.describe(include="all"))

    return cleaned_df


def get_top_products(df: pd.DataFrame) -> pd.DataFrame:
    """Return the top 10 products by revenue."""
    return (
        df.groupby("Product Name", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly sales and a three-month moving average."""
    indexed_sales = df.set_index("Order Date")

    try:
        monthly_sales = (
            indexed_sales.resample("M")["Sales"]
            .sum()
            .to_frame(name="Sales")
        )
    except ValueError as exc:
        if "Invalid frequency: M" not in str(exc):
            raise
        logging.info(
            "Pandas no longer supports frequency 'M'; using 'ME' instead."
        )
        monthly_sales = (
            indexed_sales.resample("ME")["Sales"]
            .sum()
            .to_frame(name="Sales")
        )

    monthly_sales["Moving Average"] = (
        monthly_sales["Sales"].rolling(window=3, min_periods=1).mean()
    )
    return monthly_sales


def get_top_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Return the top 5 customers by total spending."""
    return (
        df.groupby("Customer Name", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(5)
    )


def get_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return total sales by category."""
    return (
        df.groupby("Category", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Calculate executive KPI card values."""
    total_revenue = float(df["Sales"].sum())
    total_orders = int(df["Order ID"].nunique())
    unique_customers = int(df["Customer Name"].nunique())
    average_order_value = total_revenue / total_orders if total_orders else 0.0

    return {
        "total_revenue": total_revenue,
        "total_orders": float(total_orders),
        "unique_customers": float(unique_customers),
        "average_order_value": average_order_value,
    }


def annotate_bars(ax: plt.Axes, values: pd.Series, horizontal: bool = False) -> None:
    """Add readable value annotations to bar charts."""
    for patch, value in zip(ax.patches, values):
        if horizontal:
            ax.text(
                patch.get_width() * 1.01,
                patch.get_y() + patch.get_height() / 2,
                format_currency(value),
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() * 1.01,
                format_currency(value),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )


def add_kpi_cards(fig: plt.Figure, kpis: dict[str, float]) -> None:
    """Add KPI cards across the top of the dashboard."""
    cards = [
        ("Total Revenue", format_currency(kpis["total_revenue"]), "#1F77B4"),
        ("Total Orders", f"{kpis['total_orders']:,.0f}", "#2CA02C"),
        ("Unique Customers", f"{kpis['unique_customers']:,.0f}", "#9467BD"),
        (
            "Average Order Value",
            format_currency(kpis["average_order_value"]),
            "#D62728",
        ),
    ]

    x_positions = [0.145, 0.385, 0.625, 0.865]
    for x_position, (label, value, color) in zip(x_positions, cards):
        fig.text(
            x_position,
            0.885,
            value,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "white",
                "edgecolor": "#D9DEE7",
                "linewidth": 1.0,
            },
        )
        fig.text(
            x_position,
            0.84,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#4B5563",
            fontweight="bold",
        )


def create_dashboard(
    top_products: pd.DataFrame,
    monthly_sales: pd.DataFrame,
    top_customers: pd.DataFrame,
    category_sales: pd.DataFrame,
    kpis: dict[str, float],
) -> None:
    """Create and save the 2 x 2 analytics dashboard."""
    plt.style.use("seaborn-v0_8")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes.flatten():
        ax.set_facecolor("white")

    fig.suptitle(
        "Retail Sales Analytics Dashboard",
        fontsize=26,
        fontweight="bold",
        color="#111827",
        y=0.975,
    )
    fig.text(
        0.5,
        0.94,
        "Revenue performance, customer value, product mix, and category trends",
        ha="center",
        fontsize=12,
        color="#4B5563",
    )
    add_kpi_cards(fig, kpis)

    # Top 10 products by revenue.
    product_ax = axes[0, 0]
    product_plot = top_products.sort_values("Sales", ascending=True).copy()
    product_plot["Product Label"] = product_plot["Product Name"].apply(
        lambda value: shorten_label(value, 48)
    )
    product_colors = sns.color_palette("viridis", len(product_plot))
    product_ax.barh(
        product_plot["Product Label"],
        product_plot["Sales"],
        color=product_colors,
    )
    annotate_bars(product_ax, product_plot["Sales"], horizontal=True)
    product_ax.set_title("Top 10 Products by Revenue", fontweight="bold", pad=14)
    product_ax.set_xlabel("Revenue")
    product_ax.set_ylabel("")
    product_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_currency(x)))
    product_ax.grid(axis="x", alpha=0.25)
    product_ax.grid(axis="y", visible=False)
    product_ax.margins(x=0.16)

    # Monthly sales trend with moving average.
    trend_ax = axes[0, 1]
    trend_ax.plot(
        monthly_sales.index,
        monthly_sales["Sales"],
        marker="o",
        linewidth=2.6,
        color="#2563EB",
        label="Monthly Sales",
    )
    trend_ax.plot(
        monthly_sales.index,
        monthly_sales["Moving Average"],
        linestyle="--",
        linewidth=2.4,
        color="#F97316",
        label="3-Month Moving Average",
    )
    trend_ax.fill_between(
        monthly_sales.index,
        monthly_sales["Sales"],
        color="#93C5FD",
        alpha=0.22,
    )
    last_month = monthly_sales.index[-1]
    last_sales = monthly_sales["Sales"].iloc[-1]
    trend_ax.annotate(
        format_currency(last_sales),
        xy=(last_month, last_sales),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#1D4ED8",
    )
    trend_ax.set_title("Monthly Sales Trend", fontweight="bold", pad=14)
    trend_ax.set_xlabel("Order Month")
    trend_ax.set_ylabel("Revenue")
    trend_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_currency(x)))
    trend_ax.legend(frameon=False, loc="upper left")
    trend_ax.grid(alpha=0.25)

    # Top 5 customers by spend.
    customer_ax = axes[1, 0]
    customer_plot = top_customers.copy()
    customer_plot["Customer Label"] = customer_plot["Customer Name"].apply(
        lambda value: shorten_label(value, 24)
    )
    customer_colors = sns.color_palette("crest", len(customer_plot))
    customer_ax.bar(
        customer_plot["Customer Label"],
        customer_plot["Sales"],
        color=customer_colors,
    )
    annotate_bars(customer_ax, customer_plot["Sales"], horizontal=False)
    customer_ax.set_title("Top 5 Customers by Spending", fontweight="bold", pad=14)
    customer_ax.set_xlabel("Customer")
    customer_ax.set_ylabel("Revenue")
    customer_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_currency(x)))
    customer_ax.tick_params(axis="x", rotation=18)
    customer_ax.grid(axis="y", alpha=0.25)
    customer_ax.grid(axis="x", visible=False)
    customer_ax.margins(y=0.18)

    # Category-wise sales donut chart.
    category_ax = axes[1, 1]
    category_colors = sns.color_palette("Set2", len(category_sales))
    wedges, _, autotexts = category_ax.pie(
        category_sales["Sales"],
        labels=category_sales["Category"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        colors=category_colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10, "color": "#111827"},
    )
    for autotext in autotexts:
        autotext.set_color("#111827")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    category_ax.text(
        0,
        0.04,
        format_currency(category_sales["Sales"].sum()),
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    category_ax.text(
        0,
        -0.11,
        "Total Sales",
        ha="center",
        va="center",
        fontsize=10,
        color="#4B5563",
    )
    category_ax.set_title("Category-wise Sales", fontweight="bold", pad=14)
    category_ax.axis("equal")

    # Keep a reference so linters do not mark the wedges as unused in editors.
    _ = wedges

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(
        0.5,
        0.02,
        f"Generated on {generated_at} | Source: Superstore Sales Dataset",
        ha="center",
        fontsize=10,
        color="#6B7280",
    )

    plt.tight_layout(rect=[0.04, 0.055, 0.98, 0.82])
    fig.savefig(DASHBOARD_PATH, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.info("Dashboard image saved to %s", DASHBOARD_PATH)


def write_summary_report(
    df: pd.DataFrame,
    top_products: pd.DataFrame,
    top_customers: pd.DataFrame,
    category_sales: pd.DataFrame,
    kpis: dict[str, float],
) -> None:
    """Write a clean executive summary report."""
    top_product = top_products.iloc[0]
    top_customer = top_customers.iloc[0]
    best_category = category_sales.iloc[0]
    start_date = df["Order Date"].min().strftime("%B %d, %Y")
    end_date = df["Order Date"].max().strftime("%B %d, %Y")
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    report = f"""Retail Sales Analytics Summary Report
Generated: {generated_at}

Executive Summary
The retail business generated {format_currency(kpis["total_revenue"])} in total revenue across {kpis["total_orders"]:,.0f} orders from {kpis["unique_customers"]:,.0f} unique customers. The average order value was {format_currency(kpis["average_order_value"])}.

Key Findings
Total Revenue: {format_currency(kpis["total_revenue"])}
Top Product: {top_product["Product Name"]} ({format_currency(top_product["Sales"])})
Top Customer: {top_customer["Customer Name"]} ({format_currency(top_customer["Sales"])})
Best Category: {best_category["Category"]} ({format_currency(best_category["Sales"])})
Number of Orders: {kpis["total_orders"]:,.0f}
Date Range: {start_date} to {end_date}

Business Interpretation
Revenue is concentrated among a small set of high-value products and customers. The top product and top customer should be reviewed for retention, inventory planning, and promotional opportunities. Category-level performance highlights where the business is strongest and where merchandising or pricing strategies may need closer review.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    logging.info("Summary report saved to %s", REPORT_PATH)


def run_pipeline() -> None:
    """Execute the full dashboard generation pipeline."""
    raw_df = load_data(DATA_PATH)
    cleaned_df = clean_data(raw_df)

    top_products = get_top_products(cleaned_df)
    monthly_sales = get_monthly_sales(cleaned_df)
    top_customers = get_top_customers(cleaned_df)
    category_sales = get_category_sales(cleaned_df)
    kpis = calculate_kpis(cleaned_df)

    create_dashboard(
        top_products=top_products,
        monthly_sales=monthly_sales,
        top_customers=top_customers,
        category_sales=category_sales,
        kpis=kpis,
    )
    write_summary_report(
        df=cleaned_df,
        top_products=top_products,
        top_customers=top_customers,
        category_sales=category_sales,
        kpis=kpis,
    )

    print("\nDashboard Successfully Generated")
    print("Location: output/dashboard.png")


def main() -> None:
    """Entrypoint with exception handling for portfolio-ready execution."""
    configure_logging()
    try:
        run_pipeline()
    except Exception as exc:
        logging.exception("Dashboard generation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
