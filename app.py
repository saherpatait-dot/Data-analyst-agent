import re
import os
import json
import math
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.caption("Upload any CSV or Excel dataset. The analysis adapts to the columns and data available.")
st.info(
    "Core calculations run locally with Pandas. Optional Groq AI explains verified business findings when GROQ_API_KEY is configured."
)

MAX_FILE_MB = 200

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO
    buffer = BytesIO(file_bytes)
    lower = filename.lower()
    if lower.endswith(".csv"):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"Could not decode this CSV file: {last_error}")
    if lower.endswith((".xlsx", ".xls")):
        buffer.seek(0)
        return pd.read_excel(buffer)
    raise ValueError("Please upload a CSV or Excel file.")

def parse_dates(series):
    """Parse common date formats, including DD/MM/YYYY and MM/DD/YYYY."""
    if len(series.dropna()) == 0:
        return None
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed if parsed.notna().mean() >= 0.75 else None

    # Use mixed-format parsing so a column containing dates such as 08/11/2017
    # is evaluated consistently instead of failing on rows with day > 12.
    values = series.dropna().astype(str).str.strip()
    slash_parts = values.str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")
    first = pd.to_numeric(slash_parts[0], errors="coerce")
    second = pd.to_numeric(slash_parts[1], errors="coerce")
    dayfirst_evidence = bool((first > 12).any())
    monthfirst_evidence = bool((second > 12).any())

    if dayfirst_evidence and not monthfirst_evidence:
        candidates = [True, False]
    elif monthfirst_evidence and not dayfirst_evidence:
        candidates = [False, True]
    else:
        candidates = [False, True]

    best = None
    best_score = -1.0
    for dayfirst in candidates:
        try:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        score = float(parsed.notna().mean())
        if score > best_score:
            best, best_score = parsed, score
    return best if best is not None and best_score >= 0.75 else None

def get_schema(df):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    date_map = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_map[c] = pd.to_datetime(df[c], errors="coerce")
        elif any(token in str(c).lower() for token in ("date", "time", "timestamp")):
            parsed = parse_dates(df[c])
            if parsed is not None:
                date_map[c] = parsed
    dates = list(date_map)
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    return numeric, categorical, dates, date_map

def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

def mentioned_columns(question, columns):
    q = norm(question)
    hits = []
    for col in columns:
        n = norm(col)
        if n and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", q):
            hits.append(col)
    return sorted(hits, key=lambda c: len(norm(c)), reverse=True)

def choose_measure(question, numeric, explicit):
    for col in explicit:
        if col in numeric:
            return col
    q = norm(question)
    # Semantic names are useful when the question says "sales" but the column is "Sales Amount".
    hints = [
        (("sales", "revenue", "turnover", "income"), ("sales", "revenue", "turnover", "amount")),
        (("profit", "earnings", "margin"), ("profit", "earnings", "margin")),
        (("quantity", "units", "items sold", "volume"), ("quantity", "units", "volume")),
        (("discount",), ("discount",)),
        (("cost", "expense", "spend"), ("cost", "expense", "spend")),
        (("price",), ("price",)),
        (("age",), ("age",)),
        (("score", "rating"), ("score", "rating")),
    ]
    for triggers, tokens in hints:
        if any(t in q for t in triggers):
            for c in numeric:
                if any(t in norm(c).split() for t in tokens) or any(t in norm(c) for t in tokens):
                    return c
    if len(numeric) == 1:
        return numeric[0]
    return None

def choose_category(question, categorical, explicit):
    for col in explicit:
        if col in categorical:
            return col
    q = norm(question)
    hints = [
        (("product", "products", "item", "items", "sku"), ("product", "item", "sku")),
        (("customer", "customers", "client", "clients"), ("customer", "client", "buyer")),
        (("region", "regional", "area", "territory"), ("region", "area", "territory", "state", "country")),
        (("ship mode", "shipping", "shipment"), ("ship mode", "ship", "shipping", "shipment")),
        (("category", "categories"), ("category",)),
        (("segment", "segments"), ("segment",)),
        (("department", "departments"), ("department", "dept")),
        (("city", "cities"), ("city",)),
        (("state", "states"), ("state",)),
        (("country", "countries"), ("country",)),
    ]
    for triggers, tokens in hints:
        if any(t in q for t in triggers):
            for c in categorical:
                if any(t in norm(c) for t in tokens):
                    return c
    if len(categorical) == 1:
        return categorical[0]
    return None

def answer_question(df, question):
    q = question.lower().strip()
    numeric, categorical, dates, date_map = get_schema(df)
    explicit = mentioned_columns(question, list(df.columns))
    measure = choose_measure(question, numeric, explicit)
    category = choose_category(question, categorical, explicit)
    date_col = next((c for c in explicit if c in dates), None)
    if date_col is None and len(dates) == 1:
        date_col = dates[0]

    def result(answer, table=None, fig=None):
        return {"answer": answer, "table": table, "figure": fig}

    if any(x in q for x in ("help", "what can you do", "example questions")):
        return result(
            "### Try questions using your own column names\n"
            "- What is the total of [numeric column]?\n"
            "- Which are the top 5 [categories] by [numeric column]?\n"
            "- Compare [numeric column] by [category column].\n"
            "- Show monthly trends using [date column].\n"
            "- Which columns have missing values?\n"
            "- Show correlations between numeric columns.\n"
            "- Summarize this dataset."
        )

    if any(x in q for x in ("overview", "summarize dataset", "dataset summary", "summary of dataset", "data quality", "missing values", "duplicates", "how many rows", "how many columns")):
        missing = df.isna().sum()
        table = pd.DataFrame({
            "Column": [str(c) for c in df.columns],
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing values": [int(missing[c]) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        return result(
            f"### Dataset overview\n- **Rows:** {len(df):,}\n- **Columns:** {len(df.columns):,}\n"
            f"- **Missing cells:** {int(missing.sum()):,}\n- **Duplicate rows:** {int(df.duplicated().sum()):,}",
            table
        )

    if any(x in q for x in ("correlation", "correlations", "correlate", "relationship between")):
        if len(numeric) < 2:
            return result("Correlation analysis needs at least two numeric columns.")
        corr = df[numeric].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix")
        return result("### Correlation matrix\nValues range from -1 to +1. Correlation does not prove causation.", corr, fig)

    if any(x in q for x in ("trend", "over time", "monthly", "daily", "weekly", "yearly", "by month", "by year", "quarterly")):
        if measure is None or date_col is None:
            return result(
                "To make a trend chart, I need a numeric measure and a date column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Date columns detected: {', '.join(map(str, dates)) or 'none'}. "
                "Try naming both, e.g. 'Show monthly Sales by Order Date'."
            )
        temp = pd.DataFrame({"date": date_map[date_col], "value": pd.to_numeric(df[measure], errors="coerce")}).dropna()
        if temp.empty:
            return result(f"No valid dates and numeric values were found for {date_col} and {measure}.")
        freq = "D" if "daily" in q or "by day" in q else "W" if "weekly" in q or "by week" in q else "Y" if "yearly" in q or "by year" in q else "Q" if "quarter" in q else "M"
        temp = temp.set_index("date")["value"].resample(freq).sum().rename(measure).reset_index()
        fig = px.line(temp, x="date", y=measure, markers=True, title=f"{measure} over time")
        return result(f"### Trend: {measure} over time\nAggregated using the sum at the requested time frequency.", temp, fig)

    rank_match = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    is_rank = rank_match or any(x in q for x in ("highest", "lowest", "rank", "best", "worst", "most sales", "least sales"))
    if is_rank:
        if measure is None or category is None:
            return result(
                "To rank items, I need a numeric measure and a category column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Category columns: {', '.join(map(str, categorical)) or 'none'}. "
                "Example: 'Which are the top 5 products by Sales?'"
            )
        agg = df.groupby(category, dropna=False)[measure].sum(min_count=1).reset_index()
        descending = not ("bottom" in q or "lowest" in q or "worst" in q or "least" in q)
        agg = agg.sort_values(measure, ascending=not descending, na_position="last")
        limit = min(int(rank_match.group(2)), 100) if rank_match else 10
        table = agg.head(limit)
        label = "Top" if descending else "Bottom"
        fig = px.bar(table, x=category, y=measure, title=f"{label} {len(table)} {category} by {measure}")
        return result(f"### {label} {len(table)} {category} by {measure}\nValues are summed by category.", table, fig)

    if any(x in q for x in ("compare", "comparison", "breakdown", "by category", "by region", "by product", "across")):
        if measure is None or category is None:
            return result(
                "To compare groups, I need a numeric measure and a category column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Category columns: {', '.join(map(str, categorical)) or 'none'}."
            )
        agg_name = "mean" if any(x in q for x in ("average", "mean")) else "sum"
        table = df.groupby(category, dropna=False)[measure].agg(agg_name).reset_index()
        table = table.sort_values(measure, ascending=False, na_position="last")
        fig = px.bar(table.head(40), x=category, y=measure, title=f"{agg_name.title()} {measure} by {category}")
        return result(f"### {agg_name.title()} {measure} by {category}", table, fig)

    if any(x in q for x in ("missing", "null", "empty cells")):
        table = df.isna().sum().rename("Missing values").to_frame()
        table = table[table["Missing values"] > 0].sort_values("Missing values", ascending=False)
        return result("### Missing values\nNo missing values found." if table.empty else "### Missing values by column", table)

    if any(x in q for x in ("duplicate", "duplicates")):
        return result(f"### Duplicate rows\nFound **{int(df.duplicated().sum()):,}** duplicate row(s). No rows have been removed.")

    if measure is not None and any(x in q for x in ("total", "sum", "average", "mean", "median", "minimum", "maximum", "max", "min", "standard deviation", "describe", "summary", "how much", "how many")):
        s = pd.to_numeric(df[measure], errors="coerce").dropna()
        if s.empty:
            return result(f"No valid numeric values found in {measure}.")
        stats = pd.DataFrame({"Statistic": ["Count", "Sum", "Mean", "Median", "Minimum", "Maximum", "Standard deviation"], "Value": [s.count(), s.sum(), s.mean(), s.median(), s.min(), s.max(), s.std()]})
        if "average" in q or "mean" in q:
            answer = f"### Average {measure}: **{s.mean():,.4g}**"
        elif "median" in q:
            answer = f"### Median {measure}: **{s.median():,.4g}**"
        elif "minimum" in q or re.search(r"\bmin\b", q):
            answer = f"### Minimum {measure}: **{s.min():,.4g}**"
        elif "maximum" in q or re.search(r"\bmax\b", q):
            answer = f"### Maximum {measure}: **{s.max():,.4g}**"
        elif "total" in q or "sum" in q or "how much" in q:
            answer = f"### Total {measure}: **{s.sum():,.4g}**"
        else:
            answer = f"### Summary statistics for {measure}"
        return result(answer, stats)

    if any(x in q for x in ("distribution", "histogram", "spread")):
        if measure is None:
            return result("Please name the numeric column whose distribution you want to inspect.")
        fig = px.histogram(df, x=measure, title=f"Distribution of {measure}")
        return result(f"### Distribution of {measure}", df[measure].describe().to_frame(), fig)

    return result(
        "I couldn't confidently identify the analysis you want. Try **help** for examples, "
        "or name the measure and category explicitly (for example, 'top 5 products by Sales').\n\n"
        "This free-hosting version uses a safe question interpreter rather than a large language model, "
        "so it supports common analysis requests but not every possible conversational question."
    )


# ---------------- BUSINESS ANALYSIS REPORT HELPERS ----------------

def _candidate_date_columns(df):
    found = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col)
        else:
            parsed = parse_dates(df[col])
            if parsed is not None:
                # Date-like column names are a strong signal; otherwise accept
                # columns only when nearly all non-null values parse as dates.
                name_hint = any(token in str(col).lower() for token in ("date", "time", "timestamp"))
                threshold = 0.75 if name_hint else 0.95
                if parsed.notna().mean() >= threshold:
                    found.append(col)
    return found


def _column_discovery(df):
    """Discover dates, measures, identifiers and dimensions from dataset metadata.

    This is dataset-agnostic: column names are used only as weak semantic hints;
    data types, cardinality and observed values drive the classification.
    """
    date_cols = _candidate_date_columns(df)
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and c not in date_cols
    ]

    identifier_words = {
        "id", "identifier", "key", "postal", "postcode", "zip", "phone",
        "latitude", "longitude", "longitude", "index", "row number", "row"
    }
    metric_cols, identifier_cols = [], []
    for col in numeric_cols:
        name_tokens = set(norm(col).split())
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        unique_ratio = values.nunique() / max(len(values), 1)
        looks_like_identifier = bool(name_tokens & identifier_words)
        # A near-unique integer sequence is likely an identifier even if unnamed.
        if not looks_like_identifier and len(values) >= 20 and unique_ratio >= 0.995:
            looks_like_identifier = bool(np.all(np.isclose(values % 1, 0)))
        if looks_like_identifier:
            identifier_cols.append(col)
        else:
            metric_cols.append(col)

    dimensions = []
    text_identifier_words = {"id", "identifier", "key", "uuid", "guid", "record"}
    for col in df.columns:
        if col in date_cols or col in metric_cols or col in identifier_cols:
            continue
        if set(norm(col).split()) & text_identifier_words:
            identifier_cols.append(col)
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            cardinality = series.nunique(dropna=True)
        elif (pd.api.types.is_object_dtype(series)
              or pd.api.types.is_string_dtype(series)
              or isinstance(series.dtype, pd.CategoricalDtype)):
            cardinality = series.nunique(dropna=True)
        else:
            continue
        if cardinality >= 2:
            dimensions.append(col)

    return {
        "dates": date_cols,
        "metrics": metric_cols,
        "identifiers": identifier_cols,
        "dimensions": dimensions,
    }


def _role_score(df, col, role):
    """Rank a discovered dimension for a reporting role; never filters the options."""
    name = norm(col)
    primary_hints = {
        "region": ("region", "territory", "geography", "market", "location"),
        "product": ("product", "item", "sku", "service", "brand", "offering"),
        "customer": ("customer", "client", "buyer", "subscriber", "member"),
    }
    secondary_hints = {
        "region": ("country", "state", "province", "city", "zone", "area"),
        "product": ("category", "plan", "package"),
        "customer": ("account", "user", "merchant"),
    }
    tokens = set(name.split())
    score = sum(5 for token in primary_hints.get(role, ()) if token in tokens)
    score += sum(2 for token in secondary_hints.get(role, ()) if token in tokens)
    if any(token in name for token in primary_hints.get(role, ()) + secondary_hints.get(role, ())):
        score += 1
    # A useful dimension has multiple groups; moderate cardinality is preferred,
    # while high-cardinality names such as product/customer remain eligible.
    cardinality = int(df[col].nunique(dropna=True))
    if 2 <= cardinality <= 50:
        score += 2
    elif cardinality <= 500:
        score += 1
    return score


def _candidate_dimensions(df, role=None):
    discovered = _column_discovery(df)["dimensions"]
    if role:
        return sorted(discovered, key=lambda c: (-_role_score(df, c, role), str(c).lower()))
    return discovered


def _candidate_business_metrics(df):
    return _column_discovery(df)["metrics"]


def _safe_period_frame(df, date_col, metric_cols, period_type):
    work = df.copy()
    parsed_dates = parse_dates(work[date_col])
    if parsed_dates is None:
        return None, None
    work["_report_date"] = parsed_dates
    for metric in metric_cols:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["_report_date"])
    if work.empty:
        return None, None
    frequency_map = {
        "Week-over-Week (WoW)": "W-SUN",
        "Month-over-Month (MoM)": "M",
        "Quarter-over-Quarter (QoQ)": "Q",
        "Year-over-Year (YoY)": "Y",
    }
    work["_period"] = work["_report_date"].dt.to_period(frequency_map.get(period_type, "W-SUN"))
    return work, sorted(work["_period"].dropna().unique())


def _period_aggregate(work, period, metric_cols, dimension=None):
    subset = work[work["_period"] == period]
    group_cols = [dimension] if dimension else []
    if group_cols:
        return subset.groupby(group_cols, dropna=False)[metric_cols].sum(min_count=1).reset_index()
    return pd.DataFrame([{metric: subset[metric].sum(min_count=1) for metric in metric_cols}])


def _pct_change(current, previous):
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    if previous == 0:
        return np.nan
    return ((current - previous) / abs(previous)) * 100


def _compare_periods(work, current_period, previous_period, metric_cols, dimension=None):
    current = _period_aggregate(work, current_period, metric_cols, dimension)
    previous = _period_aggregate(work, previous_period, metric_cols, dimension)
    if dimension:
        merged = current.merge(previous, on=dimension, how="outer", suffixes=("_current", "_previous"))
        for metric in metric_cols:
            c, p = f"{metric}_current", f"{metric}_previous"
            merged[c] = merged[c].fillna(0)
            merged[p] = merged[p].fillna(0)
            merged[f"{metric}_change"] = merged[c] - merged[p]
            merged[f"{metric}_change_pct"] = [
                _pct_change(a, b) for a, b in zip(merged[c], merged[p])
            ]
        return merged
    rows = []
    for metric in metric_cols:
        c = current.iloc[0][metric] if len(current) else np.nan
        p = previous.iloc[0][metric] if len(previous) else np.nan
        rows.append({
            "Metric": metric, "Current": c, "Previous": p,
            "Absolute change": c - p if pd.notna(c) and pd.notna(p) else np.nan,
            "Change %": _pct_change(c, p)
        })
    return pd.DataFrame(rows)


def _contributor_table(work, current_period, previous_period, metric, dimension, top_n=15):
    if not dimension:
        return pd.DataFrame()
    comparison = _compare_periods(work, current_period, previous_period, [metric], dimension)
    delta_col = f"{metric}_change"
    pct_col = f"{metric}_change_pct"
    comparison = comparison.rename(columns={
        f"{metric}_current": "Current value",
        f"{metric}_previous": "Previous value",
        delta_col: "Absolute contribution",
        pct_col: "Contributor change %"
    })
    total_change = comparison["Absolute contribution"].sum()
    if pd.notna(total_change) and not np.isclose(total_change, 0):
        comparison["Share of total change %"] = comparison["Absolute contribution"] / total_change * 100
    else:
        comparison["Share of total change %"] = np.nan
    comparison = comparison.sort_values("Absolute contribution", key=lambda s: s.abs(), ascending=False)
    return comparison.head(top_n)


def _full_contributor_summary(work, current_period, previous_period, metric, dimension):
    """Calculate complete-population contributor aggregates, not top-N extracts."""
    if not dimension:
        return {}
    comparison = _compare_periods(work, current_period, previous_period, [metric], dimension)
    if comparison.empty:
        return {}

    change_col = f"{metric}_change"
    current_col = f"{metric}_current"
    previous_col = f"{metric}_previous"
    comparison[change_col] = pd.to_numeric(comparison[change_col], errors="coerce").fillna(0)
    positive = comparison[comparison[change_col] > 0].copy()
    negative = comparison[comparison[change_col] < 0].copy()
    total_positive = float(positive[change_col].sum())
    total_negative = float(negative[change_col].sum())
    net_change = float(comparison[change_col].sum())

    def records(frame, ascending=False):
        frame = frame.sort_values(change_col, ascending=ascending).head(5)
        output = []
        for _, row in frame.iterrows():
            output.append({
                str(dimension): str(row[dimension]),
                "Current value": float(row[current_col]),
                "Previous value": float(row[previous_col]),
                "Absolute contribution": float(row[change_col]),
                "Share of net change %": (
                    float(row[change_col] / net_change * 100)
                    if not np.isclose(net_change, 0) else None
                ),
            })
        return output

    return {
        "dimension": str(dimension),
        "metric": str(metric),
        "groups_analyzed": int(len(comparison)),
        "groups_increased": int((comparison[change_col] > 0).sum()),
        "groups_decreased": int((comparison[change_col] < 0).sum()),
        "groups_unchanged": int((comparison[change_col] == 0).sum()),
        "total_positive_contribution": total_positive,
        "total_negative_contribution": total_negative,
        "net_change": net_change,
        "top_5_increases": records(positive, ascending=False),
        "top_5_decreases": records(negative, ascending=True),
    }


def _period_label(period):
    return str(period)


def _build_business_report(df, date_col, metric_cols, period_type, dimensions, current_label=None):
    """Build a dataset-agnostic report using only columns present in the upload."""
    if not metric_cols:
        raise ValueError("Select at least one numeric business metric.")

    if date_col:
        work, periods = _safe_period_frame(df, date_col, metric_cols, period_type)
        if work is None or len(periods) < 2:
            raise ValueError("At least two populated periods are required for a period comparison.")
        comparable = [p for p in periods if (p - 1) in periods]
        if not comparable:
            raise ValueError("No adjacent populated periods were found. Choose another frequency or check date coverage.")
        selected = next((p for p in comparable if str(p) == current_label), comparable[-1])
        previous = selected - 1
        overall = _compare_periods(work, selected, previous, metric_cols)
        dimension_results = {}
        for dimension in dimensions:
            if dimension not in work.columns or dimension in (date_col, "_period", "_report_date"):
                continue
            # Avoid an unreadable table for near-unique IDs; the user can still
            # explicitly choose such a dimension if that is meaningful.
            comparison = _compare_periods(work, selected, previous, metric_cols, dimension)
            by_metric = {}
            summaries = {}
            for metric in metric_cols:
                table = _contributor_table(work, selected, previous, metric, dimension, top_n=15)
                summary = _full_contributor_summary(work, selected, previous, metric, dimension)
                if not table.empty:
                    by_metric[metric] = table
                if summary:
                    summaries[metric] = summary
            if by_metric or summaries:
                dimension_results[str(dimension)] = {"tables": by_metric, "summaries": summaries}

        return {
            "analysis_mode": "period_comparison",
            "current_period": _period_label(selected),
            "previous_period": _period_label(previous),
            "period_type": period_type,
            "date_column": date_col,
            "metrics": list(metric_cols),
            "overall": overall,
            "dimensions": dimension_results,
            "work": work,
        }

    # Snapshot mode: no date column is required. It produces descriptive metrics
    # and optional dimension breakdowns without inventing a time comparison.
    overall_rows = []
    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        overall_rows.append({
            "Metric": str(metric), "Count": int(values.count()),
            "Sum": float(values.sum()), "Mean": float(values.mean()),
            "Median": float(values.median()), "Minimum": float(values.min()),
            "Maximum": float(values.max()), "Standard deviation": float(values.std()) if len(values) > 1 else None,
        })
    dimension_results = {}
    for dimension in dimensions:
        if dimension not in df.columns:
            continue
        by_metric = {}
        for metric in metric_cols:
            table = df.groupby(dimension, dropna=False)[metric].agg(
                Count="count", Total="sum", Average="mean"
            ).reset_index().sort_values("Total", ascending=False).head(15)
            if not table.empty:
                by_metric[metric] = table
        if by_metric:
            dimension_results[str(dimension)] = {"tables": by_metric, "summaries": {}}
    return {
        "analysis_mode": "snapshot", "current_period": None, "previous_period": None,
        "period_type": "Snapshot", "date_column": None, "metrics": list(metric_cols),
        "overall": pd.DataFrame(overall_rows), "dimensions": dimension_results,
        "work": df.copy(),
    }


def _audit_ai_numeric_claims(text, verified_data):
    """Flag currency/percentage figures not traceable to supplied verified facts.

    Normalize typographic minus/dash characters first. AI-generated Markdown often
    uses U+2011/U+2212 instead of the ASCII hyphen, which otherwise makes a valid
    negative percentage look like a positive value and creates false warnings.
    """
    allowed = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
            if math.isfinite(float(value)):
                number = float(value)
                for digits in (0, 1, 2):
                    allowed.add(round(number, digits))

    collect(verified_data)

    normalized_text = str(text).translate(str.maketrans({
        "−": "-", "–": "-", "—": "-", "‑": "-", "‒": "-", "﹣": "-", "－": "-"
    }))

    # Only audit explicit currency and percentage claims; ordinary section numbers
    # and years are intentionally excluded.
    pattern = re.compile(
        # Currency signs may appear before or after the symbol, e.g. -$1,234.56
        # or $-1,234.56. Both forms must preserve the negative sign.
        r'(?P<currency>[-+]?\s*[$₹€£]\s*[-+]?\d[\d,]*(?:\.\d+)?)'
        r'|(?P<percent>[-+]?\d[\d,]*(?:\.\d+)?\s*%)'
    )
    unsupported = []
    for match in pattern.finditer(normalized_text):
        token = match.group(0)
        # Ignore generic mathematical phrases such as "does not sum to 100%";
        # these are not claims about the uploaded dataset.
        context_before = normalized_text[max(0, match.start() - 50):match.start()].lower()
        if match.group("percent") and re.search(
            r"(?:sum|add|total|equal|amount)\s+(?:up\s+)?to\s*$", context_before
        ):
            continue
        numeric = re.sub(r'[$₹€£,%\s]', '', token).replace(",", "")
        try:
            value = float(numeric)
        except ValueError:
            continue
        if not any(abs(value - candidate) <= max(0.02, abs(candidate) * 0.0005)
                   for candidate in allowed):
            unsupported.append(token)

    return list(dict.fromkeys(unsupported))


def _groq_explanation(report):
    """Generate a narrative from dynamically discovered, Pandas-verified facts."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None, "GROQ_API_KEY was not found in Streamlit Cloud Secrets."

    try:
        from openai import OpenAI
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=90.0, max_retries=1)
        compact = {
            "analysis_type": report["analysis_mode"],
            "comparison": (f"{report['previous_period']} to {report['current_period']}"
                           if report["analysis_mode"] == "period_comparison" else "Single-period snapshot; no date comparison performed"),
            "period_type": report["period_type"], "date_column": report["date_column"],
            "verified_metric_results": report["overall"].replace({np.nan: None}).to_dict(orient="records"),
            "available_dimension_analyses": {
                dimension: {
                    "top_contributors_by_metric": {
                        metric: table.replace({np.nan: None}).to_dict(orient="records")
                        for metric, table in details["tables"].items()
                    },
                    "full_population_summaries_by_metric": details["summaries"],
                } for dimension, details in report["dimensions"].items()
            },
        }
        prompt = (
            "Act as a careful, dataset-agnostic senior data analyst. Use ONLY the supplied JSON. "
            "The uploaded dataset may describe any business or operational domain. Do not assume it is sales, "
            "retail, HR, finance, or any other domain unless the column names and values support that. "
            "Use the actual metric and dimension names. Discuss only dimensions present in the JSON. "
            "Never invent missing regions, products, customers, categories, causes, or business context. "
            "For period comparisons, explain what changed and quantify it using supplied verified figures. "
            "For snapshot analysis, describe distributions and group differences without claiming a trend. "
            "Distinguish observed facts from hypotheses. You may suggest plausible explanations only as "
            "hypotheses requiring validation; do not state causes as facts. Tables under dimensions are "
            "top-15 extracts and must not be summed to represent the full population. Use full-population "
            "summaries only when they are actually present. If a dimension or summary is absent, omit that "
            "section rather than printing an unavailable template. Mention relevant data limitations and "
            "specific next checks based on the available columns. Include an executive summary, key findings, "
            "dimension analysis only where available, limitations, and recommended next checks. Keep every "
            "numeric figure exactly as supplied; do not calculate new numbers or guess.\n\n"
            f"VERIFIED DATA JSON:\n{json.dumps(compact, default=str, ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Numerical fidelity and evidence boundaries are more important than persuasive prose."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_completion_tokens=3000, stream=False,
        )
        if not getattr(response, "choices", None):
            return None, f"Groq returned an empty response for model '{model}'."
        content = getattr(response.choices[0].message, "content", None)
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
        if not isinstance(content, str) or not content.strip():
            return None, "Groq connected but returned no displayable answer. Please try again."
        content = content.strip()

        # Add deterministic full-population summaries only when they exist.
        verified_sections = []
        for dimension, details in report["dimensions"].items():
            for metric, summary in details["summaries"].items():
                verified_sections.extend([
                    f"### {dimension} — {metric}",
                    f"- Groups analyzed: {summary['groups_analyzed']:,}",
                    f"- Groups increased: {summary['groups_increased']:,}",
                    f"- Groups decreased: {summary['groups_decreased']:,}",
                    f"- Groups unchanged: {summary['groups_unchanged']:,}",
                    f"- Total positive contribution: {summary['total_positive_contribution']:,.2f}",
                    f"- Total negative contribution: {summary['total_negative_contribution']:,.2f}",
                    f"- Net change: {summary['net_change']:,.2f}", "",
                ])
        if verified_sections:
            content += "\n\n## Verified full-population summaries\n\n" + "\n".join(verified_sections)

        unsupported = _audit_ai_numeric_claims(content, compact)
        if unsupported:
            content += "\n\n---\n\n**Automatic numeric review:** These figures could not be matched to verified calculations: " + ", ".join(unsupported[:12]) + ". Review before sharing."
        else:
            content += "\n\n---\n\n**Automatic numeric review:** All explicit currency and percentage figures were matched to values in the supplied calculation data. This checks numeric traceability, not the correctness of every interpretation."
        return content, None
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            detail = "Groq rejected the API key (401). Check GROQ_API_KEY in Streamlit Secrets."
        elif status_code == 403:
            detail = "Groq denied access (403). Check model access and account limits."
        elif status_code == 404:
            detail = f"Groq could not find model '{os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}'."
        elif status_code == 429:
            detail = "Groq rate limit reached. Wait a little and try again."
        else:
            detail = f"Groq request failed ({type(exc).__name__}): {exc}"
        return None, detail


def _business_interpretation_docx(markdown_text):
    """Convert the Groq Markdown interpretation into a readable Word document."""
    from io import BytesIO
    from docx import Document
    from docx.shared import Inches, Pt
    import re

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    document.styles["Normal"].font.name = "Aptos"
    document.styles["Normal"].font.size = Pt(10)

    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue

        # Markdown table: collect rows, skipping the separator row.
        if raw.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", cell or "-") for cell in cells):
                    table_rows.append(cells)
                i += 1
            if table_rows:
                col_count = max(len(row) for row in table_rows)
                table = document.add_table(rows=0, cols=col_count)
                table.style = "Light Shading Accent 1"
                for row_index, values in enumerate(table_rows):
                    cells = table.add_row().cells
                    for col_index in range(col_count):
                        value = values[col_index] if col_index < len(values) else ""
                        value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
                        value = re.sub(r"`(.*?)`", r"\1", value)
                        cells[col_index].text = value
                        if row_index == 0:
                            for run in cells[col_index].paragraphs[0].runs:
                                run.bold = True
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", raw)
        if heading_match:
            level = min(len(heading_match.group(1)), 3)
            text_value = heading_match.group(2)
            document.add_heading(text_value, level=level)
        elif raw.startswith(("- ", "* ")):
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.add_run(raw[2:])
        elif re.match(r"^\d+[.)]\s+", raw):
            paragraph = document.add_paragraph(style="List Number")
            paragraph.add_run(re.sub(r"^\d+[.)]\s+", "", raw))
        elif raw.startswith("> "):
            paragraph = document.add_paragraph(style="Quote")
            paragraph.add_run(raw[2:])
        elif raw in ("---", "***", "___"):
            document.add_paragraph()
        else:
            paragraph = document.add_paragraph()
            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", raw)
            clean = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", clean)
            clean = re.sub(r"`(.*?)`", r"\1", clean)
            paragraph.add_run(clean)
        i += 1

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def render_business_analysis(df):
    st.header("📑 Dynamic Business Analysis")
    st.caption("Analysis options are built from the columns in your uploaded dataset. No fixed industry template is assumed.")
    discovery = _column_discovery(df)
    metric_options = discovery["metrics"]
    date_options = discovery["dates"]
    dimension_options = discovery["dimensions"]

    with st.expander("What the app detected", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.write("**Date/time columns**")
        c1.write(date_options or "None detected")
        c2.write("**Numeric measures**")
        c2.write(metric_options or "None detected")
        c3.write("**Potential dimensions**")
        c3.write(dimension_options or "None detected")
        st.caption("Column roles are suggestions based on data types, cardinality and column names. You can change the selections below.")

    if not metric_options:
        st.warning("No numeric measures were detected. This version needs at least one numeric column for business metric analysis.")
        return

    with st.expander("Configure analysis", expanded=True):
        metric_cols = st.multiselect("Numeric measures to analyze", metric_options, default=metric_options[:min(3, len(metric_options))], key="dyn_metrics")
        date_choices = ["(Snapshot — no date comparison)"] + date_options
        date_choice = st.selectbox("Date/time column (optional)", date_choices, key="dyn_date")
        date_col = None if date_choice.startswith("(") else date_choice
        if date_col:
            period_type = st.selectbox("Comparison frequency", ["Week-over-Week (WoW)", "Month-over-Month (MoM)", "Quarter-over-Quarter (QoQ)", "Year-over-Year (YoY)"], key="dyn_period_type")
        else:
            period_type = "Snapshot"
        dimensions = st.multiselect("Dimensions to break down by (optional)", dimension_options, default=[], key="dyn_dimensions")

    if not metric_cols:
        st.info("Select at least one numeric measure to continue.")
        return

    current_label = None
    if date_col:
        work, periods = _safe_period_frame(df, date_col, metric_cols, period_type)
        if work is None or len(periods) < 2:
            st.warning("This date column does not contain at least two populated periods.")
            return
        comparable = [p for p in periods if (p - 1) in periods]
        if not comparable:
            st.warning("No adjacent populated periods were found for this frequency. Try another frequency or inspect the date coverage.")
            return
        labels = [_period_label(p) for p in comparable]
        current_label = st.selectbox("Current period (compared with the immediately preceding calendar period)", labels, index=len(labels)-1, key="dyn_current_period")

    if st.button("Generate dynamic analysis", type="primary", key="generate_dynamic_report"):
        try:
            report = _build_business_report(df, date_col, metric_cols, period_type, dimensions, current_label)
            st.session_state["business_report_result"] = report
            st.session_state["business_report_ai"] = None
            st.session_state["business_report_ai_error"] = None
            st.session_state["dynamic_report_signature"] = (date_col, tuple(metric_cols), period_type, tuple(dimensions), current_label)
        except ValueError as exc:
            st.error(str(exc))
            return

    report = st.session_state.get("business_report_result")
    if not report:
        st.info("Choose the measures and optional dimensions, then generate the analysis.")
        return
    signature = (date_col, tuple(metric_cols), period_type, tuple(dimensions), current_label)
    if st.session_state.get("dynamic_report_signature") != signature:
        st.info("Your analysis settings changed. Generate the analysis again to refresh the results.")
        return

    if report["analysis_mode"] == "period_comparison":
        st.subheader(f"{report['previous_period']} → {report['current_period']}")
        st.markdown("#### Overall metric movement")
        st.dataframe(report["overall"].style.format({"Current": "{:,.2f}", "Previous": "{:,.2f}", "Absolute change": "{:+,.2f}", "Change %": "{:+.2f}%"}, na_rep="Not calculable"), use_container_width=True, hide_index=True)
    else:
        st.subheader("Dataset snapshot")
        st.caption("No date comparison was selected or detected. These are descriptive statistics, not trends.")
        st.dataframe(report["overall"], use_container_width=True, hide_index=True)

    if report["dimensions"]:
        st.markdown("#### Dimension breakdowns")
        for dimension, details in report["dimensions"].items():
            with st.expander(f"{dimension} analysis", expanded=True):
                for metric, table in details["tables"].items():
                    st.markdown(f"**{metric} — selected groups**")
                    st.dataframe(table, use_container_width=True, hide_index=True)
                for metric, summary in details["summaries"].items():
                    st.markdown(f"**{metric} — full-population summary**")
                    summary_row = pd.DataFrame([{
                        "Groups analyzed": summary["groups_analyzed"],
                        "Increased": summary["groups_increased"],
                        "Decreased": summary["groups_decreased"],
                        "Unchanged": summary["groups_unchanged"],
                        "Positive contribution": summary["total_positive_contribution"],
                        "Negative contribution": summary["total_negative_contribution"],
                        "Net change": summary["net_change"],
                    }])
                    st.dataframe(summary_row, use_container_width=True, hide_index=True)
    else:
        st.info("No dimensions were selected or detected for breakdowns. The report contains metric-level analysis only.")

    st.markdown("#### AI-written interpretation")
    if st.button("Explain verified findings with Groq", key="explain_groq"):
        st.session_state["business_report_ai"] = None
        st.session_state["business_report_ai_error"] = None
        with st.spinner("Interpreting the available dataset findings..."):
            explanation, error = _groq_explanation(report)
        st.session_state["business_report_ai"] = explanation
        st.session_state["business_report_ai_error"] = error

    if st.session_state.get("business_report_ai"):
        st.success("AI interpretation completed.")
        st.markdown(st.session_state["business_report_ai"])
        try:
            docx_data = _business_interpretation_docx(st.session_state["business_report_ai"])
            st.download_button("⬇️ Download AI interpretation (Word)", data=docx_data,
                file_name="ai_business_interpretation.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_ai_interpretation_docx")
        except ImportError:
            st.warning("Word export requires python-docx. Add it to requirements.txt and redeploy.")
    elif st.session_state.get("business_report_ai_error"):
        st.error(st.session_state["business_report_ai_error"])

    export_frames = {"overall": report["overall"]}
    for dimension, details in report["dimensions"].items():
        for metric, table in details["tables"].items():
            export_frames[f"dimension_{dimension}_{metric}"] = table
        for metric, summary in details["summaries"].items():
            export_frames[f"full_population_{dimension}_{metric}"] = pd.DataFrame([{k: v for k, v in summary.items() if k not in ("top_5_increases", "top_5_decreases")}])
    csv_parts = [f"# {name}\n{frame.to_csv(index=False)}" for name, frame in export_frames.items()]
    st.download_button("⬇️ Download analysis calculations (CSV)", data="\n".join(csv_parts).encode("utf-8"),
        file_name="business_analysis_report.csv", mime="text/csv", key="download_dynamic_csv")


analysis_mode = st.radio("Workspace", ["💬 Data Analyst", "📑 Business Analysis Report"], horizontal=True)

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"Please upload a file smaller than {MAX_FILE_MB} MB.")
        st.stop()
    try:
        df = load_dataset(uploaded.getvalue(), uploaded.name)
        dataset_signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("business_dataset_signature") != dataset_signature:
            st.session_state["business_dataset_signature"] = dataset_signature
            st.session_state.pop("business_report_result", None)
            st.session_state.pop("business_report_ai", None)
            st.session_state.pop("business_report_ai_error", None)
        if df.empty or len(df.columns) == 0:
            st.error("This file has no usable rows or columns.")
            st.stop()
        st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns):,} columns")
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        with st.expander("Preview data", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)
        with st.expander("Column profile"):
            numeric, categorical, dates, _ = get_schema(df)
            profile = pd.DataFrame([{
                "Column": str(col),
                "Type": str(df[col].dtype),
                "Missing": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
                "Examples": ", ".join(df[col].dropna().astype(str).head(3).tolist())
            } for col in df.columns])
            st.dataframe(profile, use_container_width=True)
            st.write("**Numeric columns:**", ", ".join(map(str, numeric)) or "None")
            st.write("**Category columns:**", ", ".join(map(str, categorical)) or "None")
            st.write("**Date columns:**", ", ".join(map(str, dates)) or "None")

        with st.expander("Explore a numeric column"):
            numeric, _, _, _ = get_schema(df)
            if numeric:
                selected = st.selectbox("Numeric column", numeric)
                st.dataframe(df[selected].describe().to_frame(), use_container_width=True)
                chart = st.radio("Chart", ["Histogram", "Box plot"], horizontal=True)
                fig = px.histogram(df, x=selected, title=f"Distribution of {selected}") if chart == "Histogram" else px.box(df, y=selected, title=f"Box plot of {selected}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected.")

        if analysis_mode == "💬 Data Analyst":
            st.header("💬 Ask the Data Analyst")
            st.caption("Ask using the column names in your upload, e.g. “total of [measure]”, “top 5 [groups] by [measure]”, or “show trends by [date]”.")
            file_signature = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("dataset_signature") != file_signature:
                st.session_state.dataset_signature = file_signature
                st.session_state.chat_history = []
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for item in st.session_state.chat_history:
                with st.chat_message(item["role"]):
                    st.markdown(item["content"])
                    if item.get("table") is not None:
                        st.dataframe(item["table"], use_container_width=True)
                    if item.get("figure") is not None:
                        st.plotly_chart(item["figure"], use_container_width=True)

            question = st.chat_input("Ask a question about the uploaded dataset…")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your data…"):
                        output = answer_question(df, question)
                    st.markdown(output["answer"])
                    if output.get("table") is not None:
                        st.dataframe(output["table"], use_container_width=True)
                    if output.get("figure") is not None:
                        st.plotly_chart(output["figure"], use_container_width=True)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": output["answer"],
                    "table": output.get("table"),
                    "figure": output.get("figure"),
                })
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            render_business_analysis(df)
    except Exception as exc:
        st.error(f"Could not read or analyze this file: {exc}")
else:
    st.markdown("### What you can ask")
    st.write("- What is the total of [numeric column]?")
    st.write("- Which are the top 5 [groups] by [numeric column]?")
    st.write("- Compare [numeric column] by [category column].")
    st.write("- Show trends using [date column].")
    st.write("- Which columns have missing values?")
    st.write("- Show correlations between numeric columns.")
    st.write("- Summarize this dataset.")
