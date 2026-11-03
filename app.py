import re
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.caption("Upload a CSV or Excel file, explore it, and generate evidence-based business performance reports.")
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
            "### Try questions like these\n"
            "- What is the total Sales?\n"
            "- Which are the top 5 products by Sales?\n"
            "- Compare Sales by Ship Mode.\n"
            "- Show monthly Sales trends.\n"
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
    if period_type == "Week-over-Week (WoW)":
        work["_period"] = work["_report_date"].dt.to_period("W-SUN")
    else:
        work["_period"] = work["_report_date"].dt.to_period("Q")
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


def _period_label(period):
    return str(period)


def _build_business_report(df, date_col, metric_cols, period_type, region_col, product_col, customer_col, current_label):
    work, periods = _safe_period_frame(df, date_col, metric_cols, period_type)
    if work is None or len(periods) < 2:
        raise ValueError("At least two distinct populated weeks or quarters are required for a comparison.")
    selected = next((p for p in periods if str(p) == current_label), periods[-1])
    previous = selected - 1
    if previous not in periods:
        raise ValueError(
            f"No data was found in the immediately preceding period ({_period_label(previous)}). "
            "Choose a period with an adjacent comparison period or verify the date coverage."
        )
    result = {
        "current_period": _period_label(selected),
        "previous_period": _period_label(previous),
        "period_type": period_type,
        "date_column": date_col,
        "metrics": metric_cols,
        "region_column": region_col,
        "product_column": product_col,
        "customer_column": customer_col,
        "overall": _compare_periods(work, selected, previous, metric_cols),
        "regional": _compare_periods(work, selected, previous, metric_cols, region_col) if region_col else pd.DataFrame(),
        "products": {m: _contributor_table(work, selected, previous, m, product_col) for m in metric_cols} if product_col else {},
        "customers": {m: _contributor_table(work, selected, previous, m, customer_col) for m in metric_cols} if customer_col else {},
        "work": work
    }
    return result


def _groq_explanation(report):
    """Request a business narrative from Groq using only verified report calculations."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        return None, (
            "GROQ_API_KEY was not found. Add it to Streamlit Cloud Secrets "
            "or to your local .streamlit/secrets.toml file."
        )

    try:
        from openai import OpenAI

        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            timeout=90.0,
            max_retries=1,
        )

        compact = {
            "comparison": f"{report['previous_period']} to {report['current_period']}",
            "period_type": report["period_type"],
            "date_column": report["date_column"],
            "metrics": report["overall"].replace({np.nan: None}).to_dict(orient="records"),
            "regional_results": (
                report["regional"].head(30).replace({np.nan: None}).to_dict(orient="records")
                if not report["regional"].empty else []
            ),
            "product_contributions": {
                m: t.head(10).replace({np.nan: None}).to_dict(orient="records")
                for m, t in report["products"].items()
            },
            "customer_contributions": {
                m: t.head(10).replace({np.nan: None}).to_dict(orient="records")
                for m, t in report["customers"].items()
            },
        }

        prompt = (
            "Write a clear, executive-ready business analysis report using ONLY the verified "
            "calculations in the JSON below. Do not invent causes, columns, values, entities, "
            "or business context. Distinguish observed facts from hypotheses. Explain regional "
            "differences and identify products and customers that contributed to changes using "
            "the supplied absolute contribution and share values. If a section has no data, say "
            "it could not be assessed. Do not recalculate or alter numbers. Include: Executive "
            "summary, period performance, regional findings, product drivers, customer drivers, "
            "risks/data limitations, and recommended next checks. Do not imply causation from "
            "contribution alone. Use concise Markdown and preserve the direction and meaning "
            "of every supplied result.\n\n"
            f"VERIFIED DATA JSON:\n{json.dumps(compact, default=str, ensure_ascii=False)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful senior business analyst. Never fabricate data.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3000,
            reasoning_effort="low",
            stream=False,
        )

        if not getattr(response, "choices", None):
            return None, f"Groq returned an empty response for model '{model}'."

        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)

        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            ).strip()

        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice, "finish_reason", "unknown")
            return None, (
                "Groq connected, but the model returned no displayable answer "
                f"(finish reason: {finish_reason}). Try again or check the selected model."
            )

        return content.strip(), None

    except Exception as exc:
        error_text = str(exc)
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            detail = "Groq rejected the API key (401). Check that GROQ_API_KEY is correct."
        elif status_code == 403:
            detail = "Groq denied access (403). Check model access and account limits."
        elif status_code == 404:
            detail = (
                f"Groq could not find the model '{os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}'. "
                "Check the model ID in the Groq Console."
            )
        elif status_code == 429:
            detail = "Groq rate limit reached. Wait a little and try again."
        elif "timeout" in error_text.lower() or "timed out" in error_text.lower():
            detail = "The Groq request timed out. Try again in a moment."
        else:
            detail = f"Groq request failed ({type(exc).__name__}): {error_text}"
        return None, detail


def render_business_analysis(df):
    st.header("📑 Business Analysis Report")
    st.caption("Period-over-period performance, regional metric divergence, and product/customer contribution analysis.")
    discovery = _column_discovery(df)
    metric_options = discovery["metrics"]
    date_cols = discovery["dates"]
    with st.expander("Automatic column discovery", expanded=False):
        st.write({
            "Date columns": discovery["dates"],
            "Business metrics": discovery["metrics"],
            "Dimensions": discovery["dimensions"],
            "Excluded identifier-like columns": discovery["identifiers"],
        })
    if not metric_options:
        st.warning("No suitable numeric business measures were detected. Identifier-like numeric columns are excluded.")
        return
    if not date_cols:
        st.warning("No reliable date column was detected. Add or select a date field to enable WoW/QoQ reporting.")
        return

    with st.expander("Configure report", expanded=True):
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Date column", date_cols, key="biz_date")
        period_type = c2.selectbox("Comparison frequency", ["Week-over-Week (WoW)", "Quarter-over-Quarter (QoQ)"])
        metric_cols = c3.multiselect("Business metrics", metric_options, default=metric_options[:min(3, len(metric_options))], key="biz_metrics")
        if not metric_cols:
            st.info("Select at least one numeric metric.")
            return
        c4, c5, c6 = st.columns(3)
        all_dimensions = _candidate_dimensions(df)
        region_ranked = _candidate_dimensions(df, "region")
        product_ranked = _candidate_dimensions(df, "product")
        customer_ranked = _candidate_dimensions(df, "customer")
        region_default = region_ranked[0] if region_ranked and _role_score(df, region_ranked[0], "region") >= 3 else "(Not available)"
        product_default = product_ranked[0] if product_ranked and _role_score(df, product_ranked[0], "product") >= 3 else "(Not available)"
        customer_default = customer_ranked[0] if customer_ranked and _role_score(df, customer_ranked[0], "customer") >= 3 else "(Not available)"
        dimension_options = ["(Not available)"] + all_dimensions
        region_choice = c4.selectbox("Regional dimension", dimension_options, index=dimension_options.index(region_default), key="biz_region")
        product_choice = c5.selectbox("Product dimension", dimension_options, index=dimension_options.index(product_default), key="biz_product")
        customer_choice = c6.selectbox("Customer dimension", dimension_options, index=dimension_options.index(customer_default), key="biz_customer")
        if all_dimensions:
            with st.expander("See all discovered dimensions"):
                st.write(", ".join(map(str, all_dimensions)))
        region_col = None if region_choice == "(Not available)" else region_choice
        product_col = None if product_choice == "(Not available)" else product_choice
        customer_col = None if customer_choice == "(Not available)" else customer_choice

    work, periods = _safe_period_frame(df, date_col, metric_cols, period_type)
    if work is None or len(periods) < 2:
        st.warning("At least two populated weeks or quarters are required.")
        return
    comparable_periods = [p for p in periods if (p - 1) in periods]
    if not comparable_periods:
        st.warning("No adjacent populated periods were found. WoW/QoQ requires data in consecutive calendar periods.")
        return
    labels = [_period_label(p) for p in comparable_periods]
    default_idx = len(labels) - 1
    current_label = st.selectbox("Current period (compared with the immediately preceding calendar period)", labels, index=default_idx, key="biz_current")
    if st.button("Generate business report", type="primary", key="generate_business_report"):
        try:
            report = _build_business_report(
                df, date_col, metric_cols, period_type, region_col, product_col, customer_col, current_label
            )
            st.session_state["business_report_result"] = report
            st.session_state["business_report_ai"] = None
        except ValueError as exc:
            st.error(str(exc))
            return

    report = st.session_state.get("business_report_result")
    if not report:
        st.info("Choose your dimensions and metrics, then generate the report.")
        return

    # Invalidate stale report if controls have changed.
    if (report["date_column"] != date_col or report["metrics"] != metric_cols
            or report["period_type"] != period_type or report["current_period"] != current_label
            or report.get("region_column") != region_col
            or report.get("product_column") != product_col
            or report.get("customer_column") != customer_col):
        st.info("Your report settings changed. Generate the report again to refresh the calculations.")
        return

    st.subheader(f"{report['previous_period']} → {report['current_period']}")
    st.markdown("#### Overall metric movement")
    st.dataframe(report["overall"].style.format({
        "Current": "{:,.2f}", "Previous": "{:,.2f}",
        "Absolute change": "{:+,.2f}", "Change %": "{:+.2f}%"
    }, na_rep="Not calculable"), use_container_width=True, hide_index=True)

    if not report["regional"].empty:
        st.markdown("#### Regional performance")
        regional = report["regional"]
        st.dataframe(regional, use_container_width=True, hide_index=True)
        for metric in metric_cols:
            change_col = f"{metric}_change"
            pct_col = f"{metric}_change_pct"
            if change_col in regional:
                positive = regional[change_col] > 0
                negative = regional[change_col] < 0
                if positive.any() and negative.any():
                    st.warning(f"{metric}: regions moved in different directions. Review the regional table for the actual increases and decreases.")
                elif positive.any():
                    st.info(f"{metric}: all regions with non-zero movement increased.")
                elif negative.any():
                    st.info(f"{metric}: all regions with non-zero movement decreased.")

    for section_name, section_data in (("Product contribution", report["products"]), ("Customer contribution", report["customers"])):
        if section_data:
            st.markdown(f"#### {section_name}")
            for metric, table in section_data.items():
                st.markdown(f"**{metric}**")
                st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### AI-written business interpretation")
    st.session_state.setdefault("business_report_ai", None)
    st.session_state.setdefault("business_report_ai_error", None)

    if st.button("Explain verified findings with Groq", key="explain_groq"):
        # Clear any earlier response so the user can see the outcome of this attempt.
        st.session_state["business_report_ai"] = None
        st.session_state["business_report_ai_error"] = None
        with st.spinner("Connecting to Groq and preparing your business narrative..."):
            explanation, error = _groq_explanation(report)
        st.session_state["business_report_ai"] = explanation
        st.session_state["business_report_ai_error"] = error

    if st.session_state.get("business_report_ai"):
        st.success("Groq analysis completed.")
        st.markdown(st.session_state["business_report_ai"])
    elif st.session_state.get("business_report_ai_error"):
        st.error(st.session_state["business_report_ai_error"])
    else:
        st.caption("Click the button above to generate an AI explanation of the verified report calculations.")

    # Download calculations as an auditable workbook-like CSV bundle.
    export_frames = {"overall": report["overall"]}
    if not report["regional"].empty:
        export_frames["regional"] = report["regional"]
    for metric, table in report["products"].items():
        export_frames[f"product_{metric}"] = table
    for metric, table in report["customers"].items():
        export_frames[f"customer_{metric}"] = table
    csv_parts = []
    for name, frame in export_frames.items():
        csv_parts.append(f"# {name}\n{frame.to_csv(index=False)}")
    st.download_button(
        "⬇️ Download report calculations (CSV)",
        data="\n".join(csv_parts).encode("utf-8"),
        file_name="business_analysis_report.csv",
        mime="text/csv"
    )



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
            st.caption("Examples: “top 5 products by Sales”, “compare Sales by Ship Mode”, “show monthly Sales trend”, “summarize dataset”.")
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
    st.write("- What is the total Sales?")
    st.write("- Which are the top 5 products by Sales?")
    st.write("- Compare Sales by Ship Mode.")
    st.write("- Show monthly Sales trends.")
    st.write("- Which columns have missing values?")
    st.write("- Show correlations between numeric columns.")
    st.write("- Summarize this dataset.")
