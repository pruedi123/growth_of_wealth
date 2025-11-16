import pandas as pd
import streamlit as st

# Streamlit setup
st.set_page_config(page_title="Growth of Wealth", layout="wide")


@st.cache_data
def load_monthly_returns(path: str) -> pd.DataFrame:
    """
    Load monthly return factors from Excel, clean column names, and parse dates.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_rolling_growth(
    df: pd.DataFrame,
    months: int,
    start_value: float,
    columns: list[str],
    equity_share: float,
) -> pd.DataFrame:
    """
    Compute ending wealth for every rolling window of `months` for the
    selected return columns.
    """
    result = pd.DataFrame({"Start Date": df["Date"]})

    for col in columns:
        # Blend equity factor with cash (factor of 1.0) using the chosen equity share
        monthly_mult = 1 + equity_share * (df[col] - 1)
        # Rolling product gives cumulative growth for each window; shift aligns to window start
        growth = (
            monthly_mult.rolling(window=months, min_periods=months)
            .apply(lambda x: x.prod(), raw=True)
            .shift(-(months - 1))
        )
        result[f"{col} Ending Value"] = growth * start_value

    # End date is the date `months - 1` rows after the start date
    result["End Date"] = df["Date"].shift(-(months - 1))

    # Drop only the rows that do not have a full window (tail of the dataset after the shift)
    value_cols = [c for c in result.columns if c.endswith("Ending Value")]
    result = result.dropna(subset=value_cols + ["End Date"]).reset_index(drop=True)
    return result


def main():
    st.title("Growth of Wealth from Monthly Returns")
    st.caption(
        "Uses monthly return factors starting June 1927 to compute every possible rolling period."
    )

    # Inputs
    start_value = st.number_input("Beginning portfolio value", value=1000.0, min_value=0.01)
    dataset_options = ["global_mo_factors.xlsx", "spx_mo_factors.xlsx"]
    labels = {
        "global_mo_factors.xlsx": "Global factors",
        "spx_mo_factors.xlsx": "S&P 500 factors",
    }
    selected_datasets = st.multiselect(
        "Datasets",
        options=dataset_options,
        default=dataset_options,
        format_func=lambda x: labels[x],
    )
    equity_pct = st.slider("Equity allocation (%)", min_value=0, max_value=100, value=100, step=1)
    period_unit = st.radio("Period unit", ["Months", "Years"], horizontal=True)
    period_length = st.number_input(
        f"Number of {period_unit.lower()}",
        value=120 if period_unit == "Months" else 10,
        min_value=1,
        step=1,
    )

    if not selected_datasets:
        st.warning("Pick at least one dataset to run the analysis.")
        st.stop()

    months = period_length if period_unit == "Months" else period_length * 12
    equity_share = equity_pct / 100

    results_long = []

    for ds in selected_datasets:
        df = load_monthly_returns(ds)
        return_columns = [c for c in df.columns if c != "Date"]
        selected = st.multiselect(
            f"Return series from {labels[ds]}",
            options=return_columns,
            default=[return_columns[0]],
            key=f"{ds}_series",
        )

        if not selected:
            st.warning(f"Select at least one return series for {labels[ds]}.")
            st.stop()

        with st.spinner(f"Computing rolling growth for {labels[ds]}..."):
            res = compute_rolling_growth(df, months, start_value, selected, equity_share)

        value_cols = [c for c in res.columns if c.endswith("Ending Value")]
        tidy = res.melt(
            id_vars=["Start Date", "End Date"],
            value_vars=value_cols,
            var_name="Series",
            value_name="Ending Value",
        )
        tidy["Series"] = tidy["Series"].str.replace(" Ending Value", "", regex=False)
        tidy["Dataset"] = labels[ds]
        results_long.append(tidy)

    results_long = (
        pd.concat(results_long, ignore_index=True) if results_long else pd.DataFrame()
    )

    st.subheader("Ending values for each rolling window")
    st.dataframe(results_long, height=500, use_container_width=True)

    if not results_long.empty:
        chart_data = (
            results_long.pivot_table(
                index="Start Date", columns=["Dataset", "Series"], values="Ending Value"
            ).sort_index()
        )
        chart_data.columns = [f"{ds} - {series}" for ds, series in chart_data.columns]
        st.line_chart(chart_data, use_container_width=True)

    st.info(
        "Every row represents a full rolling period starting at the given date and ending "
        "after the selected duration."
    )


if __name__ == "__main__":
    main()
