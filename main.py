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
    expense_per_period: float,
) -> pd.DataFrame:
    """
    Compute ending wealth for every rolling window of `months` for the
    selected return columns.
    """
    result = pd.DataFrame({"Start Date": df["Date"]})

    for col in columns:
        # Apply per-period expense directly to the monthly factor
        monthly_mult = df[col] * (1 - expense_per_period)
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
    start_value = st.number_input("Beginning portfolio value", value=10000.0, min_value=0.01)
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
    expense_bps = st.slider("Annual expense (bps)", min_value=0, max_value=100, value=0, step=5)
    period_unit = st.radio("Period unit", ["Months", "Years"], horizontal=True)
    period_length = st.number_input(
        f"Number of {period_unit.lower()}",
        value=120 if period_unit == "Months" else 10,
        min_value=1,
        step=1,
    )
    display_window = st.selectbox(
        "Show only the most recent period of results",
        options=["All history", 12, 24, 36, 48, 60, 120],
        format_func=lambda x: "All history" if x == "All history" else f"{x} months",
    )

    if not selected_datasets:
        st.warning("Pick at least one dataset to run the analysis.")
        st.stop()

    months = period_length if period_unit == "Months" else period_length * 12
    expense_rate = expense_bps / 10000  # convert bps to decimal annual
    expense_per_period = expense_rate / 12 if period_unit == "Months" else expense_rate
    results_long = []
    cumulative_paths = []

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
            res = compute_rolling_growth(
                df, months, start_value, selected, expense_per_period
            )

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
        if display_window != "All history":
            window_months = int(display_window)
            latest_date = df["Date"].max()
            cutoff_date = latest_date - pd.DateOffset(months=window_months - 1)
            period_df = df[df["Date"] >= cutoff_date]
            for col in selected:
                monthly_mult = period_df[col] * (1 - expense_per_period)
                cumulative_value = monthly_mult.cumprod() * start_value
                cumulative_paths.append(
                    pd.DataFrame(
                        {
                            "Date": period_df["Date"],
                            "Ending Value": cumulative_value,
                            "Series": col,
                            "Dataset": labels[ds],
                        }
                    )
                )

    results_long = (
        pd.concat(results_long, ignore_index=True) if results_long else pd.DataFrame()
    )

    if not results_long.empty:
        # Optionally restrict to the most recent N start dates for the table
        if display_window != "All history":
            window_months = int(display_window)
            latest_start = results_long["Start Date"].max()
            cutoff = latest_start - pd.DateOffset(months=window_months - 1)
            results_long = results_long[results_long["Start Date"] >= cutoff]

        if results_long.empty:
            st.warning("No rows left after applying the display window.")
            st.stop()

        if display_window == "All history":
            chart_data = (
                results_long.pivot_table(
                    index="Start Date", columns=["Dataset", "Series"], values="Ending Value"
                ).sort_index()
            )
            chart_data.columns = [f"{ds} - {series}" for ds, series in chart_data.columns]
            st.line_chart(chart_data, use_container_width=True)
        else:
            path_df = (
                pd.concat(cumulative_paths, ignore_index=True) if cumulative_paths else pd.DataFrame()
            )
            if path_df.empty:
                st.warning("No data available to plot the cumulative path.")
            else:
                path_chart = (
                    path_df.pivot_table(
                        index="Date", columns=["Dataset", "Series"], values="Ending Value"
                    ).sort_index()
                )
                path_chart.columns = [f"{ds} - {series}" for ds, series in path_chart.columns]
                st.line_chart(path_chart, use_container_width=True)

    st.subheader("Ending values for each rolling window")
    st.dataframe(results_long, height=500, use_container_width=True)

    st.info(
        "Every row represents a full rolling period starting at the given date and ending "
        "after the selected duration."
    )


if __name__ == "__main__":
    main()
