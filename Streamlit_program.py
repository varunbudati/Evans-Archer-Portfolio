# streamlit_diversification.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# tqdm provides console progress bar (useful for local runs, invisible in deployed app)
# from tqdm import tqdm # Can comment out if not needed
import io # Needed internally by Streamlit/Pandas for uploaded files
import datetime

# --- Page Configuration (Set Title and Layout) ---
st.set_page_config(page_title="Portfolio Diversification Analysis", layout="wide")
st.title("📈 Portfolio Diversification Analysis")
st.markdown("""
Replicates the analysis by **Evans & Archer (1968)** to show how portfolio risk
(standard deviation) decreases as the number of randomly selected stocks increases.
**Upload your own stock return data (CSV format)** using the sidebar and adjust parameters interactively.
""")

# --- Helper Functions ---

# Cache data loading and preparation to speed up app re-runs with same inputs
@st.cache_data
def load_and_prepare_data(uploaded_file_content, filename, date_col, permno_col, ret_col, shrcd_col, common_stock_codes, start_date, end_date):
    """Loads, filters, and pivots the uploaded data."""
    file_buffer = io.BytesIO(uploaded_file_content)
    st.write(f"Reading `{filename}`...")
    try:
        try:
            df = pd.read_csv(file_buffer, low_memory=False)
        except UnicodeDecodeError:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, low_memory=False, encoding='latin1')
        st.write(f"Successfully read `{filename}`.")

        required_cols = {date_col, permno_col, ret_col}
        optional_cols = {shrcd_col} if shrcd_col else set()
        present_cols = set(df.columns)

        missing_required = required_cols - present_cols
        if missing_required:
            st.error(f"Error: Missing required columns in CSV: {missing_required}")
            return None, 0, 0, None, None

        missing_optional = optional_cols - present_cols
        share_code_filtering_active = bool(shrcd_col) and not missing_optional

        st.write("Preprocessing data...")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
        df = df.dropna(subset=[ret_col])

        st.write(f"Filtering by date: {start_date} to {end_date}...")
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

        if share_code_filtering_active:
            st.write(f"Filtering using Share Code Column '{shrcd_col}' with codes {common_stock_codes}.")
            df = df[df[shrcd_col].isin(common_stock_codes)]
        elif shrcd_col and missing_optional:
            st.warning(f"Specified Share Code Column '{shrcd_col}' not found in CSV. Proceeding without stock type filtering.")
        elif not shrcd_col:
             st.info("No Share Code Column specified. Proceeding without stock type filtering.")

        if df.empty:
            st.error("Error: No data remaining after initial filtering (date, share code). Check settings, date range, or data format.")
            return None, 0, 0, None, None

        st.write("Pivoting data (Securities as columns, Time as rows)...")
        try:
            returns_pivot = df.pivot_table(index=date_col, columns=permno_col, values=ret_col)
        except Exception as e:
            st.error(f"Error during pivoting: {e}. Check for duplicate Security IDs for the same date.")
            duplicates = df[df.duplicated(subset=[date_col, permno_col], keep=False)]
            if not duplicates.empty:
                 st.error(f"Found {len(duplicates)} duplicate entries (example below):")
                 st.dataframe(duplicates.head())
            return None, 0, 0, None, None

        st.write("Checking for securities with complete data within the period...")
        initial_securities = returns_pivot.shape[1]
        returns_pivot = returns_pivot.dropna(axis=1, how='any')
        final_securities = returns_pivot.shape[1]
        num_dates = len(returns_pivot)
        st.write(f"Removed {initial_securities - final_securities} securities due to missing data points.")

        if final_securities == 0:
             st.error(f"Error: No securities found with complete return data for the entire period {start_date} to {end_date}. Try adjusting the date range or checking the input data.")
             return None, 0, 0, None, None

        if num_dates < 2:
            st.error(f"Error: Need at least 2 time periods with data to calculate standard deviation (found {num_dates}). Try adjusting the date range.")
            return None, 0, 0, None, None

        min_date_in_data = returns_pivot.index.min().date()
        max_date_in_data = returns_pivot.index.max().date()
        st.write("Data preparation complete.")
        return returns_pivot, num_dates, final_securities, min_date_in_data, max_date_in_data

    except Exception as e:
        st.error(f"An error occurred during data loading/preparation: {e}")
        # import traceback # Uncomment for detailed debug logs if needed
        # st.error(traceback.format_exc())
        return None, 0, 0, None, None


# --- Simulation Function (No caching for simplicity now) ---
# Takes the actual prepared DataFrame as input
def run_simulation(_returns_pivot, max_portfolio_size_sim, num_simulations_sim):
    """Runs the portfolio simulation using the prepared returns_pivot DataFrame."""

    # --- Input Validation ---
    # This check is crucial: ensure the simulation receives valid data
    if _returns_pivot is None or _returns_pivot.empty:
        # This error message should ideally not be hit if the workflow buttons disable correctly,
        # but it's a safeguard.
        st.error("Internal Error: run_simulation called with invalid data.")
        return None, None

    available_securities = _returns_pivot.columns.tolist()
    num_available_securities = len(available_securities)
    num_dates = len(_returns_pivot)

    if num_available_securities <= 0:
         st.error("Cannot run simulation: 0 securities available after filtering.")
         return None, None
    if num_dates < 2:
         st.error(f"Cannot run simulation: Need at least 2 time periods for std dev calculation (found {num_dates}).")
         return None, None

    # Adjust max size if needed
    actual_max_portfolio_size = min(max_portfolio_size_sim, num_available_securities)
    if actual_max_portfolio_size < max_portfolio_size_sim:
         st.warning(f"Adjusted Max Portfolio Size from {max_portfolio_size_sim} to {actual_max_portfolio_size} due to available securities ({num_available_securities}).")
    if actual_max_portfolio_size <= 0:
         st.error(f"Cannot run simulation: Max portfolio size adjusted to {actual_max_portfolio_size}.")
         return None, None

    st.write(f"Starting simulation: {num_simulations_sim} runs for sizes 1 to {actual_max_portfolio_size}...")
    portfolio_sizes = range(1, actual_max_portfolio_size + 1)
    average_std_devs = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Simulation Loop ---
    for i, m in enumerate(portfolio_sizes):
        status_text.text(f"Simulating Portfolio Size: {m}/{actual_max_portfolio_size}")
        portfolio_std_devs_for_size_m = []
        if m > num_available_securities: # Safety check (shouldn't be needed with adjustment above)
            continue
        for k in range(num_simulations_sim):
            try:
                selected_permnos = random.sample(available_securities, m)
                # Access the passed DataFrame directly
                portfolio_component_returns = _returns_pivot[selected_permnos]
                portfolio_period_returns = portfolio_component_returns.mean(axis=1)

                # Check if std dev calculation is possible (at least 2 data points)
                if len(portfolio_period_returns) < 2:
                     std_dev = np.nan # Cannot calculate std dev
                else:
                     log_value_relatives = np.log(1 + portfolio_period_returns + 1e-10) # Add epsilon for log(0)
                     std_dev = log_value_relatives.std(ddof=1) # Use sample std dev

                portfolio_std_devs_for_size_m.append(std_dev)
            except Exception as e:
                 # Catch potential errors during sampling or calculation for a single run
                 st.warning(f"Warning during simulation run {k+1} for size {m}: {e}. Appending NaN.")
                 portfolio_std_devs_for_size_m.append(np.nan)


        # Calculate average std dev for size 'm', ignoring NaNs
        valid_std_devs_m = [s for s in portfolio_std_devs_for_size_m if pd.notna(s)]
        if valid_std_devs_m:
             average_std_dev = np.mean(valid_std_devs_m)
             average_std_devs.append(average_std_dev)
        else:
             # If ALL runs for size m failed (unlikely but possible), append NaN
             st.warning(f"All {num_simulations_sim} simulation runs failed to produce a valid std. dev. for portfolio size {m}.")
             average_std_devs.append(np.nan)

        progress_bar.progress((i + 1) / actual_max_portfolio_size)

    status_text.text("Simulation loop complete.")
    progress_bar.empty()

    # --- Final Result Check ---
    # Filter out any potential NaNs if std dev failed for some size *entirely*
    valid_indices = ~np.isnan(average_std_devs)
    portfolio_sizes_final = np.array(portfolio_sizes)[valid_indices]
    average_std_devs_final = np.array(average_std_devs)[valid_indices]

    if len(portfolio_sizes_final) == 0:
         st.error("Simulation Error: No valid results generated. Standard deviation calculation might be failing consistently. Check data quality and time periods.")
         return None, None # Return None to indicate failure

    st.write("Simulation finished successfully.")
    return portfolio_sizes_final.tolist(), average_std_devs_final.tolist()


def hyperbolic_func(x, a, b):
    """Function for curve fitting: Y = A + B/X"""
    return a + b / x

# Keep curve fitting cached based on inputs (results of simulation)
@st.cache_data
def fit_curve(portfolio_sizes, average_std_devs):
    """Fits the hyperbolic curve to the simulation results."""
    if not portfolio_sizes or not average_std_devs or len(portfolio_sizes) < 2:
        st.warning("Need at least 2 data points for curve fitting.")
        return None, None, None, None

    x_data = np.array(portfolio_sizes)
    y_data = np.array(average_std_devs)

    valid_data = ~np.isnan(y_data) & ~np.isinf(y_data) # Also check for inf
    if not np.all(valid_data):
        st.warning(f"Invalid (NaN/Inf) values found in average standard deviations. Fitting on valid points only.")
        x_data = x_data[valid_data]
        y_data = y_data[valid_data]
        if len(x_data) < 2:
             st.warning("Not enough valid data points remaining for curve fitting.")
             return None, None, None, None

    try:
        st.write("Attempting curve fit...")
        initial_guess = [y_data[-1], (y_data[0] - y_data[-1])]
        params, covariance = curve_fit(hyperbolic_func, x_data, y_data, p0=initial_guess, maxfev=5000)
        a_fit, b_fit = params

        residuals = y_data - hyperbolic_func(x_data, a_fit, b_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0
        st.write("Curve fitting successful.")
        return a_fit, b_fit, r_squared, x_data
    except Exception as e:
        st.warning(f"Curve fitting failed: {e}")
        return None, None, None, None

# Plotting function remains largely the same
def plot_results(portfolio_sizes, average_std_devs, a_fit, b_fit, r_squared, x_fit_data, data_start, data_end):
    """Generates the diversification plot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    if portfolio_sizes and average_std_devs:
        ax.plot(portfolio_sizes, average_std_devs, 'bo-', label='Avg Portfolio Std Dev (Simulated)', markersize=5)

    fit_label = "Fitted Curve (Failed)"
    asymptote_label = "Asymptote (Systematic Risk)"
    if a_fit is not None and b_fit is not None and x_fit_data is not None:
         fit_label = f'Fitted Curve: Y = {a_fit:.4f} + {b_fit:.4f}/X'
         if r_squared is not None:
              fit_label += f'\n$R^2 = {r_squared:.4f}$'
         ax.plot(x_fit_data, hyperbolic_func(x_fit_data, a_fit, b_fit), 'r--', label=fit_label)

    if a_fit is not None:
         asymptote_label = f'Asymptote (Systematic Risk) = {a_fit:.4f}'
         xmin, xmax = 0, max(portfolio_sizes if portfolio_sizes else [1]) * 1.05 # Determine plot range
         if x_fit_data is not None:
              xmax = max(xmax, max(x_fit_data)*1.05)
         ax.hlines(y=a_fit, xmin=xmin, xmax=xmax, color='g', linestyle=':', label=asymptote_label)
         ax.set_xlim(xmin, xmax)


    ax.set_xlabel("Number of Securities in Portfolio (X)")
    ax.set_ylabel("Average Standard Deviation of Log Value Relatives (Y)")
    ax.set_title(f"Portfolio Diversification Effect ({data_start} to {data_end})")
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if average_std_devs:
        valid_std_devs = [s for s in average_std_devs if pd.notna(s)]
        if valid_std_devs:
             min_val = min(valid_std_devs)
             max_val = max(valid_std_devs)
             # Adjust y-axis limits, ensure min < max
             min_y_plot = min_val * 0.95 if a_fit is None else min(min_val, a_fit if pd.notna(a_fit) else min_val) * 0.95
             max_y_plot = max_val * 1.05
             if min_y_plot >= max_y_plot:
                  min_y_plot = max_y_plot * 0.9
             ax.set_ylim(min_y_plot, max_y_plot)

    st.pyplot(fig)
    plt.close(fig)


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Upload Stock Return CSV", type=["csv", "txt"])

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'returns_pivot' not in st.session_state:
    st.session_state.returns_pivot = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'num_securities' not in st.session_state:
     st.session_state.num_securities = 0
if 'data_min_date' not in st.session_state:
     st.session_state.data_min_date = None
if 'data_max_date' not in st.session_state:
     st.session_state.data_max_date = None


st.sidebar.subheader("2. Data Columns")
date_col = st.sidebar.text_input("Date Column Name", value="date", key="date_col_input")
permno_col = st.sidebar.text_input("Security ID Column Name", value="PERMNO", key="permno_col_input")
ret_col = st.sidebar.text_input("Return Column Name", value="RET", key="ret_col_input")
shrcd_col = st.sidebar.text_input("Share Code Column Name (Optional)", value="SHRCD", help="Leave blank if not filtering by share code.", key="shrcd_col_input")
common_codes_str = st.sidebar.text_input("Common Stock Codes (comma-sep)", value="10,11", help="Used if Share Code Column is provided.", key="common_codes_input")

common_stock_codes = []
if shrcd_col and common_codes_str:
    try:
        common_stock_codes = [int(code.strip()) for code in common_codes_str.split(',') if code.strip()]
    except ValueError:
        st.sidebar.error("Invalid Common Stock Codes format. Use comma-separated integers.")

st.sidebar.subheader("3. Analysis Period & Simulation")
default_start = datetime.date(2010, 1, 1)
default_end = datetime.date(2020, 12, 31)
start_date = st.sidebar.date_input("Start Date", value=default_start, key="start_date_input")
end_date = st.sidebar.date_input("End Date", value=default_end, key="end_date_input")

can_run_analysis = True
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    can_run_analysis = False

max_portfolio_size = st.sidebar.slider(
    "Max Portfolio Size (X)", min_value=2, max_value=200, value=40, step=1, key="max_size_slider", help="Minimum 2 required for std dev."
)
num_simulations = st.sidebar.slider(
    "Number of Simulations per Size", min_value=10, max_value=1000, value=100, step=10, key="num_sim_slider"
)

# --- Main Area Logic ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📊 Analysis Setup & Results")

    # Button to trigger data load and prep
    if st.button("Load and Prepare Data", key="load_data_button"):
        # Reset state before loading new data
        st.session_state.data_loaded = False
        st.session_state.returns_pivot = None
        st.session_state.simulation_results = None
        st.session_state.num_securities = 0

        if not can_run_analysis:
             st.error("Please fix the date range in the sidebar before loading data.")
        else:
            uploaded_file_content = uploaded_file.getvalue()
            filename = uploaded_file.name
            with st.spinner(f"Loading and preparing data from `{filename}`..."):
                returns_pivot, num_dates, num_securities, data_min_date, data_max_date = load_and_prepare_data(
                    uploaded_file_content, filename, date_col, permno_col, ret_col, shrcd_col, common_stock_codes, start_date, end_date
                )

            if returns_pivot is not None:
                st.success(f"Data loaded and prepared successfully!")
                st.markdown(f"""
                *   **Analysis Period Used:** `{start_date}` to `{end_date}`
                *   **Date Range Found in Filtered Data:** `{data_min_date}` to `{data_max_date}` (`{num_dates}` periods)
                *   **Securities Available for Simulation:** `{num_securities}`
                """)
                # Store prepared data in session state
                st.session_state.data_loaded = True
                st.session_state.returns_pivot = returns_pivot
                st.session_state.num_securities = num_securities
                st.session_state.data_min_date = data_min_date
                st.session_state.data_max_date = data_max_date
            else:
                st.warning("Data could not be loaded or prepared. Check file, columns, dates, and data integrity.")
                st.session_state.data_loaded = False # Ensure state reflects failure

    # Only show simulation button if data is loaded successfully
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("🚀 Run Simulation")
        st.write(f"Ready to simulate using **{st.session_state.num_securities}** securities.")

        # Disable button if conditions aren't met
        run_sim_disabled = st.session_state.num_securities < 1 or max_portfolio_size < 1

        if st.button("Run Diversification Simulation", key="run_sim_button", disabled=run_sim_disabled):
            st.session_state.simulation_results = None # Clear previous results

            returns_pivot_df = st.session_state.returns_pivot # Get DF from state

            # Check again before running
            if returns_pivot_df is None or returns_pivot_df.empty:
                 st.error("Cannot run simulation: Prepared data is missing or invalid.")
            else:
                with st.spinner("Running simulations... This may take a while."):
                     # Pass the actual dataframe directly
                     portfolio_sizes, average_std_devs = run_simulation(
                         returns_pivot_df,
                         max_portfolio_size,
                         num_simulations
                     )

                if portfolio_sizes is not None and average_std_devs is not None:
                     st.session_state.simulation_results = (portfolio_sizes, average_std_devs)
                     st.success("Simulation complete!")
                else:
                    # Error message already shown inside run_simulation
                    st.error("Simulation failed to produce results. Please check the console/logs or data settings.")

    # Display results if simulation has run successfully
    if st.session_state.simulation_results:
        st.markdown("---")
        st.subheader("📈 Simulation Results")
        portfolio_sizes, average_std_devs = st.session_state.simulation_results

        st.markdown("#### Curve Fitting")
        a_fit, b_fit, r_squared, x_fit_data = fit_curve(portfolio_sizes, average_std_devs)

        col1, col2, col3 = st.columns(3)
        col1.metric("Systematic Risk (A)", f"{a_fit:.5f}" if a_fit is not None else "N/A")
        col2.metric("Diversification Coeff (B)", f"{b_fit:.5f}" if b_fit is not None else "N/A")
        col3.metric("R-squared of Fit", f"{r_squared:.4f}" if r_squared is not None else "N/A")

        st.markdown("#### Diversification Plot")
        plot_results(
             portfolio_sizes, average_std_devs,
             a_fit, b_fit, r_squared, x_fit_data,
             st.session_state.data_min_date, st.session_state.data_max_date
        )

elif not uploaded_file:
    st.info("👈 **Please upload a CSV file using the sidebar to begin.**")

st.markdown("---")
st.markdown("App based on Evans & Archer (1968) | Built with [Streamlit](https://streamlit.io)")
