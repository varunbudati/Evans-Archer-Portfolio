import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm # Optional for console progress, not visible in Streamlit app
import io # To handle uploaded file bytes
import datetime

# --- Page Configuration (Set Title and Layout) ---
st.set_page_config(page_title="Portfolio Diversification Analysis", layout="wide")
st.title("📈 Portfolio Diversification Analysis")
st.markdown("""
Replicates the analysis by **Evans & Archer (1968)** to show how portfolio risk
(standard deviation) decreases as the number of randomly selected stocks increases.
Upload your own stock return data (CSV format) and adjust parameters interactively.
""")

# --- Helper Functions ---

# Cache data loading and preparation to speed up app
@st.cache_data
def load_and_prepare_data(uploaded_file, date_col, permno_col, ret_col, shrcd_col, common_stock_codes, start_date, end_date):
    """Loads, filters, and pivots the uploaded data."""
    try:
        # Determine the correct encoding
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
        except UnicodeDecodeError:
            # Reset buffer position after failed read attempt
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False, encoding='latin1')

        # --- Basic Column Checks ---
        required_cols = {date_col, permno_col, ret_col}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Error: Missing required columns in CSV: {missing}")
            return None, 0, 0, None, None # Indicate error

        # --- Preprocessing ---
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        # Convert return column
        df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
        # Drop rows with NaN returns early, as they can't be used
        df = df.dropna(subset=[ret_col])

        # Filter by date range first
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

        # Filter for common stocks (if column exists)
        if shrcd_col and shrcd_col in df.columns:
            df = df[df[shrcd_col].isin(common_stock_codes)]
        elif shrcd_col:
            st.warning(f"Share code column '{shrcd_col}' not found. Proceeding without stock type filtering.")

        if df.empty:
            st.error("Error: No data remaining after initial filtering (date, common stock). Check settings or data.")
            return None, 0, 0, None, None

        # --- Pivoting ---
        try:
            returns_pivot = df.pivot_table(index=date_col, columns=permno_col, values=ret_col)
        except Exception as e:
            st.error(f"Error during pivoting: {e}. Check for duplicate PERMNO entries for the same date.")
            duplicates = df[df.duplicated(subset=[date_col, permno_col], keep=False)]
            if not duplicates.empty:
                 st.error("Found duplicate entries (example):")
                 st.dataframe(duplicates.head())
            return None, 0, 0, None, None

        # --- Final Data Cleaning (Completeness Check) ---
        initial_securities = returns_pivot.shape[1]
        returns_pivot = returns_pivot.dropna(axis=1, how='any')
        final_securities = returns_pivot.shape[1]
        num_dates = len(returns_pivot)

        if final_securities == 0:
             st.error(f"Error: No securities found with complete return data for the period {start_date} to {end_date}.")
             st.info(f"Started with {initial_securities} securities before checking for NaNs.")
             return None, 0, 0, None, None

        if num_dates < 2:
            st.error("Error: Need at least 2 time periods with data to calculate standard deviation.")
            return None, 0, 0, None, None

        min_date_in_data = returns_pivot.index.min().date()
        max_date_in_data = returns_pivot.index.max().date()

        return returns_pivot, num_dates, final_securities, min_date_in_data, max_date_in_data

    except Exception as e:
        st.error(f"An error occurred during data loading/preparation: {e}")
        return None, 0, 0, None, None


# Cache the simulation results based on input parameters
@st.cache_data
def run_simulation(_returns_pivot, max_portfolio_size_sim, num_simulations_sim):
    """Runs the portfolio simulation."""
    if _returns_pivot is None or _returns_pivot.empty:
        return None, None

    available_securities = _returns_pivot.columns.tolist()
    num_available_securities = len(available_securities)

    # Adjust max size if needed
    actual_max_portfolio_size = min(max_portfolio_size_sim, num_available_securities)
    if actual_max_portfolio_size < max_portfolio_size_sim:
         st.warning(f"Adjusted Max Portfolio Size from {max_portfolio_size_sim} to {actual_max_portfolio_size} due to available securities.")
    if actual_max_portfolio_size == 0:
         st.error("Cannot run simulation with 0 available securities.")
         return None, None

    portfolio_sizes = range(1, actual_max_portfolio_size + 1)
    average_std_devs = []
    all_simulation_results = {size: [] for size in portfolio_sizes}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, m in enumerate(portfolio_sizes):
        status_text.text(f"Simulating Portfolio Size: {m}/{actual_max_portfolio_size}")
        portfolio_std_devs_for_size_m = []
        for _ in range(num_simulations_sim):
            selected_permnos = random.sample(available_securities, m)
            portfolio_component_returns = _returns_pivot[selected_permnos]
            portfolio_period_returns = portfolio_component_returns.mean(axis=1)
            log_value_relatives = np.log(1 + portfolio_period_returns + 1e-10) # Add epsilon for log(0)
            std_dev = log_value_relatives.std(ddof=1) # Use sample std dev
            portfolio_std_devs_for_size_m.append(std_dev)
            all_simulation_results[m].append(std_dev)

        average_std_dev = np.mean(portfolio_std_devs_for_size_m)
        average_std_devs.append(average_std_dev)
        progress_bar.progress((i + 1) / actual_max_portfolio_size)

    status_text.text("Simulation complete!")
    progress_bar.empty() # Clear progress bar

    return portfolio_sizes, average_std_devs


def hyperbolic_func(x, a, b):
    """Function for curve fitting: Y = A + B/X"""
    return a + b / x

def fit_curve(portfolio_sizes, average_std_devs):
    """Fits the hyperbolic curve to the simulation results."""
    if not portfolio_sizes or not average_std_devs:
        return None, None, None, None

    x_data = np.array(portfolio_sizes)
    y_data = np.array(average_std_devs)

    try:
        initial_guess = [y_data[-1], (y_data[0] - y_data[-1])] # A ~ last std dev, B ~ initial drop
        params, covariance = curve_fit(hyperbolic_func, x_data, y_data, p0=initial_guess)
        a_fit, b_fit = params

        # Calculate R-squared
        residuals = y_data - hyperbolic_func(x_data, a_fit, b_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        # Handle ss_tot == 0 case (if all y_data points are the same)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return a_fit, b_fit, r_squared, x_data # Return x_data used for fit
    except Exception as e:
        st.warning(f"Curve fitting failed: {e}")
        return None, None, None, None

def plot_results(portfolio_sizes, average_std_devs, a_fit, b_fit, r_squared, x_fit_data, data_start, data_end):
    """Generates the diversification plot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    if portfolio_sizes and average_std_devs:
        ax.plot(portfolio_sizes, average_std_devs, 'bo-', label='Average Portfolio Std Dev (Simulated)', markersize=5)

    fit_label = "Fitted Curve"
    asymptote_label = "Asymptote (Systematic Risk)"
    if a_fit is not None and b_fit is not None and x_fit_data is not None:
         fit_label = f'Fitted Curve: Y = {a_fit:.4f} + {b_fit:.4f}/X'
         if r_squared is not None:
              fit_label += f'\n$R^2 = {r_squared:.4f}$'
         ax.plot(x_fit_data, hyperbolic_func(x_fit_data, a_fit, b_fit), 'r--', label=fit_label)

    if a_fit is not None:
         asymptote_label = f'Asymptote (Systematic Risk) = {a_fit:.4f}'
         ax.axhline(y=a_fit, color='g', linestyle=':', label=asymptote_label)


    ax.set_xlabel("Number of Securities in Portfolio (X)")
    ax.set_ylabel("Average Standard Deviation of Log Value Relatives (Y)")
    ax.set_title(f"Portfolio Diversification Effect ({data_start} to {data_end})")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Dynamic limits
    if portfolio_sizes:
         ax.set_xlim(0, max(portfolio_sizes) * 1.05)
    if average_std_devs:
        min_y_plot = min(average_std_devs) * 0.95 if a_fit is None else min(min(average_std_devs), a_fit) * 0.95
        max_y_plot = max(average_std_devs) * 1.05
        ax.set_ylim(min_y_plot, max_y_plot)

    st.pyplot(fig)
    # Clear the current figure to prevent overlaps in subsequent runs if plt state persists
    plt.clf()
    plt.close(fig)


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Stock Return CSV", type="csv")

# --- Default values ---
default_start = datetime.date(2010, 1, 1)
default_end = datetime.date(2020, 12, 31)
default_max_size = 40
default_simulations = 100 # Reduced default for faster interaction in streamlit

# --- Input Fields (disabled until file is uploaded) ---
st.sidebar.subheader("Data Columns")
date_col = st.sidebar.text_input("Date Column Name", value="date", disabled=not uploaded_file)
permno_col = st.sidebar.text_input("Security ID Column Name", value="PERMNO", disabled=not uploaded_file)
ret_col = st.sidebar.text_input("Return Column Name", value="RET", disabled=not uploaded_file)
shrcd_col = st.sidebar.text_input("Share Code Column Name (Optional)", value="SHRCD", disabled=not uploaded_file)
common_codes_str = st.sidebar.text_input("Common Stock Codes (comma-sep)", value="10,11", disabled=not uploaded_file)
try:
    common_stock_codes = [int(code.strip()) for code in common_codes_str.split(',') if code.strip()]
except ValueError:
    st.sidebar.error("Invalid Common Stock Codes format. Use comma-separated integers.")
    common_stock_codes = [10, 11] # Fallback

st.sidebar.subheader("Analysis Period & Simulation")
start_date = st.sidebar.date_input("Start Date", value=default_start, disabled=not uploaded_file)
end_date = st.sidebar.date_input("End Date", value=default_end, disabled=not uploaded_file)

# Validate date range
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    can_run = False
else:
    can_run = True


max_portfolio_size = st.sidebar.slider(
    "Max Portfolio Size (X)",
    min_value=5,
    max_value=100, # Reasonable upper limit for slider
    value=default_max_size,
    step=1,
    disabled=not uploaded_file
)
num_simulations = st.sidebar.slider(
    "Number of Simulations per Size",
    min_value=10,
    max_value=1000, # Reasonable upper limit
    value=default_simulations,
    step=10,
    disabled=not uploaded_file
)

# --- Main Area ---
if uploaded_file is not None and can_run:
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Load data using the helper function (cached)
    with st.spinner("Loading and preparing data..."):
        returns_pivot, num_dates, num_securities, data_min_date, data_max_date = load_and_prepare_data(
            uploaded_file, date_col, permno_col, ret_col, shrcd_col, common_stock_codes, start_date, end_date
        )

    if returns_pivot is not None:
        st.success(f"Data loaded successfully!")
        st.markdown(f"""
        *   **Analysis Period in Data:** {data_min_date} to {data_max_date}
        *   **Number of Time Periods:** {num_dates}
        *   **Number of Securities w/ Complete Data:** {num_securities}
        """)

        # Add a button to run the potentially long simulation
        if st.button("🚀 Run Diversification Simulation"):
            with st.spinner("Running simulations... This may take a while."):
                 portfolio_sizes, average_std_devs = run_simulation(returns_pivot, max_portfolio_size, num_simulations)

            if portfolio_sizes and average_std_devs:
                st.markdown("#### Curve Fitting")
                a_fit, b_fit, r_squared, x_fit_data = fit_curve(portfolio_sizes, average_std_devs)

                col1, col2, col3 = st.columns(3)
                if a_fit is not None:
                    col1.metric("Systematic Risk (A)", f"{a_fit:.5f}")
                else:
                    col1.metric("Systematic Risk (A)", "N/A")
                if b_fit is not None:
                     col2.metric("Diversification Coeff (B)", f"{b_fit:.5f}")
                else:
                     col2.metric("Diversification Coeff (B)", "N/A")
                if r_squared is not None:
                    col3.metric("R-squared of Fit", f"{r_squared:.4f}")
                else:
                    col3.metric("R-squared of Fit", "N/A")


                st.markdown("#### Diversification Plot")
                plot_results(portfolio_sizes, average_std_devs, a_fit, b_fit, r_squared, x_fit_data, data_min_date, data_max_date)
            else:
                st.error("Simulation did not produce results. Check data or settings.")

    else:
        st.warning("Data could not be loaded or prepared. Please check the file format and configuration settings.")

elif not uploaded_file:
    st.info("👈 Please upload a CSV file using the sidebar to begin.")

st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io)")
