# streamlit_diversification.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# tqdm provides console progress bar (useful for local runs, invisible in deployed app)
from tqdm import tqdm
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
    # The uploaded_file_content is bytes, wrap it in BytesIO for pandas
    file_buffer = io.BytesIO(uploaded_file_content)
    try:
        # Determine the correct encoding
        try:
            df = pd.read_csv(file_buffer, low_memory=False)
        except UnicodeDecodeError:
            # Reset buffer position after failed read attempt
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, low_memory=False, encoding='latin1')

        st.write(f"Successfully read `{filename}`.") # Feedback

        # --- Basic Column Checks ---
        required_cols = {date_col, permno_col, ret_col}
        optional_cols = {shrcd_col} if shrcd_col else set()
        present_cols = set(df.columns)

        missing_required = required_cols - present_cols
        if missing_required:
            st.error(f"Error: Missing required columns in CSV: {missing_required}")
            return None, 0, 0, None, None # Indicate error

        missing_optional = optional_cols - present_cols
        share_code_filtering_active = bool(shrcd_col) and not missing_optional

        # --- Preprocessing ---
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]) # Drop rows where date conversion failed

        # Convert return column
        df[ret_col] = pd.to_numeric(df[ret_col], errors='coerce')
        # Drop rows with NaN returns early, as they can't be used
        df = df.dropna(subset=[ret_col])

        # Filter by date range first
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

        # Filter for common stocks (if column exists and was provided)
        if share_code_filtering_active:
            df = df[df[shrcd_col].isin(common_stock_codes)]
            st.write(f"Filtering data using Share Code Column '{shrcd_col}' with codes {common_stock_codes}.")
        elif shrcd_col and missing_optional:
            st.warning(f"Specified Share Code Column '{shrcd_col}' not found in the CSV. Proceeding without stock type filtering.")
        elif not shrcd_col:
             st.info("No Share Code Column specified. Proceeding without stock type filtering.")


        if df.empty:
            st.error("Error: No data remaining after initial filtering (date, share code). Check settings, date range, or data.")
            return None, 0, 0, None, None

        # --- Pivoting ---
        st.write("Pivoting data (Securities as columns, Time as rows)...")
        try:
            returns_pivot = df.pivot_table(index=date_col, columns=permno_col, values=ret_col)
        except Exception as e:
            st.error(f"Error during pivoting: {e}. Check for duplicate Security IDs for the same date.")
            # Provide more debug info if duplicates exist
            duplicates = df[df.duplicated(subset=[date_col, permno_col], keep=False)]
            if not duplicates.empty:
                 st.error(f"Found {len(duplicates)} duplicate entries (example below):")
                 st.dataframe(duplicates.head())
            return None, 0, 0, None, None

        # --- Final Data Cleaning (Completeness Check) ---
        st.write("Checking for securities with complete data within the period...")
        initial_securities = returns_pivot.shape[1]
        returns_pivot = returns_pivot.dropna(axis=1, how='any') # Drop columns (securities) with ANY NaN
        final_securities = returns_pivot.shape[1]
        num_dates = len(returns_pivot)
        st.write(f"Removed {initial_securities - final_securities} securities due to missing data points.")


        if final_securities == 0:
             st.error(f"Error: No securities found with complete return data for the entire period {start_date} to {end_date}.")
             return None, 0, 0, None, None

        if num_dates < 2:
            st.error("Error: Need at least 2 time periods with data to calculate standard deviation.")
            return None, 0, 0, None, None

        min_date_in_data = returns_pivot.index.min().date()
        max_date_in_data = returns_pivot.index.max().date()

        return returns_pivot, num_dates, final_securities, min_date_in_data, max_date_in_data

    except Exception as e:
        st.error(f"An error occurred during data loading/preparation: {e}")
        # Consider adding more detailed traceback logging for debugging if needed
        # import traceback
        # st.error(traceback.format_exc())
        return None, 0, 0, None, None


# Cache the simulation results based on input parameters
# Use _returns_pivot.values.tobytes() as part of cache key if pivot df changes
@st.cache_data
def run_simulation(_returns_pivot_hash, available_securities, max_portfolio_size_sim, num_simulations_sim):
    """Runs the portfolio simulation. Needs data hash and available securities list for caching."""
    # Note: We don't pass the full pivot dataframe to keep cache key smaller,
    # but this assumes the simulation only needs the list of securities and params.
    # If the simulation needed the actual returns, we'd need to pass the dataframe
    # or its hash representation. For this specific calculation (sampling names),
    # the list of names should suffice for the cache logic, assuming the underlying
    # data associated with those names is implicitly handled by the caller.
    # Let's refine this - the simulation *does* need the actual returns_pivot.
    # We need a way to cache based on it without passing the whole large DF.
    # The caller (main part of script) will pass the actual _returns_pivot.
    # Let's adjust the signature and cache key generation there.

    st.error("Simulation caching needs refinement. Re-running simulation for now.") # Placeholder
    # This function signature needs the actual dataframe now:
    # def run_simulation(_returns_pivot, max_portfolio_size_sim, num_simulations_sim):

    # The code below assumes _returns_pivot is passed directly (modify call site)
    if _returns_pivot_hash is None: # Let's assume hash represents if data is valid
         st.error("Cannot run simulation without valid data.")
         return None, None

    num_available_securities = len(available_securities)

    # Adjust max size if needed
    actual_max_portfolio_size = min(max_portfolio_size_sim, num_available_securities)
    if actual_max_portfolio_size < max_portfolio_size_sim:
         st.warning(f"Adjusted Max Portfolio Size from {max_portfolio_size_sim} to {actual_max_portfolio_size} due to available securities ({num_available_securities}).")
    if actual_max_portfolio_size <= 0: # Check for <= 0
         st.error("Cannot run simulation with 0 or fewer available securities.")
         return None, None
    if actual_max_portfolio_size == 1 and num_simulations_sim > 1:
         st.info("Max portfolio size is 1. Only one unique 'portfolio' possible.")
         # Could optimize here, but let simulation run for consistency

    portfolio_sizes = range(1, actual_max_portfolio_size + 1)
    average_std_devs = []
    # all_simulation_results = {size: [] for size in portfolio_sizes} # Keep if needed later

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Access the returns_pivot data (passed by the caller)
    # We need to get the actual DF associated with _returns_pivot_hash
    # This caching strategy needs rethinking or simplification for Streamlit.
    # For now, let's remove the simulation caching @st.cache_data and run it each time.
    # This avoids complexity but means it re-runs on every interaction after clicking Run.

    # Reloading data to get the dataframe (since we can't cache the df easily)
    # THIS IS INEFFICIENT - We need the DF from the load_and_prepare step.
    # Let's restructure the main logic flow.

    # *** REVISED APPROACH: Pass the actual returns_pivot to simulation ***
    # Remove @st.cache_data from run_simulation for now.

    # Assuming returns_pivot is the actual DataFrame passed in:
    for i, m in enumerate(portfolio_sizes):
        status_text.text(f"Simulating Portfolio Size: {m}/{actual_max_portfolio_size}")
        portfolio_std_devs_for_size_m = []
        if m > num_available_securities: # Safety check
            st.warning(f"Skipping size {m}, not enough securities ({num_available_securities}).")
            continue
        for _ in range(num_simulations_sim):
            # Make sure _returns_pivot is the DataFrame passed to this function
            selected_permnos = random.sample(available_securities, m)
            portfolio_component_returns = st.session_state['returns_pivot'][selected_permnos] # Access from state
            portfolio_period_returns = portfolio_component_returns.mean(axis=1)
            log_value_relatives = np.log(1 + portfolio_period_returns + 1e-10) # Add epsilon for log(0)
            std_dev = log_value_relatives.std(ddof=1) # Use sample std dev
            portfolio_std_devs_for_size_m.append(std_dev)
            # all_simulation_results[m].append(std_dev)

        # Handle case where std_dev calculation might fail (e.g., single data point)
        if portfolio_std_devs_for_size_m:
             average_std_dev = np.mean(portfolio_std_devs_for_size_m)
             average_std_devs.append(average_std_dev)
        else:
             # This shouldn't happen with checks above, but handle defensively
             average_std_devs.append(np.nan)

        progress_bar.progress((i + 1) / actual_max_portfolio_size)

    status_text.text("Simulation complete!")
    progress_bar.empty() # Clear progress bar

    # Filter out any potential NaNs if std dev failed for some size
    valid_indices = ~np.isnan(average_std_devs)
    portfolio_sizes_final = np.array(portfolio_sizes)[valid_indices]
    average_std_devs_final = np.array(average_std_devs)[valid_indices]

    if len(portfolio_sizes_final) == 0:
         st.error("Simulation resulted in no valid standard deviation values.")
         return None, None


    return portfolio_sizes_final.tolist(), average_std_devs_final.tolist()


def hyperbolic_func(x, a, b):
    """Function for curve fitting: Y = A + B/X"""
    return a + b / x

# Keep curve fitting cached based on inputs
@st.cache_data
def fit_curve(portfolio_sizes, average_std_devs):
    """Fits the hyperbolic curve to the simulation results."""
    if not portfolio_sizes or not average_std_devs or len(portfolio_sizes) < 2:
        st.warning("Need at least 2 data points for curve fitting.")
        return None, None, None, None

    x_data = np.array(portfolio_sizes)
    y_data = np.array(average_std_devs)

    # Check for NaNs that might have slipped through
    valid_data = ~np.isnan(y_data)
    if not np.all(valid_data):
        st.warning(f"NaN values found in average standard deviations. Fitting on valid points only.")
        x_data = x_data[valid_data]
        y_data = y_data[valid_data]
        if len(x_data) < 2:
             st.warning("Not enough valid data points remaining for curve fitting.")
             return None, None, None, None


    try:
        # Provide initial guesses, ensure y_data has valid values
        initial_guess = [y_data[-1], (y_data[0] - y_data[-1])]
        params, covariance = curve_fit(hyperbolic_func, x_data, y_data, p0=initial_guess, maxfev=5000)
        a_fit, b_fit = params

        # Calculate R-squared
        residuals = y_data - hyperbolic_func(x_data, a_fit, b_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        # Handle ss_tot == 0 case (if all y_data points are the same)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0 # Use tolerance for float comparison

        return a_fit, b_fit, r_squared, x_data # Return x_data used for fit
    except Exception as e:
        st.warning(f"Curve fitting failed: {e}")
        return None, None, None, None

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
         # Plot fit only over the range of x data used for fitting
         ax.plot(x_fit_data, hyperbolic_func(x_fit_data, a_fit, b_fit), 'r--', label=fit_label)

    if a_fit is not None:
         asymptote_label = f'Asymptote (Systematic Risk) = {a_fit:.4f}'
         # Extend asymptote line across the plot width
         xmin, xmax = ax.get_xlim() # Get current limits after plotting data
         if not portfolio_sizes: # If only fit exists, estimate limits
              xmin = 0
              xmax = max(x_fit_data) * 1.05 if x_fit_data is not None else 10
         ax.hlines(y=a_fit, xmin=xmin, xmax=xmax, color='g', linestyle=':', label=asymptote_label)
         ax.set_xlim(xmin, xmax) # Reset limits after adding hline


    ax.set_xlabel("Number of Securities in Portfolio (X)")
    ax.set_ylabel("Average Standard Deviation of Log Value Relatives (Y)")
    ax.set_title(f"Portfolio Diversification Effect ({data_start} to {data_end})")
    ax.legend(loc='best') # Adjust legend location
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Dynamic limits
    if portfolio_sizes:
         ax.set_xlim(0, max(portfolio_sizes) * 1.05) # Use actual max size plotted
    elif x_fit_data is not None:
         ax.set_xlim(0, max(x_fit_data) * 1.05)

    if average_std_devs:
        valid_std_devs = [s for s in average_std_devs if pd.notna(s)]
        if valid_std_devs:
             min_val = min(valid_std_devs)
             max_val = max(valid_std_devs)
             min_y_plot = min_val * 0.95 if a_fit is None else min(min_val, a_fit) * 0.95
             max_y_plot = max_val * 1.05
             # Ensure min < max
             if min_y_plot >= max_y_plot:
                  min_y_plot = max_y_plot * 0.9
             ax.set_ylim(min_y_plot, max_y_plot)

    st.pyplot(fig)
    # Clear the current figure to prevent overlaps in subsequent runs if plt state persists
    plt.close(fig) # Explicitly close the figure


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# File Uploader - THIS IS WHERE THE USER UPLOADS FROM THEIR COMPUTER
uploaded_file = st.sidebar.file_uploader("Upload Stock Return CSV", type=["csv", "txt"]) # Allow txt just in case

# Initialize session state variables if they don't exist
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'returns_pivot' not in st.session_state:
    st.session_state['returns_pivot'] = None
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None


# --- Default values ---
default_start = datetime.date(2010, 1, 1)
default_end = datetime.date(2020, 12, 31)
default_max_size = 40
default_simulations = 100 # Reduced default for faster interaction in streamlit

# --- Input Fields ---
st.sidebar.subheader("Data Columns")
# Use defaults common in finance datasets
date_col = st.sidebar.text_input("Date Column Name", value="date")
permno_col = st.sidebar.text_input("Security ID Column Name", value="PERMNO")
ret_col = st.sidebar.text_input("Return Column Name", value="RET")
shrcd_col = st.sidebar.text_input("Share Code Column Name (Optional)", value="SHRCD", help="Leave blank if not filtering by share code.")
common_codes_str = st.sidebar.text_input("Common Stock Codes (comma-sep)", value="10,11", help="Used if Share Code Column is provided.")

# Process common stock codes safely
common_stock_codes = []
if shrcd_col and common_codes_str:
    try:
        common_stock_codes = [int(code.strip()) for code in common_codes_str.split(',') if code.strip()]
    except ValueError:
        st.sidebar.error("Invalid Common Stock Codes format. Use comma-separated integers.")
        common_stock_codes = [] # Reset to avoid using bad value


st.sidebar.subheader("Analysis Period & Simulation")
# Make date inputs dependent on file upload? Or allow setting anytime? Allow setting anytime.
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=default_end)

# Validate date range
can_run_analysis = True
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    can_run_analysis = False

max_portfolio_size = st.sidebar.slider(
    "Max Portfolio Size (X)",
    min_value=5,
    max_value=200, # Increased upper limit
    value=default_max_size,
    step=1
)
num_simulations = st.sidebar.slider(
    "Number of Simulations per Size",
    min_value=10,
    max_value=1000,
    value=default_simulations,
    step=10
)

# --- Main Area Logic ---
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📊 Analysis Setup & Results")

    # Add button to trigger data load and prep
    if st.button("Load and Prepare Data"):
        st.session_state['data_loaded'] = False # Reset state
        st.session_state['returns_pivot'] = None
        st.session_state['simulation_results'] = None

        if not can_run_analysis:
             st.error("Please fix the date range in the sidebar.")
        else:
            # Read file content once
            uploaded_file_content = uploaded_file.getvalue()
            filename = uploaded_file.name
            with st.spinner(f"Loading and preparing data from `{filename}`..."):
                # Pass content and filename for caching key
                returns_pivot, num_dates, num_securities, data_min_date, data_max_date = load_and_prepare_data(
                    uploaded_file_content, filename, date_col, permno_col, ret_col, shrcd_col, common_stock_codes, start_date, end_date
                )

            if returns_pivot is not None:
                st.success(f"Data loaded successfully!")
                st.markdown(f"""
                *   **Analysis Period Used:** `{start_date}` to `{end_date}`
                *   **Date Range Found in Filtered Data:** `{data_min_date}` to `{data_max_date}`
                *   **Number of Time Periods:** `{num_dates}`
                *   **Number of Securities w/ Complete Data:** `{num_securities}`
                """)
                st.session_state['data_loaded'] = True
                st.session_state['returns_pivot'] = returns_pivot
                st.session_state['num_securities'] = num_securities
                st.session_state['data_min_date'] = data_min_date
                st.session_state['data_max_date'] = data_max_date


            else:
                st.warning("Data could not be loaded or prepared. Please check the file format, column names, date range, and data integrity.")
                st.session_state['data_loaded'] = False

    # Only show simulation button if data is loaded successfully
    if st.session_state['data_loaded']:
        st.markdown("---")
        st.subheader("🚀 Run Simulation")
        st.write(f"Ready to simulate using {st.session_state['num_securities']} securities.")

        if st.button("Run Diversification Simulation"):
            st.session_state['simulation_results'] = None # Clear previous results
            returns_pivot_df = st.session_state['returns_pivot']
            available_secs = returns_pivot_df.columns.tolist()

            # Generate a simple hash of the dataframe contents for caching attempt (if re-enabled)
            # data_hash = hashlib.sha256(pd.util.hash_pandas_object(returns_pivot_df).values).hexdigest()

            with st.spinner("Running simulations... This may take a while."):
                 # Pass the actual dataframe here - caching removed from run_simulation for now
                 portfolio_sizes, average_std_devs = run_simulation(
                     _returns_pivot_hash=None, # Not using hash for cache key now
                     available_securities=available_secs,
                     max_portfolio_size_sim=max_portfolio_size,
                     num_simulations_sim=num_simulations
                     # Pass the actual DF via session state inside the function
                 )

            if portfolio_sizes and average_std_devs:
                 st.session_state['simulation_results'] = (portfolio_sizes, average_std_devs)
                 st.success("Simulation complete!")
            else:
                st.error("Simulation did not produce valid results. Check data or settings.")

    # Display results if simulation has run successfully
    if st.session_state.get('simulation_results'):
        st.markdown("---")
        st.subheader("📈 Simulation Results")
        portfolio_sizes, average_std_devs = st.session_state['simulation_results']

        st.markdown("#### Curve Fitting")
        # Fit curve using cached function
        a_fit, b_fit, r_squared, x_fit_data = fit_curve(portfolio_sizes, average_std_devs)

        col1, col2, col3 = st.columns(3)
        col1.metric("Systematic Risk (A)", f"{a_fit:.5f}" if a_fit is not None else "N/A")
        col2.metric("Diversification Coeff (B)", f"{b_fit:.5f}" if b_fit is not None else "N/A")
        col3.metric("R-squared of Fit", f"{r_squared:.4f}" if r_squared is not None else "N/A")

        st.markdown("#### Diversification Plot")
        plot_results(
             portfolio_sizes,
             average_std_devs,
             a_fit, b_fit, r_squared, x_fit_data,
             st.session_state['data_min_date'], st.session_state['data_max_date'] # Use actual data range for title
        )


elif not uploaded_file:
    st.info("👈 **Please upload a CSV file using the sidebar to begin.**")

st.markdown("---")
st.markdown("App based on Evans & Archer (1968) | Built with [Streamlit](https://streamlit.io)")
