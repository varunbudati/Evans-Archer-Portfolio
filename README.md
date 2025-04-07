# Portfolio Diversification Analysis (Evans & Archer Replication) 📊

This Streamlit application provides an interactive way to replicate and visualize the core empirical analysis presented in the paper:

**Evans, John L., and Stephen H. Archer. "Diversification and the Reduction of Dispersion: An Empirical Analysis." *The Journal of Finance*, vol. 23, no. 5, 1968, pp. 761–67.**

The app demonstrates how portfolio risk (measured by the standard deviation of returns) decreases as the number of randomly selected securities in the portfolio increases. It fits a hyperbolic curve to the simulation results to estimate the level of non-diversifiable systematic risk.

## Features ✨

*   **Interactive Data Upload:** Upload your own historical stock return data in CSV format directly through the app.
*   **Configurable Parameters:** Adjust settings via sidebar widgets:
    *   Specify column names for Date, Security ID, Return, and optional Share Code.
    *   Select the analysis start and end dates.
    *   Control the maximum portfolio size to simulate.
    *   Set the number of random portfolio simulations per size.
    *   Define common stock share codes for filtering (if using Share Code).
*   **Simulation Engine:** Randomly selects securities, calculates equally-weighted portfolio returns, and computes the standard deviation of log value relatives (`log(1 + Return)`) as per the original paper's methodology.
*   **Curve Fitting:** Fits the hyperbolic function `Y = A + B/X` to estimate systematic risk (Asymptote `A`) and the diversification coefficient (`B`).
*   **Visualization:** Generates a plot showing the relationship between portfolio size (X-axis) and average portfolio standard deviation (Y-axis), including the fitted curve and the systematic risk asymptote.
*   **Results Summary:** Displays key metrics like the number of securities used, time periods, fitted parameters (A, B), and the R-squared value of the fit.

## Requirements 📋

*   Python 3.x
*   Libraries listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Setup ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-directory>
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional) Prepare Example Data:** If example data is provided *within this repository* (e.g., in a `.zip` file), **you must extract the CSV file from the archive first.** The Streamlit app requires you to upload the `.csv` file itself, not the zip archive.

## Data Requirements 📄

The application requires a CSV file containing historical return data with the following structure:

1.  **Date Column:** A column containing the date for each return observation (e.g., `date`). The script will attempt to parse various date formats.
2.  **Security ID Column:** A column with a unique identifier for each security (e.g., `PERMNO`, `Ticker`, `CUSIP`).
3.  **Return Column:** A column with the periodic return for the security (e.g., `RET`). Total return (including dividends) is preferred. Ensure returns are in decimal format (e.g., 0.05 for 5%).

**Optional:**

4.  **Share Code Column:** A column identifying the type of security (e.g., `SHRCD`). If provided in the sidebar configuration, the app can filter data to include only specified codes (default: `10, 11` for CRSP common stock). Leave the 'Share Code Column Name' field blank in the app sidebar if you don't want to use this filter or if your data lacks this column.

**Important Notes on Data:**

*   **Completeness:** The analysis methodology (specifically the `pivot_table` and `dropna(axis=1)`) requires that **each security included in the final simulation must have a non-missing return value for *every single period* within the selected Start and End Date range.** Securities with any missing data points during the period will be excluded.
*   **File Format:** Ensure the file is a standard CSV. The app attempts to read UTF-8 and latin1 encodings.
*   **GitHub Data:** If you cloned this repository and it included zipped sample data, remember to **unzip it first** to get the `.csv` file before trying to upload it to the running Streamlit application.

## Usage Instructions 🚀

1.  **Run the Streamlit App:** Open your terminal, navigate to the repository directory, and run:
    ```bash
    streamlit run streamlit_diversification.py
    ```
    Your web browser should open with the application.
2.  **Upload Data:** Use the "Upload Stock Return CSV" button in the sidebar to upload your **extracted** `.csv` data file.
3.  **Configure Parameters:** Adjust the settings in the sidebar:
    *   Verify/correct the column names to match your CSV.
    *   Select the desired Start and End Dates.
    *   Use the sliders to set the "Max Portfolio Size" and "Number of Simulations".
4.  **Load Data:** Click the **"Load and Prepare Data"** button. Review the summary statistics displayed (number of securities, periods) to ensure the data was processed as expected.
5.  **Run Simulation:** If data loading was successful, click the **"Run Diversification Simulation"** button. This step might take some time depending on the configuration.
6.  **View Results:** Once the simulation completes, the app will display:
    *   Fitted curve parameters (Systematic Risk A, Coefficient B, R-squared).
    *   The diversification plot visualizing the results.

## Output 📈

*   **Summary Statistics:** Information about the data used after filtering (date range, number of periods, number of eligible securities).
*   **Fitted Parameters:** Estimates for systematic risk (`A`), the diversification coefficient (`B`), and the goodness-of-fit (`R-squared`).
*   **Plot:** A graph showing:
    *   Average portfolio standard deviation (Y-axis) vs. the number of securities (X-axis).
    *   The fitted hyperbolic curve.
    *   A horizontal line representing the estimated systematic risk level (Asymptote `A`).

## Underlying Concepts 🧠

*   **Diversification:** Reducing risk by investing in a variety of assets.
*   **Systematic Risk (Market Risk):** Non-diversifiable risk inherent to the overall market.
*   **Unsystematic Risk (Specific Risk):** Diversifiable risk associated with individual assets.
*   **Log Value Relatives:** The standard deviation is calculated on `log(1 + Return)`, a common practice in finance for analyzing compounded returns and stabilizing variance.

## Limitations ⚠️

*   Results are sensitive to the input data quality, date range, and chosen parameters.
*   Assumes equal weighting of securities within each simulated portfolio.
*   Random sampling introduces slight variations in results between runs.
*   Historical data is not necessarily indicative of future performance.
*   The `dropna(axis=1)` step imposes a strict requirement for complete data history, potentially excluding many securities.

## Citation 🙏

Please cite the original work when referencing this analysis:

Evans, John L., and Stephen H. Archer. "Diversification and the Reduction of Dispersion: An Empirical Analysis." *The Journal of Finance*, vol. 23, no. 5, 1968, pp. 761–67. JSTOR, https://doi.org/10.2307/2325905.

## Deployment ☁️

This application is built with Streamlit and can be easily deployed using services like Streamlit Community Cloud by connecting to your GitHub repository.