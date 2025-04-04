import warnings
from datetime import datetime

import pandas as pd
import pandas_datareader as pdr


def get_fred_data(series_id, start_year=1950):
    """
    Get data from FRED for a specific series.

    Args:
        series_id: FRED series ID
        start_year: First year to include in the data

    Returns:
        pandas.DataFrame: Data from FRED
    """
    try:
        # Download data
        end_date = datetime.now()
        start_date = f"{start_year}-01-01"

        print(
            f"Downloading data from FRED (series: {series_id}) from {start_date} to {end_date.strftime('%Y-%m-%d')}..."
        )
        data = pdr.fred.FredReader(series_id, start=start_date, end=end_date).read()

        return data
    except Exception as e:
        print(f"Error downloading data for {series_id}: {e}")
        return pd.DataFrame()


def check_data_series(equity_series_id, bond_series_id, start_year=1950):
    """
    Check the availability of data for FRED series.

    Args:
        equity_series_id: FRED series ID for equity data
        bond_series_id: FRED series ID for bond data
        start_year: First year to include in the data

    Returns:
        dict: Information about available years
    """
    # Suppress warnings during this check
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Get equity data
        equity_data = get_fred_data(equity_series_id, start_year)

        # Get bond data
        bond_data = get_fred_data(bond_series_id, start_year)

        # Calculate annual returns for equity data
        if not equity_data.empty:
            annual_equity_data = equity_data.resample("YE").last()
            annual_equity_returns = annual_equity_data.pct_change().dropna()
            equity_years = set(annual_equity_returns.index.year)
        else:
            equity_years = set()

        # Calculate annual returns for bond data
        if not bond_data.empty:
            annual_bond_data = bond_data.resample("YE").last()
            annual_bond_returns = annual_bond_data.pct_change().dropna()
            bond_years = set(annual_bond_returns.index.year)
        else:
            bond_years = set()

        # Find common years
        common_years = sorted(equity_years.intersection(bond_years))

        # Check if we have enough data for bootstrapping
        bootstrap_period_length = 10
        valid_start_years = (
            common_years[: -bootstrap_period_length + 1]
            if len(common_years) >= bootstrap_period_length
            else []
        )

        return {
            "equity_series_id": equity_series_id,
            "bond_series_id": bond_series_id,
            "equity_years": sorted(equity_years) if equity_years else [],
            "bond_years": sorted(bond_years) if bond_years else [],
            "common_years": common_years,
            "equity_start": min(equity_years) if equity_years else None,
            "equity_end": max(equity_years) if equity_years else None,
            "bond_start": min(bond_years) if bond_years else None,
            "bond_end": max(bond_years) if bond_years else None,
            "common_start": min(common_years) if common_years else None,
            "common_end": max(common_years) if common_years else None,
            "common_count": len(common_years),
            "valid_start_years": valid_start_years,
            "valid_start_years_count": len(valid_start_years),
            "can_bootstrap_10_years": len(valid_start_years) > 0,
        }


def print_results(results):
    """
    Print the results of the check.

    Args:
        results: dict with the results
    """
    print(
        f"\nResults for {results['equity_series_id']} (equity) and {results['bond_series_id']} (bonds):"
    )
    print(
        f"Equity years: {results['equity_start']}-{results['equity_end']} ({len(results['equity_years'])} years)"
    )
    print(
        f"Bond years: {results['bond_start']}-{results['bond_end']} ({len(results['bond_years'])} years)"
    )
    print(
        f"Common years: {results['common_start']}-{results['common_end']} ({results['common_count']} years)"
    )
    print(
        f"Valid start years for 10-year bootstrapping: {results['valid_start_years_count']}"
    )
    print(f"Can bootstrap 10 years: {results['can_bootstrap_10_years']}")
    if results["valid_start_years"]:
        print(f"Valid start years: {results['valid_start_years']}")
    else:
        print("No valid start years for 10-year bootstrapping.")


def check_multiple_combinations():
    """
    Check multiple combinations of equity and bond series.
    """
    # List of series to check
    equity_series = ["SP500", "DJIA", "NASDAQCOM", "WILL5000PR", "WILL5000IND", "SPXTR"]
    bond_series = ["GS10", "AAA", "BAA", "DGS10", "BAMLCC0A0CMTRIV"]

    # Check each combination
    results = []
    for equity_id in equity_series:
        for bond_id in bond_series:
            result = check_data_series(equity_id, bond_id, start_year=1950)
            results.append(result)
            print_results(result)

    # Find the best combination
    best_result = max(results, key=lambda x: x["valid_start_years_count"])

    print("\n\nBest combination for 10-year bootstrapping:")
    print_results(best_result)

    return best_result


if __name__ == "__main__":
    # Check the current configuration
    from params import Params

    print("Checking current configuration...")
    current_result = check_data_series(
        Params.equity_series_id,
        Params.bond_series_id,
        start_year=Params.data_start_year,
    )
    print_results(current_result)

    # Check if we can find a better combination
    print("\nChecking for better combinations...")
    best_result = check_multiple_combinations()

    # Recommend changes if needed
    if (
        not current_result["can_bootstrap_10_years"]
        and best_result["can_bootstrap_10_years"]
    ):
        print("\nRecommended changes to params.py:")
        print(f'equity_series_id = "{best_result["equity_series_id"]}"')
        print(f'bond_series_id = "{best_result["bond_series_id"]}"')
        print(f"data_start_year = {best_result['common_start']}")
    elif (
        not current_result["can_bootstrap_10_years"]
        and not best_result["can_bootstrap_10_years"]
    ):
        print("\nNo combination found that allows for 10-year bootstrapping.")
        print("Consider reducing bootstrap_period_length in params.py.")
    else:
        print("\nCurrent configuration is already optimal for 10-year bootstrapping.")
