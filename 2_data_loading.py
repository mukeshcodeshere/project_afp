import pandas as pd
import sqlite3
import gc

from consts import (DB_PATH, START_DATE, END_DATE)

def load_distinct_tickers_from_db():
    """Load distinct tickers from the database within the specified date range."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DISTINCT ticker
        FROM merged_data
        WHERE date BETWEEN ? AND ?
    """
    distinct_tickers = pd.read_sql(query, conn, params=(START_DATE, END_DATE))
    conn.close()
    return distinct_tickers


def load_merged_data_from_db(tickers_list, chunk_size=100000):
    """Load data from the database within the specified date range, excluding tickers with no data."""
    conn = sqlite3.connect(DB_PATH)

    # Convert tickers_list into a format suitable for the SQL IN clause (comma-separated string)
    tickers_tuple = tuple(tickers_list)

    # Make sure tickers_tuple is not empty to prevent SQL errors
    if not tickers_tuple:
        return pd.DataFrame()  # Return an empty DataFrame if no tickers are provided

    query = """
        SELECT *
        FROM merged_data
        WHERE date BETWEEN ? AND ?
        AND ticker IN ({})
    """.format(','.join(['?'] * len(tickers_tuple)))  # Dynamically insert placeholders for each ticker

    # Initialize an empty DataFrame to hold the data
    merged_data = pd.DataFrame()

    # Run the query with the tickers_list as parameters
    params = (START_DATE, END_DATE) + tickers_tuple

    # Use SQLite's `fetchmany` to load data in chunks to minimize memory usage
    cursor = conn.cursor()
    cursor.execute(query, params)

    while True:
        # Fetch a chunk of rows
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break  # Stop if no more rows are returned

        # Convert the chunk of rows to a DataFrame
        chunk_df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

        # Append the chunk to the final DataFrame
        merged_data = pd.concat([merged_data, chunk_df], ignore_index=True)

        del chunk_df

    conn.close()

    return merged_data


def load_equities_data_from_db(tickers_list, chunk_size=100000):
    """Load data from the database within the specified date range, excluding tickers with no data."""
    conn = sqlite3.connect(DB_PATH)

    # Convert tickers_list into a format suitable for the SQL IN clause (comma-separated string)
    tickers_tuple = tuple(tickers_list)

    # Make sure tickers_tuple is not empty to prevent SQL errors
    if not tickers_tuple:
        return pd.DataFrame()  # Return an empty DataFrame if no tickers are provided

    query = """
        SELECT *
        FROM equities_data
        WHERE date BETWEEN ? AND ?
    """.format(','.join(['?'] * len(tickers_tuple)))  # Dynamically insert placeholders for each ticker

    # Initialize an empty DataFrame to hold the data
    equities_data = pd.DataFrame()

    # Run the query with the tickers_list as parameters
    params = (START_DATE, END_DATE)

    # Use SQLite's `fetchmany` to load data in chunks to minimize memory usage
    cursor = conn.cursor()
    cursor.execute(query, params)

    while True:
        # Fetch a chunk of rows
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break  # Stop if no more rows are returned

        # Convert the chunk of rows to a DataFrame
        chunk_df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

        # Append the chunk to the final DataFrame
        equities_data = pd.concat([equities_data, chunk_df], ignore_index=True)

        del chunk_df

    conn.close()

    return equities_data


def load_options_data_from_db(tickers_list, chunk_size=100000):
    """Load data from the database within the specified date range, excluding tickers with no data."""
    conn = sqlite3.connect(DB_PATH)

    # Convert tickers_list into a format suitable for the SQL IN clause (comma-separated string)
    tickers_tuple = tuple(tickers_list)

    # Make sure tickers_tuple is not empty to prevent SQL errors
    if not tickers_tuple:
        return pd.DataFrame()  # Return an empty DataFrame if no tickers are provided

    query = """
        SELECT *
        FROM volume_threshold_options_data
        WHERE date BETWEEN ? AND ?
    """.format(','.join(['?'] * len(tickers_tuple)))  # Dynamically insert placeholders for each ticker

    # Initialize an empty DataFrame to hold the data
    options_data = pd.DataFrame()

    # Run the query with the tickers_list as parameters
    params = (START_DATE, END_DATE)

    # Use SQLite's `fetchmany` to load data in chunks to minimize memory usage
    cursor = conn.cursor()
    cursor.execute(query, params)

    while True:
        # Fetch a chunk of rows
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break  # Stop if no more rows are returned

        # Convert the chunk of rows to a DataFrame
        chunk_df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

        # Append the chunk to the final DataFrame
        options_data = pd.concat([options_data, chunk_df], ignore_index=True)

        del chunk_df

    conn.close()

    return options_data



