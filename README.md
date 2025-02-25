# AFP Project

This repository contains code for analyzing and backtesting financial strategies, particularly focused on options trading. The project is structured into several Python scripts and Jupyter notebooks for data cleaning, backtesting, signal generation, and combining return streams.

## Table of Contents
1. [Overview](#overview)
2. [File Descriptions](#file-descriptions)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Contributors](#contributors)

---

## Overview

The goal of this project is to develop and backtest options trading strategies using historical data. The repository includes tools for:
- Cleaning and downloading financial data.
- Backtesting strategies with customizable parameters.
- Generating trading signals based on different methodologies.
- Combining return streams from multiple strategies.

---

## File Descriptions

### Data Preparation
- **`1_clean_download_data.ipynb`**: Jupyter notebook for downloading and cleaning raw financial data to prepare it for analysis.

### Backtesting
- **`2_backtest.ipynb`**: Jupyter notebook for running backtests on trading strategies.
- **`2_backtester.py`**: Python script containing reusable functions for backtesting strategies.
- **`2_bt_strategy_v2.py`**: Python script defining specific backtesting strategies.

### Constants and Data Loading
- **`2_consts.py`**: A file defining constants used across the project (e.g., API keys, file paths).
- **`2_data_loading.py`**: Script for loading and preprocessing data into the required format.

### Signal Generation
- **`2_options_signals_leo.py`**: Signal generation logic based on Leo's methodology.
- **`2_options_signals_michelle.py`**: Signal generation logic based on Michelle's methodology.

### Options Trading Code
- **`3_options_trading_code.ipynb`**: Jupyter notebook containing code for executing options trading strategies.

### Return Streams
- **`4_combine_returnstreams.ipynb`**: Notebook for combining return streams from multiple strategies into a unified performance report.

### Documentation
- **`README.md`**: This file provides an overview of the project structure and usage instructions.

---

## Setup Instructions

1. Clone the repository:
