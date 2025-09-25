# Intraday Trading System

This repository contains a prototype implementation of the intraday trading system
described in the accompanying research report.  The goal of this project is to
provide a reproducible, modular pipeline that takes raw minute‐bar market data,
generates a rich set of features, labels the data using the triple barrier
method, trains predictive models and evaluates their performance via a walk
forward backtest.  The system is designed to operate on a commodity laptop or
desktop and makes use of freely available data sources such as Yahoo Finance
(via the `yfinance` Python package) and Alpha Vantage.

## Project layout

```
trading_app/
├── README.md          – This file
├── config.py          – Global configuration and constants
├── data_pipeline.py   – Fetches, stores and loads intraday market data
├── features.py        – Computes technical and novel features from raw data
├── labeling.py        – Implements the triple barrier labeling scheme
├── model.py           – Training routines for logistic regression, XGBoost and
│                        walk‑forward prediction with probability metrics
├── backtest.py        – Simplified trade‑level backtesting engine using the triple
│                        barrier for exits
├── evaluation.py      – Utility functions to compute performance metrics such
│                        as cumulative return, Sharpe ratio and max drawdown
├── run.py             – Example script tying everything together
└── requirements.txt   – Python dependencies
```

## Getting started

1. Install the required packages (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Update the values in `config.py` to specify the tickers you wish to
   investigate, your Alpha Vantage API key (optional) and various strategy
   parameters.

3. Run the demonstration script to download data, compute features, label
   observations, train a model and run a simple backtest:

   ```bash
   python run.py
   ```

   The script will print a summary of the model’s classification performance
   during cross‑validation as well as backtest metrics such as CAGR, Sharpe
   ratio and maximum drawdown.

## Disclaimer

This code is provided for educational purposes only and is not meant to be
trading advice.  The research report accompanying this code discusses the
limitations of backtesting and the importance of proper risk management.  Past
performance is not indicative of future results.  Use this software at your
own risk.

## Google Drive Sync (optional)

You can upload/sync the project to a Google Drive folder via the Drive API using a service account.

1. Create a service account and download its JSON key.
2. In Google Drive, create a destination folder and share it with the service account email.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set credentials and run the sync:
   - PowerShell:
     ```powershell
     $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service_account.json"
     cd trading_app
     python gdrive_sync.py --src . --dest-folder-name trading_app
     ```
   - Or use an existing folder ID:
     ```powershell
     python gdrive_sync.py --src . --dest-folder-id <FOLDER_ID>
     ```

Re-run the script to push local edits; it updates existing files in Drive.