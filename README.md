# Airline Delay Analysis | Data Engineering Internship at Global Maximum

A data engineering project built to analyze airline delays and evaluate flight performance from multiple perspectives. Using the **Airline Delay Cause** dataset (398,233 rows), I designed a **Medallion Architecture** with Bronze, Silver, and Gold layers.

## Overview

I worked with **Python, SQL, Apache Spark, Apache Iceberg, and DuckDB** to ingest, process, transform, and model the raw data. I built fact and dimension tables to develop a data warehouse structure, and prepared Gold-layer reporting tables (PBL) that support analyses such as flight performance, delay causes, and airport and carrier comparisons.

Throughout the project, I gained hands-on experience in data quality, transformation pipelines, data modeling, and analytical reporting.

## Architecture

| Layer | Purpose |
|---|---|
| **Bronze** | Raw data ingested from the source dataset |
| **Silver** | Cleaned, validated, and transformed data |
| **Gold** | Fact/dimension tables and reporting tables for analytics |

## Synthetic Data

The original Kaggle dataset contains no financial or satisfaction data. To support financial performance and passenger satisfaction analyses, I generated two synthetic tables and linked them to the existing data model:

| File | Description | Key Columns |
|---|---|---|
| `FactFinance.csv` | Financial metrics per carrier, airport, and month | `Revenue`, `OperationCost`, `DelayCost` |
| `FactSatisfaction.csv` | Passenger volume and satisfaction scores per carrier, airport, and month | `PassengerCount`, `PassengerSatisfaction`, `BaggageSatisfaction`, `CabinCrewSatisfaction`, `ComfortSatisfaction`, `FoodBeverageSatisfaction` |

Both tables share the same keys as the source data (`year`, `month`, `carrier`, `carrier_name`, `airport`, `airport_name`), so they can be joined with the original dataset and with each other. They were generated with a Python script (`airline_synthetic_data.py`) using a fixed random seed (`42`) for reproducibility, and are derived from real columns such as flight counts, delay minutes, cancellations, and diversions.

> ⚠️ **Disclaimer:** These tables are intended for analysis practice and demonstration. They do **not** represent real airline financials or real passenger feedback, and relationships found in them (for example, between delay cost and satisfaction) reflect the assumptions built into the generation script.

## Technologies

Python, SQL, Apache Spark, Apache Iceberg, DuckDB, Pandas

## Key Concepts

Data Engineering, Medallion Architecture, ETL/ELT, Data Modeling, Data Warehouse, Synthetic Data Generation
