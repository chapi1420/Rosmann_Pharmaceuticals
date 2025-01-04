# Rossmann Sales Prediction Project

## Overview

This project aims to forecast sales for Rossmann Pharmaceuticals stores across various cities six weeks ahead of time. The finance team relies on accurate predictions to make informed decisions regarding promotions and inventory management. The project encompasses data exploration, machine learning model building, deep learning techniques, and deploying a REST API for real-time predictions.

## Table of Contents

- [Business Need](#business-need)
- [Data and Features](#data-and-features)
- [Learning Outcomes](#learning-outcomes)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task 1: Exploration of Customer Purchasing Behavior](#task-1-exploration-of-customer-purchasing-behavior)
  - [Task 2: Prediction of Store Sales](#task-2-prediction-of-store-sales)
  - [Task 3: Model Serving API Call](#task-3-model-serving-api-call)
- [Key Dates](#key-dates)
- [References](#references)

## Business Need

The finance team at Rossmann Pharmaceuticals needs a reliable sales forecasting model to enhance decision-making processes related to promotions, inventory, and staffing across multiple store locations.

## Data and Features

The dataset used for this project is sourced from Kaggle's [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition. Key features include:

- **Store**: Unique ID of each store
- **Sales**: Daily turnover (target variable)
- **Customers**: Number of customers on a given day
- **Open**: Indicator if the store was open
- **StateHoliday**: State holiday indicator
- **SchoolHoliday**: School holiday indicator
- **StoreType**: Differentiates between store models
- **Assortment**: Describes assortment levels
- **CompetitionDistance**: Distance to the nearest competitor
- **Promo**: Indicates if a store is running a promotion

## Learning Outcomes

The project focuses on developing competencies in:

- Advanced use of scikit-learn
- Feature engineering
- ML model building and fine-tuning
- CI/CD deployment of ML models
- Building dashboards and model management

## Project Structure

```plaintext
- data/
  - raw/                   # Raw data files
  - processed/            # Processed data files
- notebooks/              # Jupyter notebooks for exploration and modeling
- src/                    # Source code for data processing, modeling, and API
  - EDA.py  # Data cleaning and preprocessing scripts
  - model_training.py      # Scripts for training machine learning models
  - api.py                 # REST API implementation
- requirements.txt         # Python package dependencies
- README.md                # Project documentation
