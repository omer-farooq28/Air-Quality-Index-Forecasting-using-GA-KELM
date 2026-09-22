# Air Quality Index Prediction using GA-KELM

A machine learning project for **Air Quality Index (AQI) prediction** using a **Genetic Algorithm-based Kernel Extreme Learning Machine (GA-KELM)**. The project uses Ahmedabad air-quality data and evaluates GA-KELM against **Support Vector Regression (SVR)** as a baseline.

This repository is developed as an **academic project** demonstrating data preprocessing, regression modelling, optimization, and model evaluation for AQI prediction.

## Project Overview

The project investigates a GA-KELM-based regression approach for predicting AQI from pollutant measurements.

### Workflow

1. Load the Ahmedabad air-quality dataset.
2. Handle missing values during preprocessing.
3. Select pollutant measurements as input features and AQI as the target.
4. Normalize the input and target values using `MinMaxScaler`.
5. Split the data into training and testing sets.
6. Train SVR as the baseline model.
7. Train the GA-KELM model.
8. Calculate MSE and RMSE.
9. Compare the model results using tables and visualizations.

## Models

### GA-KELM

GA-KELM combines a **Kernel Extreme Learning Machine (KELM)** with a **Genetic Algorithm (GA)**. The Genetic Algorithm searches for suitable model parameters, while KELM is used for nonlinear regression between pollutant measurements and AQI.

### SVR

**Support Vector Regression (SVR)** is used as the baseline regression model for comparison.

## Dataset

The project uses an **Ahmedabad air-quality dataset**.

### Input Features

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

### Target

- **AQI**

## Results

The current notebook reports the following results on the **normalized AQI target scale**:

| Algorithm | MSE | RMSE |
|---|---:|---:|
| SVR | 0.025847 | 0.160771 |
| GA-KELM | 0.019424 | 0.139371 |

Because the target was normalized before evaluation, these values represent **normalized-scale errors** and should not be interpreted directly as AQI-point errors.

## Project Structure

~~~text
Air-Quality-Index-using--GA-KELM/
│
├── dataset/
│   ├── Dataset.csv
│   └── testData.csv
│
├── models/
│   ├── elm.npy
│   └── extension_weights.hdf5
│
├── notebooks/
│   └── AirQuality.ipynb
│
├── src/
│   └── GAKELM.py
│
├── .gitignore
├── requirements.txt
└── README.md
~~~

## Technologies Used

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Genetic Algorithm
- Kernel Extreme Learning Machine
- Support Vector Regression

## Installation

### 1. Clone the repository

~~~bash
git clone https://github.com/omer-farooq28/Air-Quality-Index-using--GA-KELM.git
cd Air-Quality-Index-using--GA-KELM
~~~

### 2. Create a Python 3.11 virtual environment

~~~bash
py -3.11 -m venv venv
~~~

Activate the environment on Windows:

~~~bash
venv\Scripts\activate
~~~

### 3. Install dependencies

~~~bash
py -3.11 -m pip install -r requirements.txt
~~~

### 4. Launch Jupyter Notebook

~~~bash
jupyter notebook
~~~

Open:

~~~text
notebooks/AirQuality.ipynb
~~~

Run the notebook cells in sequence to reproduce the preprocessing, training, evaluation, and comparison workflow.

## Evaluation Metrics

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted target values.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and expresses prediction error on the same scale as the evaluated target values.

In the current implementation, both metrics are calculated using the normalized AQI target.

## Important Files

| File | Description |
|---|---|
| `dataset/Dataset.csv` | Main Ahmedabad air-quality dataset |
| `dataset/testData.csv` | Test dataset |
| `notebooks/AirQuality.ipynb` | Main experimentation and evaluation notebook |
| `src/GAKELM.py` | GA-KELM implementation |
| `models/` | Stored model-related files |
| `requirements.txt` | Python dependencies |

## Future Improvements

Possible extensions include:

- Further Genetic Algorithm tuning
- Feature-selection experiments
- Evaluation on the original AQI scale
- Comparison with additional regression and deep-learning models
- Improved model validation strategies
- Development of an AQI prediction web application
- Deployment of the trained model through an API

## Academic Project

This repository is developed as an **academic machine learning project** focused on **AQI prediction using GA-KELM**.

It demonstrates the practical application of data preprocessing, regression modelling, optimization techniques, and performance evaluation to an environmental data prediction problem.

## Author

**Mohammed Omer Farooq**

GitHub: [omer-farooq28](https://github.com/omer-farooq28)

## License

This project is intended for academic and educational purposes.
