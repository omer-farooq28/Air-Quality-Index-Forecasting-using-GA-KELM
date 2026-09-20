# Air Quality Index Prediction using GA-KELM

A machine learning project for **Air Quality Index (AQI) prediction** using a **Genetic Algorithm-based Kernel Extreme Learning Machine (GA-KELM)**.

The system uses air pollutant measurements to learn patterns in air quality and predict AQI values. The repository includes the dataset, trained model files, Jupyter Notebook, and GA-KELM implementation.

## Project Overview

This project applies GA-KELM to model the relationship between pollutant concentrations and AQI.

### Workflow

1. Load the air quality dataset.
2. Handle missing values and preprocess the data.
3. Normalize the input features.
4. Split the data into training and testing sets.
5. Train the GA-KELM model.
6. Optimize model parameters using a Genetic Algorithm.
7. Evaluate prediction performance.
8. Generate AQI predictions for test data.

## Model

### GA-KELM

GA-KELM combines:

- **Genetic Algorithm (GA)** for optimization
- **Kernel Extreme Learning Machine (KELM)** for nonlinear regression

The Genetic Algorithm searches for suitable model parameters using prediction error as the fitness criterion, while KELM uses a kernel function to model nonlinear relationships between air pollutant measurements and AQI.

## Input Features

The project works with major air-quality parameters:

- SO₂
- NO₂
- PM2.5
- PM10
- CO
- O₃

The target variable is **Air Quality Index (AQI)**.

## Dataset

The repository contains:

- `Dataset/Dataset.csv` — main dataset used for model development
- `Dataset/testData.csv` — test data used for prediction and evaluation

## Project Structure

```text
Air-Quality-Index-using--GA-KELM/
│
├── AIR-QUALITY/
│   ├── Dataset/
│   │   ├── Dataset.csv
│   │   └── testData.csv
│   ├── model/
│   │   ├── elm.npy
│   │   └── extension_weights.hdf5
│   ├── AirQuality.ipynb
│   ├── AirQuality.html
│   ├── GAKELM.py
│   ├── requirements.txt
│   └── .gitignore
│
└── README.md
```

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Keras
- TensorFlow
- Jupyter Notebook
- Genetic Algorithm
- Kernel Extreme Learning Machine

## Installation

The repository contains the original project environment with legacy package versions.

### 1. Clone the repository

```bash
git clone https://github.com/omer-farooq28/Air-Quality-Index-using--GA-KELM.git
cd Air-Quality-Index-using--GA-KELM
```

### 2. Open the project directory

```bash
cd AIR-QUALITY
```

### 3. Create a virtual environment

The original project was developed using Python 3.7.

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: The pinned dependencies in `requirements.txt` are from the original project environment and include older versions of TensorFlow, Keras, NumPy, Pandas, and Scikit-learn. A modern Python environment may require updated dependencies.

## Running the Notebook

Start Jupyter Notebook from the `AIR-QUALITY` directory:

```bash
jupyter notebook
```

Then open:

```text
AirQuality.ipynb
```

Run the notebook cells sequentially to reproduce preprocessing, model training, evaluation, and prediction.

## Evaluation

The project can evaluate regression performance using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

RMSE measures the average magnitude of prediction errors and is useful for evaluating AQI prediction performance.

## Key Files

| File | Description |
|---|---|
| `AirQuality.ipynb` | Main Jupyter Notebook containing the project workflow |
| `AirQuality.html` | HTML export of the notebook |
| `GAKELM.py` | GA-KELM implementation |
| `Dataset/Dataset.csv` | Main dataset |
| `Dataset/testData.csv` | Test dataset |
| `model/elm.npy` | Saved model-related parameters |
| `model/extension_weights.hdf5` | Saved model weights |
| `requirements.txt` | Python package dependencies |

## Future Enhancements

- Improve GA-KELM hyperparameter optimization.
- Reduce prediction error through improved preprocessing and feature selection.
- Compare GA-KELM with additional machine learning and deep learning models.
- Develop a web-based interface for AQI prediction.
- Deploy the trained model as an API.

## Author

**Mohammed Omer Farooq**

GitHub: https://github.com/omer-farooq28

## License

This project is intended for academic and educational purposes.
