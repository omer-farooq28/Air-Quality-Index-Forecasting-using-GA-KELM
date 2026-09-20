# Air Quality Index Prediction using GA-KELM

This project focuses on predicting the **Air Quality Index (AQI)** from air-quality measurements using a **Genetic Algorithm-based Kernel Extreme Learning Machine (GA-KELM)**.

The notebook has been updated to run with **Python 3.11** and uses a cleaner, modern scikit-learn workflow. It also compares GA-KELM with **Support Vector Regression (SVR)** so the model performance can be viewed side by side.

## What the project does

The workflow in the notebook is straightforward:

1. Load the Ahmedabad air-quality dataset.
2. Handle missing values during preprocessing.
3. Select pollutant measurements as model inputs and AQI as the target.
4. Scale the data using MinMaxScaler.
5. Split the data into training and testing sets.
6. Train an SVR baseline.
7. Train the GA-KELM model.
8. Calculate MSE and RMSE on the normalized target values.
9. Compare both models using a results table and bar chart.

The main goal is to build and evaluate an AQI prediction model rather than simply classify air-quality categories.

## Model

### GA-KELM

GA-KELM combines two ideas:

- **Genetic Algorithm (GA):** searches for suitable model parameters.
- **Kernel Extreme Learning Machine (KELM):** uses a kernel-based approach to learn nonlinear relationships.

The Genetic Algorithm uses prediction error as its fitness measure and searches for parameters that improve the KELM model.

### SVR baseline

**Support Vector Regression (SVR)** is included as a baseline model. This gives the project a direct comparison between a standard regression approach and the proposed GA-KELM model.

## Input data

The dataset contains air-quality measurements collected for **Ahmedabad**.

The notebook works with pollutant-related features including:

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

The prediction target is:

- **AQI**

## Dataset

The repository includes:

- `Dataset/Dataset.csv` — dataset used by the notebook
- `Dataset/testData.csv` — test data available in the project

## Current results

The latest saved notebook output reports the following results on the **normalized target scale**:

| Algorithm | MSE | RMSE |
|---|---:|---:|
| SVR | 0.025847 | 0.160771 |
| GA-KELM | 0.019424 | 0.139371 |

These values are calculated before converting the target back to the original AQI scale. This keeps the evaluation consistent with the normalized values used during model training.

The notebook also includes a bar chart comparing the MSE and RMSE of SVR and GA-KELM.

## Project structure

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

## Technologies used

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Genetic Algorithm
- Kernel Extreme Learning Machine

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/omer-farooq28/Air-Quality-Index-using--GA-KELM.git
cd Air-Quality-Index-using--GA-KELM
```

### 2. Open the project folder

```bash
cd AIR-QUALITY
```

### 3. Create a Python 3.11 virtual environment

```bash
py -3.11 -m venv venv
```

On Windows, activate it with:

```bash
venv\Scripts\activate
```

### 4. Install the dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 5. Start Jupyter

```jupyter notebook```

Open:

```text
AirQuality.ipynb
```

Run the cells in order so that the preprocessing, SVR model, GA-KELM model, metrics, and comparison chart are generated correctly.

## Evaluation metrics

The notebook uses two regression metrics:

**Mean Squared Error (MSE)**

MSE measures the average squared difference between the actual and predicted AQI values.

**Root Mean Squared Error (RMSE)**

RMSE is the square root of MSE and is easier to interpret because it remains on the same scale as the evaluated target values.

In the current notebook, both metrics are calculated on the normalized target values.

## Key files

| File | Purpose |
|---|---|
| `AirQuality.ipynb` | Main notebook containing preprocessing, training, evaluation, and comparison |
| `AirQuality.html` | HTML version of the notebook |
| `GAKELM.py` | GA-KELM implementation |
| `Dataset/Dataset.csv` | Main dataset |
| `Dataset/testData.csv` | Test data |
| `requirements.txt` | Python dependencies |

## Future improvements

Some useful next steps for the project are:

- Tune the GA-KELM search further.
- Experiment with feature selection and preprocessing.
- Evaluate additional regression models.
- Report both normalized metrics and AQI-scale metrics when presenting model performance.
- Build a simple interface where a user can enter pollutant values and receive an AQI prediction.
- Deploy the trained model as a small web application or API.

## Author

**Mohammed Omer Farooq**

GitHub: https://github.com/omer-farooq28

## License

This project was developed for academic and educational purposes.
