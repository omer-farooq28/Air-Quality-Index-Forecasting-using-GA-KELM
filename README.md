# Air Quality Index Prediction using GA-KELM

This project is about predicting the **Air Quality Index (AQI)** from air-quality measurements using a **Genetic Algorithm-based Kernel Extreme Learning Machine (GA-KELM)**.

The project uses an Ahmedabad air-quality dataset and compares GA-KELM with **Support Vector Regression (SVR)**. The notebook is set up for **Python 3.11** and keeps the workflow simple: prepare the data, train the models, measure their errors, and compare the results.

## Project workflow

The notebook follows these steps:

1. Load the Ahmedabad air-quality dataset.
2. Handle missing values during preprocessing.
3. Select pollutant measurements as input features and AQI as the target.
4. Normalize the input and target values using MinMaxScaler.
5. Split the data into training and testing sets.
6. Train SVR as the baseline model.
7. Train the GA-KELM model.
8. Calculate MSE and RMSE.
9. Compare the model results using a table and chart.

## Models

### GA-KELM

GA-KELM combines a Kernel Extreme Learning Machine with a Genetic Algorithm. The Genetic Algorithm searches for suitable model parameters, while the kernel-based ELM is used to learn the relationship between the pollutant measurements and AQI.

### SVR

Support Vector Regression is used as the baseline model. Including SVR makes it easier to compare the proposed GA-KELM approach with a commonly used regression method.

## Dataset

The project uses air-quality data for **Ahmedabad**.

The main input features include:

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

The prediction target is **AQI**.

## Results

The current notebook reports the following results on the **normalized target scale**:

| Algorithm | MSE | RMSE |
|---|---:|---:|
| SVR | 0.025847 | 0.160771 |
| GA-KELM | 0.019424 | 0.139371 |

The notebook also contains a bar chart that makes the MSE and RMSE comparison easier to view.

These values are calculated on normalized AQI values. They should therefore be interpreted as normalized-scale errors, not as direct AQI-point errors.

## Repository structure

```text
Air-Quality-Index-using--GA-KELM/
│
├── AIR-QUALITY/
│   ├── data/
│   │   ├── Dataset.csv
│   │   └── testData.csv
│   │
│   ├── models/
│   │   ├── elm.npy
│   │   └── extension_weights.hdf5
│   │
│   ├── notebooks/
│   │   ├── AirQuality.ipynb
│   │   └── AirQuality.html
│   │
│   ├── src/
│   │   └── GAKELM.py
│   │
│   ├── .gitignore
│   └── requirements.txt
│
└── README.md
```

The folders are separated by purpose so that the dataset, trained model files, notebooks, and Python source code are easier to find and maintain.

## Technologies used

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Genetic Algorithm
- Kernel Extreme Learning Machine
- Support Vector Regression

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

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install the dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/AirQuality.ipynb
```

Run the notebook cells in order.

## Evaluation metrics

**MSE (Mean Squared Error)** measures the average squared difference between actual and predicted values.

**RMSE (Root Mean Squared Error)** is the square root of MSE. It is easier to interpret because it uses the same scale as the values being evaluated.

The current notebook calculates both metrics using the normalized AQI target.

## Important files

| File | Purpose |
|---|---|
| `data/Dataset.csv` | Main Ahmedabad air-quality dataset |
| `data/testData.csv` | Test data included in the project |
| `notebooks/AirQuality.ipynb` | Main notebook for preprocessing, training and evaluation |
| `notebooks/AirQuality.html` | HTML version of the notebook |
| `src/GAKELM.py` | GA-KELM implementation |
| `models/` | Stored model-related files |
| `requirements.txt` | Python dependencies |

## Future improvements

Possible next steps include:

- Further tuning of the Genetic Algorithm.
- Testing additional regression models.
- Trying feature-selection techniques.
- Reporting both normalized and original AQI-scale metrics.
- Building a simple interface for AQI prediction.
- Deploying the trained model as a web application or API.

## Author

**Mohammed Omer Farooq**

GitHub: https://github.com/omer-farooq28

## License

This project was developed for academic and educational purposes.
