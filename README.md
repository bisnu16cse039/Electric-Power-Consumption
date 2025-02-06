# Electric-Power-Consumption


## Overview

This project focuses on analyzing and predicting forcast electric power consumption at Tetouan city in Morocco using xgboost and transformers. By leveraging historical data, the aim is to develop models that can forecast future power usage patterns, aiding in efficient energy management.

## Dataset

The analysis utilizes the "Electric Power Consumption" dataset from Kaggle. This dataset comprises over 50 thousands measurements of electric power consumption energy consumption of the Tétouan city in Morocco over nearly one years. Key features include:

- **Date Time**: Time window of ten minutes.
- **Temperature**: Weather Temperature.
- **Humidity**: Weather Humidity.
- **Wind Speed**: Wind Speed.
- **Zone 1 Power Consumption**
- **Zone 2 Power Consumption**
- **Zone 3 Power Consumption**

*Note:Therer are no missing values. So we didn't require handle such case.


## Project Structure

The repository is organized as follows:

- `data/`: Contains the dataset and any data preprocessing scripts.
- `notebooks/`: Jupyter notebooks detailing the exploratory data analysis (EDA) and model development.
- `src/`: Saved models and training script.
- `README.md`: Project overview and instructions.

## Installation

To replicate this analysis, ensure you have Python 3.x installed. Follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bisnu16cse039/Electric-Power-Consumption.git
   cd Electric-Power-Consumption
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

*Note: Ensure that the `requirements.txt` file lists all necessary dependencies.*

## Usage

After setting up the environment:

1. **Data Preprocessing**:

   Run the data preprocessing script to handle missing values and perform necessary transformations:

   ```bash
   python scripts/preprocess_data.py
   ```

2. **Model Training and Evaluation**:

   Train the machine learning model using the prepared data:

   ```bash
   python scripts/train_evaluate.py
   ```



*Detailed analysis and visualizations can be found in the Jupyter notebooks within the `notebooks/` directory.*

## Results

The machine learning models were trained and evaluated on the dataset, yielding the following performance metrics:

| Model                | MAE   | MSE   |
|----------------------|-------|-------|
| XGBoost             | 0.4156 |  0.2803 |
| Vanilla Transformers | 0.3593 |  0.2199 |

These results suggest that the transformer based model outperforms XGBOOST.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

"I would like to acknowledge the helpful resources I consulted, including the  [Electric Power Consumption Forecasting](https://www.kaggle.com/code/nechbamohammed/electric-power-consumption-forecasting) notebook and the assistance provided by Deepseek, ChatGPT, and Gemini in code generation..

