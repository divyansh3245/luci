
# Wine Quality Prediction

This project uses a machine learning model to predict the quality of red wine based on physicochemical properties. The model is trained on the `winequality-red.csv` dataset and deployed via a Streamlit web app.

---

## Dataset

**Source:** UCI Machine Learning Repository  
**File:** `data/winequality-red.csv`  
**Attributes (11 features):**

- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

**Target:** `quality` (score between 0 and 10)

---

## Project Structure

```
wine-quality-predictor/
├── data/
│   └── winequality-red.csv
├── app.py
├── data_visualization.py
├── model.py
├── scaler.pkl
├── wine_model.pkl
├── requirements.txt
└── README.md

```

---

## How It Works

- **model.py**: Trains the model and saves the scaler and model as `.pkl` files.
- **data_visualization.py**: Can be run standalone to visualize data distributions and correlations.
- **app.py**: Streamlit web app that loads the model and takes user input for wine quality prediction.

---

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (if not already trained)**
   ```bash
   python model.py
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **View Data Visualizations**
   ```bash
   python data_visualization.py
   ```

--- 

## Demo

[View Demo](https://files.catbox.moe/d9z0jn.mp4)