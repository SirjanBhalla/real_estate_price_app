# **REAL ESTATE PRICE PREDICTOR**
A web app that estimates the sale price of a property based on key attributes such as year sold, bedrooms, lot size, and property type.

### OVERVIEW:

Input: Year Sold, Property Tax, Insurance, Bedrooms, Bathrooms, Sqft, Year Built, Lot Size, Basement, Popular, Recession, Property Type (Condo/Bungalow)

Output: Predicted property sale price in USD

Built with: Python, Streamlit, scikit-learn

----------------------------------------------------------------------------------------------------------------------------

### PROJECT STRUCTURE:

real_estate_price_app/
├── app.py
├── real_estate_price/
│    ├── __init__.py
│    ├── config.py
│    ├── data_management.py
│    ├── pipeline.py
│    ├── trained_models/
│    │    └── real_estate_pipeline.joblib
├── requirements.txt
├── README.md

----------------------------------------------------------------------------------------------------------------------------

### HOW TO RUN THE APP:

##### 1. Clone the repo and change directory:
    ```terminal
    git clone <your-repo-link>
    cd real_estate_price_app
    ```

##### 2. Create & activate a virtual environment (recommended):
    ```terminal
    python -m venv .venv
    source .venv/bin/activate        # On Windows: .venv\Scripts\activate
    ```

##### 3. Install dependencies:
    ```terminal
    pip install -r requirements.txt
    ```

##### 4. Start the Streamlit app:
    ```terminal
    streamlit run app.py
    ```

##### 5. Open the browser link

----------------------------------------------------------------------------------------------------------------------------

### FEATURES:

1. Intuitive input forms with tooltips explaining each field.

2. Radio and dropdowns for categorical features (like property type).

3. Outputs an easy-to-read price estimate with formatting.

4. Friendly info reminders about the estimate’s limitations.

----------------------------------------------------------------------------------------------------------------------------

### MODEL INFO:

1. Model type: Random Forest regression

2. Input features: Year Sold, Property Tax, Insurance, Bedrooms, Bathrooms, Sqft, Year Built, Lot Size, Basement, Popular, Recession, Property Type (Condo/Bungalow)

3. Output: Sale price in USD (rounded, formatted)

4. Best reproducible R²: ~0.79

5. Random seed: Set for reproducibility

----------------------------------------------------------------------------------------------------------------------------

*See requirements.txt for exact versions.*


