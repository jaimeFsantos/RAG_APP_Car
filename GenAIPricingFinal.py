import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from openai import OpenAI
import tiktoken
import faiss
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# OpenAI API Configuration
OPENAI_API_KEY = "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"  # Replace with your actual OpenAI API key

class CarPriceAnalyst:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def build_index(self, total_data: pd.DataFrame, filtered_data: pd.DataFrame):
        """Placeholder for compatibility"""
        pass
        
    def format_metrics_explanation(self, metrics: Dict) -> str:
        """Format model performance metrics explanation"""
        avg_rmse = np.mean([m['rmse'] for m in metrics.values()])
        avg_r2 = np.mean([m['r2'] for m in metrics.values()])
        avg_mape = np.mean([m['mape'] for m in metrics.values()]) * 100
        
        return f"""
        Model Performance Metrics:
        - RMSE (Root Mean Square Error): ${avg_rmse:,.2f} - This represents the average prediction error in dollars
        - R² Score: {avg_r2:.3f} - This indicates that {avg_r2*100:.1f}% of price variations are explained by the model
        - MAPE: {avg_mape:.1f}% - This means predictions are typically within {avg_mape:.1f}% of actual prices
        """

    def analyze_feature_importance(self, shap_values: Dict[str, float]) -> str:
        """Format feature importance analysis"""
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:5]
        
        analysis = []
        for feature, impact in top_features:
            direction = "increases" if impact > 0 else "decreases"
            feature_name = feature.replace('_', ' ').title()
            analysis.append(f"- {feature_name}: {direction} price by ${abs(impact):,.2f}")
            
        return "\n".join(analysis)

    def format_claude_response(self, response):
        """Format the response with proper styling"""
        try:
            text = str(response)
            
            # Process bold text
            text = text.replace('\\n', '\n')
            while '**' in text:
                text = text.replace('**', '<strong>', 1)
                text = text.replace('**', '</strong>', 1)
            
            # Process paragraphs and bullets
            lines = []
            paragraphs = text.split('\n\n')
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                    
                lines_in_paragraph = paragraph.split('\n')
                has_bullets = any(line.strip().startswith(('-', '*')) for line in lines_in_paragraph)
                
                if has_bullets:
                    lines.append('<ul>')
                    for line in lines_in_paragraph:
                        line = line.strip()
                        if line.startswith(('-', '*')):
                            content = line[1:].strip()
                            if content.startswith(' '):
                                content = content[1:]
                            lines.append(f'<li>{content}</li>')
                    lines.append('</ul>')
                else:
                    lines.append(f'<p>{paragraph.strip()}</p>')
            
            formatted_text = '\n'.join(lines)
            
            # Add styling
            st.markdown("""
                <style>
                p {
                    margin-bottom: 1em;
                    line-height: 1.6;
                }
                strong {
                    color: #1f77b4;
                    font-weight: 600;
                }
                ul {
                    margin: 1em 0;
                    padding-left: 2em;
                    list-style-type: disc;
                }
                li {
                    margin-bottom: 0.5em;
                    line-height: 1.6;
                    padding-left: 0.5em;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown(formatted_text, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error formatting response: {str(e)}")
            return str(response)

    def get_response(self, query: str, total_data: pd.DataFrame, filtered_data: pd.DataFrame, 
                    prediction_result: Dict[str, Any] = None, shap_values: Dict[str, float] = None) -> str:
        """Generate focused analysis of model results and predictions"""
        
        # Prepare prediction context if available
        prediction_context = ""
        if prediction_result:
            prediction_context = f"""
            The model predicts a price of ${prediction_result['predicted_price']:,.2f}
            Confidence range: ${prediction_result['prediction_interval'][0]:,.2f} to ${prediction_result['prediction_interval'][1]:,.2f}
            Model confidence: {(1 - prediction_result['mape']) * 100:.1f}%

            Model Performance:
            {self.format_metrics_explanation(st.session_state.get('metrics', {}))}
            """

        # Prepare feature importance if available
        feature_context = ""
        if shap_values:
            sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            feature_impacts = []
            for feature, impact in sorted_features[:5]:
                direction = "increases" if impact > 0 else "decreases"
                feature_name = feature.replace('_', ' ').title()
                feature_impacts.append(f"- {feature_name} {direction} price by ${abs(impact):,.2f}")
            feature_context = "\n".join(feature_impacts)

        prompt = f"""As a data science expert, explain this car price analysis clearly and directly. Here are the key details:

        {prediction_context}

        Top price factors:
        {feature_context}

        Question: {query}

        Guidelines:
        - Be direct and concise
        - Use bold (**) for key numbers and findings
        - Focus only on what's relevant to the question
        - If a feature isn't important, say "the model did not find [feature] to be important"
        - Keep response to 2-3 short paragraphs
        - Write conversationally
        - Avoid technical jargon unless explaining a specific metric
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=400
        )

        response_text = response.choices[0].message.content
        self.format_claude_response(response_text)
        return ""

class CarPricePredictor:
    def __init__(self, models=None, fast_mode=False, max_samples=None):
        self.scaler = StandardScaler()
        self.best_models = {}
        self.fast_mode = fast_mode
        self.max_samples = max_samples
        self.feature_columns = None
        self.is_trained = False
        self.metrics = {}
        self.unique_values = {}
        
        self.available_models = {
            'ridge': {'speed': 1, 'name': 'Ridge Regression'},
            'lasso': {'speed': 2, 'name': 'Lasso Regression'},
            'gbm': {'speed': 3, 'name': 'Gradient Boosting'},
            'rf': {'speed': 4, 'name': 'Random Forest'},
            'xgb': {'speed': 5, 'name': 'XGBoost'}
        }
        
        self.selected_models = models if models else list(self.available_models.keys())
        
        self.param_grids = {
            'regular': {
                'rf': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gbm': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'xgb': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'fast': {
                'rf': {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'gbm': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'xgb': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'min_child_weight': [3],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
            }
        }

    def update_unique_values(self, df):
        def safe_sort(values):
            cleaned_values = [str(x) for x in values if pd.notna(x)]
            return sorted(cleaned_values)
        
        self.unique_values = {
            'state': safe_sort(df['state'].unique()),
            'body': safe_sort(df['body'].unique()),
            'transmission': safe_sort(df['transmission'].unique()),
            'color': safe_sort(df['color'].unique()),
            'interior': safe_sort(df['interior'].unique())
        }

    def remove_outliers(self, df, threshold=1.5):
        initial_rows = len(df)
        
        Q1 = df['sellingprice'].quantile(0.25)
        Q3 = df['sellingprice'].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (
            (df['sellingprice'] < (Q1 - threshold * IQR)) | 
            (df['sellingprice'] > (Q3 + threshold * IQR))
        )
        
        df_cleaned = df[~outlier_condition]
        
        rows_removed = initial_rows - len(df_cleaned)
        return df_cleaned, {
            'initial_rows': initial_rows,
            'final_rows': len(df_cleaned),
            'rows_removed': rows_removed,
            'removal_percentage': (rows_removed/initial_rows)*100,
            'price_range': (Q1, Q3),
            'outliers': df[outlier_condition]['sellingprice'].tolist()
        }

    def prepare_data(self, df):
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)
        
        string_columns = ['state', 'body', 'transmission', 'color', 'interior']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self.update_unique_values(df)
        
        drop_cols = ['datetime', 'Day of Sale', 'Weekend', 'vin', 'seller', 'make', 'model', 'trim', 'saledate']
        df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        
        fill_values = {
            'transmission': 'unknown',
            'interior': 'unknown',
            'condition': df['condition'].median(),
            'odometer': df['odometer'].median()
        }
        
        for col, fill_value in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        
        df = df[df['sellingprice'] > 0]
        
        df, outlier_stats = self.remove_outliers(df)
        
        return df, outlier_stats

    def engineer_features(self, df):
        self.original_features = df.copy()
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['sellingprice', 'mmr']]
        
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
        
        key_numeric = ['odometer', 'year', 'condition']
        for col in key_numeric:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        categorical_cols = ['body', 'transmission', 'state', 'color', 'interior']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        df['age_miles'] = df['vehicle_age'] * df['odometer']
        df['age_squared'] = df['vehicle_age'] ** 2
        
        return df

    def remove_multicollinearity(self, X, threshold=0.95):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            return X.drop(columns=to_drop)
        return X

    def tune_model(self, model_type, X, y):
        param_grid = self.param_grids['fast' if self.fast_mode else 'regular'][model_type]
        
        if model_type == 'rf':
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1 if not self.fast_mode else 1)
        elif model_type == 'gbm':
            base_model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'xgb':
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1 if not self.fast_mode else 1)
        
        cv_folds = min(3, len(X)) if self.fast_mode else min(5, len(X))
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    def fit(self, X, y):
        self.feature_columns = X.columns.tolist()
        
        for model_type in self.selected_models:
            if model_type in ['rf', 'gbm', 'xgb']:
                self.best_models[model_type] = self.tune_model(model_type, X, y)
            elif model_type == 'lasso':
                self.best_models['lasso'] = LassoCV(
                    cv=3 if self.fast_mode else 5,
                    random_state=42,
                    max_iter=2000
                ).fit(X, y)
            elif model_type == 'ridge':
                self.best_models['ridge'] = RidgeCV(
                    cv=3 if self.fast_mode else 5
                ).fit(X, y)
        
        if len(self.best_models) > 1:
            self.ensemble = VotingRegressor([
                (name, model) for name, model in self.best_models.items()
            ])
            self.ensemble.fit(X, y)
        
        self.is_trained = True

    def evaluate(self, X, y):
        metrics = {}
        predictions = {}
        
        for name, model in self.best_models.items():
            pred = model.predict(X)
            predictions[name] = pred
            metrics[name] = {
                'r2': r2_score(y, pred),
                'rmse': np.sqrt(mean_squared_error(y, pred)),
                'mape': mean_absolute_percentage_error(y, pred)
            }
        
        if len(self.best_models) > 1:
            ensemble_pred = self.ensemble.predict(X)
            predictions['ensemble'] = ensemble_pred
            metrics['ensemble'] = {
                'r2': r2_score(y, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
                'mape': mean_absolute_percentage_error(y, ensemble_pred)
            }
        
        self.metrics = metrics
        return metrics, predictions

    def prepare_prediction_data(self, input_data):
        """Prepare input data for prediction"""
        df = pd.DataFrame([input_data])
        
        df['vehicle_age'] = 2024 - df['year']
        df['age_miles'] = df['vehicle_age'] * df['odometer']
        df['age_squared'] = df['vehicle_age'] ** 2
        
        numeric_cols = ['odometer', 'year', 'condition']
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
            df[f'{col}_squared'] = df[col] ** 2
        
        categorical_cols = ['body', 'transmission', 'state', 'color', 'interior']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        missing_cols = set(self.feature_columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
            
        df_encoded = df_encoded[self.feature_columns]
        
        return df_encoded

    def create_what_if_prediction(self, input_data):
        if not self.is_trained or self.feature_columns is None:
            raise ValueError("Model must be trained before making predictions.")
        
        df_encoded = self.prepare_prediction_data(input_data)
        
        X_scaled = self.scaler.transform(df_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        predictions = []
        model_predictions = {}
        for model_name, model in self.best_models.items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            model_predictions[model_name] = pred
        
        if len(self.best_models) > 1:
            ensemble_pred = self.ensemble.predict(X_scaled)[0]
            predictions.append(ensemble_pred)
            model_predictions['ensemble'] = ensemble_pred
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        confidence_interval = (
            mean_pred - (1.96 * std_pred),
            mean_pred + (1.96 * std_pred)
        )
        
        mape = np.mean([metrics['mape'] for metrics in self.metrics.values()])
        prediction_interval = (
            mean_pred * (1 - mape),
            mean_pred * (1 + mape)
        )
        
        return {
            'predicted_price': mean_pred,
            'confidence_interval': confidence_interval,
            'prediction_interval': prediction_interval,
            'std_dev': std_pred,
            'model_predictions': model_predictions,
            'mape': mape
        }

def analyze_shap_values(predictor, input_data):
    """Generate SHAP analysis for the prediction"""
    explainer = shap.TreeExplainer(predictor.best_models['rf'])
    df_encoded = predictor.prepare_prediction_data(input_data)
    shap_values = explainer.shap_values(df_encoded)
    
    feature_importance = dict(zip(predictor.feature_columns, shap_values[0]))
    return feature_importance

def main():
    st.set_page_config(
        page_title="Car Price Predictor",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Price Predictor")
    
    # File upload in sidebar
    st.sidebar.header("Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Car Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            total_data = pd.read_csv(uploaded_file)
            
            string_cols = ['make', 'model', 'trim', 'state', 'body', 'transmission', 'color', 'interior']
            for col in string_cols:
                if col in total_data.columns:
                    total_data[col] = total_data[col].astype(str)
            
            # Initialize predictor
            predictor = CarPricePredictor(
                models=['gbm', 'rf', 'xgb'],
                fast_mode=True
            )
            
            predictor.update_unique_values(total_data)
            
            # Model settings
            st.sidebar.subheader("Model Settings")
            fast_mode = st.sidebar.checkbox("Fast Mode (Quick Training)", value=True)
            predictor.fast_mode = fast_mode
            
            # Vehicle selection interface
            st.header("Select Vehicle")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                make = st.selectbox("Make", options=sorted(total_data['make'].unique()))
            
            filtered_models = total_data[total_data['make'] == make]['model'].unique()
            with col2:
                model = st.selectbox("Model", options=sorted(filtered_models))
            
            filtered_trims = total_data[
                (total_data['make'] == make) & 
                (total_data['model'] == model)
            ]['trim'].unique()
            with col3:
                trim = st.selectbox("Trim", options=sorted(filtered_trims))
            
            # Filter data for selected vehicle
            filtered_data = total_data[
                (total_data['make'] == make) &
                (total_data['model'] == model) &
                (total_data['trim'] == trim)
            ]
            
            st.info(f"Number of samples for this vehicle: {len(filtered_data)}")
            
            # Model training section
            if len(filtered_data) > 5:  # Minimum samples needed for training
                if st.button("Train Models", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            # Prepare and engineer features
                            df, outlier_stats = predictor.prepare_data(filtered_data)
                            df_engineered = predictor.engineer_features(df)
                            
                            # Split features and target
                            X = df_engineered.drop(['sellingprice', 'mmr'] if 'mmr' in df_engineered.columns else ['sellingprice'], axis=1)
                            y = df_engineered['sellingprice']
                            
                            # Remove multicollinearity
                            X = predictor.remove_multicollinearity(X)
                            
                            # Train-test split and scaling
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            X_train_scaled = predictor.scaler.fit_transform(X_train)
                            X_test_scaled = predictor.scaler.transform(X_test)
                            
                            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                            
                            # Fit and evaluate models
                            predictor.fit(X_train_scaled, y_train)
                            metrics, predictions = predictor.evaluate(X_test_scaled, y_test)
                            
                            # Store in session state
                            st.session_state['predictor'] = predictor
                            st.session_state['metrics'] = metrics
                            st.session_state['filtered_data'] = filtered_data
                            st.session_state['total_data'] = total_data
                            
                            st.success("Models trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
            else:
                st.warning("Not enough samples to train models. Please select a different vehicle with more data.")
            
            # Display model performance metrics
            if 'metrics' in st.session_state:
                st.header("Model Performance")
                
                avg_metrics = {
                    'RMSE': np.mean([m['rmse'] for m in st.session_state['metrics'].values()]),
                    'R²': np.mean([m['r2'] for m in st.session_state['metrics'].values()]),
                    'Error %': np.mean([m['mape'] for m in st.session_state['metrics'].values()]) * 100
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Error", f"{avg_metrics['Error %']:.1f}%")
                with col2:
                    st.metric("RMSE", f"${avg_metrics['RMSE']:,.0f}")
                with col3:
                    st.metric("R² Score", f"{avg_metrics['R²']:.3f}")
            
            # Price estimator section
            st.header("Price Estimator")
            
            if 'predictor' in st.session_state:
                predictor = st.session_state['predictor']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
                    condition = st.number_input("Condition (1-50)", min_value=1.0, max_value=50.0, value=25.0, step=1.0)
                    odometer = st.number_input("Mileage", min_value=0, value=50000, step=1000)
                    state = st.selectbox("State", options=predictor.unique_values['state'])
                
                with col2:
                    body = st.selectbox("Body Style", options=predictor.unique_values['body'])
                    transmission = st.selectbox("Transmission", options=predictor.unique_values['transmission'])
                    color = st.selectbox("Color", options=predictor.unique_values['color'])
                    interior = st.selectbox("Interior", options=predictor.unique_values['interior'])
                
                if st.button("Get Price Estimate", type="primary"):
                    try:
                        input_data = {
                            'state': state,
                            'body': body,
                            'transmission': transmission,
                            'color': color,
                            'interior': interior,
                            'year': year,
                            'condition': condition,
                            'odometer': odometer
                        }
                        
                        prediction_result = predictor.create_what_if_prediction(input_data)
                        st.session_state['last_prediction'] = input_data
                        st.session_state['last_prediction_result'] = prediction_result
                        
                        mean_price = prediction_result['predicted_price']
                        mape = prediction_result['mape']
                        
                        low_estimate = mean_price * (1 - mape)
                        high_estimate = mean_price * (1 + mape)
                        
                        st.subheader("Price Estimates")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Low Estimate", f"${low_estimate:,.0f}")
                        with col2:
                            st.metric("Best Estimate", f"${mean_price:,.0f}")
                        with col3:
                            st.metric("High Estimate", f"${high_estimate:,.0f}")
                        
                        st.info(f"Estimated error margin: ±{mape*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            
            # AI Analyst Section
            if 'predictor' in st.session_state:
                st.header("💡 AI Insights")

                if 'analyst' not in st.session_state:
                    st.session_state.analyst = CarPriceAnalyst(OPENAI_API_KEY)
                    st.session_state.analyst.build_index(total_data, filtered_data)
                
                query = st.text_input(
                    "Ask me anything about the market, predictions, or pricing factors:",
                    placeholder="E.g., What factors most affect this car's value? How does it compare to similar models?"
                )
                
                if query:
                    with st.spinner("Analyzing data and generating insights..."):
                        try:
                            shap_values = None
                            if 'last_prediction' in st.session_state:
                                shap_values = analyze_shap_values(
                                    predictor,
                                    st.session_state.last_prediction
                                )
                            
                            response = st.session_state.analyst.get_response(
                                query,
                                total_data,
                                filtered_data,
                                st.session_state.get('last_prediction_result'),
                                shap_values
                            )
                            
                            st.write(response)
                            
                            # Add visualization of SHAP values
                            if shap_values:
                                st.subheader("Feature Impact Visualization")
                                
                                feature_importance = pd.DataFrame({
                                    'Feature': list(shap_values.keys()),
                                    'Impact': list(shap_values.values())
                                })
                                feature_importance = feature_importance.sort_values('Impact', key=abs, ascending=True)
                                
                                fig = go.Figure(go.Bar(
                                    x=feature_importance['Impact'],
                                    y=feature_importance['Feature'],
                                    orientation='h'
                                ))
                                
                                fig.update_layout(
                                    title="Feature Impact on Price Prediction",
                                    xaxis_title="Impact on Price ($)",
                                    yaxis_title="Feature",
                                    height=max(400, len(feature_importance) * 20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file containing car sales data to begin.")

if __name__ == "__main__":
    main()