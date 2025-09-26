"""
Sales Forecasting Models
========================

Comprehensive forecasting algorithms for business sales prediction
Created by: Ayush Jaggi
Date: September 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Statistical and time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Prophet for advanced time series (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')


class SalesForecaster:
    """
    Comprehensive sales forecasting system supporting multiple algorithms:
    - ARIMA (Statistical time series)
    - Prophet (Facebook's forecasting tool)
    - Random Forest (Machine learning)
    - Linear Regression (Baseline)
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SalesForecaster
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Data storage
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # Model storage
        self.models = {}
        self.model_results = {}
        self.best_model = None
        
        # Feature columns for ML models
        self.feature_columns = ['year', 'month', 'day', 'weekday', 'is_weekend', 
                               'is_holiday_season', 'sales_lag1', 'sales_lag7', 
                               'sales_ma7', 'sales_ma30']
        
        # Fitted flags
        self.is_fitted = False
        
        print("📈 SalesForecaster initialized!")
        print(f"   Prophet available: {PROPHET_AVAILABLE}")
        print(f"   Random state: {random_state}")
    
    def load_data(self, data_df, date_column='date', sales_column='sales'):
        """
        Load and preprocess sales data
        
        Args:
            data_df (pd.DataFrame): Sales data
            date_column (str): Name of date column
            sales_column (str): Name of sales column
        """
        print("📥 Loading sales data...")
        
        self.data = data_df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Set date as index
        self.data = self.data.set_index(date_column).sort_index()
        
        # Rename sales column for consistency
        if sales_column != 'sales':
            self.data = self.data.rename(columns={sales_column: 'sales'})
        
        # Generate time-based features
        self._generate_time_features()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"   Total days: {len(self.data):,}")
        print(f"   Average daily sales: ${self.data['sales'].mean():,.0f}")
        print(f"   Total revenue: ${self.data['sales'].sum():,.0f}")
    
    def _generate_time_features(self):
        """Generate time-based features for machine learning models"""
        print("🔄 Generating time-based features...")
        
        # Basic time features
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['weekday'] = self.data.index.weekday
        self.data['is_weekend'] = (self.data.index.weekday >= 5).astype(int)
        
        # Holiday season indicator (November-December)
        self.data['is_holiday_season'] = (
            (self.data.index.month == 11) | (self.data.index.month == 12)
        ).astype(int)
        
        # Lag features
        self.data['sales_lag1'] = self.data['sales'].shift(1)
        self.data['sales_lag7'] = self.data['sales'].shift(7)
        
        # Moving averages
        self.data['sales_ma7'] = self.data['sales'].rolling(window=7).mean()
        self.data['sales_ma30'] = self.data['sales'].rolling(window=30).mean()
        
        print(f"   Generated {len(self.feature_columns)} features")
    
    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of data for testing
        """
        print(f"🔄 Splitting data (test size: {test_size:.1%})...")
        
        split_index = int(len(self.data) * (1 - test_size))
        
        self.train_data = self.data.iloc[:split_index].copy()
        self.test_data = self.data.iloc[split_index:].copy()
        
        print(f"   Training data: {len(self.train_data)} days")
        print(f"   Testing data: {len(self.test_data)} days")
    
    def fit_arima(self, order=(1, 1, 1)):
        """
        Fit ARIMA model
        
        Args:
            order (tuple): ARIMA order (p, d, q)
        """
        print(f"🔄 Training ARIMA model with order {order}...")
        
        try:
            # Fit ARIMA model
            model = ARIMA(self.train_data['sales'], order=order)
            fitted_model = model.fit()
            
            # Store model
            self.models['ARIMA'] = {
                'model': fitted_model,
                'order': order
            }
            
            print(f"   ✅ ARIMA model trained successfully")
            print(f"   AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"   ❌ ARIMA training failed: {e}")
            return None
    
    def fit_prophet(self):
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            print("⚠️ Prophet not available - skipping")
            return None
        
        print("🔄 Training Prophet model...")
        
        try:
            # Prepare data for Prophet
            prophet_data = self.train_data.reset_index()[['date', 'sales']].rename(
                columns={'date': 'ds', 'sales': 'y'}
            )
            
            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(prophet_data)
            
            # Store model
            self.models['Prophet'] = {
                'model': model,
                'train_data': prophet_data
            }
            
            print("   ✅ Prophet model trained successfully")
            return model
            
        except Exception as e:
            print(f"   ❌ Prophet training failed: {e}")
            return None
    
    def fit_random_forest(self, n_estimators=100, max_depth=15):
        """
        Fit Random Forest model
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
        """
        print(f"🔄 Training Random Forest model...")
        
        try:
            # Prepare features
            X_train = self.train_data[self.feature_columns].dropna()
            y_train = self.train_data.loc[X_train.index, 'sales']
            
            print(f"   Training samples: {len(X_train)}")
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store model
            self.models['Random Forest'] = {
                'model': model,
                'feature_importance': feature_importance
            }
            
            print(f"   ✅ Random Forest trained successfully")
            print(f"   Top feature: {feature_importance.iloc[0]['feature']}")
            
            return model
            
        except Exception as e:
            print(f"   ❌ Random Forest training failed: {e}")
            return None
    
    def fit_linear_regression(self):
        """Fit Linear Regression model"""
        print("🔄 Training Linear Regression model...")
        
        try:
            # Prepare features
            X_train = self.train_data[self.feature_columns].dropna()
            y_train = self.train_data.loc[X_train.index, 'sales']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Store model
            self.models['Linear Regression'] = {
                'model': model,
                'scaler': scaler
            }
            
            print("   ✅ Linear Regression trained successfully")
            print(f"   Model score: {model.score(X_train_scaled, y_train):.3f}")
            
            return model
            
        except Exception as e:
            print(f"   ❌ Linear Regression training failed: {e}")
            return None
    
    def fit_all_models(self):
        """Train all available models"""
        print("🚀 Training all forecasting models...")
        
        # Train each model
        self.fit_arima()
        self.fit_prophet()
        self.fit_random_forest()
        self.fit_linear_regression()
        
        self.is_fitted = True
        print(f"✅ Training completed! {len(self.models)} models available")
    
    def predict(self, model_name, steps=None):
        """
        Generate predictions using specified model
        
        Args:
            model_name (str): Name of model to use
            steps (int): Number of future steps to predict (None for test set)
            
        Returns:
            np.array: Predictions
        """
        if not self.is_fitted or model_name not in self.models:
            raise ValueError(f"Model {model_name} not available or not fitted")
        
        if steps is None:
            # Predict on test set
            return self._predict_test_set(model_name)
        else:
            # Predict future values
            return self._predict_future(model_name, steps)
    
    def _predict_test_set(self, model_name):
        """Predict on test set"""
        model_info = self.models[model_name]
        
        if model_name == 'ARIMA':
            model = model_info['model']
            predictions = model.forecast(steps=len(self.test_data))
            
        elif model_name == 'Prophet':
            model = model_info['model']
            future_dates = pd.DataFrame({
                'ds': self.test_data.index
            })
            forecast = model.predict(future_dates)
            predictions = forecast['yhat'].values
            
        elif model_name in ['Random Forest', 'Linear Regression']:
            model = model_info['model']
            
            # Prepare test features
            X_test = self.test_data[self.feature_columns].dropna()
            
            if model_name == 'Linear Regression':
                scaler = model_info['scaler']
                X_test_scaled = scaler.transform(X_test)
                predictions = model.predict(X_test_scaled)
            else:
                predictions = model.predict(X_test)
        
        return predictions
    
    def _predict_future(self, model_name, steps):
        """Predict future values"""
        model_info = self.models[model_name]
        last_date = self.data.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                   periods=steps, freq='D')
        
        if model_name == 'ARIMA':
            model = model_info['model']
            predictions = model.forecast(steps=steps)
            
        elif model_name == 'Prophet':
            model = model_info['model']
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values
            
        elif model_name in ['Random Forest', 'Linear Regression']:
            # For ML models, we need to simulate future features
            predictions = []
            current_data = self.data.copy()
            
            for i in range(steps):
                future_date = future_dates[i]
                
                # Create features for this future date
                features = {
                    'year': future_date.year,
                    'month': future_date.month,
                    'day': future_date.day,
                    'weekday': future_date.weekday(),
                    'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                    'is_holiday_season': 1 if future_date.month in [11, 12] else 0,
                    'sales_lag1': current_data['sales'].iloc[-1],
                    'sales_lag7': current_data['sales'].iloc[-7] if len(current_data) >= 7 else current_data['sales'].iloc[-1],
                    'sales_ma7': current_data['sales'].tail(7).mean(),
                    'sales_ma30': current_data['sales'].tail(30).mean()
                }
                
                # Predict
                X_future = pd.DataFrame([features])[self.feature_columns]
                
                if model_name == 'Linear Regression':
                    scaler = model_info['scaler']
                    X_future_scaled = scaler.transform(X_future)
                    pred = model_info['model'].predict(X_future_scaled)[0]
                else:
                    pred = model_info['model'].predict(X_future)[0]
                
                predictions.append(pred)
                
                # Add this prediction to data for next iteration
                new_row = pd.Series({
                    'sales': pred,
                    **features
                }, name=future_date)
                current_data = pd.concat([current_data, new_row.to_frame().T])
        
        return np.array(predictions)
    
    def evaluate_models(self):
        """Evaluate all trained models on test set"""
        print("📊 Evaluating model performance...")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                # Get predictions
                predictions = self._predict_test_set(model_name)
                
                # Align with actual values
                if model_name in ['Random Forest', 'Linear Regression']:
                    # These models might have different indices due to NaN handling
                    X_test = self.test_data[self.feature_columns].dropna()
                    actual_values = self.test_data.loc[X_test.index, 'sales'].values
                    predictions = predictions[:len(actual_values)]
                else:
                    actual_values = self.test_data['sales'].values
                    predictions = predictions[:len(actual_values)]
                
                # Calculate metrics
                mae = mean_absolute_error(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                r2 = r2_score(actual_values, predictions)
                
                # Direction accuracy
                if len(actual_values) > 1:
                    actual_direction = np.diff(actual_values) > 0
                    pred_direction = np.diff(predictions) > 0
                    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                else:
                    direction_accuracy = 0
                
                results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'predictions': predictions,
                    'actual': actual_values
                }
                
                print(f"   {model_name}: MAPE = {mape:.2f}%, R² = {r2:.3f}")
                
            except Exception as e:
                print(f"   ❌ {model_name} evaluation failed: {e}")
                continue
        
        self.model_results = results
        
        # Find best model (lowest MAPE)
        if results:
            self.best_model = min(results.keys(), key=lambda x: results[x]['mape'])
            print(f"\n🏆 Best model: {self.best_model} (MAPE: {results[self.best_model]['mape']:.2f}%)")
        
        return results
    
    def get_forecast(self, days=30, model_name=None):
        """
        Generate future forecast
        
        Args:
            days (int): Number of days to forecast
            model_name (str): Model to use (None for best model)
            
        Returns:
            pd.DataFrame: Forecast results
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Please call fit_all_models() first.")
        
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        print(f"🔮 Generating {days}-day forecast using {model_name}...")
        
        # Generate predictions
        predictions = self._predict_future(model_name, days)
        
        # Create forecast DataFrame
        last_date = self.data.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                   periods=days, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions,
            'model': model_name
        })
        
        # Add confidence intervals (simple approach)
        historical_std = self.data['sales'].std()
        forecast_df['lower_bound'] = forecast_df['forecast'] - 1.96 * historical_std
        forecast_df['upper_bound'] = forecast_df['forecast'] + 1.96 * historical_std
        
        # Ensure non-negative forecasts
        forecast_df['forecast'] = np.maximum(forecast_df['forecast'], 0)
        forecast_df['lower_bound'] = np.maximum(forecast_df['lower_bound'], 0)
        
        print(f"   📊 Average daily forecast: ${predictions.mean():,.0f}")
        print(f"   💰 Total forecasted revenue: ${predictions.sum():,.0f}")
        
        return forecast_df
    
    def plot_forecast(self, forecast_df, show_historical_days=90):
        """
        Plot forecast with historical data
        
        Args:
            forecast_df (pd.DataFrame): Forecast results
            show_historical_days (int): Days of historical data to show
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Historical data
        historical = self.data.tail(show_historical_days)
        ax.plot(historical.index, historical['sales'], 
               label='Historical Sales', color='navy', linewidth=2, alpha=0.8)
        
        # Forecast
        ax.plot(forecast_df['date'], forecast_df['forecast'],
               label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Confidence intervals
        ax.fill_between(forecast_df['date'], 
                       forecast_df['lower_bound'], 
                       forecast_df['upper_bound'],
                       alpha=0.2, color='red', label='95% Confidence')
        
        # Formatting
        ax.set_title('Sales Forecast', fontsize=16, fontweight='bold')
        ax.set_ylabel('Sales ($)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate historical from forecast
        ax.axvline(x=self.data.index.max(), color='black', 
                  linestyle=':', alpha=0.7, label='Forecast Start')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_model_comparison(self):
        """Get model comparison DataFrame"""
        if not self.model_results:
            print("⚠️ No model results available. Run evaluate_models() first.")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': f"${metrics['mae']:,.0f}",
                'RMSE': f"${metrics['rmse']:,.0f}",
                'MAPE': f"{metrics['mape']:.2f}%",
                'R²': f"{metrics['r2']:.3f}",
                'Direction_Accuracy': f"{metrics['direction_accuracy']:.1f}%"
            })
        
        return pd.DataFrame(comparison_data)
    
    def analyze_seasonality(self):
        """Analyze seasonal patterns in the data"""
        print("🎭 Analyzing seasonal patterns...")
        
        # Weekly seasonality
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_pattern = self.data.groupby(self.data.index.weekday)['sales'].mean()
        
        # Monthly seasonality
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pattern = self.data.groupby(self.data.index.month)['sales'].mean()
        
        print("📅 Weekly Pattern:")
        for i, (day, avg_sales) in enumerate(weekly_pattern.items()):
            print(f"   {weekday_names[day]}: ${avg_sales:,.0f}")
        
        print("\n📅 Monthly Pattern:")
        for month, avg_sales in monthly_pattern.items():
            print(f"   {month_names[month-1]}: ${avg_sales:,.0f}")
        
        return {
            'weekly': weekly_pattern,
            'monthly': monthly_pattern
        }
    
    def get_business_insights(self, forecast_df):
        """Generate business insights from forecast"""
        print("💼 Generating business insights...")
        
        insights = {}
        
        # Forecast summary
        insights['forecast_summary'] = {
            'total_revenue': forecast_df['forecast'].sum(),
            'avg_daily_sales': forecast_df['forecast'].mean(),
            'forecast_period': f"{forecast_df['date'].min()} to {forecast_df['date'].max()}",
            'model_used': forecast_df['model'].iloc[0]
        }
        
        # Compare to historical
        historical_avg = self.data['sales'].tail(30).mean()
        forecast_avg = forecast_df['forecast'].mean()
        change_pct = ((forecast_avg - historical_avg) / historical_avg) * 100
        
        insights['trend_analysis'] = {
            'historical_avg_30d': historical_avg,
            'forecast_avg': forecast_avg,
            'change_percent': change_pct,
            'trend_direction': 'Up' if change_pct > 0 else 'Down' if change_pct < 0 else 'Stable'
        }
        
        # Seasonality insights
        seasonality = self.analyze_seasonality()
        
        insights['seasonality'] = {
            'best_weekday': seasonality['weekly'].idxmax(),
            'best_month': seasonality['monthly'].idxmax(),
            'weekend_boost': ((seasonality['weekly'][5:].mean() - seasonality['weekly'][:5].mean()) / seasonality['weekly'][:5].mean()) * 100
        }
        
        return insights


# Example usage and testing
if __name__ == "__main__":
    print("📈 Testing Sales Forecasting System")
    print("=" * 50)
    
    # Generate sample data
    def create_sample_sales_data():
        """Create sample sales data for testing"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', end='2024-09-30', freq='D')
        base_sales = 3000
        trend = np.linspace(0, 500, len(dates))
        seasonal = 400 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 300, len(dates))
        
        sales = base_sales + trend + seasonal + weekly + noise
        sales = np.maximum(sales, 100)  # No negative sales
        
        return pd.DataFrame({
            'date': dates,
            'sales': sales
        })
    
    # Test the forecaster
    sample_data = create_sample_sales_data()
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    forecaster.load_data(sample_data)
    forecaster.split_data(test_size=0.2)
    
    # Train models
    forecaster.fit_all_models()
    
    # Evaluate models
    results = forecaster.evaluate_models()
    
    # Show comparison
    comparison = forecaster.get_model_comparison()
    print("\n📊 Model Comparison:")
    print(comparison.to_string(index=False))
    
    # Generate forecast
    forecast = forecaster.get_forecast(days=30)
    print(f"\n🔮 30-Day Forecast Summary:")
    print(f"   Total Revenue: ${forecast['forecast'].sum():,.0f}")
    print(f"   Average Daily: ${forecast['forecast'].mean():,.0f}")
    
    # Business insights
    insights = forecaster.get_business_insights(forecast)
    print(f"\n💼 Business Insights:")
    print(f"   Trend: {insights['trend_analysis']['trend_direction']} ({insights['trend_analysis']['change_percent']:+.1f}%)")
    print(f"   Best weekday: {insights['seasonality']['best_weekday']}")
    print(f"   Weekend boost: {insights['seasonality']['weekend_boost']:+.1f}%")
    
    print("\n✅ Testing completed successfully!")
