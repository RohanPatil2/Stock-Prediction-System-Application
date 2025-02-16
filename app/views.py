from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.conf import settings
from django.contrib.auth.decorators import login_required

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import pandas as pd


import numpy as np
import json
import requests
import math
import logging
import traceback
import os
import time
import csv
import re

from datetime import datetime, timedelta
from collections import deque
from statistics import mean, median, stdev
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import yfinance as yf
import talib
import joblib
import qrcode
import smtplib
from email.mime.text import MIMEText
from django.views.decorators.http import require_http_methods
from django.views.decorators.gzip import gzip_page
from django.db import connection, transaction
from django.core.serializers.json import DjangoJSONEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import holidays

# Extended configuration
MODEL_CONFIG.update({
    'lstm': None,  # Will be defined separately
    'prophet': Prophet,
    'arima': ARIMA,
    'mlp': MLPRegressor
})

TECHNICAL_INDICATORS = INDICATORS + [
    'ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE', 'ICHIMOKU_SPAN_A', 'ICHIMOKU_SPAN_B',
    'WILLIAM_R', 'CCI', 'ADX', 'ATR', 'STOCH_RSI', 'MFI', 'FIBONACCI_LEVELS'
]

# Custom metrics tracking
class PredictionMetrics:
    def __init__(self):
        self.model_performance = {}
        self.execution_times = {}
        self.data_quality = {}
    
    def log_metric(self, category, name, value):
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][name] = value

metrics = PredictionMetrics()

# Enhanced error classes
class DataQualityWarning(UserWarning):
    pass

class ModelConvergenceWarning(UserWarning):
    pass

class ForecastStabilityWarning(UserWarning):
    pass

# Extended helper functions
def calculate_advanced_indicators(df):
    try:
        # Ichimoku Cloud
        conversion_period = 9
        base_period = 26
        span_b_period = 52
        df['ICHIMOKU_CONVERSION'] = (df['High'].rolling(window=conversion_period).max() + 
                                   df['Low'].rolling(window=conversion_period).min()) / 2
        df['ICHIMOKU_BASE'] = (df['High'].rolling(window=base_period).max() + 
                              df['Low'].rolling(window=base_period).min()) / 2
        df['ICHIMOKU_SPAN_A'] = (df['ICHIMOKU_CONVERSION'] + df['ICHIMOKU_BASE']) / 2
        df['ICHIMOKU_SPAN_B'] = (df['High'].rolling(window=span_b_period).max() + 
                                df['Low'].rolling(window=span_b_period).min()) / 2
        
        # Fibonacci Retracement Levels
        max_price = df['High'].max()
        min_price = df['Low'].min()
        difference = max_price - min_price
        df['FIB_23.6'] = max_price - difference * 0.236
        df['FIB_38.2'] = max_price - difference * 0.382
        df['FIB_50.0'] = max_price - difference * 0.5
        df['FIB_61.8'] = max_price - difference * 0.618
        
        # Volume-weighted MACD
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['VW_MACD'] = df['VWAP'].ewm(span=12, adjust=False).mean() - df['VWAP'].ewm(span=26, adjust=False).mean()
        
        # Seasonality Detection
        us_holidays = holidays.UnitedStates()
        df['IS_HOLIDAY'] = df.index.to_series().apply(lambda x: x in us_holidays).astype(int)
        df['DAY_OF_WEEK'] = df.index.dayofweek
        df['MONTH'] = df.index.month
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Advanced indicator calculation failed: {str(e)}")
        raise DataProcessingError("Failed to calculate advanced technical indicators")

def create_lstm_model(input_shape):
    try:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                recurrent_dropout=0.2, kernel_initializer='glorot_uniform'),
            Dropout(0.3),
            LSTM(64, return_sequences=False, 
                recurrent_dropout=0.2, kernel_initializer='glorot_uniform'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model
    except Exception as e:
        raise ModelTrainingError(f"LSTM model creation failed: {str(e)}")

def validate_data_quality(df):
    checks = {
        'missing_values': df.isnull().sum().sum(),
        'zero_volume_days': (df['Volume'] == 0).sum(),
        'price_spikes': (df['Close'].pct_change().abs() > 0.1).sum(),
        'flatline_periods': (df['Close'].rolling(5).std() == 0).sum()
    }
    
    metrics.log_metric('data_quality', 'basic_checks', checks)
    
    if checks['missing_values'] > len(df) * 0.1:
        raise DataQualityWarning("Excessive missing values in dataset")
    if checks['zero_volume_days'] > len(df) * 0.05:
        raise DataQualityWarning("Suspicious number of zero-volume days")
    if checks['price_spikes'] > len(df) * 0.05:
        raise DataQualityWarning("Unusual number of large price movements")
    
    return True

def generate_fibonacci_levels(high, low):
    diff = high - low
    return {
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '50.0%': high - diff * 0.5,
        '61.8%': high - diff * 0.618,
        '78.6%': high - diff * 0.786
    }

def monte_carlo_simulation(returns, days, simulations=1000):
    try:
        log_returns = np.log(1 + returns)
        drift = log_returns.mean() - 0.5 * log_returns.var()
        volatility = log_returns.std()
        
        simulation_results = []
        for _ in range(simulations):
            daily_returns = np.exp(drift + volatility * np.random.normal(size=days))
            price_path = [returns.index[-1]]
            for ret in daily_returns:
                price_path.append(price_path[-1] * ret)
            simulation_results.append(price_path)
        
        return np.array(simulation_results)
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {str(e)}")
        return None

# Enhanced prediction view
@require_http_methods(["GET"])
@gzip_page
@login_required
@transaction.atomic
def predict(request, ticker_value, number_of_days):
    full_context = {}
    debug_info = {}
    warnings = []
    
    try:
        # ======================= Advanced Input Validation =======================
        if not re.match(r'^[A-Z]{1,5}(?:\.(B|A))?$', ticker_value):
            raise ValueError("Invalid ticker format. Must be 1-5 uppercase letters with optional .A/.B suffix")
        
        number_of_days = min(max(int(number_of_days), 1), 365)
        debug_info['input_validation'] = {'ticker': ticker_value, 'days': number_of_days}
        
        # ======================= Data Acquisition & Preparation =======================
        data_start = time.time()
        df = fetch_and_preprocess_data(ticker_value)
        validate_data_quality(df)
        df = calculate_technical_indicators(df)
        df = calculate_advanced_indicators(df)
        debug_info['data_prep_time'] = time.time() - data_start
        
        # ======================= Feature Engineering =======================
        feature_start = time.time()
        features = create_feature_matrix(df)
        debug_info['feature_engineering_time'] = time.time() - feature_start
        
        # ======================= Model Training Pipeline =======================
        model_start = time.time()
        models = {
            'LSTM': train_lstm_model(features),
            'Prophet': train_prophet_model(df),
            'ARIMA': train_arima_model(df),
            'Gradient Boosting': get_model('gradient_boost').fit(features.X_train, features.y_train)
        }
        debug_info['model_training_time'] = time.time() - model_start
        
        # ======================= Ensemble Forecasting =======================
        forecast_start = time.time()
        ensemble_forecast = generate_ensemble_forecast(models, features, number_of_days)
        debug_info['forecast_generation_time'] = time.time() - forecast_start
        
        # ======================= Risk Analysis =======================
        risk_start = time.time()
        risk_metrics = calculate_risk_metrics(df, ensemble_forecast)
        debug_info['risk_analysis_time'] = time.time() - risk_start
        
        # ======================= Sentiment Integration =======================
        sentiment_start = time.time()
        sentiment_analysis = perform_sentiment_analysis(ticker_value)
        debug_info['sentiment_analysis_time'] = time.time() - sentiment_start
        
        # ======================= Report Generation =======================
        report_start = time.time()
        report = generate_comprehensive_report(
            ticker_value, ensemble_forecast, risk_metrics, sentiment_analysis
        )
        debug_info['report_generation_time'] = time.time() - report_start
        
        # ======================= Visualization Pipeline =======================
        viz_start = time.time()
        visualizations = create_advanced_visualizations(df, ensemble_forecast, risk_metrics)
        debug_info['visualization_time'] = time.time() - viz_start
        
        # ======================= Context Preparation =======================
        full_context.update({
            'ticker': ticker_value,
            'forecast': ensemble_forecast,
            'risk_metrics': risk_metrics,
            'sentiment': sentiment_analysis,
            'visualizations': visualizations,
            'report': report,
            'debug_info': debug_info,
            'model_performance': metrics.model_performance,
            'warnings': warnings
        })
        
        # ======================= Database Logging =======================
        log_prediction_to_db(request.user, ticker_value, ensemble_forecast)
        
        # ======================= Asynchronous Tasks =======================
        if settings.ENABLE_ASYNC_TASKS:
            from .tasks import (
                cache_prediction_results,
                update_historical_accuracy,
                notify_subscribed_users
            )
            transaction.on_commit(lambda: cache_prediction_results.delay(full_context))
            transaction.on_commit(lambda: update_historical_accuracy.delay(ticker_value))
            transaction.on_commit(lambda: notify_subscribed_users.delay(ticker_value))

    except ValueError as ve:
        logger.error(f"Input validation error: {str(ve)}")
        return render(request, 'input_error.html', {'error': str(ve)})
    
    except DataQualityWarning as dqw:
        logger.warning(f"Data quality issue: {str(dqw)}")
        warnings.append(str(dqw))
        full_context['warnings'] = warnings
    
    except ModelConvergenceWarning as mcw:
        logger.warning(f"Model convergence issue: {str(mcw)}")
        warnings.append(str(mcw))
        full_context['warnings'] = warnings
    
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        return render(request, 'critical_error.html', {
            'error_id': uuid.uuid4(),
            'traceback': traceback.format_exc()
        })
    
    return render(request, 'full_analysis_report.html', full_context)

logger = logging.getLogger(__name__)

# Custom exceptions
class StockPredictionException(Exception):
    pass

class DataProcessingError(StockPredictionException):
    pass

class ModelTrainingError(StockPredictionException):
    pass

# Configuration
INDICATORS = ['RSI', 'MACD', 'OBV', 'SMA_50', 'SMA_200', 'EMA_20', 'BB_UPPER', 'BB_LOWER']
MODEL_CONFIG = {
    'linear': LinearRegression,
    'svm': SVR,
    'random_forest': RandomForestRegressor,
    'gradient_boost': GradientBoostingRegressor
}

# Helper functions
def calculate_technical_indicators(df):
    try:
        # Calculate various technical indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(df['Close'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        return df
    except Exception as e:
        raise DataProcessingError(f"Technical indicator calculation failed: {str(e)}")

def create_sequence_dataset(data, window_size=60):
    try:
        X, y = [], []
        for i in range(len(data)-window_size-1):
            X.append(data[i:(i+window_size)])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)
    except Exception as e:
        raise DataProcessingError(f"Sequence creation failed: {str(e)}")

def get_model(model_name='linear'):
    try:
        model_class = MODEL_CONFIG.get(model_name.lower(), LinearRegression)
        if model_name == 'svm':
            return make_pipeline(
                MinMaxScaler(),
                GridSearchCV(
                    SVR(),
                    {'kernel': ('linear', 'rbf'), 'C': [1, 10]},
                    cv=5
                )
            )
        elif model_name == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'gradient_boost':
            return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        return model_class()
    except Exception as e:
        raise ModelTrainingError(f"Model initialization failed: {str(e)}")

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'negative'
    return 'neutral'

def send_email_notification(subject, body, recipient):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = settings.EMAIL_HOST_USER
        msg['To'] = recipient
        
        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            server.starttls()
            server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")

# Main prediction view
@login_required
def predict(request, ticker_value, number_of_days):
    start_time = time.time()
    prediction_context = {}
    
    try:
        # ======================= Input Validation =======================
        if not re.match(r'^[A-Z]{1,5}$', ticker_value):
            return render(request, 'invalid_ticker.html', {
                'error': 'Invalid ticker format. Must be 1-5 uppercase letters.'
            })

        number_of_days = int(number_of_days)
        if number_of_days < 1 or number_of_days > 365:
            return render(request, 'invalid_days.html', {
                'error': 'Prediction days must be between 1 and 365'
            })

        # ======================= Data Acquisition =======================
        cache_key = f"stock_data_{ticker_value}"
        df = cache.get(cache_key)
        
        if not df:
            try:
                df = yf.download(
                    tickers=ticker_value,
                    period='5y',
                    interval='1d',
                    prepost=True,
                    threads=True
                )
                if df.empty:
                    raise StockPredictionException("No data available for this ticker")
                
                df = calculate_technical_indicators(df)
                cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour
            except Exception as e:
                logger.error(f"Data acquisition failed: {str(e)}")
                return render(request, 'data_error.html', {
                    'error': 'Failed to fetch stock data'
                })

        # ======================= Advanced Charting =======================
        technical_fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.2, 0.2])
        
        # Candlestick Chart
        technical_fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        # Volume
        technical_fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(100, 100, 200, 0.6)'
        ), row=2, col=1)

        # RSI
        technical_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=1)
        ), row=3, col=1)

        # MACD
        technical_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue', width=1)
        ), row=4, col=1)
        
        technical_fig.update_layout(
            title=f'{ticker_value} Technical Analysis',
            height=900,
            showlegend=True,
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white"
        )
        prediction_context['technical_plot'] = plot(technical_fig, output_type='div')

        # ======================= Multi-Model Training =======================
        models = {
            'Linear Regression': get_model('linear'),
            'Support Vector Machine': get_model('svm'),
            'Random Forest': get_model('random_forest'),
            'Gradient Boosting': get_model('gradient_boost')
        }

        model_results = {}
        forecast_data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(forecast_data)

        X, y = create_sequence_dataset(scaled_data)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results[model_name] = {
                    'mse': mse,
                    'r2': r2,
                    'forecast': []
                }
            except Exception as e:
                logger.error(f"Model {model_name} training failed: {str(e)}")

        # ======================= Ensemble Prediction =======================
        last_sequence = scaled_data[-60:]
        future_predictions = []
        
        for _ in range(number_of_days):
            current_sequence = last_sequence[-60:].reshape(1, -1)
            day_predictions = []
            
            for model_name, result in model_results.items():
                if models[model_name]:
                    pred = models[model_name].predict(current_sequence)
                    day_predictions.append(pred[0])
            
            avg_prediction = np.mean(day_predictions)
            future_predictions.append(avg_prediction)
            last_sequence = np.append(last_sequence, avg_prediction)
        
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten().tolist()

        # ======================= News Sentiment Analysis =======================
        try:
            news_articles = get_stock_news(ticker_value, settings.NEWS_API_KEY)
            sentiment_scores = []
            
            for article in news_articles[:10]:  # Analyze top 10 articles
                sentiment = analyze_sentiment(article['title'] + ' ' + article['description'])
                article['sentiment'] = sentiment
                sentiment_scores.append(1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            prediction_context['news_sentiment'] = avg_sentiment
            prediction_context['market_news'] = news_articles[:5]
        except Exception as e:
            logger.error(f"News analysis failed: {str(e)}")

        # ======================= Risk Analysis =======================
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * math.sqrt(252)
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
        
        prediction_context['risk_metrics'] = {
            'volatility': f"{volatility:.2%}",
            'sharpe_ratio': f"{sharpe_ratio:.2f}",
            'max_drawdown': f"{(df['Close'].max() - df['Close'].min()) / df['Close'].max():.2%}",
            'var_95': f"{np.percentile(daily_returns, 5):.2%}"
        }

        # ======================= Generate Report =======================
        report_data = {
            'ticker': ticker_value,
            'prediction_days': number_of_days,
            'final_prediction': future_predictions[-1],
            'confidence_interval': f"± {np.std(future_predictions):.2f}",
            'execution_time': f"{(time.time() - start_time):.2f} seconds"
        }
        
        # ======================= Prepare Context =======================
        prediction_context.update({
            'plot_div': plot_div,
            'ensemble_forecast': future_predictions,
            'model_comparison': model_results,
            'ticker_info': get_ticker_info(ticker_value),
            'report': report_data,
            'prediction_chart': generate_prediction_chart(future_predictions, number_of_days),
            'performance_metrics': calculate_performance_metrics(df, future_predictions)
        })

        # Send email notification
        if request.user.email:
            email_body = f"Stock prediction report for {ticker_value}\n\n" + \
                         "\n".join([f"{k}: {v}" for k, v in report_data.items()])
            send_email_notification(
                subject=f"Prediction Report for {ticker_value}",
                body=email_body,
                recipient=request.user.email
            )

    except StockPredictionException as e:
        logger.error(f"Prediction error: {str(e)}")
        return render(request, 'prediction_error.html', {
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        return render(request, 'error.html', {
            'error': 'An unexpected error occurred',
            'traceback': traceback.format_exc()
        })

    return render(request, "advanced_result.html", prediction_context)

# Additional helper functions
def get_ticker_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'name': info.get('longName', ''),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': f"${info.get('marketCap', 0):,}",
            'peRatio': info.get('trailingPE', 'N/A'),
            'dividendYield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A'
        }
    except:
        return {}

def generate_prediction_chart(predictions, days):
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    fig = go.Figure(data=go.Scatter(x=dates, y=predictions, mode='lines+markers'))
    fig.update_layout(
        title='Ensemble Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Price',
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    return plot(fig, output_type='div')

def calculate_performance_metrics(historical_data, predictions):
    actual_prices = historical_data['Close'].values[-len(predictions):]
    mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    return {
        'MAPE': f"{mape:.2f}%",
        'RMSE': f"{rmse:.2f}",
        'Directional_Accuracy': f"{np.mean(np.sign(actual_prices[1:] - actual_prices[:-1]) == np.sign(predictions[1:] - predictions[:-1])) * 100:.2f}%"
    }
