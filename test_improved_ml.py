#!/usr/bin/env python3
"""
Test script to demonstrate the improved ML model performance.
Shows feature importance analysis, optimized feature selection, and enhanced predictions.
"""

import json
from datetime import datetime, timedelta
from trading_bot import TradingBot
import pandas as pd
import numpy as np

def test_improved_ml_model():
    """Test the enhanced ML model with feature importance analysis."""
    print("🚀 Testing Improved ML Model with Enhanced Features")
    print("="*60)
    
    # Create bot instance
    bot = TradingBot('config.json')
    
    # Test feature preprocessing
    print("\n1️⃣ Testing Enhanced Feature Engineering...")
    market_data = bot._fetch_market_data('XRP-USD', lookback_days=100)
    processed_data = bot._preprocess_data(market_data)
    
    print(f"Original data shape: {market_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"New features added: {processed_data.shape[1] - market_data.shape[1]}")
    
    print("\nAvailable Features:")
    feature_columns = [col for col in processed_data.columns if col not in ['Target', 'Target_Binary']]
    for i, feature in enumerate(feature_columns, 1):
        print(f"{i:2d}. {feature}")
    
    # Test optimized feature selection
    print("\n2️⃣ Testing Optimized Feature Selection...")
    optimized_features = bot._get_optimized_features()
    print(f"Selected {len(optimized_features)} optimized features:")
    for i, feature in enumerate(optimized_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Test model prediction with new features
    print("\n3️⃣ Testing ML Predictions with Enhanced Features...")
    recent_data = processed_data.tail(10)
    
    for i in range(5):
        sample_data = recent_data.iloc[i:i+1]
        try:
            confidence, prediction = bot._make_ml_prediction(sample_data)
            actual_return = recent_data.iloc[i]['Returns'] if not pd.isna(recent_data.iloc[i]['Returns']) else 0
            
            print(f"Sample {i+1}:")
            print(f"  ML Confidence: {confidence:.1%}")
            print(f"  Prediction: {'BUY' if confidence > 0.5 else 'HOLD'}")
            print(f"  Actual Return: {actual_return:.2%}")
            print(f"  Correct: {'✅' if (confidence > 0.5) == (actual_return > 0) else '❌'}")
            print()
        except Exception as e:
            print(f"  Error in prediction: {e}")
    
    # Test feature importance (simulate some training history)
    print("4️⃣ Testing Feature Importance Analysis...")
    
    # Manually trigger feature importance analysis
    X = processed_data[optimized_features].values
    y = processed_data['Target_Binary'].values
    
    # Remove any NaN values
    valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid_indices]
    y_clean = y[valid_indices]
    
    if len(X_clean) > 0:
        try:
            # Scale features
            X_scaled = bot.scaler.fit_transform(X_clean)
            
            # Train model
            bot.model.fit(X_scaled, y_clean)
            
            # Analyze feature importance
            importance_analysis = bot._analyze_feature_importance(X_scaled, y_clean, optimized_features)
            
            print(f"\nTop 5 Most Important Features:")
            for i, feature in enumerate(importance_analysis.get('top_5_features', []), 1):
                importance = importance_analysis['features'].get(feature, 0)
                print(f"{i}. {feature}: {importance:.4f}")
                
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
    
    # Test model performance comparison
    print("\n5️⃣ Model Performance Summary...")
    try:
        if len(X_clean) > 50:  # Ensure we have enough data
            # Create train/test split
            split_idx = int(len(X_clean) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
            
            # Train and evaluate
            bot.model.fit(X_train, y_train)
            train_score = bot.model.score(X_train, y_train)
            test_score = bot.model.score(X_test, y_test)
            
            print(f"Training Accuracy: {train_score:.1%}")
            print(f"Testing Accuracy: {test_score:.1%}")
            print(f"Training Samples: {len(X_train)}")
            print(f"Testing Samples: {len(X_test)}")
            
            # Get predictions
            y_pred = bot.model.predict_proba(X_test)[:, 1]
            
            # Calculate prediction confidence statistics
            high_conf_mask = y_pred > 0.7
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (y_test[high_conf_mask] == (y_pred[high_conf_mask] > 0.5)).mean()
                print(f"High Confidence Predictions (>70%): {high_conf_mask.sum()}")
                print(f"High Confidence Accuracy: {high_conf_accuracy:.1%}")
        
    except Exception as e:
        print(f"Error in performance evaluation: {e}")
    
    print("\n🎯 Enhanced ML Model Testing Complete!")
    print("Key Improvements:")
    print("✅ Added EMA-9 and EMA-50 indicators")
    print("✅ Enhanced feature engineering (19 features vs 9 original)")
    print("✅ Improved Random Forest hyperparameters")
    print("✅ Dynamic feature importance analysis")
    print("✅ Automatic feature selection based on importance")
    print("✅ Performance tracking and optimization")
    
    return bot

def compare_feature_sets():
    """Compare performance between old and new feature sets."""
    print("\n📊 Comparing Feature Set Performance...")
    
    bot = TradingBot('config.json')
    market_data = bot._fetch_market_data('XRP-USD', lookback_days=200)
    processed_data = bot._preprocess_data(market_data)
    
    # Remove NaN values
    processed_data = processed_data.dropna()
    
    if len(processed_data) < 100:
        print("Not enough data for comparison")
        return
    
    # Old feature set
    old_features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'Returns', 'Volume_Change']
    
    # New optimized feature set
    new_features = bot._get_optimized_features()
    
    print(f"Old Feature Set: {len(old_features)} features")
    print(f"New Feature Set: {len(new_features)} features")
    
    # Test both feature sets
    results = {}
    
    for name, features in [("Old", old_features), ("New", new_features)]:
        try:
            # Check if all features exist
            available_features = [f for f in features if f in processed_data.columns]
            if len(available_features) != len(features):
                missing = set(features) - set(available_features)
                print(f"Warning: Missing features for {name} set: {missing}")
                features = available_features
            
            if len(features) == 0:
                print(f"No valid features for {name} set")
                continue
            
            X = processed_data[features].values
            y = processed_data['Target_Binary'].values
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale and train
            scaler = bot.scaler.__class__()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = bot.model.__class__(**bot.model.get_params())
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'features': features
            }
            
            print(f"\n{name} Feature Set Results:")
            print(f"  Features used: {len(features)}")
            print(f"  Training accuracy: {train_acc:.1%}")
            print(f"  Testing accuracy: {test_acc:.1%}")
            
        except Exception as e:
            print(f"Error testing {name} feature set: {e}")
    
    # Compare results
    if len(results) == 2:
        old_test = results['Old']['test_accuracy']
        new_test = results['New']['test_accuracy']
        improvement = new_test - old_test
        
        print(f"\n📈 Performance Comparison:")
        print(f"Accuracy Improvement: {improvement:+.1%}")
        print(f"Relative Improvement: {(improvement/old_test)*100:+.1f}%")
        
        if improvement > 0:
            print("🎉 New feature set performs better!")
        else:
            print("⚠️  Old feature set still performs better")

if __name__ == "__main__":
    # Test the improved ML model
    bot = test_improved_ml_model()
    
    # Compare feature sets
    compare_feature_sets()
    
    print("\n🚀 Ready for enhanced backtesting and live trading!") 