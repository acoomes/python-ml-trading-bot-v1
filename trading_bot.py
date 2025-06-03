print('[DEBUG] trading_bot.py loaded (top of file)')

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from coinbase.rest import RESTClient
from coinbase.rest import RESTClient as AdvancedRESTClient
from coinbase.rest import RESTClient as RESTClient
import random
import signal
import sys
import atexit
from sklearn.model_selection import RandomizedSearchCV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Try to import coinbase, handle gracefully if not available
try:
    from coinbase.rest import RESTClient
    COINBASE_AVAILABLE = True
except ImportError:
    print("Coinbase Advanced Trading API not available. Live trading disabled.")
    COINBASE_AVAILABLE = False
    RESTClient = None

class TradingBot:
    def __init__(self, config_file='config.json'):
        """Initialize the trading bot with configuration."""
        self.config = self.load_config(config_file)
        self._setup_logging()

        # Initialize trading state first
        self.portfolio_value = self.config['trading']['starting_portfolio_amount']
        self.initial_portfolio_value = self.portfolio_value
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.cooldown_until = None

        # Position state management
        self.position_state_file = 'current_position.json'
        self.current_position = None
        self.monitoring_active = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Register cleanup function
        atexit.register(self._cleanup_on_exit)
        
        # Setup Coinbase API client if credentials are provided
        cb_cfg = self.config.get('coinbase', {})
        self.cb_client = None
        self.live_trading = cb_cfg.get('enabled', False)
        
        # Load API credentials from cdp_api_key.json
        try:
            # Use the key_file parameter with the new PEM-formatted key
            self.cb_client = RESTClient(key_file='cdp_api_key.json')
            
            # Test the connection by getting account information
            accounts = self.cb_client.get_accounts()
            if accounts:
                logging.info("Successfully connected to Coinbase Advanced Trading API")
                # Enable live trading for testing with configurable risk
                self.live_trading = True
                risk_pct = self.config['trading']['risk_per_trade_percent']
                logging.warning(f"LIVE TRADING ENABLED: Using {risk_pct}% risk per trade (~${self.portfolio_value * risk_pct/100:.0f} trades)")
            else:
                logging.warning("Connected to API but no accounts found")
                self.live_trading = False
        except Exception as e:
            logging.error(f"Failed to initialize Coinbase client: {str(e)}")
            self.cb_client = None
            self.live_trading = False
        
        # Initialize ML model
        self.model = RandomForestClassifier(
            n_estimators=200,           # Increased from 100
            max_depth=15,               # Prevent overfitting
            min_samples_split=5,        # Prevent overfitting
            min_samples_leaf=2,         # Prevent overfitting
            max_features='sqrt',        # Feature subsampling
            bootstrap=True,             # Bootstrap sampling
            random_state=42,
            n_jobs=-1                   # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.last_retraining = datetime.now()
        self.total_trades = 0
        self.next_retraining_trade = 1  # Start with retraining after first trade
        
        # Feature importance tracking
        self.feature_importance_history = []
        self.current_features = None
        
        # Initialize model with historical data
        self._initialize_model()
        
        # Trading history
        self.trade_history: List[Dict] = []
        
        # Always clear retraining_history.csv on bot creation
        retraining_file = 'retraining_history.csv'
        retraining_header = [
            'timestamp', 'trigger_trade_timestamp', 'total_trades', 'next_retraining_trade', 'model_accuracy',
            'win_rate', 'avg_risk_units', 'total_risk_units', 'portfolio_rpnl', 'ml_threshold',
            'profit_target', 'max_trade_duration', 'portfolio_value'
        ]
        try:
            pd.DataFrame(columns=retraining_header).to_csv(retraining_file, index=False)
            print(f"[DEBUG] Cleared {os.path.abspath(retraining_file)} at bot initialization.")
        except Exception as e:
            print(f"[ERROR] Could not clear {retraining_file}: {e}")
        
        # Check for existing position on startup
        self._check_existing_position()
        
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            level=getattr(logging, self.config['logging']['alert_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _fetch_market_data(self, symbol: str, interval: str = '1d', 
                          lookback_days: int = 100) -> pd.DataFrame:
        """Fetch market data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            return df
        except Exception as e:
            self._send_alert(f"Error fetching market data: {str(e)}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data and engineer features."""
        df = data.copy()
        if df.empty:
            print("No data available for this symbol and date range.")
            return None
        
        # Add traditional technical indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Add EMA indicators (new)
        df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # Add momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        
        # Add volatility indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']  # Normalized BB width
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Add price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']  # Price relative to SMA20
        df['Price_vs_EMA9'] = (df['Close'] - df['EMA_9']) / df['EMA_9']     # Price relative to EMA9
        
        # Add trend indicators
        df['SMA_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)  # SMA crossover signal
        df['EMA_Cross'] = (df['EMA_9'] > df['EMA_50']).astype(int)   # EMA crossover signal
        
        # Add volatility-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']  # Intraday volatility
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()  # Simple volume moving average
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']  # Volume relative to average
        
        # Add target variable (multi-class: strong down, down, up, strong up)
        future_returns = df['Returns'].shift(-1)
        
        # Create categorical target, handling NaN values
        df['Target'] = pd.cut(future_returns, 
                             bins=[-np.inf, -0.02, 0, 0.02, np.inf], 
                             labels=[0, 1, 2, 3])
        
        # Convert to int, but keep NaN as NaN initially
        df['Target'] = pd.to_numeric(df['Target'], errors='coerce')
        
        # For backward compatibility, also keep binary target
        df['Target_Binary'] = (future_returns > 0).astype(float)  # Use float to handle NaN
        
        # Drop NaN values at the end
        return df.dropna()
    
    def _calculate_risk_units(self, trade_outcome: Dict) -> float:
        """Calculate risk units for a trade outcome.
        
        Risk units = (Net PnL) / (Risk Amount)
        Example: If $100 was risked and $150 was gained, that's +1.5 risk units
        """
        print("\n=== Risk Units Calculation ===")
        
        # Get values from trade outcome dictionary
        net_pnl = float(trade_outcome.get('net_pnl', 0.0))
        risk_amount = float(trade_outcome.get('risk_amount', 0.0))
        
        # Try to get risk amount from trade details if not found in main dictionary
        if risk_amount <= 0 and 'trade_details' in trade_outcome:
            risk_amount = float(trade_outcome['trade_details'].get('risk_amount', 0.0))
        
        print(f"Net PnL: ${net_pnl:.2f}")
        print(f"Risk Amount: ${risk_amount:.2f}")
        
        if risk_amount <= 0:
            print("Warning: Risk amount is 0 or negative, returning 0 risk units")
            return 0.0
        
        # Calculate risk units
        risk_units = net_pnl / risk_amount
        print(f"Calculated Risk Units: {risk_units:.2f}")
        
        # Store the risk units in the trade_outcome dictionary
        trade_outcome['risk_units'] = float(risk_units)
        
        # Also store in the trade details if they exist
        if 'trade_details' in trade_outcome:
            trade_outcome['trade_details']['risk_units'] = float(risk_units)
        
        print(f"Final Risk Units: {risk_units:.2f}")
        return float(risk_units)

    def _calculate_next_retraining_trade(self) -> int:
        """Calculate the next trade number at which to retrain the model.
        
        Schedule:
        - First 5 trades: Retrain after each trade (trades 1-5)
        - After that: Increment interval by 1 every 5 trades
        Example: 1,2,3,4,5,7,9,11,13,15,18,21,24,27,30,34,38,42,46,50,...
        """
        if self.total_trades < 5:
            return self.total_trades + 1
        
        # Calculate which group of 5 trades we're in after the initial 5
        group = (self.total_trades - 5) // 5
        
        # Calculate the interval for this group
        interval = group + 1
        
        # Calculate the next retraining point
        last_retraining = self.total_trades
        while last_retraining < self.total_trades + interval:
            last_retraining += interval
        
        return last_retraining

    def _ml_feedback_loop(self, trade_outcome: Dict):
        """Update ML model based on trade outcome."""
        try:
            # Get values from trade outcome
            net_pnl = float(trade_outcome.get('net_pnl', 0.0))
            risk_amount = float(trade_outcome.get('risk_amount', 0.0))
            risk_units = float(trade_outcome.get('risk_units', 0.0))  # Get risk units from trade outcome
            trade_timestamp = trade_outcome.get('exit_timestamp') or trade_outcome.get('timestamp')
            
            print(f"\nML Feedback Loop Debug:")
            print(f"Net PnL: ${net_pnl:.2f}")
            print(f"Risk Amount: ${risk_amount:.2f}")
            print(f"Risk Units: {risk_units:.2f}")
            
            # Store trade outcome for retraining
            trade_details = {
                'timestamp': datetime.now(),
                'entry_timestamp': trade_outcome.get('entry_timestamp'),
                'exit_timestamp': trade_outcome.get('exit_timestamp'),
                'entry_price': trade_outcome.get('entry_price'),
                'exit_price': trade_outcome.get('exit_price'),
                'position_size': trade_outcome.get('position_size'),
                'risk_amount': float(risk_amount),  # Ensure risk amount is stored as float
                'net_pnl': net_pnl,
                'fees': trade_outcome.get('fees'),
                'slippage': trade_outcome.get('slippage'),
                'risk_reward_ratio': trade_outcome.get('risk_reward_ratio'),
                'risk_units': risk_units,  # Store risk units from trade outcome
                'current_portfolio_value': self.portfolio_value,
                'exit_reason': trade_outcome.get('exit_reason')
            }
            
            print("\nTrade Details Debug:")
            print(f"Risk Amount: ${trade_details['risk_amount']:.2f}")
            print(f"Net PnL: ${trade_details['net_pnl']:.2f}")
            print(f"Risk Units: {trade_details['risk_units']:.2f}")
            
            # Store in trade history
            self.trade_history.append(trade_details)
            
            # Increment total trades
            self.total_trades += 1
            
            # Check if it's time to retrain (progressive schedule)
            if self.total_trades == self.next_retraining_trade:
                self._retrain_model(trigger_trade_timestamp=trade_timestamp)
                self.next_retraining_trade = self._calculate_next_retraining_trade()
                print(f"Next retraining scheduled at trade {self.next_retraining_trade}")
        except Exception as e:
            self._send_alert(f"Error in ML feedback loop: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self, recent_trades: List[Dict]) -> Dict:
        """Calculate various performance metrics for recent trades."""
        if not recent_trades:
            return {
                'win_rate': 0.0,
                'avg_risk_units': 0.0,
                'total_risk_units': 0.0,
                'portfolio_rpnl': 0.0,
                'total_pnl': 0.0
            }
        
        print("\nPerformance Metrics Calculation:")
        print(f"Number of recent trades: {len(recent_trades)}")
        
        # Initialize metrics
        total_pnl = 0.0
        risk_units_list = []
        winning_trades = 0
        
        # Process each trade
        for i, trade in enumerate(recent_trades):
            # Get trade values
            net_pnl = float(trade.get('net_pnl', 0.0))
            risk_amount = float(trade.get('risk_amount', 0.0))
            
            # Update total PnL
            total_pnl += net_pnl
            
            # Count winning trades
            if net_pnl > 0:
                winning_trades += 1
            
            # Calculate risk units
            trade_risk_units = net_pnl / risk_amount if risk_amount > 0 else 0.0
            risk_units_list.append(trade_risk_units)
            
            print(f"\nTrade {i+1} Details:")
            print(f"  Net PnL: ${net_pnl:.2f}")
            print(f"  Risk Amount: ${risk_amount:.2f}")
            print(f"  Risk Units: {trade_risk_units:.2f}")
        
        # Calculate aggregate metrics
        win_rate = winning_trades / len(recent_trades)
        avg_risk_units = sum(risk_units_list) / len(risk_units_list)
        total_risk_units = sum(risk_units_list)
        portfolio_rpnl = recent_trades[-1]['current_portfolio_value'] - recent_trades[0]['current_portfolio_value']
        
        print(f"\nAggregate Metrics:")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Risk Units: {avg_risk_units:.2f}")
        print(f"Total Risk Units: {total_risk_units:.2f}")
        print(f"Portfolio R&PL: ${portfolio_rpnl:.2f}")
        
        return {
            'win_rate': win_rate,
            'avg_risk_units': avg_risk_units,
            'total_risk_units': total_risk_units,
            'portfolio_rpnl': portfolio_rpnl,
            'total_pnl': total_pnl
        }

    def _retrain_model(self, trigger_trade_timestamp=None):
        """Retrain the ML model with accumulated trade history. Optionally log the triggering trade's timestamp."""
        try:
            print("\n=== Retraining ML Model ===")
            logging.info("\n=== Model Retraining Started ===")
            
            # Log retraining trigger
            logging.info(f"Retraining triggered at trade {self.total_trades}")
            logging.info(f"Next scheduled retraining at trade {self.next_retraining_trade}")
            
            # Fetch recent market data
            market_data = self._fetch_market_data(
                self.config['trading']['symbol'],
                lookback_days=365  # Use 1 year of data for retraining
            )
            logging.info(f"Fetched {len(market_data)} days of market data for retraining")
            
            processed_data = self._preprocess_data(market_data)
            logging.info("Data preprocessing completed")
            
            # Get optimized features
            features = self._get_optimized_features()
            
            X = processed_data[features].values
            y = processed_data['Target_Binary'].values  # Use binary target for now
            logging.info(f"Prepared {len(features)} optimized features for retraining")
            
            # Adjust strategy based on trade history
            metrics = {'win_rate': 0, 'avg_risk_units': 0, 'total_risk_units': 0, 'portfolio_rpnl': 0}
            if len(self.trade_history) > 0:
                # Calculate performance metrics for recent trades
                recent_trades = self.trade_history[-min(50, len(self.trade_history)):]
                metrics = self._calculate_performance_metrics(recent_trades)
                
                # Log performance metrics
                perf_log = (
                    f"\nRecent Performance Metrics:\n"
                    f"Win Rate: {metrics['win_rate']:.2%}\n"
                    f"Average Risk Units: {metrics['avg_risk_units']:.2f}\n"
                    f"Total Risk Units: {metrics['total_risk_units']:.2f}\n"
                    f"Portfolio R&PL: ${metrics['portfolio_rpnl']:.2f}"
                )
                logging.info(perf_log)
                
                # Log parameter adjustments
                old_threshold = self.config['ml_model']['prediction_threshold']
                old_profit_target = self.config['trading']['profit_target_percent']
                old_duration = self.config['trading']['max_trade_duration_hours']
                
                if metrics['total_risk_units'] < -1.0:
                    logging.info("Poor risk-adjusted performance detected, adjusting parameters conservatively")
                    new_threshold = min(max(old_threshold * 1.05, 0.08), 0.25)  # Less aggressive threshold increases
                    print(f"[DEBUG] Conservative: threshold {old_threshold:.2f} -> {new_threshold:.2f}")
                    self.config['ml_model']['prediction_threshold'] = new_threshold
                    self.config['trading']['profit_target_percent'] = max(2.0, self.config['trading']['profit_target_percent'] * 0.95)  # Smaller adjustments
                    self.config['trading']['max_trade_duration_hours'] = max(8, self.config['trading']['max_trade_duration_hours'] * 0.9)  # Longer minimum duration
                elif metrics['total_risk_units'] > 1.0:
                    logging.info("Good risk-adjusted performance detected, optimizing parameters aggressively")
                    new_threshold = min(max(old_threshold * 0.98, 0.08), 0.25)  # Smaller threshold decreases
                    print(f"[DEBUG] Aggressive: threshold {old_threshold:.2f} -> {new_threshold:.2f}")
                    self.config['ml_model']['prediction_threshold'] = new_threshold
                    # Always adjust profit target for good performance
                    self.config['trading']['profit_target_percent'] = min(8.0, self.config['trading']['profit_target_percent'] * 1.05)  # Smaller increases
                    # Also increase trade duration for good performance
                    self.config['trading']['max_trade_duration_hours'] = min(24.0, self.config['trading']['max_trade_duration_hours'] * 1.05)
                else:
                    logging.info("Moderate performance, making minor parameter adjustments")
                    # Make small adjustments for moderate performance
                    if metrics['win_rate'] > 0.55:  # Slightly above break-even
                        new_threshold = min(max(old_threshold * 0.99, 0.08), 0.25)
                        self.config['trading']['profit_target_percent'] = min(8.0, self.config['trading']['profit_target_percent'] * 1.01)
                        self.config['trading']['max_trade_duration_hours'] = min(24.0, self.config['trading']['max_trade_duration_hours'] * 1.01)
                        print(f"[DEBUG] Moderate positive: slight optimization")
                    elif metrics['win_rate'] < 0.45:  # Below break-even
                        new_threshold = min(max(old_threshold * 1.01, 0.08), 0.25)
                        self.config['trading']['profit_target_percent'] = max(2.0, self.config['trading']['profit_target_percent'] * 0.99)
                        self.config['trading']['max_trade_duration_hours'] = max(8, self.config['trading']['max_trade_duration_hours'] * 0.99)
                        print(f"[DEBUG] Moderate negative: slight conservation")
                    else:
                        new_threshold = old_threshold  # No change
                        print(f"[DEBUG] Neutral performance: no adjustments")
                    self.config['ml_model']['prediction_threshold'] = new_threshold
                
                # Log parameter changes
                param_log = (
                    f"\nParameter Adjustments:\n"
                    f"ML Threshold: {old_threshold:.2%} -> {self.config['ml_model']['prediction_threshold']:.2%}\n"
                    f"Profit Target: {old_profit_target:.1f}% -> {self.config['trading']['profit_target_percent']:.1f}%\n"
                    f"Max Duration: {old_duration:.1f}h -> {self.config['trading']['max_trade_duration_hours']:.1f}h"
                )
                logging.info(param_log)
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            logging.info("Feature scaling completed")
            
            # Train model
            self.model.fit(X_scaled, y)
            logging.info("Model retraining completed")
            
            # Analyze feature importance
            importance_analysis = self._analyze_feature_importance(X_scaled, y, features)
            
            # Save feature importance analysis
            self._save_feature_importance_analysis()
            
            # Update last retraining timestamp
            self.last_retraining = datetime.now()
            
            # Log retraining results
            model_score = self.model.score(X_scaled, y)
            
            # Create detailed retraining log
            retrain_log = (
                f"\n=== Model Retraining Complete ===\n"
                f"Time: {self.last_retraining}\n"
                f"Total Trades: {self.total_trades}\n"
                f"Next Retraining at Trade: {self.next_retraining_trade}\n"
                f"Model Accuracy: {model_score:.2%}\n"
                f"Training Data Period: {market_data.index[0]} to {market_data.index[-1]}\n"
                f"Number of Training Samples: {len(X)}\n"
                f"Current Portfolio Value: ${self.portfolio_value:.2f}\n"
                f"Adjusted Parameters:\n"
                f"  ML Prediction Threshold: {self.config['ml_model']['prediction_threshold']:.2%}\n"
                f"  Profit Target: {self.config['trading']['profit_target_percent']:.1f}%\n"
                f"  Max Trade Duration: {self.config['trading']['max_trade_duration_hours']:.1f} hours\n"
                f"{'='*50}"
            )
            logging.info(retrain_log)
            
            # Save retraining record to CSV
            retraining_record = {
                'timestamp': self.last_retraining,
                'trigger_trade_timestamp': trigger_trade_timestamp,
                'total_trades': self.total_trades,
                'next_retraining_trade': self.next_retraining_trade,
                'model_accuracy': model_score,
                'win_rate': metrics['win_rate'],
                'avg_risk_units': metrics['avg_risk_units'],
                'total_risk_units': metrics['total_risk_units'],
                'portfolio_rpnl': metrics['portfolio_rpnl'],
                'ml_threshold': self.config['ml_model']['prediction_threshold'],
                'profit_target': self.config['trading']['profit_target_percent'],
                'max_trade_duration': self.config['trading']['max_trade_duration_hours'],
                'portfolio_value': self.portfolio_value
            }
            retraining_file = 'retraining_history.csv'
            retraining_header = [
                'timestamp', 'trigger_trade_timestamp', 'total_trades', 'next_retraining_trade', 'model_accuracy',
                'win_rate', 'avg_risk_units', 'total_risk_units', 'portfolio_rpnl', 'ml_threshold',
                'profit_target', 'max_trade_duration', 'portfolio_value'
            ]
            if not os.path.exists(retraining_file) or os.path.getsize(retraining_file) == 0:
                pd.DataFrame([retraining_record], columns=retraining_header).to_csv(retraining_file, index=False)
            else:
                pd.DataFrame([retraining_record], columns=retraining_header).to_csv(retraining_file, mode='a', header=False, index=False)
            logging.info(f"Retraining record saved to {retraining_file}")
            
        except Exception as e:
            error_msg = f"Error retraining model: {str(e)}"
            logging.error(error_msg)
            self._send_alert(error_msg)
            raise

    def _initialize_model(self):
        """Initialize and train the ML model with historical data."""
        try:
            print("\n=== Initializing ML Model ===")
            logging.info("\n=== Model Initialization Started ===")
            
            # Fetch historical data
            market_data = self._fetch_market_data(
                self.config['trading']['symbol'],
                lookback_days=365  # Use 1 year of data for initial training
            )
            logging.info(f"Fetched {len(market_data)} days of historical data")
            
            # Preprocess data
            processed_data = self._preprocess_data(market_data)
            if processed_data is None:
                error_msg = f"No data available for {self.config['trading']['symbol']}. Please check the symbol and date range."
                logging.error(error_msg)
                self._send_alert(error_msg)
                raise ValueError(error_msg)
            
            logging.info("Data preprocessing completed")
            
            # Get optimized features for initial training
            features = self._get_optimized_features()
            
            X = processed_data[features].values
            y = processed_data['Target_Binary'].values  # Use binary target for now
            logging.info(f"Prepared {len(features)} optimized features for training")
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            logging.info("Feature scaling completed")
            
            # Train model
            self.model.fit(X_scaled, y)
            logging.info("Model training completed")
            
            # Analyze initial feature importance
            initial_importance = self._analyze_feature_importance(X_scaled, y, features)
            
            # Log initialization results
            model_score = self.model.score(X_scaled, y)
            print(f"Model initialized successfully. Accuracy: {model_score:.2%}")
            
            # Create detailed initialization log
            init_log = (
                f"\n=== Model Initialization Complete ===\n"
                f"Time: {datetime.now()}\n"
                f"Training Data Period: {market_data.index[0]} to {market_data.index[-1]}\n"
                f"Number of Training Samples: {len(X)}\n"
                f"Features Used: {len(features)} optimized features\n"
                f"Top 5 Features: {initial_importance.get('top_5_features', [])}\n"
                f"Model Accuracy: {model_score:.2%}\n"
                f"Initial Portfolio Value: ${self.portfolio_value:.2f}\n"
                f"ML Prediction Threshold: {self.config['ml_model']['prediction_threshold']:.2%}\n"
                f"Risk Per Trade: {self.config['trading']['risk_per_trade_percent']:.1f}%\n"
                f"Profit Target: {self.config['trading']['profit_target_percent']:.1f}%\n"
                f"Max Trade Duration: {self.config['trading']['max_trade_duration_hours']:.1f} hours\n"
                f"{'='*50}"
            )
            logging.info(init_log)
            
            self._send_alert("ML model initialized successfully")
        except Exception as e:
            error_msg = f"Error initializing ML model: {str(e)}"
            logging.error(error_msg)
            self._send_alert(error_msg)
            raise
    
    def _make_ml_prediction(self, data: pd.DataFrame) -> Tuple[float, bool]:
        """Make ML prediction with confidence score and additional market filters."""
        try:
            if self.model is None:
                self._initialize_model()
            
            # Use the optimized features that the model was actually trained with
            features_to_use = self.current_features if self.current_features else self._get_optimized_features()
            
            # Prepare features (ensure we have the right number)
            feature_data = data[features_to_use].iloc[-1:].values
            
            # Scale the features
            feature_data_scaled = self.scaler.transform(feature_data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(feature_data_scaled)[0]
            confidence = max(prediction_proba) - min(prediction_proba)
            prediction = prediction_proba[1] > 0.5  # Assuming binary classification
            
            # Additional market state filters for better entry conditions
            latest_data = data.iloc[-1]
            
            # Filter 1: Trend alignment (EMA and SMA agreement)
            ema9_above_ema50 = latest_data.get('EMA_9', 0) > latest_data.get('EMA_50', 0)
            price_above_sma20 = latest_data.get('Price_vs_SMA20', 0) > 0
            trend_aligned = ema9_above_ema50 and price_above_sma20
            
            # Filter 2: Momentum indicators
            rsi = latest_data.get('RSI', 50)
            rsi_favorable = 30 < rsi < 70  # Not oversold/overbought
            
            # Filter 3: Volatility filter (avoid very low volatility periods)
            bb_width = latest_data.get('BB_width', 0)
            sufficient_volatility = bb_width > 0.02  # Minimum 2% width
            
            # Filter 4: Volume confirmation
            volume_ratio = latest_data.get('Volume_Ratio', 1.0)
            volume_confirmation = volume_ratio > 0.8  # At least 80% of average
            
            # Apply tiered quality filters for enhanced profitability
            if prediction and confidence > self.config['ml_model']['prediction_threshold']:
                # Count how many filters pass
                filters_passed = sum([trend_aligned, rsi_favorable, sufficient_volatility, volume_confirmation])
                
                # Implement tiered entry quality system
                if filters_passed == 4:
                    # Perfect setup - all filters pass
                    setup_quality = "PERFECT"
                    quality_multiplier = 1.5  # Increase profit targets and position size
                    self._last_setup_quality = setup_quality  # Store for position sizing
                    print(f"🌟 PERFECT SETUP: {filters_passed}/4 filters passed")
                    print(f"   Trend Aligned: {trend_aligned} | RSI Favorable: {rsi_favorable}")
                    print(f"   Sufficient Vol: {sufficient_volatility} | Volume Confirm: {volume_confirmation}")
                    print(f"   Quality Multiplier: {quality_multiplier:.1f}x")
                    return confidence * quality_multiplier, prediction
                elif filters_passed == 3:
                    # Good setup - most filters pass
                    setup_quality = "GOOD"
                    quality_multiplier = 1.2  # Slight increase for good setups
                    self._last_setup_quality = setup_quality  # Store for position sizing
                    print(f"✅ GOOD SETUP: {filters_passed}/4 filters passed")
                    print(f"   Trend Aligned: {trend_aligned} | RSI Favorable: {rsi_favorable}")
                    print(f"   Sufficient Vol: {sufficient_volatility} | Volume Confirm: {volume_confirmation}")
                    print(f"   Quality Multiplier: {quality_multiplier:.1f}x")
                    return confidence * quality_multiplier, prediction
                elif filters_passed == 2:
                    # Marginal setup - allow with reduced size
                    setup_quality = "MARGINAL"
                    quality_multiplier = 0.8  # Reduce position size for marginal setups
                    self._last_setup_quality = setup_quality  # Store for position sizing
                    print(f"⚠️ MARGINAL SETUP: {filters_passed}/4 filters passed")
                    print(f"   Trend Aligned: {trend_aligned} | RSI Favorable: {rsi_favorable}")
                    print(f"   Sufficient Vol: {sufficient_volatility} | Volume Confirm: {volume_confirmation}")
                    print(f"   Quality Multiplier: {quality_multiplier:.1f}x")
                    return confidence * quality_multiplier, prediction
                else:
                    # Poor setup - reject
                    setup_quality = "REJECTED"
                    self._last_setup_quality = setup_quality  # Store for position sizing
                    print(f"❌ SETUP REJECTED: Only {filters_passed}/4 filters passed")
                    print(f"   Trend Aligned: {trend_aligned} | RSI Favorable: {rsi_favorable}")
                    print(f"   Sufficient Vol: {sufficient_volatility} | Volume Confirm: {volume_confirmation}")
                    return confidence * 0.3, False  # Significantly reduce confidence for poor setups
            
            return confidence, prediction
            
        except Exception as e:
            logging.error(f"Error in ML prediction: {str(e)}")
            return 0.0, False
    
    def _calculate_trade_parameters(self, current_price: float, 
                                  confidence: float) -> Dict:
        """Calculate position size, stop loss, and take profit with dynamic volatility-based targets."""
        # Calculate optimal position sizing using Kelly Criterion
        def calculate_kelly_position_size(confidence: float, historical_trades: List[Dict]) -> float:
            """Calculate optimal position size using Kelly Criterion."""
            try:
                if len(historical_trades) < 10:
                    # Not enough history, use conservative base sizing
                    return self.config['trading']['risk_per_trade_percent']
                
                # Analyze recent trade performance
                recent_trades = historical_trades[-20:]  # Last 20 trades
                wins = [t for t in recent_trades if t.get('net_pnl', 0) > 0]
                losses = [t for t in recent_trades if t.get('net_pnl', 0) <= 0]
                
                if not wins or not losses:
                    return self.config['trading']['risk_per_trade_percent']
                
                win_prob = len(wins) / len(recent_trades)
                
                # Calculate average win and loss amounts (as percentage of risk)
                avg_win_risk_units = sum(t.get('risk_units', 0) for t in wins) / len(wins)
                avg_loss_risk_units = abs(sum(t.get('risk_units', 0) for t in losses) / len(losses))
                
                if avg_loss_risk_units == 0:
                    return self.config['trading']['risk_per_trade_percent']
                
                # Kelly formula: f = (bp - q) / b
                # where: b = odds received (avg_win / avg_loss), p = win_prob, q = 1 - p
                b = avg_win_risk_units / avg_loss_risk_units  # Risk/reward ratio
                p = win_prob
                q = 1 - p
                
                kelly_fraction = (b * p - q) / b
                
                # Apply Kelly with safety multiplier and confidence scaling
                safety_multiplier = 0.5  # Use half Kelly for safety (common practice)
                confidence_scaling = min(2.0, confidence * 5)  # Scale by confidence
                
                optimal_risk_pct = kelly_fraction * safety_multiplier * confidence_scaling * 100
                
                # Clamp between reasonable bounds
                base_risk = self.config['trading']['risk_per_trade_percent']
                min_risk = base_risk * 0.5  # 2.5% minimum
                max_risk = base_risk * 3.0  # 15% maximum for great setups
                
                optimal_risk_pct = max(min_risk, min(max_risk, optimal_risk_pct))
                
                print(f"\nKelly Position Sizing:")
                print(f"Win Probability: {win_prob:.1%}")
                print(f"Avg Win R:R: {avg_win_risk_units:.2f}")
                print(f"Avg Loss R:R: {avg_loss_risk_units:.2f}")
                print(f"Risk/Reward Ratio: {b:.2f}")
                print(f"Raw Kelly: {kelly_fraction:.2%}")
                print(f"Confidence Scaled: {optimal_risk_pct:.1f}%")
                
                return optimal_risk_pct
                
            except Exception as e:
                print(f"Error calculating Kelly position size: {e}")
                return self.config['trading']['risk_per_trade_percent']
        
        # Calculate risk per trade using Kelly Criterion
        kelly_risk_percent = calculate_kelly_position_size(confidence, self.trade_history)
        max_risk_amount = self.portfolio_value * (kelly_risk_percent / 100)
        
        # Apply confidence scaling to final position size with setup quality adjustment
        setup_quality_multiplier = 1.0
        if hasattr(self, '_last_setup_quality'):
            if self._last_setup_quality == "PERFECT":
                setup_quality_multiplier = 1.0  # Full size for perfect setups
            elif self._last_setup_quality == "GOOD":
                setup_quality_multiplier = 0.7  # Reduced size for good setups
            elif self._last_setup_quality == "MARGINAL":
                setup_quality_multiplier = 0.4  # Much smaller size for marginal setups
        
        actual_risk_amount = max_risk_amount * confidence * setup_quality_multiplier
        
        print(f"\nDynamic Position Sizing:")
        print(f"Base Risk Amount: ${max_risk_amount:.2f}")
        print(f"Confidence Scaling: {confidence:.1%}")
        print(f"Setup Quality: {getattr(self, '_last_setup_quality', 'UNKNOWN')}")
        print(f"Quality Multiplier: {setup_quality_multiplier:.1f}x")
        print(f"Final Risk Amount: ${actual_risk_amount:.2f}")
        
        # Add minimum expected value filter
        def check_minimum_expected_value(historical_trades: List[Dict], confidence: float) -> bool:
            """Check if trade has positive expected value based on historical performance."""
            try:
                # Be very lenient in early trading - allow first 15 trades regardless
                if len(historical_trades) < 15:
                    print(f"\nEarly Trading Mode: {len(historical_trades)}/15 trades completed - EV filter disabled")
                    return True
                
                # Calculate expected value based on recent performance
                recent_trades = historical_trades[-20:]  # Last 20 trades
                wins = [t for t in recent_trades if t.get('net_pnl', 0) > 0]
                losses = [t for t in recent_trades if t.get('net_pnl', 0) <= 0]
                
                if not wins or not losses:
                    print("Missing win or loss data - allowing trade")
                    return True  # Allow if we don't have both wins and losses
                
                win_prob = len(wins) / len(recent_trades)
                avg_win_risk_units = sum(t.get('risk_units', 0) for t in wins) / len(wins)
                avg_loss_risk_units = abs(sum(t.get('risk_units', 0) for t in losses) / len(losses))
                
                # Calculate expected value per trade
                expected_value = (win_prob * avg_win_risk_units) - ((1 - win_prob) * avg_loss_risk_units)
                
                # Scale expected value by confidence (higher confidence should have better EV)
                confidence_adjusted_ev = expected_value * (confidence * 2)  # Further reduced scaling
                
                print(f"\nExpected Value Analysis:")
                print(f"Win Probability: {win_prob:.1%}")
                print(f"Avg Win R:R: {avg_win_risk_units:.2f}")
                print(f"Avg Loss R:R: {avg_loss_risk_units:.2f}")
                print(f"Raw Expected Value: {expected_value:.3f}")
                print(f"Confidence Adjusted EV: {confidence_adjusted_ev:.3f}")
                
                # Very lenient threshold and multiple escape hatches
                required_ev = -0.02  # Allow even slightly negative EV (was 0.01)
                print(f"Minimum Required EV: {required_ev}")
                
                # Escape hatch 1: High win rate (>50%) overrides EV concerns
                if win_prob > 0.5:
                    print("High win rate (>50%) - allowing trade despite EV")
                    return True
                
                # Escape hatch 2: Very high confidence overrides EV concerns
                if confidence > 0.25:
                    print("Very high confidence (>25%) - allowing trade despite EV")
                    return True
                
                # Escape hatch 3: If we haven't traded in a while, reset and allow
                if len(historical_trades) >= 20:
                    # Check if we've been too conservative (few recent trades)
                    recent_10_days = historical_trades[-10:]  # Even more recent
                    if len(recent_10_days) < 3:  # Less than 3 trades in last 10 attempts
                        print("Trading frequency too low - resetting EV requirements")
                        return True
                
                # Escape hatch 4: Perfect setups always allowed
                if hasattr(self, '_last_setup_quality') and self._last_setup_quality == "PERFECT":
                    print("Perfect setup detected - allowing trade regardless of EV")
                    return True
                
                # Main EV check (very lenient now)
                result = confidence_adjusted_ev > required_ev
                if not result:
                    print(f"EV check failed: {confidence_adjusted_ev:.3f} <= {required_ev}")
                else:
                    print(f"EV check passed: {confidence_adjusted_ev:.3f} > {required_ev}")
                
                return result
                
            except Exception as e:
                print(f"Error checking expected value: {e}")
                return True  # Allow trade if calculation fails
        
        # Check if trade meets minimum expected value requirement
        if not check_minimum_expected_value(self.trade_history, confidence):
            print("❌ TRADE REJECTED: Negative expected value")
            raise ValueError("Trade rejected due to negative expected value")
        
        # Calculate volatility-based profit target
        def calculate_dynamic_profit_target(market_data: pd.DataFrame) -> float:
            """Calculate profit target based on recent volatility and ML confidence."""
            try:
                # Get recent price data for volatility calculation
                recent_returns = market_data['Close'].pct_change().dropna()
                
                # Calculate various volatility measures
                daily_volatility = recent_returns.tail(20).std()  # 20-day volatility
                weekly_volatility = recent_returns.tail(5).std()   # 5-day (weekly) volatility
                
                # Calculate volatility percentiles to understand current market state
                rolling_vol = recent_returns.rolling(window=20).std().dropna()
                current_vol_percentile = (rolling_vol.tail(1).iloc[0] > rolling_vol.tail(20)).mean() * 100
                
                # Base profit target from config
                base_target = self.config['trading']['profit_target_percent']
                
                # Enhanced asymmetric targeting based on confidence and volatility
                confidence_multiplier = 1.0
                if confidence > 0.20:  # Very high confidence trades
                    confidence_multiplier = 1.15  # Reduced from 1.2 to 1.15
                elif confidence > 0.17:  # High confidence trades
                    confidence_multiplier = 1.1   # Reduced from 1.15 to 1.1
                elif confidence > 0.15:  # Medium confidence trades
                    confidence_multiplier = 1.05  # Reduced from 1.1 to 1.05
                else:  # Low confidence trades
                    confidence_multiplier = 1.02  # Reduced from 1.05 to 1.02
                
                # Adjust based on volatility conditions with better scaling
                if daily_volatility > 0.04:  # High volatility (>4% daily moves)
                    volatility_multiplier = 1.15  # Reduced from 1.2 to 1.15
                    target = base_target * volatility_multiplier * confidence_multiplier
                elif daily_volatility > 0.025:  # Medium volatility (2.5-4% daily moves)
                    volatility_multiplier = 1.1   # Reduced from 1.3 to 1.1
                    target = base_target * volatility_multiplier * confidence_multiplier
                elif daily_volatility < 0.015:  # Low volatility (<1.5% daily moves)
                    volatility_multiplier = 1.05  # Kept at 1.05
                    target = base_target * volatility_multiplier * confidence_multiplier
                else:  # Normal volatility
                    volatility_multiplier = 1.08  # Reduced from 1.2 to 1.08
                    target = base_target * volatility_multiplier * confidence_multiplier
                
                # Ensure minimum viable R:R ratio (must be at least 2:1 to overcome fees and losses)
                min_target_for_viability = 1.2  # Reduced from 1.5% to 1.2% - ultra achievable
                target = max(min_target_for_viability, min(2.5, target))  # Reduced max from 3.0% to 2.5% - micro-scalping approach
                
                print(f"\nAsymmetric Profit Target Calculation:")
                print(f"Daily Volatility: {daily_volatility:.3f} ({daily_volatility*100:.1f}%)")
                print(f"ML Confidence: {confidence:.1%}")
                print(f"Base Target: {base_target:.1f}%")
                print(f"Confidence Multiplier: {confidence_multiplier:.1f}x")
                print(f"Volatility Multiplier: {volatility_multiplier:.1f}x")
                print(f"Final Asymmetric Target: {target:.1f}%")
                print(f"Minimum Viable Target: {min_target_for_viability:.1f}%")
                
                return target
                
            except Exception as e:
                print(f"Error calculating dynamic profit target: {e}")
                return max(4.0, self.config['trading']['profit_target_percent'])  # Ensure minimum 4%
        
        # Get market data for volatility calculation
        try:
            market_data = self._fetch_market_data(self.config['trading']['symbol'])
            dynamic_profit_target = calculate_dynamic_profit_target(market_data)
        except Exception as e:
            print(f"Could not fetch market data for volatility calculation: {e}")
            dynamic_profit_target = self.config['trading']['profit_target_percent']
        
        # Calculate stop loss (use dynamic percentage based on volatility)
        try:
            # Get market volatility for dynamic stop loss
            historical_returns = market_data['Close'].pct_change().dropna()
            daily_volatility = historical_returns.tail(20).std()
            
            # Dynamic stop loss based on volatility (2-3x daily volatility, min 5%, max 12%)
            stop_loss_percent = max(5.0, min(12.0, daily_volatility * 200))  # 2x daily vol as %
            stop_loss = current_price * (1 - stop_loss_percent / 100)
            
            print(f"Dynamic Stop Loss: {stop_loss_percent:.1f}% (based on {daily_volatility*100:.1f}% daily volatility)")
        except Exception as e:
            print(f"Error calculating dynamic stop loss: {e}")
            stop_loss_percent = 8.0  # Fallback to 8% stop loss
            stop_loss = current_price * (1 - stop_loss_percent / 100)
        
        # Calculate take profit using dynamic target
        take_profit = current_price * (1 + dynamic_profit_target / 100)
        
        # Calculate position size based on risk amount and stop loss distance
        risk_per_share = current_price - stop_loss
        if risk_per_share <= 0:
            raise ValueError("Invalid stop loss - too close to current price")
        
        position_size = actual_risk_amount / risk_per_share
        
        print(f"\nTrade Parameters Calculation:")
        print(f"Current Price: ${current_price:.4f}")
        print(f"ML Confidence: {confidence:.1%}")
        print(f"Max Risk Amount: ${max_risk_amount:.2f}")
        print(f"Actual Risk (confidence scaled): ${actual_risk_amount:.2f}")
        print(f"Dynamic Profit Target: {dynamic_profit_target:.1f}%")
        print(f"Position Size: {position_size:.4f} {self.config['trading']['symbol'].split('-')[0]}")
        print(f"Stop Loss: ${stop_loss:.4f} (-{stop_loss_percent:.1f}%)")
        print(f"Take Profit: ${take_profit:.4f} (+{dynamic_profit_target:.1f}%)")
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': actual_risk_amount,
            'confidence': confidence,
            'dynamic_profit_target': dynamic_profit_target  # Store for logging
        }
    
    def _execute_trade(self, symbol: str, order_type: str, 
                      quantity: float, price: Optional[float] = None,
                      is_mock: bool = True) -> Dict:
        """Execute a trade (mock implementation)."""
        print(f"\n[DEBUG] _execute_trade called: symbol={symbol}, order_type={order_type}, quantity={quantity:.4f}, price={price}, is_mock={is_mock}")
        
        if is_mock:
            # Simulate trade execution
            execution_price = float(price if price else self._fetch_market_data(symbol).iloc[-1]['Close'])
            
            # Apply realistic slippage simulation (market impact)
            if order_type.upper() == 'BUY':
                execution_price *= (1 + self.config['risk_management']['slippage_percent'] / 100)
            else:  # SELL
                execution_price *= (1 - self.config['risk_management']['slippage_percent'] / 100)
            
            slippage_amount = abs(execution_price - float(price if price else execution_price)) * quantity
            fees = float(execution_price * quantity * (self.config['risk_management']['transaction_fee_percent'] / 100))
            
            # Calculate risk amount for this trade
            risk_amount = float(self.portfolio_value * (self.config['trading']['risk_per_trade_percent'] / 100))
            
            print(f"\nExecute Trade Debug:")
            print(f"Order Type: {order_type}")
            print(f"Quantity: {quantity:.4f}")
            print(f"Execution Price: ${execution_price:.4f} (after slippage)")
            print(f"Slippage Amount: ${slippage_amount:.4f}")
            print(f"Fees: ${fees:.4f}")
            print(f"Risk Amount: ${risk_amount:.2f}")
            
            # Create trade info with explicit float conversions
            trade_info = {
                'execution_price': float(execution_price),
                'slippage': float(slippage_amount),
                'fees': float(fees),
                'timestamp': datetime.now(),
                'quantity': float(quantity),
                'risk_amount': float(risk_amount),
                'order_type': order_type,
                'trade_details': {
                    'risk_amount': float(risk_amount),
                    'quantity': float(quantity),
                    'execution_price': float(execution_price),
                    'slippage': float(slippage_amount),
                    'fees': float(fees),
                    'portfolio_value': float(self.portfolio_value),
                    'risk_percent': float(self.config['trading']['risk_per_trade_percent'])
                }
            }
            
            print(f"\nTrade Info Final Check:")
            print(f"Risk Amount: ${trade_info['risk_amount']:.2f}")
            print(f"Trade Details Risk Amount: ${trade_info['trade_details']['risk_amount']:.2f}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Risk Percent: {self.config['trading']['risk_per_trade_percent']:.1f}%")
            
            return trade_info
        else:
            print(f"\n[DEBUG] LIVE TRADING MODE - Attempting real order placement")
            if not self.cb_client:
                raise Exception("Coinbase client not configured")

            try:
                # Round price and quantity to Coinbase requirements
                # XRP-USD: price_increment = 0.0001, base_increment = 0.000001
                rounded_price = round(price, 4) if price is not None else None
                rounded_quantity = round(quantity, 6)
                
                print(f"[DEBUG] Original price: {price}, rounded: {rounded_price}")
                print(f"[DEBUG] Original quantity: {quantity}, rounded: {rounded_quantity}")
                print(f"[DEBUG] Preparing order: {order_type} {rounded_quantity} {symbol}")
                
                side = 'BUY' if order_type.upper() == 'BUY' else 'SELL'
                # Generate a unique client order ID using timestamp and random number
                client_order_id = f"{int(time.time())}_{side}_{rounded_quantity}"
                print(f"[DEBUG] Client Order ID: {client_order_id}")
                
                order_result = None
                if rounded_price is not None:
                    print(f"[DEBUG] Placing limit order at ${rounded_price:.4f}")
                    # Place limit order
                    order_result = self.cb_client.create_order(
                        product_id=symbol,
                        client_order_id=client_order_id,
                        side=side,
                        order_configuration={
                            'limit_limit_gtc': {
                                'base_size': str(rounded_quantity),
                                'limit_price': str(rounded_price)
                            }
                        }
                    )
                else:
                    print(f"[DEBUG] Placing market order")
                    # Place market order
                    order_result = self.cb_client.create_order(
                        product_id=symbol,
                        client_order_id=client_order_id,
                        side=side,
                        order_configuration={
                            'market_market_ioc': {
                                'base_size': str(rounded_quantity)
                            }
                        }
                    )

                print(f"[DEBUG] Order response received: {order_result}")

                exec_price = None
                if order_result and hasattr(order_result, 'average_filled_price') and order_result.average_filled_price:
                    exec_price = float(order_result.average_filled_price)
                elif rounded_price is not None:
                    exec_price = float(rounded_price)
                else:
                    # Fallback to current market price
                    current_data = self._fetch_market_data(symbol)
                    exec_price = float(current_data.iloc[-1]['Close'])

                fees = 0.0
                if order_result and hasattr(order_result, 'total_fees'):
                    fees = float(order_result.total_fees or 0)
                
                risk_amount = float(self.portfolio_value * (self.config['trading']['risk_per_trade_percent'] / 100))

                print(f"[DEBUG] Trade completed - exec_price: ${exec_price}, fees: ${fees}")

                return {
                    'execution_price': exec_price,
                    'slippage': 0.0,
                    'fees': fees,
                    'timestamp': datetime.now(),
                    'quantity': float(rounded_quantity),
                    'risk_amount': risk_amount,
                    'order_id': getattr(order_result, 'order_id', None) if order_result else None,
                    'order_type': order_type,
                    'trade_details': {
                        'risk_amount': risk_amount,
                        'quantity': float(rounded_quantity),
                        'execution_price': exec_price
                    }
                }
            except Exception as e:
                print(f"[DEBUG] Order placement failed: {e}")
                logging.error(f"Live trading order failed: {e}")
                
                # Instead of raising an exception and hanging, return a mock trade result
                print(f"[DEBUG] Falling back to mock execution due to order failure")
                fallback_price = price if price is not None else self._fetch_market_data(symbol).iloc[-1]['Close']
                risk_amount = float(self.portfolio_value * (self.config['trading']['risk_per_trade_percent'] / 100))
                
                return {
                    'execution_price': float(fallback_price),
                    'slippage': 0.0,
                    'fees': 0.0,
                    'timestamp': datetime.now(),
                    'quantity': float(quantity),
                    'risk_amount': risk_amount,
                    'order_id': None,
                    'order_type': order_type,
                    'failed_order': True,  # Flag to indicate this was a failed order
                    'trade_details': {
                        'risk_amount': risk_amount,
                        'quantity': float(quantity),
                        'execution_price': float(fallback_price)
                    }
                }
    
    def _evaluate_trade(self, entry_info: Dict, exit_info: Dict) -> Dict:
        """Evaluate a trade's outcome and calculate risk-adjusted metrics."""
        print("\n=== Trade Evaluation ===")
        
        # Get basic trade information
        entry_price = float(entry_info['execution_price'])
        exit_price = float(exit_info['execution_price'])
        quantity = float(entry_info['quantity'])
        
        # Get risk amount from entry info, fallback to exit info if not present
        risk_amount = float(entry_info.get('risk_amount', 0.0))
        if risk_amount <= 0:
            risk_amount = float(exit_info.get('risk_amount', 0.0))
        if risk_amount <= 0:
            # Try to get from trade details
            risk_amount = float(entry_info.get('trade_details', {}).get('risk_amount', 0.0))
        if risk_amount <= 0:
            risk_amount = float(exit_info.get('trade_details', {}).get('risk_amount', 0.0))
        if risk_amount <= 0:
            # Calculate risk amount from portfolio value and risk percent
            portfolio_value = float(entry_info.get('trade_details', {}).get('portfolio_value', self.portfolio_value))
            risk_percent = float(entry_info.get('trade_details', {}).get('risk_percent', self.config['trading']['risk_per_trade_percent']))
            risk_amount = float(portfolio_value * (risk_percent / 100))
        
        print(f"\nTrade Evaluation Details:")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Exit Price: ${exit_price:.2f}")
        print(f"Quantity: {quantity:.4f}")
        print(f"Risk Amount: ${risk_amount:.2f}")
        
        # Calculate PnL
        gross_pnl = float((exit_price - entry_price) * quantity)
        fees = float(abs(quantity * entry_price * self.config['risk_management']['transaction_fee_percent'] / 100))
        net_pnl = float(gross_pnl - fees)
        
        print(f"Gross PnL: ${gross_pnl:.2f}")
        print(f"Fees: ${fees:.2f}")
        print(f"Net PnL: ${net_pnl:.2f}")
        
        # Calculate risk/reward ratio and risk units
        risk_reward_ratio = 0.0
        risk_units = 0.0
        if risk_amount > 0:
            risk_reward_ratio = float(net_pnl / risk_amount)
            risk_units = float(net_pnl / risk_amount)
            print(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")
            print(f"Risk Units: {risk_units:.2f}")
        
        # Determine if trade was a win
        is_win = net_pnl > 0
        
        # Create evaluation dictionary
        evaluation = {
            'is_win': is_win,
            'gross_pnl': float(gross_pnl),
            'fees': float(fees),
            'net_pnl': float(net_pnl),
            'risk_amount': float(risk_amount),  # Ensure risk amount is stored as float
            'risk_reward_ratio': float(risk_reward_ratio),
            'risk_units': float(risk_units),  # Add risk units to evaluation
            'exit_reason': exit_info.get('exit_reason', 'unknown'),
            'entry_timestamp': entry_info['timestamp'],
            'exit_timestamp': exit_info['timestamp'],
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'quantity': float(quantity),
            'trade_details': {  # Add trade details for risk units calculation
                'risk_amount': float(risk_amount),
                'net_pnl': float(net_pnl),
                'risk_units': float(risk_units),
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'quantity': float(quantity),
                'gross_pnl': float(gross_pnl),
                'fees': float(fees),
                'portfolio_value': float(self.portfolio_value),
                'risk_percent': float(self.config['trading']['risk_per_trade_percent'])
            }
        }
        
        print("\nTrade Evaluation Results:")
        print(f"Risk Amount: ${evaluation['risk_amount']:.2f}")
        print(f"Net PnL: ${evaluation['net_pnl']:.2f}")
        print(f"Risk/Reward Ratio: {evaluation['risk_reward_ratio']:.2f}")
        print(f"Risk Units: {evaluation['risk_units']:.2f}")
        
        return evaluation
    
    def _update_portfolio_value(self, amount_won_lost: float):
        """Update portfolio value and check drawdown limits."""
        self.portfolio_value += amount_won_lost
        self._check_drawdown()
    
    def _check_drawdown(self):
        """Check if portfolio drawdown exceeds limit."""
        drawdown = (self.initial_portfolio_value - self.portfolio_value) / self.initial_portfolio_value * 100
        if drawdown > self.config['trading']['max_portfolio_drawdown_percent']:
            self._send_alert(f"Maximum drawdown limit reached: {drawdown:.2f}%")
            raise Exception("Maximum drawdown limit exceeded")
    
    def _check_cool_down(self) -> bool:
        """Check if bot should be in cooldown period."""
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return True
        
        if (self.consecutive_losses >= self.config['trading']['consecutive_losses_threshold'] or
            self.portfolio_value < self.initial_portfolio_value * 
            (1 - self.config['trading']['intraday_loss_threshold_percent']/100)):
            
            self.cooldown_until = datetime.now() + timedelta(
                hours=self.config['trading']['cooldown_period_hours'])
            self._send_alert("Entering cooldown period")
            return True
        
        return False
    
    def _log_trade(self, trade_details: Dict):
        print("[DEBUG] Entering _log_trade (top of function)...")
        # Ensure numeric values are float
        net_pnl = float(trade_details.get('net_pnl', 0.0))
        risk_amount = float(trade_details.get('risk_amount', 0.0))
        risk_units = float(trade_details.get('risk_units', 0.0))  # Get risk units from trade details
        
        print(f"[DEBUG] Trade details received:")
        print(f"  Net PnL: ${net_pnl:.2f}")
        print(f"  Risk Amount: ${risk_amount:.2f}")
        print(f"  Risk Units: {risk_units:.2f}")
        
        # Update trade details
        trade_details['net_pnl'] = net_pnl
        trade_details['risk_amount'] = risk_amount
        trade_details['risk_units'] = risk_units
        
        # Log trade
        log_message = (
            f"\nTrade Executed:\n"
            f"Entry Time: {trade_details['entry_timestamp']}\n"
            f"Exit Time: {trade_details['exit_timestamp']}\n"
            f"Entry Price: ${trade_details['entry_price']:.2f}\n"
            f"Exit Price: ${trade_details['exit_price']:.2f}\n"
            f"Position Size: {trade_details['position_size']:.4f}\n"
            f"Risk Amount: ${risk_amount:.2f}\n"
            f"Net PnL: ${net_pnl:.2f}\n"
            f"Fees: ${trade_details['fees']:.2f}\n"
            f"Slippage: ${trade_details['slippage']:.2f}\n"
            f"Risk/Reward Ratio: {trade_details['risk_reward_ratio']:.2f}\n"
            f"Risk Units: {risk_units:.2f}\n"  # Ensure risk units are included in log message
            f"Portfolio Value: ${trade_details['current_portfolio_value']:.2f}\n"
            f"Exit Reason: {trade_details['exit_reason']}\n"
            f"{'='*50}"
        )
        print(f"[DEBUG] Writing trade log with risk units: {risk_units:.2f}")
        logging.info(log_message)
        
        # Also write to trades.log file
        try:
            with open('trades.log', 'a') as f:
                f.write(log_message + '\n')
            print(f"[DEBUG] Wrote trade to trades.log with risk units: {risk_units:.2f}")
        except Exception as e:
            print(f"[ERROR] Failed to write to trades.log: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_alert(self, message: str):
        """Send alert message."""
        logging.warning(message)
        print(f"ALERT: {message}")
    
    def run_daily_cycle(self):
        print("[DEBUG] Entering run_daily_cycle (top of function)...")
        try:
            print("[DEBUG] Entering run_daily_cycle try block...")
            print("\n=== Starting Daily Trading Cycle ===")
            print(f"Current Portfolio Value: ${self.portfolio_value:.2f}")
            
            # Check cooldown
            if self._check_cool_down():
                print("Bot is in cooldown period. No trading today.")
                logging.info("\n=== No Trade - Cooldown Period ===\n" + "="*50)
                return
            
            # Fetch and preprocess market data
            print("\nFetching market data...")
            market_data = self._fetch_market_data(self.config['trading']['symbol'])
            processed_data = self._preprocess_data(market_data)
            print(f"Latest price: ${market_data.iloc[-1]['Close']:.2f}")
            
            trades_today = 0
            max_trades = self.config['trading'].get('max_trades_per_day', 1)
            min_qty = self.config['trading'].get('min_trade_quantity', 0.0001)
            
            while trades_today < max_trades:
                # Generate trading signal
                print(f"\nGenerating trading signal for trade {trades_today+1}...")
                
                # Detect market regime first
                market_regime = self._detect_market_regime(processed_data)
                
                # Get base ML prediction
                base_confidence, prediction = self._make_ml_prediction(processed_data)
                
                # Adapt strategy based on market regime
                adapted_confidence, profit_target_multiplier = self._adapt_strategy_to_regime(market_regime, base_confidence)
                
                print(f"Base ML Confidence: {base_confidence:.2%}")
                print(f"Regime-Adapted Confidence: {adapted_confidence:.2%}")
                
                if adapted_confidence >= self.config['ml_model']['prediction_threshold']:
                    print("\n=== Trade Setup ===")
                    current_price = float(market_data.iloc[-1]['Close'])
                    
                    # Calculate trade parameters
                    trade_params = self._calculate_trade_parameters(current_price, adapted_confidence)
                    risk_amount = float(trade_params['risk_amount'])
                    
                    print("\nTrade Parameters Debug:")
                    print(f"Portfolio Value: ${self.portfolio_value:.2f}")
                    print(f"Risk Per Trade %: {self.config['trading']['risk_per_trade_percent']:.1f}%")
                    print(f"Calculated Risk Amount: ${risk_amount:.2f}")
                    
                    # Enforce minimum trade quantity
                    position_size = max(round(float(trade_params['position_size']), 4), 0.0)
                    if position_size < min_qty:
                        print(f"Trade size {position_size} is below minimum {min_qty}, skipping.")
                        break
                    
                    print(f"Entry Price: ${current_price:.2f}")
                    print(f"Position Size: {position_size:.4f} units")
                    print(f"Stop Loss: ${trade_params['stop_loss']:.2f}")
                    print(f"Take Profit: ${trade_params['take_profit']:.2f}")
                    print(f"Risk Amount: ${risk_amount:.2f}")
                    
                    # Execute entry
                    print("\nExecuting entry trade...")
                    entry_info = self._execute_trade(
                        self.config['trading']['symbol'],
                        'BUY',
                        position_size,
                        current_price,
                        is_mock=not self.live_trading
                    )
                    
                    # Ensure risk amount is set in entry info
                    entry_info['quantity'] = float(position_size)
                    entry_info['risk_amount'] = float(risk_amount)
                    entry_info['trade_details'] = {
                        'risk_amount': float(risk_amount),
                        'quantity': float(position_size),
                        'execution_price': float(current_price)
                    }
                    
                    print("\nEntry Trade Debug:")
                    print(f"Entry Info Risk Amount: ${entry_info['risk_amount']:.2f}")
                    print(f"Entry Trade Details Risk Amount: ${entry_info['trade_details']['risk_amount']:.2f}")
                    
                    # Save position state immediately after entry
                    self._save_position_state(entry_info, trade_params)
                    self.current_position = {
                        'entry_timestamp': entry_info['timestamp'].isoformat(),
                        'symbol': self.config['trading']['symbol'],
                        'position_size': float(entry_info['quantity']),
                        'entry_price': float(entry_info['execution_price']),
                        'stop_loss': float(trade_params['stop_loss']),
                        'take_profit': float(trade_params['take_profit']),
                        'risk_amount': float(entry_info['risk_amount']),
                        'portfolio_value': float(self.portfolio_value),
                        'is_live_trading': self.live_trading
                    }
                    
                    # Monitor trade
                    print("\n=== Monitoring Trade ===")
                    start_time = datetime.now()
                    
                    if not self.live_trading:
                        # Mock mode: simulate a trade outcome
                        print("Mock mode: Simulating trade outcome...")
                        import random
                        
                        # Simulate some time passing (1-30 minutes)
                        simulated_duration_minutes = random.randint(1, 30)
                        print(f"Simulating {simulated_duration_minutes} minutes of trading...")
                        
                        # Simulate price movement (random walk)
                        price_change_percent = random.uniform(-0.05, 0.05)  # -5% to +5%
                        current_price = float(current_price * (1 + price_change_percent))
                        
                        # Determine exit reason based on simulated price
                        if current_price <= float(trade_params['stop_loss']):
                            exit_reason = "Stop Loss"
                            print(f"\nSimulated Stop Loss triggered at ${current_price:.2f}")
                        elif current_price >= float(trade_params['take_profit']):
                            exit_reason = "Take Profit"
                            print(f"\nSimulated Take Profit triggered at ${current_price:.2f}")
                        else:
                            exit_reason = "Time Exit"
                            print(f"\nSimulated Time Exit at ${current_price:.2f}")
                    else:
                        # Live trading mode: use new position monitoring system
                        self.monitoring_active = True
                        exit_reason = None
                        
                        try:
                            print(f"Live monitoring started. Stop Loss: ${trade_params['stop_loss']:.4f}, Take Profit: ${trade_params['take_profit']:.4f}")
                            print(f"Max trade duration: {self.config['trading']['max_trade_duration_hours']} hours")
                            
                            while self.monitoring_active:
                                # Get real-time price using yfinance with 1-minute intervals
                                try:
                                    # Use 1-minute intervals for live monitoring (not daily)
                                    current_data = self._fetch_market_data(
                                        self.config['trading']['symbol'], 
                                        interval='1m',  # 1-minute data for real-time updates
                                        lookback_days=1  # Only need recent data for monitoring
                                    )
                                    current_price = float(current_data.iloc[-1]['Close'])
                                except Exception as e:
                                    print(f"   Warning: Could not fetch live price ({e}), using last known price")
                                    # Keep using last known current_price
                                
                                # Calculate time elapsed
                                elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                                remaining_hours = self.config['trading']['max_trade_duration_hours'] - elapsed_hours
                                
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring: Price=${current_price:.4f} | "
                                      f"SL=${trade_params['stop_loss']:.4f} | TP=${trade_params['take_profit']:.4f} | "
                                      f"Time Left: {remaining_hours:.1f}h")
                                
                                # Check stop loss and take profit
                                if current_price <= float(trade_params['stop_loss']):
                                    print(f"\n🔴 STOP LOSS TRIGGERED at ${current_price:.4f}!")
                                    exit_reason = "Stop Loss"
                                    self.monitoring_active = False
                                    break
                                elif current_price >= float(trade_params['take_profit']):
                                    print(f"\n🟢 TAKE PROFIT TRIGGERED at ${current_price:.4f}!")
                                    exit_reason = "Take Profit"
                                    self.monitoring_active = False
                                    break
                                elif elapsed_hours >= self.config['trading']['max_trade_duration_hours']:
                                    print(f"\n⏰ MAXIMUM TRADE DURATION REACHED ({elapsed_hours:.1f}h)!")
                                    exit_reason = "Time Exit"
                                    self.monitoring_active = False
                                    break
                                else:
                                    print("   Status: Monitoring... (checking again in 60 seconds)")
                                    time.sleep(60)  # Check every minute
                                    continue
                        except KeyboardInterrupt:
                            print("\n[SIGNAL] Monitoring interrupted by user. Position remains open.")
                            return  # Exit without completing trade - position state preserved
                        except Exception as e:
                            print(f"[ERROR] Monitoring error: {e}")
                            # On critical error, attempt emergency close
                            self._emergency_close_position(self.current_position)
                            return
                    
                    # Execute exit
                    print(f"Executing exit trade at ${current_price:.2f}...")
                    exit_info = self._execute_trade(
                        self.config['trading']['symbol'],
                        'SELL',
                        position_size,
                        current_price,
                        is_mock=not self.live_trading
                    )
                    
                    # Ensure risk amount is set in exit info
                    exit_info['quantity'] = float(position_size)
                    exit_info['risk_amount'] = float(risk_amount)
                    exit_info['exit_reason'] = exit_reason
                    exit_info['trade_details'] = {
                        'risk_amount': float(risk_amount),
                        'quantity': float(position_size),
                        'execution_price': float(current_price)
                    }
                    
                    print("\nExit Trade Debug:")
                    print(f"Exit Info Risk Amount: ${exit_info['risk_amount']:.2f}")
                    print(f"Exit Trade Details Risk Amount: ${exit_info['trade_details']['risk_amount']:.2f}")
                    
                    # Evaluate trade
                    trade_evaluation = self._evaluate_trade(entry_info, exit_info)
                    
                    print("\nTrade Evaluation Debug:")
                    print(f"Entry Info Risk Amount: ${entry_info['risk_amount']:.2f}")
                    print(f"Exit Info Risk Amount: ${exit_info['risk_amount']:.2f}")
                    print(f"Trade Evaluation Risk Amount: ${trade_evaluation['risk_amount']:.2f}")
                    print(f"Net PnL: ${trade_evaluation['net_pnl']:.2f}")
                    
                    # Add additional trade information
                    trade_evaluation.update({
                        'entry_timestamp': entry_info['timestamp'],
                        'exit_timestamp': exit_info['timestamp'],
                        'entry_price': float(entry_info['execution_price']),
                        'exit_price': float(exit_info['execution_price']),
                        'position_size': float(position_size),
                        'risk_amount': float(risk_amount),  # Ensure risk amount is preserved
                        'slippage': float(entry_info.get('slippage', 0.0)) + float(exit_info.get('slippage', 0.0)),
                        'trade_details': {
                            'risk_amount': float(risk_amount),
                            'position_size': float(position_size),
                            'entry_price': float(entry_info['execution_price']),
                            'exit_price': float(exit_info['execution_price']),
                            'net_pnl': float(trade_evaluation['net_pnl'])
                        }
                    })
                    
                    # Calculate risk units
                    risk_units = self._calculate_risk_units(trade_evaluation)
                    print(f"\nRisk Units Calculation:")
                    print(f"Net PnL: ${trade_evaluation['net_pnl']:.2f}")
                    print(f"Risk Amount: ${trade_evaluation['risk_amount']:.2f}")
                    print(f"Risk Units: {risk_units:.2f}")
                    
                    # Log trade
                    trade_details = {
                        'entry_timestamp': entry_info['timestamp'],
                        'exit_timestamp': exit_info['timestamp'],
                        'entry_price': float(entry_info['execution_price']),
                        'exit_price': float(exit_info['execution_price']),
                        'position_size': float(position_size),
                        'risk_amount': float(risk_amount),
                        'net_pnl': float(trade_evaluation['net_pnl']),
                        'fees': float(trade_evaluation['fees']),
                        'slippage': float(trade_evaluation['slippage']),
                        'risk_reward_ratio': float(trade_evaluation['risk_reward_ratio']),
                        'risk_units': float(trade_evaluation['risk_units']),  # Get risk units directly from evaluation
                        'current_portfolio_value': float(self.portfolio_value),
                        'exit_reason': trade_evaluation['exit_reason']
                    }
                    
                    print("\nFinal Trade Details Debug:")
                    print(f"Risk Amount: ${trade_details['risk_amount']:.2f}")
                    print(f"Net PnL: ${trade_details['net_pnl']:.2f}")
                    print(f"Risk Units: {trade_details['risk_units']:.2f}")
                    
                    self._log_trade(trade_details)
                    
                    # Update portfolio and ML model
                    self._update_portfolio_value(float(trade_evaluation['net_pnl']))
                    self._ml_feedback_loop(trade_evaluation)
                    
                    # Clear position state after successful completion
                    self._clear_position_state()
                    self.current_position = None
                    self.monitoring_active = False
                    
                    trades_today += 1
                    break
                else:
                    print("\nNo trade taken - ML confidence below threshold")
                    print(f"Required confidence: {self.config['ml_model']['prediction_threshold']:.2%}")
                    print(f"Current confidence: {adapted_confidence:.2%}")
                    break
        except Exception as e:
            print(f"[ERROR] Exception in run_daily_cycle: {e}")
            import traceback
            traceback.print_exc()
            self._send_alert(f"Error in daily cycle: {str(e)}")
            raise
        finally:
            print("[DEBUG] Entering finally block of run_daily_cycle...")
            try:
                pass  # Removed backtest summary logging, now handled in backtest.py
            except Exception as e:
                print(f"[ERROR] Exception in finally block of run_daily_cycle: {e}")
                import traceback
                traceback.print_exc()

    def plot_retraining_history(self):
        """Plot retraining history and performance metrics."""
        try:
            # Read retraining history
            retraining_file = 'retraining_history.csv'
            if not os.path.exists(retraining_file):
                print("No retraining history found.")
                return
            
            df = pd.read_csv(retraining_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            fig.suptitle('Model Retraining History', fontsize=16)
            
            # Plot 1: Portfolio Value and Model Accuracy
            ax1 = axes[0]
            ax1.plot(df['timestamp'], df['portfolio_value'], 'b-', label='Portfolio Value')
            ax1.set_ylabel('Portfolio Value ($)', color='b')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df['timestamp'], df['model_accuracy'], 'r--', label='Model Accuracy')
            ax1_twin.set_ylabel('Model Accuracy', color='r')
            
            ax1.set_title('Portfolio Value and Model Accuracy Over Time')
            ax1.grid(True)
            
            # Plot 2: Performance Metrics
            ax2 = axes[1]
            ax2.plot(df['timestamp'], df['win_rate'], 'g-', label='Win Rate')
            ax2.plot(df['timestamp'], df['avg_risk_units'], 'b--', label='Avg Risk Units')
            ax2.plot(df['timestamp'], df['total_risk_units'], 'r:', label='Total Risk Units')
            ax2.set_ylabel('Metrics')
            ax2.set_title('Performance Metrics Over Time')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Trading Parameters
            ax3 = axes[2]
            ax3.plot(df['timestamp'], df['ml_threshold'], 'g-', label='ML Threshold')
            ax3.plot(df['timestamp'], df['profit_target'], 'b--', label='Profit Target')
            ax3.plot(df['timestamp'], df['max_trade_duration'], 'r:', label='Max Duration')
            ax3.set_ylabel('Parameter Values')
            ax3.set_title('Trading Parameters Over Time')
            ax3.legend()
            ax3.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('retraining_history.png')
            print("\nRetraining history plot saved to 'retraining_history.png'")
            
        except Exception as e:
            print(f"Error plotting retraining history: {str(e)}")

    def plot_portfolio_with_retraining(self, portfolio_history_file='portfolio_history.csv', retraining_history_file='retraining_history.csv'):
        """Plot portfolio value over time and overlay retraining events."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        if not os.path.exists(portfolio_history_file):
            print(f"No portfolio history found at {portfolio_history_file}.")
            return
        df = pd.read_csv(portfolio_history_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(14, 7))
        plt.plot(df['timestamp'], df['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')

        # Overlay retraining points
        if os.path.exists(retraining_history_file):
            retrain_df = pd.read_csv(retraining_history_file)
            retrain_df['timestamp'] = pd.to_datetime(retrain_df['timestamp'])
            for idx, row in retrain_df.iterrows():
                plt.axvline(row['timestamp'], color='red', linestyle='--', alpha=0.6)
                plt.scatter(row['timestamp'], row['portfolio_value'], color='red', zorder=5)
                plt.annotate(f"Retrain\nT{int(row['total_trades'])}\nAcc {row['model_accuracy']:.2%}",
                             (row['timestamp'], row['portfolio_value']),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
        else:
            print(f"No retraining history found at {retraining_history_file}.")

        plt.legend()
        plt.tight_layout()
        plt.savefig('backtest_results_with_retraining.png')
        print("Portfolio value plot with retraining points saved to 'backtest_results_with_retraining.png'")

    def plot_backtest_with_retraining(self, trade_file='backtest_results.csv', retrain_file='retraining_history.csv'):
        """Plot portfolio value from backtest_results.csv and overlay retraining events from retraining_history.csv using trigger_trade_timestamp."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        if not os.path.exists(trade_file):
            print(f"No trade history found at {trade_file}.")
            return
        trades = pd.read_csv(trade_file)
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])

        plt.figure(figsize=(14, 7))
        plt.plot(trades['timestamp'], trades['current_portfolio_value'], label='Portfolio Value', color='blue')
        plt.title('Portfolio Value Over Time (with Retraining Events)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')

        # Overlay retraining points using trigger_trade_timestamp
        if os.path.exists(retrain_file):
            retrain_df = pd.read_csv(retrain_file)
            # Use trigger_trade_timestamp if available, else fallback to timestamp
            if 'trigger_trade_timestamp' in retrain_df.columns:
                retrain_df['event_time'] = pd.to_datetime(retrain_df['trigger_trade_timestamp'])
            else:
                retrain_df['event_time'] = pd.to_datetime(retrain_df['timestamp'])
            for idx, row in retrain_df.iterrows():
                if pd.isnull(row['event_time']):
                    continue
                plt.axvline(row['event_time'], color='red', linestyle='--', alpha=0.6)
                plt.scatter(row['event_time'], row['portfolio_value'], color='red', zorder=5)
                plt.annotate(f"Retrain\nT{int(row['total_trades'])}\nAcc {row['model_accuracy']:.2%}",
                             (row['event_time'], row['portfolio_value']),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
        else:
            print(f"No retraining history found at {retrain_file}.")

        plt.legend()
        plt.tight_layout()
        plt.savefig('backtest_results_with_retraining.png')
        print("Backtest results plot with retraining points saved to 'backtest_results_with_retraining.png'")

    def start_new_backtest(self):
        """Prepare for a new backtest: reset retraining history and any other stateful logs."""
        import pandas as pd
        import os
        retraining_file = 'retraining_history.csv'
        retraining_header = [
            'timestamp', 'trigger_trade_timestamp', 'total_trades', 'next_retraining_trade', 'model_accuracy',
            'win_rate', 'avg_risk_units', 'total_risk_units', 'portfolio_rpnl', 'ml_threshold',
            'profit_target', 'max_trade_duration', 'portfolio_value'
        ]
        try:
            pd.DataFrame(columns=retraining_header).to_csv(retraining_file, index=False)
            print(f"[DEBUG] Cleared {os.path.abspath(retraining_file)} at start of backtest.")
        except Exception as e:
            print(f"[ERROR] Could not clear {retraining_file}: {e}")
        self.trade_history = []
        self.total_trades = 0
        self.next_retraining_trade = 1
        self.last_retraining = None
        print('[DEBUG] start_new_backtest() called.')

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n[SIGNAL] Received signal {signum}. Initiating graceful shutdown...")
            self._emergency_exit()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _check_existing_position(self):
        """Check for existing position on bot startup and offer recovery options."""
        if os.path.exists(self.position_state_file):
            try:
                with open(self.position_state_file, 'r') as f:
                    position_data = json.load(f)
                
                print(f"\n🚨 EXISTING POSITION DETECTED! 🚨")
                print(f"Entry Time: {position_data['entry_timestamp']}")
                print(f"Symbol: {position_data['symbol']}")
                print(f"Position Size: {position_data['position_size']:.4f}")
                print(f"Entry Price: ${position_data['entry_price']:.4f}")
                print(f"Stop Loss: ${position_data['stop_loss']:.4f}")
                print(f"Take Profit: ${position_data['take_profit']:.4f}")
                print(f"Risk Amount: ${position_data['risk_amount']:.2f}")
                
                # Calculate time elapsed
                entry_time = datetime.fromisoformat(position_data['entry_timestamp'])
                elapsed_hours = (datetime.now() - entry_time).total_seconds() / 3600
                print(f"Time Elapsed: {elapsed_hours:.2f} hours")
                
                if self.live_trading:
                    choice = input("\nOptions:\n1) Resume monitoring\n2) Close position immediately\n3) Ignore (dangerous!)\nChoice (1/2/3): ")
                    
                    if choice == '1':
                        self.current_position = position_data
                        print("✅ Resuming position monitoring...")
                        self._resume_position_monitoring()
                    elif choice == '2':
                        print("🔴 Closing position immediately...")
                        self._emergency_close_position(position_data)
                    else:
                        print("⚠️  WARNING: Ignoring existing position. This is dangerous!")
                        os.remove(self.position_state_file)
                else:
                    print("📝 Mock mode detected. Clearing position file.")
                    os.remove(self.position_state_file)
                    
            except Exception as e:
                print(f"[ERROR] Failed to read position file: {e}")
                print("🗑️  Removing corrupted position file.")
                os.remove(self.position_state_file)
    
    def _save_position_state(self, entry_info: Dict, trade_params: Dict):
        """Save current position state to file."""
        position_data = {
            'entry_timestamp': entry_info['timestamp'].isoformat(),
            'symbol': self.config['trading']['symbol'],
            'position_size': float(entry_info['quantity']),
            'entry_price': float(entry_info['execution_price']),
            'stop_loss': float(trade_params['stop_loss']),
            'take_profit': float(trade_params['take_profit']),
            'risk_amount': float(entry_info['risk_amount']),
            'portfolio_value': float(self.portfolio_value),
            'is_live_trading': self.live_trading
        }
        
        try:
            with open(self.position_state_file, 'w') as f:
                json.dump(position_data, f, indent=2)
            print(f"💾 Position state saved to {self.position_state_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save position state: {e}")
    
    def _clear_position_state(self):
        """Clear position state file after successful trade completion."""
        try:
            if os.path.exists(self.position_state_file):
                os.remove(self.position_state_file)
                print("🗑️  Position state cleared.")
        except Exception as e:
            print(f"[ERROR] Failed to clear position state: {e}")
    
    def _emergency_close_position(self, position_data: Dict = None):
        """Emergency close current position."""
        if position_data is None:
            position_data = self.current_position
            
        if position_data is None:
            print("ℹ️  No position to close.")
            return
        
        try:
            print(f"\n🚨 EMERGENCY POSITION CLOSE 🚨")
            
            # Get current market price
            current_data = self._fetch_market_data(position_data['symbol'])
            current_price = float(current_data.iloc[-1]['Close'])
            
            print(f"Current Price: ${current_price:.4f}")
            print(f"Position Size: {position_data['position_size']:.4f}")
            
            # Execute emergency sell
            exit_info = self._execute_trade(
                position_data['symbol'],
                'SELL',
                position_data['position_size'],
                current_price,
                is_mock=not self.live_trading
            )
            
            # Check if the order actually succeeded
            if exit_info.get('failed_order', False):
                # Order failed - position is still open on Coinbase!
                print(f"\n❌ EMERGENCY EXIT ORDER FAILED!")
                print(f"⚠️  POSITION REMAINS OPEN ON COINBASE")
                print(f"💾 Keeping position saved in current_position.json")
                
                failure_log = (
                    f"\n❌ EMERGENCY EXIT FAILED ❌\n"
                    f"Time: {datetime.now()}\n"
                    f"Entry Time: {position_data['entry_timestamp']}\n"
                    f"Entry Price: ${position_data['entry_price']:.4f}\n"
                    f"Attempted Exit Price: ${current_price:.4f}\n"
                    f"Position Size: {position_data['position_size']:.4f}\n"
                    f"ERROR: Position remains OPEN on Coinbase!\n"
                    f"ACTION REQUIRED: Manual close or bot restart needed\n"
                    f"{'='*50}"
                )
                logging.error(failure_log)
                print(failure_log)
                
                # DO NOT clear position state - keep it saved!
                # DO NOT update portfolio value - position is still open!
                print("🚨 Position state preserved - manual intervention required!")
                return
            
            # Order succeeded - proceed with normal exit processing
            entry_price = position_data['entry_price']
            gross_pnl = (current_price - entry_price) * position_data['position_size']
            fees = abs(position_data['position_size'] * entry_price * self.config['trading']['fee_percent'] / 100)
            net_pnl = gross_pnl - fees
            
            print(f"Emergency Exit Summary:")
            print(f"Entry Price: ${entry_price:.4f}")
            print(f"Exit Price: ${current_price:.4f}")
            print(f"Gross P&L: ${gross_pnl:.2f}")
            print(f"Fees: ${fees:.2f}")
            print(f"Net P&L: ${net_pnl:.2f}")
            
            # Update portfolio value only if order succeeded
            self.portfolio_value += net_pnl
            
            # Log successful emergency exit
            emergency_log = (
                f"\n✅ EMERGENCY POSITION CLOSED ✅\n"
                f"Time: {datetime.now()}\n"
                f"Entry Time: {position_data['entry_timestamp']}\n"
                f"Entry Price: ${entry_price:.4f}\n"
                f"Exit Price: ${current_price:.4f}\n"
                f"Position Size: {position_data['position_size']:.4f}\n"
                f"Net P&L: ${net_pnl:.2f}\n"
                f"Portfolio Value: ${self.portfolio_value:.2f}\n"
                f"Exit Reason: Emergency Close\n"
                f"{'='*50}"
            )
            logging.warning(emergency_log)
            
            # Clear position state only if order succeeded
            self._clear_position_state()
            self.current_position = None
            
            print("✅ Emergency close completed.")
            
        except Exception as e:
            # Any exception means position is likely still open
            print(f"❌ Emergency close failed: {e}")
            print(f"⚠️  POSITION REMAINS OPEN ON COINBASE")
            print(f"💾 Keeping position saved in current_position.json")
            
            failure_log = (
                f"\n❌ EMERGENCY EXIT EXCEPTION ❌\n"
                f"Time: {datetime.now()}\n"
                f"Error: {str(e)}\n"
                f"Position remains OPEN on Coinbase!\n"
                f"ACTION REQUIRED: Manual close or bot restart needed\n"
                f"{'='*50}"
            )
            logging.error(failure_log)
            # DO NOT clear position state on exception!
    
    def _resume_position_monitoring(self):
        """Resume monitoring an existing position."""
        if self.current_position is None:
            print("[ERROR] No position to resume monitoring.")
            return
        
        position_data = self.current_position
        entry_time = datetime.fromisoformat(position_data['entry_timestamp'])
        
        print(f"\n🔄 Resuming position monitoring...")
        print(f"Stop Loss: ${position_data['stop_loss']:.4f}")
        print(f"Take Profit: ${position_data['take_profit']:.4f}")
        
        # Continue monitoring with existing parameters
        self.monitoring_active = True
        start_time = entry_time  # Use original entry time for duration calculation
        
        while self.monitoring_active:
            try:
                # Get current price
                current_data = self._fetch_market_data(
                    position_data['symbol'], 
                    interval='1m',
                    lookback_days=1
                )
                current_price = float(current_data.iloc[-1]['Close'])
                
                # Calculate time elapsed from original entry
                elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                remaining_hours = self.config['trading']['max_trade_duration_hours'] - elapsed_hours
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring: Price=${current_price:.4f} | "
                      f"SL=${position_data['stop_loss']:.4f} | TP=${position_data['take_profit']:.4f} | "
                      f"Time Left: {remaining_hours:.1f}h")
                
                # Check exit conditions
                if current_price <= position_data['stop_loss']:
                    print(f"\n🔴 STOP LOSS TRIGGERED at ${current_price:.4f}!")
                    self._complete_position_exit(position_data, current_price, "Stop Loss")
                    break
                elif current_price >= position_data['take_profit']:
                    print(f"\n🟢 TAKE PROFIT TRIGGERED at ${current_price:.4f}!")
                    self._complete_position_exit(position_data, current_price, "Take Profit")
                    break
                elif elapsed_hours >= self.config['trading']['max_trade_duration_hours']:
                    print(f"\n⏰ MAXIMUM TRADE DURATION REACHED ({elapsed_hours:.1f}h)!")
                    self._complete_position_exit(position_data, current_price, "Time Exit")
                    break
                else:
                    print("   Status: Monitoring... (checking again in 60 seconds)")
                    time.sleep(60)
                    continue
                    
            except KeyboardInterrupt:
                print("\n[SIGNAL] Monitoring interrupted. Position remains open.")
                break
            except Exception as e:
                print(f"[ERROR] Monitoring error: {e}")
                time.sleep(60)
                continue
    
    def _complete_position_exit(self, position_data: Dict, exit_price: float, exit_reason: str):
        """Complete position exit and cleanup."""
        try:
            # Execute sell order
            exit_info = self._execute_trade(
                position_data['symbol'],
                'SELL',
                position_data['position_size'],
                exit_price,
                is_mock=not self.live_trading
            )
            
            # Check if the order actually succeeded
            if exit_info.get('failed_order', False):
                # Order failed - position is still open on Coinbase!
                print(f"\n❌ EXIT ORDER FAILED!")
                print(f"⚠️  POSITION REMAINS OPEN ON COINBASE")
                print(f"💾 Keeping position saved for recovery")
                
                failure_log = (
                    f"\n❌ POSITION EXIT FAILED ❌\n"
                    f"Time: {datetime.now()}\n"
                    f"Entry Time: {position_data['entry_timestamp']}\n"
                    f"Entry Price: ${position_data['entry_price']:.4f}\n"
                    f"Attempted Exit Price: ${exit_price:.4f}\n"
                    f"Position Size: {position_data['position_size']:.4f}\n"
                    f"Exit Reason: {exit_reason}\n"
                    f"ERROR: Position remains OPEN on Coinbase!\n"
                    f"ACTION REQUIRED: Manual close or bot restart needed\n"
                    f"{'='*50}"
                )
                logging.error(failure_log)
                print(failure_log)
                
                # DO NOT clear position state - keep it saved!
                # DO NOT update portfolio value - position is still open!
                self.monitoring_active = False  # Stop monitoring loop
                return
            
            # Order succeeded - proceed with normal exit processing
            entry_price = position_data['entry_price']
            gross_pnl = (exit_price - entry_price) * position_data['position_size']
            fees = abs(position_data['position_size'] * entry_price * self.config['trading']['fee_percent'] / 100)
            net_pnl = gross_pnl - fees
            
            # Update portfolio only if order succeeded
            self.portfolio_value += net_pnl
            
            # Log successful completion
            completion_log = (
                f"\n✅ POSITION COMPLETED ✅\n"
                f"Exit Time: {datetime.now()}\n"
                f"Entry Time: {position_data['entry_timestamp']}\n"
                f"Entry Price: ${entry_price:.4f}\n"
                f"Exit Price: ${exit_price:.4f}\n"
                f"Position Size: {position_data['position_size']:.4f}\n"
                f"Gross P&L: ${gross_pnl:.2f}\n"
                f"Fees: ${fees:.2f}\n"
                f"Net P&L: ${net_pnl:.2f}\n"
                f"Portfolio Value: ${self.portfolio_value:.2f}\n"
                f"Exit Reason: {exit_reason}\n"
                f"{'='*50}"
            )
            logging.info(completion_log)
            print(completion_log)
            
            # Clear position state only if order succeeded
            self._clear_position_state()
            self.current_position = None
            self.monitoring_active = False
            
        except Exception as e:
            # Any exception means position is likely still open
            print(f"❌ Failed to complete position exit: {e}")
            print(f"⚠️  POSITION REMAINS OPEN ON COINBASE")
            print(f"💾 Keeping position saved for recovery")
            
            failure_log = (
                f"\n❌ POSITION EXIT EXCEPTION ❌\n"
                f"Time: {datetime.now()}\n"
                f"Error: {str(e)}\n"
                f"Exit Reason: {exit_reason}\n"
                f"Position remains OPEN on Coinbase!\n"
                f"ACTION REQUIRED: Manual close or bot restart needed\n"
                f"{'='*50}"
            )
            logging.error(failure_log)
            self.monitoring_active = False  # Stop monitoring loop
            # DO NOT clear position state on exception!
    
    def _emergency_exit(self):
        """Handle emergency bot shutdown."""
        print("\n🚨 EMERGENCY SHUTDOWN INITIATED 🚨")
        
        if self.current_position is not None:
            print("📍 Active position detected. Initiating emergency close...")
            self._emergency_close_position()
        else:
            print("ℹ️  No active position to close.")
        
        print("🔄 Saving current state...")
        # Could save additional state here if needed
        
        print("✅ Emergency shutdown completed.")
    
    def _cleanup_on_exit(self):
        """Cleanup function called on normal exit."""
        if self.current_position is not None:
            print("\n⚠️  Bot exiting with active position!")
            print("💾 Position state preserved for recovery on next startup.")
        else:
            # Clean up position file if no active position
            self._clear_position_state()

    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Analyze feature importance and return ranked features."""
        try:
            print("\n=== Feature Importance Analysis ===")
            
            # Get feature importance from trained model
            importance_scores = self.model.feature_importances_
            
            # Create importance dictionary
            feature_importance = dict(zip(feature_names, importance_scores))
            
            # Sort by importance (descending)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("Feature Importance Ranking:")
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"{i:2d}. {feature:15s}: {importance:.4f}")
            
            # Store in history
            importance_record = {
                'timestamp': datetime.now(),
                'features': dict(sorted_features),
                'top_5_features': [f[0] for f in sorted_features[:5]],
                'total_trades': self.total_trades
            }
            self.feature_importance_history.append(importance_record)
            
            return importance_record
            
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
            return {}
    
    def _select_top_features(self, feature_names: List[str], min_features: int = 8, max_features: int = 15) -> List[str]:
        """Select top performing features based on importance analysis."""
        try:
            if not self.feature_importance_history:
                # If no history, use all features
                return feature_names
            
            # Get latest feature importance
            latest_importance = self.feature_importance_history[-1]['features']
            
            # Sort features by importance
            sorted_features = sorted(latest_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select features that are above average importance
            importance_values = list(latest_importance.values())
            avg_importance = np.mean(importance_values)
            std_importance = np.std(importance_values)
            threshold = avg_importance + (0.1 * std_importance)  # Slightly above average
            
            selected_features = []
            for feature, importance in sorted_features:
                if importance >= threshold and len(selected_features) < max_features:
                    selected_features.append(feature)
            
            # Ensure minimum number of features
            if len(selected_features) < min_features:
                selected_features = [f[0] for f in sorted_features[:min_features]]
            
            print(f"\nFeature Selection:")
            print(f"Original features: {len(feature_names)}")
            print(f"Selected features: {len(selected_features)}")
            print(f"Threshold: {threshold:.4f}")
            print(f"Selected: {selected_features}")
            
            return selected_features
            
        except Exception as e:
            print(f"Error selecting features: {e}")
            return feature_names  # Fallback to all features
    
    def _get_optimized_features(self) -> List[str]:
        """Get the current optimized feature set."""
        # Define all available features
        all_features = [
            'SMA_20', 'SMA_50', 'EMA_9', 'EMA_50', 'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'BB_width', 'ATR', 'Returns', 'Volume_Change',
            'Price_vs_SMA20', 'Price_vs_EMA9', 'SMA_Cross', 'EMA_Cross',
            'High_Low_Pct', 'Volume_Ratio'
        ]
        
        # Use feature selection if we have enough history
        if len(self.feature_importance_history) >= 2:
            optimized_features = self._select_top_features(all_features)
            self.current_features = optimized_features
            return optimized_features
        else:
            # For initial training, use a curated set of important features
            initial_features = [
                'EMA_9', 'EMA_50', 'SMA_20', 'RSI', 'MACD', 'MACD_signal',
                'BB_width', 'ATR', 'Returns', 'Price_vs_EMA9', 'EMA_Cross',
                'High_Low_Pct', 'Volume_Ratio'
            ]
            self.current_features = initial_features
            return initial_features

    def _save_feature_importance_analysis(self):
        """Save feature importance analysis to CSV for tracking."""
        try:
            if not self.feature_importance_history:
                return
            
            # Prepare data for CSV
            analysis_records = []
            for record in self.feature_importance_history:
                base_record = {
                    'timestamp': record['timestamp'],
                    'total_trades': record['total_trades'],
                    'top_5_features': ', '.join(record['top_5_features'])
                }
                
                # Add individual feature importance
                for feature, importance in record['features'].items():
                    base_record[f"importance_{feature}"] = importance
                
                analysis_records.append(base_record)
            
            # Save to CSV
            df = pd.DataFrame(analysis_records)
            importance_file = 'feature_importance_history.csv'
            df.to_csv(importance_file, index=False)
            print(f"Feature importance analysis saved to '{importance_file}'")
            
        except Exception as e:
            print(f"Error saving feature importance analysis: {e}")
    
    def plot_feature_importance_evolution(self):
        """Plot how feature importance evolves over time."""
        try:
            if len(self.feature_importance_history) < 2:
                print("Not enough feature importance data to plot evolution.")
                return
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get top features across all analyses
            all_features = set()
            for record in self.feature_importance_history:
                all_features.update(record['features'].keys())
            
            # Create matrix of importance over time
            times = [record['timestamp'] for record in self.feature_importance_history]
            feature_matrix = []
            
            for feature in sorted(all_features):
                importance_series = []
                for record in self.feature_importance_history:
                    importance_series.append(record['features'].get(feature, 0))
                feature_matrix.append(importance_series)
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(feature_matrix, 
                       xticklabels=[t.strftime('%m-%d %H:%M') for t in times],
                       yticklabels=sorted(all_features),
                       annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title('Feature Importance Evolution Over Time')
            plt.xlabel('Training Sessions')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig('feature_importance_evolution.png')
            print("Feature importance evolution plot saved to 'feature_importance_evolution.png'")
            
        except Exception as e:
            print(f"Error plotting feature importance evolution: {e}")

    def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect current market regime for strategy adaptation."""
        try:
            latest_data = data.tail(20)  # Last 20 days
            
            # Trend Detection
            sma_20 = latest_data['SMA_20'].iloc[-1]
            sma_50 = latest_data['SMA_50'].iloc[-1] if 'SMA_50' in latest_data.columns else sma_20
            current_price = latest_data['Close'].iloc[-1]
            
            # Calculate trend strength
            if current_price > sma_20 > sma_50:
                trend = "STRONG_UPTREND"
            elif current_price > sma_20:
                trend = "UPTREND"
            elif current_price < sma_20 < sma_50:
                trend = "STRONG_DOWNTREND"
            elif current_price < sma_20:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            
            # Volatility Regime Detection
            returns = latest_data['Close'].pct_change().dropna()
            current_vol = returns.std()
            vol_ma = returns.rolling(10).std().mean()
            
            if current_vol > vol_ma * 1.5:
                volatility = "HIGH_VOL"
            elif current_vol < vol_ma * 0.7:
                volatility = "LOW_VOL"
            else:
                volatility = "NORMAL_VOL"
            
            # Market State Detection
            rsi = latest_data['RSI'].iloc[-1] if 'RSI' in latest_data.columns else 50
            if rsi > 70:
                market_state = "OVERBOUGHT"
            elif rsi < 30:
                market_state = "OVERSOLD"
            else:
                market_state = "NEUTRAL"
            
            regime = {
                'trend': trend,
                'volatility': volatility,
                'market_state': market_state,
                'vol_ratio': current_vol / vol_ma
            }
            
            print(f"\n📊 Market Regime Analysis:")
            print(f"Trend: {trend}")
            print(f"Volatility: {volatility} ({current_vol:.1%} vs {vol_ma:.1%})")
            print(f"Market State: {market_state}")
            
            return regime
            
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return {'trend': 'UNKNOWN', 'volatility': 'NORMAL_VOL', 'market_state': 'NEUTRAL', 'vol_ratio': 1.0}
    
    def _adapt_strategy_to_regime(self, regime: Dict[str, str], base_confidence: float) -> Tuple[float, float]:
        """Adapt trading strategy based on market regime."""
        try:
            confidence_multiplier = 1.0
            profit_target_multiplier = 1.0
            
            # Trend-based adaptations
            if regime['trend'] in ['STRONG_UPTREND', 'UPTREND']:
                confidence_multiplier *= 1.2  # More aggressive in uptrends
                profit_target_multiplier *= 1.3  # Higher targets in trending markets
            elif regime['trend'] in ['STRONG_DOWNTREND', 'DOWNTREND']:
                confidence_multiplier *= 0.8  # More conservative in downtrends
                profit_target_multiplier *= 0.9  # Lower targets in downtrends
            else:  # SIDEWAYS
                confidence_multiplier *= 0.9  # Slightly conservative in ranging markets
                profit_target_multiplier *= 0.8  # Lower targets in ranging markets
            
            # Volatility-based adaptations
            if regime['volatility'] == 'HIGH_VOL':
                confidence_multiplier *= 0.9  # Slightly conservative in high vol
                profit_target_multiplier *= 1.4  # Higher targets in volatile markets
            elif regime['volatility'] == 'LOW_VOL':
                confidence_multiplier *= 1.1  # Slightly aggressive in low vol
                profit_target_multiplier *= 0.8  # Lower targets in low vol
            
            # Market state adaptations
            if regime['market_state'] == 'OVERBOUGHT':
                confidence_multiplier *= 0.7  # Very conservative when overbought
            elif regime['market_state'] == 'OVERSOLD':
                confidence_multiplier *= 1.3  # Aggressive when oversold
            
            adapted_confidence = base_confidence * confidence_multiplier
            
            print(f"\n🎯 Strategy Adaptation:")
            print(f"Base Confidence: {base_confidence:.1%}")
            print(f"Confidence Multiplier: {confidence_multiplier:.2f}x")
            print(f"Adapted Confidence: {adapted_confidence:.1%}")
            print(f"Profit Target Multiplier: {profit_target_multiplier:.2f}x")
            
            return adapted_confidence, profit_target_multiplier
            
        except Exception as e:
            print(f"Error adapting strategy to regime: {e}")
            return base_confidence, 1.0

if __name__ == "__main__":
    print("[DEBUG] Script started: __main__ entry point.")
    bot = TradingBot('config.json')
    bot.start_new_backtest()  # Ensure retraining_history.csv is cleared at the start
    bot.run_daily_cycle() 