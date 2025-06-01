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
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.last_retraining = datetime.now()
        self.total_trades = 0
        self.next_retraining_trade = 1  # Start with retraining after first trade
        
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
        
        # Add technical indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Add price momentum features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Add target variable (1 if next day's return is positive, 0 otherwise)
        df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
        
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
            
            # Prepare features and target
            features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 
                       'ATR', 'Returns', 'Volume_Change']
            
            X = processed_data[features].values
            y = processed_data['Target'].values
            logging.info(f"Prepared {len(features)} features for retraining")
            
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
                if metrics['total_risk_units'] < -1.0:
                    logging.info("Poor risk-adjusted performance detected, adjusting parameters conservatively")
                    new_threshold = min(max(old_threshold * 1.1, 0.1), 0.7)
                    print(f"[DEBUG] Conservative: threshold {old_threshold:.2f} -> {new_threshold:.2f}")
                    self.config['ml_model']['prediction_threshold'] = new_threshold
                    self.config['trading']['profit_target_percent'] = max(15.0, self.config['trading']['profit_target_percent'] * 0.9)
                    self.config['trading']['max_trade_duration_hours'] = max(4, self.config['trading']['max_trade_duration_hours'] * 0.8)
                elif metrics['total_risk_units'] > 1.0:
                    logging.info("Good risk-adjusted performance detected, optimizing parameters")
                    new_threshold = min(max(old_threshold * 0.95, 0.1), 0.7)
                    print(f"[DEBUG] Aggressive: threshold {old_threshold:.2f} -> {new_threshold:.2f}")
                    self.config['ml_model']['prediction_threshold'] = new_threshold
                    if metrics['win_rate'] > 0.6:
                        self.config['trading']['profit_target_percent'] = min(30.0, self.config['trading']['profit_target_percent'] * 1.1)
                else:
                    logging.info("Moderate performance, making minor parameter adjustments")
                
                # Log parameter changes
                param_log = (
                    f"\nParameter Adjustments:\n"
                    f"ML Threshold: {old_threshold:.2%} -> {self.config['ml_model']['prediction_threshold']:.2%}\n"
                    f"Profit Target: {self.config['trading']['profit_target_percent']:.1f}% -> {self.config['trading']['profit_target_percent']:.1f}%\n"
                    f"Max Duration: {self.config['trading']['max_trade_duration_hours']:.1f}h -> {self.config['trading']['max_trade_duration_hours']:.1f}h"
                )
                logging.info(param_log)
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            logging.info("Feature scaling completed")
            
            # Train model
            self.model.fit(X_scaled, y)
            logging.info("Model retraining completed")
            
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
            
            # Prepare features and target
            features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 
                       'ATR', 'Returns', 'Volume_Change']
            
            X = processed_data[features].values
            y = processed_data['Target'].values
            logging.info(f"Prepared {len(features)} features for training")
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X)
            logging.info("Feature scaling completed")
            
            # Train model
            self.model.fit(X_scaled, y)
            logging.info("Model training completed")
            
            # Log initialization results
            model_score = self.model.score(X_scaled, y)
            print(f"Model initialized successfully. Accuracy: {model_score:.2%}")
            
            # Create detailed initialization log
            init_log = (
                f"\n=== Model Initialization Complete ===\n"
                f"Time: {datetime.now()}\n"
                f"Training Data Period: {market_data.index[0]} to {market_data.index[-1]}\n"
                f"Number of Training Samples: {len(X)}\n"
                f"Features Used: {', '.join(features)}\n"
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
    
    def _make_ml_prediction(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Generate trading signal using ML model."""
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 
                   'ATR', 'Returns', 'Volume_Change']
        
        X = market_data[features].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict_proba(X_scaled)[0]
        confidence = prediction[1]  # Probability of positive return
        
        return confidence, prediction[1]
    
    def _calculate_trade_parameters(self, current_price: float, 
                                  confidence: float) -> Dict:
        """Calculate position size, stop-loss, and take-profit levels."""
        # Calculate risk amount based on portfolio value and risk percentage
        risk_amount = float(self.portfolio_value * (self.config['trading']['risk_per_trade_percent'] / 100))
        
        # Calculate position size based on risk amount and current price
        position_size = float(risk_amount / current_price)
        
        # Adjust position size based on confidence (sophisticated risk management)
        # Lower confidence = smaller position size = more conservative approach
        position_size *= float(confidence)
        actual_risk = float(position_size * current_price)
        
        # Calculate stop loss and take profit levels
        stop_loss_pct = float(self.config['trading']['risk_per_trade_percent'])
        take_profit_pct = float(self.config['trading']['profit_target_percent'])
        
        stop_loss = float(current_price * (1 - stop_loss_pct/100))
        take_profit = float(current_price * (1 + take_profit_pct/100))
        
        print(f"\nTrade Parameters Debug:")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Risk Per Trade %: {self.config['trading']['risk_per_trade_percent']:.1f}%")
        print(f"Max Risk Amount: ${risk_amount:.2f}")
        print(f"ML Confidence: {confidence:.1%}")
        print(f"Current Price: ${current_price:.4f}")
        print(f"Position Size: {position_size:.4f} XRP")
        print(f"Actual Risk Taken: ${actual_risk:.2f} ({confidence:.1%} of max)")
        print(f"Stop Loss: ${stop_loss:.4f}")
        print(f"Take Profit: ${take_profit:.4f}")
        
        # Create trade parameters dictionary with explicit float conversions
        trade_params = {
            'position_size': float(position_size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'risk_amount': float(actual_risk),  # Use actual risk taken, not max risk
            'trade_details': {  # Add trade details to preserve risk information
                'risk_amount': float(actual_risk),
                'position_size': float(position_size),
                'current_price': float(current_price)
            }
        }
        
        print(f"\nTrade Parameters Final Check:")
        print(f"Actual Risk Amount: ${trade_params['risk_amount']:.2f}")
        print(f"Trade Details Risk Amount: ${trade_params['trade_details']['risk_amount']:.2f}")
        
        return trade_params
    
    def _execute_trade(self, symbol: str, order_type: str, 
                      quantity: float, price: Optional[float] = None,
                      is_mock: bool = True) -> Dict:
        """Execute a trade (mock implementation)."""
        print(f"\n[DEBUG] _execute_trade called: symbol={symbol}, order_type={order_type}, quantity={quantity:.4f}, price={price}, is_mock={is_mock}")
        
        if is_mock:
            # Simulate trade execution
            execution_price = float(price if price else self._fetch_market_data(symbol).iloc[-1]['Close'])
            slippage = float(execution_price * (self.config['risk_management']['slippage_percent'] / 100))
            fees = float(execution_price * quantity * (self.config['risk_management']['transaction_fee_percent'] / 100))
            
            # Calculate risk amount for this trade
            risk_amount = float(self.portfolio_value * (self.config['trading']['risk_per_trade_percent'] / 100))
            
            print(f"\nExecute Trade Debug:")
            print(f"Order Type: {order_type}")
            print(f"Quantity: {quantity:.4f}")
            print(f"Execution Price: ${execution_price:.2f}")
            print(f"Risk Amount: ${risk_amount:.2f}")
            
            # Create trade info with explicit float conversions
            trade_info = {
                'execution_price': float(execution_price),
                'slippage': float(slippage),
                'fees': float(fees),
                'timestamp': datetime.now(),
                'quantity': float(quantity),
                'risk_amount': float(risk_amount),
                'order_type': order_type,
                'trade_details': {
                    'risk_amount': float(risk_amount),
                    'quantity': float(quantity),
                    'execution_price': float(execution_price),
                    'slippage': float(slippage),
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
        fees = float(abs(quantity * entry_price * self.config['trading']['fee_percent'] / 100))
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
                confidence, prediction = self._make_ml_prediction(processed_data)
                print(f"ML Model Confidence: {confidence:.2%}")
                
                if confidence >= self.config['ml_model']['prediction_threshold']:
                    print("\n=== Trade Setup ===")
                    current_price = float(market_data.iloc[-1]['Close'])
                    
                    # Calculate trade parameters
                    trade_params = self._calculate_trade_parameters(current_price, confidence)
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
                    print(f"Current confidence: {confidence:.2%}")
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
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df['timestamp'], df['model_accuracy'], 'r--', label='Model Accuracy')
            ax1_twin.set_ylabel('Model Accuracy', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
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

if __name__ == "__main__":
    print("[DEBUG] Script started: __main__ entry point.")
    bot = TradingBot('config.json')
    bot.start_new_backtest()  # Ensure retraining_history.csv is cleared at the start
    bot.run_daily_cycle() 