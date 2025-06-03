import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from trading_bot import TradingBot
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import logging
import os

class BacktestBot(TradingBot):
    def __init__(self, config_path: str, start_date: datetime, end_date: datetime):
        """Initialize backtesting with date range."""
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.backtest_results = []
        self.portfolio_history = []  # Track portfolio value over time
        super().__init__(config_path)
        
    def _fetch_market_data(self, symbol: str, interval: str = '1d', 
                          lookback_days: int = 100) -> pd.DataFrame:
        """Fetch historical market data for backtesting."""
        try:
            ticker = yf.Ticker(symbol)
            # Fetch data from before start_date to ensure we have enough for indicators
            fetch_start = self.start_date - timedelta(days=lookback_days)
            df = ticker.history(start=fetch_start, end=self.current_date, interval=interval)
            return df
        except Exception as e:
            self._send_alert(f"Error fetching market data: {str(e)}")
            raise
    
    def _execute_trade(self, symbol: str, order_type: str, 
                      quantity: float, price: Optional[float] = None,
                      is_mock: bool = True) -> Dict:
        """Simulate trade execution using historical data."""
        if price is None:
            # Get the price for the current date
            current_data = self._fetch_market_data(symbol)
            price = current_data.iloc[-1]['Close']
        
        # Simulate slippage and fees
        slippage = price * (self.config['risk_management']['slippage_percent'] / 100)
        fees = price * quantity * (self.config['risk_management']['transaction_fee_percent'] / 100)
        
        return {
            'execution_price': price,
            'slippage': slippage,
            'fees': fees,
            'timestamp': self.current_date
        }
    
    def run_backtest(self):
        """Run the backtest simulation."""
        print(f"\n=== Starting Backtest ===")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Portfolio: ${self.portfolio_value:.2f}")
        
        while self.current_date <= self.end_date:
            try:
                # Allow up to max_trades_per_day per day
                trades_today = 0
                max_trades = self.config['trading'].get('max_trades_per_day', 1)
                min_qty = self.config['trading'].get('min_trade_quantity', 0.0001)
                
                for trade_num in range(max_trades):
                    # Run one trading cycle (simulate as if it's a new signal each time)
                    # We'll call a new method to handle a single trade attempt
                    self.run_single_trade_cycle(trades_today, min_qty)
                    trades_today += 1
                
                # Record daily results
                self.backtest_results.append({
                    'date': self.current_date,
                    'portfolio_value': self.portfolio_value,
                    'trades_today': trades_today
                })
                self.portfolio_history.append({'date': self.current_date, 'value': self.portfolio_value})
                
                # Move to next day
                self.current_date += timedelta(days=1)
                
            except Exception as e:
                print(f"Error on {self.current_date.date()}: {str(e)}")
                self.current_date += timedelta(days=1)
                continue
        
        self._analyze_results()

    def run_single_trade_cycle(self, trades_today, min_qty):
        # Fetch and preprocess market data
        market_data = self._fetch_market_data(self.config['trading']['symbol'])
        processed_data = self._preprocess_data(market_data)
        
        # Generate trading signal
        confidence, prediction = self._make_ml_prediction(processed_data)
        
        if confidence > self.config['ml_model']['prediction_threshold']:
            current_price = market_data.iloc[-1]['Close']
            trade_params = self._calculate_trade_parameters(current_price, confidence)
            # Enforce minimum trade quantity
            position_size = max(round(trade_params['position_size'], 4), 0.0)
            if position_size < min_qty:
                return
            trade_params['position_size'] = position_size
            
            # Execute entry
            entry_info = self._execute_trade(
                self.config['trading']['symbol'],
                'BUY',
                trade_params['position_size'],
                current_price
            )
            entry_info['quantity'] = trade_params['position_size']
            entry_info['entry_timestamp'] = self.current_date
            
            # Monitor trade (realistic market-based exit simulation)
            # Instead of random exits, simulate realistic price movement over trading hours
            trade_duration_hours = int(self.config['trading']['max_trade_duration_hours'])
            
            # Get historical volatility for this asset to simulate realistic price movement
            historical_returns = market_data['Close'].pct_change().dropna()
            daily_volatility = historical_returns.std()
            
            # Use the dynamic profit target from trade_params instead of static config
            dynamic_take_profit = trade_params.get('take_profit', current_price * (1 + self.config['trading']['profit_target_percent'] / 100))
            dynamic_profit_target_pct = trade_params.get('dynamic_profit_target', self.config['trading']['profit_target_percent'])
            
            # Simulate hourly price movements over the trade duration
            exit_price = current_price
            exit_reason = "Time Exit"  # Default
            
            for hour in range(1, trade_duration_hours + 1):
                # Simulate hourly price movement based on historical volatility
                # Scale daily volatility to hourly (approximately daily_vol / sqrt(24))
                hourly_volatility = daily_volatility / np.sqrt(24)
                price_change = np.random.normal(0, hourly_volatility)
                exit_price = exit_price * (1 + price_change)
                
                # Check if we hit stop loss or take profit using dynamic values
                if exit_price <= trade_params['stop_loss']:
                    exit_reason = "Stop Loss"
                    exit_price = trade_params['stop_loss']
                    break
                elif exit_price >= dynamic_take_profit:
                    exit_reason = "Take Profit"
                    exit_price = dynamic_take_profit
                    break
                    
            # Log the simulated exit details with dynamic target info
            print(f"\nSimulated Trade Exit:")
            print(f"Entry: ${current_price:.4f} -> Exit: ${exit_price:.4f}")
            print(f"Duration: {hour if exit_reason != 'Time Exit' else trade_duration_hours} hours")
            print(f"Exit Reason: {exit_reason}")
            print(f"PnL: {((exit_price - current_price) / current_price) * 100:.2f}%")
            print(f"Dynamic Profit Target Used: {dynamic_profit_target_pct:.1f}%")
            
            # Execute the simulated exit trade
            exit_info = self._execute_trade(
                self.config['trading']['symbol'],
                'SELL',
                trade_params['position_size'],
                exit_price
            )
            exit_info['quantity'] = trade_params['position_size']
            exit_info['exit_timestamp'] = self.current_date
            exit_info['exit_reason'] = exit_reason
            
            # Evaluate trade
            trade_evaluation = self._evaluate_trade(entry_info, exit_info)
            # Add timestamps to trade_evaluation for ML feedback
            trade_evaluation['entry_timestamp'] = entry_info['entry_timestamp']
            trade_evaluation['exit_timestamp'] = exit_info['exit_timestamp']
            
            # Update portfolio and ML model
            self._update_portfolio_value(trade_evaluation['net_pnl'])
            self._ml_feedback_loop(trade_evaluation)
            
            # Log trade
            trade_details = {
                'entry_timestamp': entry_info['entry_timestamp'],
                'exit_timestamp': exit_info['exit_timestamp'],
                'entry_price': entry_info['execution_price'],
                'exit_price': exit_info['execution_price'],
                'position_size': trade_params['position_size'],
                'risk_amount': trade_params['risk_amount'],
                'net_pnl': trade_evaluation['net_pnl'],
                'fees': trade_evaluation['fees'],
                'slippage': entry_info.get('slippage', 0.0) + exit_info.get('slippage', 0.0),
                'risk_reward_ratio': trade_evaluation['risk_reward_ratio'],
                'current_portfolio_value': self.portfolio_value,
                'exit_reason': exit_reason
            }
            self._log_trade(trade_details)
        else:
            # Log no-trade decision
            no_trade_details = {
                'timestamp': self.current_date,
                'current_price': market_data.iloc[-1]['Close'],
                'ml_confidence': confidence,
                'required_confidence': self.config['ml_model']['prediction_threshold'],
                'portfolio_value': self.portfolio_value,
                'reason': 'ML confidence below threshold'
            }
            log_message = (
                f"\nNo Trade Decision:\n"
                f"Time: {no_trade_details['timestamp']}\n"
                f"Current Price: ${no_trade_details['current_price']:.2f}\n"
                f"ML Confidence: {no_trade_details['ml_confidence']:.2%}\n"
                f"Required Confidence: {no_trade_details['required_confidence']:.2%}\n"
                f"Portfolio Value: ${no_trade_details['portfolio_value']:.2f}\n"
                f"Reason: {no_trade_details['reason']}\n"
                f"{'='*50}"
            )
            logging.info(log_message)
            return
    
    def _analyze_results(self):
        """Analyze and display backtest results."""
        if not self.trade_history:
            print("\nNo trades were executed during the backtest period.")
            return

        # Calculate performance metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('net_pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate portfolio metrics
        initial_value = self.initial_portfolio_value
        final_value = self.portfolio_value
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Fix: Calculate total_pnl as actual portfolio change, not sum of individual trades
        total_pnl = final_value - initial_value
        
        avg_risk_units = sum(t.get('risk_reward_ratio', 0) for t in self.trade_history) / total_trades if total_trades > 0 else 0
        total_risk_units = sum(t.get('risk_reward_ratio', 0) for t in self.trade_history)
        
        # Count ML retraining runs and get final ML parameters
        retrain_file = 'retraining_history.csv'
        num_retrainings = 0
        final_ml_threshold = None
        final_profit_target = None
        final_max_trade_duration = None
        if os.path.exists(retrain_file):
            retrain_df = pd.read_csv(retrain_file)
            num_retrainings = len(retrain_df)
            if not retrain_df.empty:
                final_ml_threshold = retrain_df['ml_threshold'].iloc[-1] if 'ml_threshold' in retrain_df.columns else None
                final_profit_target = retrain_df['profit_target'].iloc[-1] if 'profit_target' in retrain_df.columns else None
                final_max_trade_duration = retrain_df['max_trade_duration'].iloc[-1] if 'max_trade_duration' in retrain_df.columns else None
        
        # Print results
        print("\n=== Backtest Results ===")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Portfolio: ${initial_value:.2f}")
        print(f"Final Portfolio: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Risk Units: {avg_risk_units:.2f}")
        print(f"Total Risk Units: {total_risk_units:.2f}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"ML Retraining Runs: {num_retrainings}")
        print(f"Final ML Threshold: {final_ml_threshold}")
        print(f"Final Profit Target: {final_profit_target}")
        print(f"Final Max Trade Duration: {final_max_trade_duration}")
        
        # Plot portfolio value over time
        if hasattr(self, 'portfolio_history') and self.portfolio_history:
            dates = [d['date'] for d in self.portfolio_history]
            values = [d['value'] for d in self.portfolio_history]
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, label='Portfolio Value')
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)

            # Overlay ML retraining runs
            if os.path.exists(retrain_file):
                retrain_df = pd.read_csv(retrain_file)
                if 'trigger_trade_timestamp' in retrain_df.columns:
                    retrain_df['event_time'] = pd.to_datetime(retrain_df['trigger_trade_timestamp'])
                else:
                    retrain_df['event_time'] = pd.to_datetime(retrain_df['timestamp'])
                for idx, row in retrain_df.iterrows():
                    if pd.isnull(row['event_time']):
                        continue
                    plt.axvline(row['event_time'], color='red', linestyle='--', alpha=0.6)
                    plt.scatter(row['event_time'], values[0], color='red', zorder=5)  # Use first value as y, will adjust below
                    plt.annotate(f"Retrain\nT{int(row['total_trades'])}",
                                 (row['event_time'], values[0]),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
            plt.legend()
            plt.savefig('backtest_results.png')
            plt.close()
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(self.trade_history)
        results_df.to_csv('backtest_results.csv', index=False)
        print("\nDetailed results saved to 'backtest_results.csv'")

        # Create and save backtest summary
        summary_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_portfolio': float(initial_value),
            'final_portfolio': float(final_value),
            'total_return_pct': float(total_return),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'avg_risk_units': float(avg_risk_units),
            'total_risk_units': float(total_risk_units),
            'total_pnl': float(total_pnl),
            'ml_retraining_runs': num_retrainings,
            'final_ml_threshold': final_ml_threshold,
            'final_profit_target': final_profit_target,
            'final_max_trade_duration': final_max_trade_duration
        }
        
        # Add config parameters
        for section, values in self.config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    summary_dict[f"{section}.{k}"] = v
            else:
                summary_dict[section] = values
        
        # Create summary DataFrame and save to CSV
        summary_df = pd.DataFrame([summary_dict])
        summary_file = 'backtest_summaries.csv'
        if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
            summary_df.to_csv(summary_file, index=False)
            print("\nCreated new backtest_summaries.csv")
        else:
            summary_df.to_csv(summary_file, mode='a', header=False, index=False)
            print("\nAppended to backtest_summaries.csv")

if __name__ == "__main__":
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Get backtest period from config, fallback to last 30 days if not present
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    if 'backtest' in config:
        if 'start_date' in config['backtest']:
            start_date = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
        if 'end_date' in config['backtest']:
            end_date = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    
    # Run backtest
    backtest = BacktestBot('config.json', start_date, end_date)
    backtest.run_backtest() 