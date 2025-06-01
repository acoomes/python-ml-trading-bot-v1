# ML-Powered Algorithmic Trading Bot

A sophisticated, enterprise-grade algorithmic trading bot that combines advanced machine learning with robust risk management and professional-grade safety features. The bot features adaptive learning, position recovery systems, and comprehensive performance analytics.

## 🚀 Key Features

### **🤖 Advanced Machine Learning**
- **Adaptive ML Model**: Random Forest Classifier with intelligent retraining
- **Progressive Retraining Schedule**: Dynamic model updates based on trade performance
- **Risk-Adjusted Position Sizing**: ML confidence multipliers for conservative risk management
- **Parameter Auto-Adjustment**: ML threshold, profit targets, and duration optimized based on performance
- **Technical Analysis**: 9 sophisticated features (SMA, RSI, MACD, Bollinger Bands, ATR, etc.)

### **🛡️ Professional Risk Management**
- **Position Recovery System**: Automatic detection and recovery of orphaned positions
- **Multi-Layer Risk Controls**: Per-trade, portfolio, and drawdown limits
- **Emergency Exit Mechanisms**: Signal handlers and graceful shutdown
- **Cool-down Periods**: Automatic trading pause after consecutive losses
- **Time-Based Exits**: Maximum trade duration enforcement
- **Confidence-Based Sizing**: Lower ML confidence = smaller position sizes

### **📊 Comprehensive Analytics & Visualization**
- **Real-Time Performance Tracking**: Risk units, win rates, portfolio R&PL
- **Advanced Plotting**: Portfolio value, ML retraining events, performance metrics
- **Detailed Trade Logging**: Complete audit trail with risk-adjusted metrics
- **Backtest Analysis**: Historical performance with ML evolution tracking
- **CSV Export**: All data exportable for external analysis

### **🔗 Live Trading Integration**
- **Coinbase Advanced Trading API**: Professional-grade live trading
- **Mock Mode**: Safe testing environment with realistic execution simulation
- **Live Position Monitoring**: Real-time price checking with stop-loss/take-profit
- **Order Management**: Limit and market orders with proper quantity/price rounding

## 📋 Requirements

- **Python 3.8+**
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: Standard desktop/laptop (no GPU required)
- **API Access**: Coinbase Advanced Trading account (optional, for live trading)

## 🛠️ Installation

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd python-ml-trading-bot-v1
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Trading Parameters**:
```bash
# Edit config.json with your preferred settings
# See Configuration section below for details
```

## ⚙️ Configuration

### **Core Trading Parameters** (`config.json`)

```json
{
  "trading": {
    "symbol": "XRP-USD",                     // Trading pair
    "starting_portfolio_amount": 100.0,      // Initial capital
    "max_trades_per_day": 10,               // Daily trade limit
    "risk_per_trade_percent": 10,           // Max risk per trade
    "profit_target_percent": 25.0,          // Take profit level
    "max_portfolio_drawdown_percent": 20.0, // Portfolio protection
    "max_trade_duration_hours": 8,          // Auto-exit time
    "fee_percent": 1.5                      // Trading fees
  },
  "ml_model": {
    "prediction_threshold": 0.10,           // ML confidence required
    "features": [                           // Technical indicators
      "SMA_20", "SMA_50", "RSI", "MACD", 
      "BB_upper", "BB_lower", "ATR", 
      "Returns", "Volume_Change"
    ]
  },
  "coinbase": {
    "enabled": false,                       // Enable live trading
    "api_url": "https://api.coinbase.com/api/v3/brokerage/"
  }
}
```

### **Live Trading Setup** (Optional)
1. Create Coinbase Advanced Trading account
2. Generate API credentials
3. Save as `cdp_api_key.json` in project root
4. Set `coinbase.enabled: true` in config.json

## 🚀 Usage

### **1. Backtesting (Recommended First Step)**
```bash
python backtest.py
```
- **Safe Testing**: No real money involved
- **Historical Analysis**: Tests strategy on past data
- **ML Evolution**: Shows how model adapts over time
- **Performance Metrics**: Win rate, risk units, portfolio returns

### **2. Live Trading**
```bash
python trading_bot.py
```
- **Real-Time Trading**: Places actual orders (if enabled)
- **Position Monitoring**: Continuous stop-loss/take-profit checking
- **Recovery System**: Handles interruptions gracefully

### **3. Position Recovery** (Automatic)
If bot is interrupted with open position:
```
🚨 EXISTING POSITION DETECTED! 🚨
Entry Time: 2025-06-01T14:32:15
Symbol: XRP-USD
Position Size: 43.2100
Entry Price: $2.3187
Stop Loss: $2.0868
Take Profit: $3.0143

Options:
1) Resume monitoring    ✅ Recommended
2) Close immediately    🔴 Emergency
3) Ignore              ⚠️ Dangerous
```

## 📁 Project Structure

```
python-ml-trading-bot-v1/
├── trading_bot.py              # Core trading engine
├── backtest.py                 # Historical testing system
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── POSITION_RECOVERY_README.md # Detailed safety documentation
├── test_position_recovery.py   # Recovery system testing
├── 
├── Generated Files:
├── trades.log                  # Detailed trade logging
├── retraining_history.csv      # ML model evolution
├── backtest_results.csv        # Historical performance
├── backtest_summaries.csv      # Aggregated results
├── current_position.json       # Active position state
└── *.png                       # Performance visualizations
```

## 🧠 Machine Learning System

### **Adaptive Learning Architecture**
- **Model**: Random Forest Classifier (100 estimators)
- **Features**: 9 technical indicators with 1-year training data
- **Retraining Schedule**: Progressive (1,2,3,4,5,7,9,11,13,15...)
- **Performance Feedback**: Risk units drive parameter adjustments

### **Intelligent Parameter Adjustment**
| Performance | ML Threshold | Profit Target | Max Duration |
|-------------|--------------|---------------|--------------|
| Poor (<-1.0 risk units) | Increase 10% | Decrease 10% | Decrease 20% |
| Good (>1.0 risk units) | Decrease 5% | Increase 10% | Maintain |
| Moderate | Minor adjustments | Maintain | Maintain |

### **Risk-Adjusted Position Sizing**
```python
# Example: 70% ML confidence = 70% of maximum position size
max_risk = portfolio_value * risk_percent
position_size = (max_risk / price) * ml_confidence
actual_risk = position_size * price  # Always ≤ max_risk
```

## 📊 Performance Analytics

### **Risk-Adjusted Metrics**
- **Risk Units**: Net PnL / Risk Amount (standardized performance)
- **Portfolio R&PL**: Risk-adjusted portfolio returns
- **Win Rate**: Percentage of profitable trades
- **Average Risk Units**: Mean risk-adjusted return per trade

### **Visualization Tools**
```python
bot.plot_retraining_history()        # ML evolution over time
bot.plot_backtest_with_retraining()  # Portfolio + ML events
bot.plot_portfolio_with_retraining() # Live trading performance
```

### **Generated Charts**
- **Portfolio Value**: Time-series with ML retraining markers
- **Model Performance**: Accuracy, win rate, risk units evolution
- **Parameter Evolution**: How ML adjusts trading parameters

## 🛡️ Safety Features

### **Position Recovery System**
- **Automatic Detection**: Finds orphaned positions on startup
- **Recovery Options**: Resume monitoring, emergency close, or ignore
- **State Persistence**: Position details saved to `current_position.json`
- **Signal Handling**: Graceful shutdown on CTRL+C, termination signals

### **Risk Management Layers**
1. **Per-Trade Risk**: Maximum percentage of portfolio per trade
2. **Portfolio Drawdown**: Global loss limits with automatic shutdown
3. **Consecutive Loss Protection**: Cool-down periods after losses
4. **Time-Based Exits**: Maximum trade duration enforcement
5. **ML Confidence Gates**: Only trade above threshold confidence

### **Emergency Protocols**
- **Signal Handlers**: SIGINT, SIGTERM, SIGBREAK (Windows)
- **Emergency Close**: Immediate position liquidation
- **Position State Cleanup**: Automatic file management
- **Error Recovery**: Graceful handling of API failures

## 📈 Usage Examples

### **Basic Backtest**
```bash
# Test strategy on historical data
python backtest.py

# Expected output:
=== Backtest Results ===
Period: 2025-04-28 to 2025-05-28
Initial Portfolio: $100.00
Final Portfolio: $125.43
Total Return: 25.43%
Total Trades: 23
Win Rate: 65.22%
ML Retraining Runs: 8
```

### **Live Trading with Position Recovery**
```bash
# Start live trading
python trading_bot.py

# If interrupted and restarted:
🚨 EXISTING POSITION DETECTED! 🚨
[Position details displayed]
Choose option 1 to resume monitoring
```

### **Performance Visualization**
```python
from trading_bot import TradingBot
bot = TradingBot()
bot.plot_retraining_history()  # Creates retraining_history.png
```

## 🔧 Advanced Configuration

### **ML Model Tuning**
```json
{
  "ml_model": {
    "prediction_threshold": 0.15,     // Higher = more selective
    "trades_before_retraining": 1,    // Retraining frequency
    "features": [...]                 // Technical indicators
  }
}
```

### **Risk Management Customization**
```json
{
  "trading": {
    "risk_per_trade_percent": 5,      // More conservative
    "consecutive_losses_threshold": 2, // Tighter controls
    "cooldown_period_hours": 48       // Longer recovery
  }
}
```

### **Live Trading Safety**
```json
{
  "trading": {
    "min_trade_quantity": 0.0001,     // Minimum order size
    "fee_percent": 1.5               // Expected trading fees
  },
  "risk_management": {
    "slippage_percent": 0.1,         // Market impact allowance
    "min_risk_reward_ratio": 1.5     // Minimum R:R for trades
  }
}
```

## 📚 Documentation

- **`POSITION_RECOVERY_README.md`**: Detailed safety system documentation
- **`test_position_recovery.py`**: Recovery system testing and examples
- **Code Comments**: Extensive inline documentation
- **Configuration**: JSON schema with parameter descriptions

## ⚠️ Important Disclaimers

### **Educational Purpose**
This trading bot is designed for **educational and research purposes**. It demonstrates advanced algorithmic trading concepts, machine learning integration, and professional risk management practices.

### **Risk Warnings**
- **Live Trading**: Always test thoroughly in mock mode before using real capital
- **Market Risk**: Past performance does not guarantee future results
- **API Limits**: Respect exchange rate limits and terms of service
- **Capital Risk**: Never risk more than you can afford to lose

### **Best Practices**
1. **Start Small**: Begin with minimal capital and conservative settings
2. **Monitor Performance**: Regularly review trade logs and metrics
3. **Update Dependencies**: Keep libraries updated for security
4. **Backup Data**: Save important configuration and log files
5. **Test Recovery**: Periodically test position recovery system

## 🤝 Contributing

Contributions welcome! Areas of interest:
- **Additional Technical Indicators**: New ML features
- **Alternative ML Models**: LSTM, reinforcement learning
- **Exchange Integrations**: Additional trading platforms
- **Risk Management**: Enhanced safety features
- **Visualization**: Advanced analytics and charts

## 📄 License

MIT License - See LICENSE file for details.

---

**⚡ Quick Start**: `python backtest.py` → Review results → Configure `config.json` → `python trading_bot.py` 
