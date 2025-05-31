# ML-Powered Trading Bot

A sophisticated algorithmic trading bot that combines machine learning with robust risk management for automated trading. The bot is designed to execute a maximum of one complete trade per trading day, with a focus on intelligent trade generation through ML feedback and comprehensive risk management.

## Features

- **ML-Powered Trading**: Uses machine learning to generate trading signals based on technical indicators and market data
- **Risk Management**:
  - Maximum 1% risk per trade
  - 20% profit target per trade
  - Global portfolio drawdown limit
  - Cool-down periods after consecutive losses
  - Time-based trade exits
- **Adaptive Learning**: ML model updates based on trade outcomes
- **Comprehensive Logging**: Detailed trade logging with performance metrics
- **Simulated Environment**: Safe testing environment with mock trade execution
- **Coinbase Integration**: Optional real trading via the Coinbase API

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python-ml-trading-bot-v1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The bot's behavior can be configured through `config.json`. Key parameters include:

- Trading symbol
- Starting portfolio amount
- Risk management parameters
- ML model settings
- Logging configuration
- Coinbase API credentials (optional)

To enable live trading, set `coinbase.enabled` to `true` in `config.json` and
provide your Coinbase API key, secret, and passphrase.

## Usage

1. Configure the bot by editing `config.json`
2. Run the bot:
```bash
python trading_bot.py
```

If `coinbase.enabled` is `true`, the bot will place real orders using your
Coinbase account. Use caution and test thoroughly before trading with real
funds.

## Project Structure

- `trading_bot.py`: Core trading bot implementation
- `config.json`: Configuration file
- `requirements.txt`: Python dependencies
- `trades.log`: Trade history and performance logs

## Risk Management

The bot implements several risk management features:

1. **Per-Trade Risk**: Maximum 1% of portfolio per trade
2. **Profit Targets**: 20% maximum profit target per trade
3. **Drawdown Protection**: Global portfolio drawdown limit
4. **Cool-down Periods**: Automatic trading pause after consecutive losses
5. **Time-Based Exits**: Maximum trade duration enforcement

## Machine Learning

The bot uses a Random Forest Classifier with the following features:

- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Price momentum
- Volume analysis
- Volatility measures

## Logging

Each trade is logged with detailed information:

- Entry/exit timestamps
- Entry/exit prices
- Position size
- Risk amount
- PnL
- Fees and slippage
- Risk-reward ratio
- Current portfolio value

## Disclaimer

This trading bot is for educational purposes only. Always test thoroughly in a simulated environment before using with real capital. Past performance is not indicative of future results.

## License

MIT License 