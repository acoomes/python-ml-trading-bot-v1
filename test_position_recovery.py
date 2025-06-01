#!/usr/bin/env python3
"""
Test script to demonstrate position recovery system.
This script simulates various scenarios of bot restarts with open positions.
"""

import json
import os
from datetime import datetime, timedelta
from trading_bot import TradingBot

def create_test_position_file():
    """Create a test position state file to simulate an interrupted bot."""
    position_data = {
        'entry_timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
        'symbol': 'XRP-USD',
        'position_size': 43.2100,
        'entry_price': 2.3187,
        'stop_loss': 2.0868,
        'take_profit': 3.0143,
        'risk_amount': 25.89,
        'portfolio_value': 258.91,
        'is_live_trading': False
    }
    
    with open('current_position.json', 'w') as f:
        json.dump(position_data, f, indent=2)
    
    print("✅ Created test position file: current_position.json")
    print(f"📊 Position details:")
    print(f"   Entry Time: {position_data['entry_timestamp']}")
    print(f"   Symbol: {position_data['symbol']}")
    print(f"   Position Size: {position_data['position_size']:.4f}")
    print(f"   Entry Price: ${position_data['entry_price']:.4f}")
    print(f"   Stop Loss: ${position_data['stop_loss']:.4f}")
    print(f"   Take Profit: ${position_data['take_profit']:.4f}")
    print(f"   Risk Amount: ${position_data['risk_amount']:.2f}")
    
def test_position_recovery():
    """Test the position recovery system."""
    print("\n🧪 Testing Position Recovery System")
    print("=" * 50)
    
    # Create test position file
    create_test_position_file()
    
    print(f"\n🤖 Initializing TradingBot with existing position...")
    print("This will demonstrate the position recovery dialog.")
    print("You'll see options to:")
    print("1) Resume monitoring")
    print("2) Close position immediately") 
    print("3) Ignore (dangerous!)")
    
    try:
        # Initialize bot (this will trigger position recovery check)
        bot = TradingBot('config.json')
        print("\n✅ Bot initialized successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Bot initialization interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during bot initialization: {e}")

def cleanup_test_files():
    """Clean up test files."""
    files_to_clean = ['current_position.json']
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Cleaned up: {file}")

if __name__ == "__main__":
    print("🧪 Position Recovery System Test")
    print("This script demonstrates how the bot handles recovery from interrupted trades.")
    
    choice = input("\nOptions:\n1) Test position recovery\n2) Clean up test files\nChoice (1/2): ")
    
    if choice == '1':
        test_position_recovery()
    elif choice == '2':
        cleanup_test_files()
    else:
        print("Invalid choice. Exiting.")
    
    print("\n✅ Test completed.") 