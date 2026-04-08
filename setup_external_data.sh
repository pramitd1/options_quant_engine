#!/bin/bash
# External Data Setup - Finnhub API Key Configuration
# Run this after getting your free Finnhub API key

set -e

echo "=========================================="
echo "  External Data Provider Setup"
echo "=========================================="
echo ""
echo "📋 STEP 1: Get Free Finnhub API Key"
echo "───────────────────────────────────────"
echo ""
echo "A browser window has opened to: https://finnhub.io/dashboard"
echo ""
echo "Follow these steps:"
echo "  1. Enter your email address"
echo "  2. Click 'Get your free API key'"
echo "  3. Check your email for verification link"
echo "  4. Click the verification link"
echo "  5. Copy your API key (looks like: pk_xxx...)"
echo ""
echo "⏳ Waiting for API key..."
read -p "Paste your Finnhub API key and press Enter: " FINNHUB_KEY

if [ -z "$FINNHUB_KEY" ]; then
    echo "❌ API key is empty. Please try again."
    exit 1
fi

echo ""
echo "✅ API key received"
echo ""

# Create or update .env file
echo "📝 Updating .env file..."
if [ -f ".env" ]; then
    # Check if FINNHUB_API_KEY already exists
    if grep -q "^FINNHUB_API_KEY" .env; then
        # Update existing key
        sed -i '' "s/^FINNHUB_API_KEY=.*/FINNHUB_API_KEY=$FINNHUB_KEY/" .env
    else
        # Add new key
        echo "FINNHUB_API_KEY=$FINNHUB_KEY" >> .env
    fi
else
    # Create new .env file
    cat > .env << EOF
# External Data API Configuration
# Created: $(date)

# Finnhub API Key (FREE)
FINNHUB_API_KEY=$FINNHUB_KEY

# Data caching
DATA_CACHE_ENABLED=true
DATA_CACHE_DIR=./data/cache

# Backtest configuration
DATA_BACKTEST_YEARS=3
DATA_HISTORICAL_DAYS=365

# Logging level
LOGGING_LEVEL=INFO
EOF
fi

echo "✅ .env file updated"
echo ""

echo "📊 STEP 2: Run Health Check"
echo "───────────────────────────────────────"
echo ""

# Run health check
cd "$(dirname "$0")" || exit 1
python -c "
import sys
sys.path.insert(0, '.')
from data.external_data_provider import ExternalDataProvider

try:
    provider = ExternalDataProvider()
    checks = provider.health_check()
    
    print('Health check results:')
    for key, status in checks.items():
        if key == 'timestamp':
            print(f'  ⏰ Timestamp: {status}')
        elif key == 'all_healthy':
            symbol = '✅' if status else '❌'
            print(f'  {symbol} Overall: {\"All sources healthy\" if status else \"Some sources down\"}')
        else:
            symbol = '✅' if status else '❌'
            print(f'  {symbol} {key}: {status}')
    
    if not checks['all_healthy']:
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Ready to fetch historical data:"
    echo "  python data/external_data_provider.py"
    echo ""
else
    echo ""
    echo "❌ Health check failed. Please verify:"
    echo "  1. API key is correct"
    echo "  2. .env file has FINNHUB_API_KEY=<your_key>"
    echo "  3. Internet connection is working"
    exit 1
fi
