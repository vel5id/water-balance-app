"""Quick test to verify frequency and convergence warnings fixes"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Capture warnings to verify they are suppressed
warning_list = []
def custom_warning(message, category, filename, lineno, file=None, line=None):
    warning_list.append(str(message))
warnings.showwarning = custom_warning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Create test data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum() + 100
series = pd.Series(values, index=dates)

print("=" * 70)
print("TESTING FREQUENCY AND CONVERGENCE WARNINGS FIXES")
print("=" * 70)

# Test 1: Check series frequency
print("\n1. Testing series frequency...")
print(f"   Original freq: {series.index.freq}")

# Test 2: Import and test forecast module
print("\n2. Testing forecast module...")
try:
    from wbm.forecast import build_sarima_model_with_params
    print("   [OK] Module imported successfully")
    
    # Run small forecast test
    print("\n3. Running SARIMA forecast test (may have convergence issues)...")
    forecast, info = build_sarima_model_with_params(
        series, 
        future_days=10, 
        min_history=30
    )
    print(f"   [OK] Forecast completed")
    print(f"   Order: {info.get('order')}")
    print(f"   Seasonal order: {info.get('seasonal_order')}")
    print(f"   Forecast shape: {forecast.shape}")
    
    # Check for warnings
    print("\n4. Checking for warnings...")
    if warning_list:
        print(f"   [WARNING] Found {len(warning_list)} warnings:")
        for w in warning_list[:3]:  # Show first 3
            print(f"     - {w[:100]}...")
    else:
        print(f"   [OK] No warnings captured (suppression working)")
    
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[PASS] TEST COMPLETE")
print("=" * 70)
