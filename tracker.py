"""Portfolio Performance Tracker - Securities return and risk analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_metrics(returns: pd.Series, risk_free_rate=0.02):
    annual_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Jahresrendite': f'{annual_return*100:.2f}%',
        'Volatilität': f'{volatility*100:.2f}%',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_drawdown*100:.2f}%'
    }

def generate_demo_portfolio():
    dates = pd.date_range('2024-01-01', '2025-12-31', freq='B')
    np.random.seed(42)
    prices = pd.DataFrame({
        'Portfolio': 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, len(dates))),
        'DAX Benchmark': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.010, len(dates))),
    }, index=dates)
    return prices

def plot_performance(prices, output='performance.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold', color='#1B4F72')
    
    # Normalized performance
    norm = prices / prices.iloc[0] * 100
    axes[0,0].plot(norm.index, norm['Portfolio'], color='#c9a84c', linewidth=2, label='Portfolio')
    axes[0,0].plot(norm.index, norm['DAX Benchmark'], color='#2E86C1', linewidth=1.5, alpha=0.7, label='Benchmark')
    axes[0,0].set_title('Kumulative Performance'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    
    # Drawdown
    returns = prices['Portfolio'].pct_change().dropna()
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    axes[0,1].fill_between(dd.index, dd.values, color='#ef4444', alpha=0.4)
    axes[0,1].set_title('Drawdown'); axes[0,1].grid(alpha=0.3)
    
    # Rolling volatility
    vol = returns.rolling(30).std() * np.sqrt(252) * 100
    axes[1,0].plot(vol.index, vol.values, color='#eab308', linewidth=1.5)
    axes[1,0].set_title('Rolling 30-Tage Volatilität (%)'); axes[1,0].grid(alpha=0.3)
    
    # Monthly returns heatmap
    monthly = returns.resample('M').apply(lambda x: (1+x).prod()-1) * 100
    axes[1,1].bar(range(len(monthly)), monthly.values,
        color=['#22c55e' if v>0 else '#ef4444' for v in monthly.values])
    axes[1,1].set_title('Monatliche Renditen (%)'); axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'✓ Chart saved: {output}')

if __name__ == '__main__':
    prices = generate_demo_portfolio()
    returns = prices['Portfolio'].pct_change().dropna()
    metrics = calculate_metrics(returns)
    print("\n📈 Portfolio-Kennzahlen:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    plot_performance(prices)
