# Portfolio Performance Tracker (Erweitert)

## Erweiterter Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

TRADING_DAYS = 252

# -----------------------------
# Core Metrics
# -----------------------------

def calculate_metrics(returns: pd.Series, risk_free_rate=0.02):
    annual_return = returns.mean() * TRADING_DAYS
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

    downside = returns[returns < 0]
    sortino = (annual_return - risk_free_rate) / (downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Jahresrendite': f'{annual_return*100:.2f}%',
        'Volatilität': f'{volatility*100:.2f}%',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Sortino Ratio': f'{sortino:.2f}',
        'Max Drawdown': f'{max_drawdown*100:.2f}%',
        'Calmar Ratio': f'{calmar:.2f}'
    }

# -----------------------------
# Additional Analysis
# -----------------------------

def calculate_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate=0.02):
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance if variance != 0 else 0

    alpha = (portfolio_returns.mean() * TRADING_DAYS) - (
        risk_free_rate + beta * ((benchmark_returns.mean() * TRADING_DAYS) - risk_free_rate)
    )

    return beta, alpha


def value_at_risk(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)


def expected_shortfall(returns, confidence=0.95):
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

# -----------------------------
# Data Generation
# -----------------------------

def generate_demo_portfolio():
    dates = pd.date_range('2024-01-01', '2025-12-31', freq='B')
    np.random.seed(42)
    prices = pd.DataFrame({
        'Portfolio': 100 * np.cumprod(1 + np.random.normal(0.0004, 0.012, len(dates))),
        'DAX Benchmark': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.010, len(dates))),
    }, index=dates)
    return prices

# -----------------------------
# Visualization
# -----------------------------

def plot_performance(prices, output='performance.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Portfolio Performance Analysis', fontsize=16)

    norm = prices / prices.iloc[0] * 100
    axes[0,0].plot(norm.index, norm['Portfolio'], linewidth=2, label='Portfolio')
    axes[0,0].plot(norm.index, norm['DAX Benchmark'], linewidth=1.5, label='Benchmark')
    axes[0,0].set_title('Kumulative Performance'); axes[0,0].legend(); axes[0,0].grid()

    returns = prices['Portfolio'].pct_change().dropna()
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    axes[0,1].fill_between(dd.index, dd.values)
    axes[0,1].set_title('Drawdown'); axes[0,1].grid()

    vol = returns.rolling(30).std() * np.sqrt(TRADING_DAYS) * 100
    axes[1,0].plot(vol.index, vol.values)
    axes[1,0].set_title('Rolling Volatilität (%)'); axes[1,0].grid()

    monthly = returns.resample('M').apply(lambda x: (1+x).prod()-1) * 100
    axes[1,1].bar(range(len(monthly)), monthly.values)
    axes[1,1].set_title('Monatliche Renditen (%)'); axes[1,1].grid()

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f'✓ Chart saved: {output}')

# -----------------------------
# Main
# -----------------------------

if __name__ == '__main__':
    prices = generate_demo_portfolio()

    portfolio_returns = prices['Portfolio'].pct_change().dropna()
    benchmark_returns = prices['DAX Benchmark'].pct_change().dropna()

    metrics = calculate_metrics(portfolio_returns)
    beta, alpha = calculate_beta_alpha(portfolio_returns, benchmark_returns)

    var = value_at_risk(portfolio_returns)
    es = expected_shortfall(portfolio_returns)

    print("\n📈 Portfolio-Kennzahlen:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print(f"   Beta: {beta:.2f}")
    print(f"   Alpha: {alpha:.4f}")
    print(f"   Value at Risk (95%): {var:.4f}")
    print(f"   Expected Shortfall: {es:.4f}")

    plot_performance(prices)
```

---

# README.md

## Überblick

Dieses Tool analysiert die Performance eines Portfolios und berechnet wichtige Finanzkennzahlen.

---

## Features

### Performance Kennzahlen
- Jahresrendite
- Volatilität
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown

### Risikoanalyse
- Value at Risk (VaR)
- Expected Shortfall (CVaR)

### Marktvergleich
- Beta (Marktrisiko)
- Alpha (Überrendite)

### Visualisierung
- Kumulative Performance
- Drawdowns
- Rolling Volatilität
- Monatliche Renditen

---

## Verwendung

```bash
python tracker.py
```

---

## Erweiterungen möglich

- Einlesen echter Marktdaten (Yahoo Finance API)
- Multi-Asset Portfolios
- Rebalancing Simulation
- Backtesting Strategien
- Web Dashboard (Streamlit)

---

## Fazit

Dieses Tool ist ein solides Fundament für Portfolio-Analyse und kann leicht zu einem professionellen Analytics-Tool ausgebaut werden.

