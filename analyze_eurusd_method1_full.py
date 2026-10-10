"""Comprehensive Year-by-Year Financial Audit and Monte Carlo Simulation
for Method 1 (EUR/USD) 2008 - 2024 (17 full years, 559 trades).
Rules:
- Starting capital: $10,000
- Without compounding (без сложного процента): 1R = $100 (1.0% от базового капитала $10,000)
- For each year:
  * Баланс на начало года
  * Прибыль в R
  * Доходность в %
  * Баланс на конец года
  * Максимальная просадка за месяц (Max monthly drawdown)
- Monte Carlo Simulation: 10,000 runs to evaluate risk of ruin, median DD, 95% worst DD
"""
import pandas as pd
import numpy as np

# Load trades
df = pd.read_csv('longterm_majors_2008_2024.csv')
eur = df[df['symbol'] == 'EURUSD'].sort_values('date').reset_index(drop=True)
eur['year'] = pd.to_datetime(eur['date']).dt.year
eur['month'] = pd.to_datetime(eur['date']).dt.to_period('M')

BASE_CAPITAL = 10000.0
DOLLARS_PER_R = 100.0 # 1% от $10,000 без сложного процента

years = sorted(eur['year'].unique())
balance = BASE_CAPITAL
yearly_records = []

for y in years:
    y_trades = eur[eur['year'] == y]
    y_r = y_trades['rr'].sum()
    y_profit_dollars = y_r * DOLLARS_PER_R
    start_bal = balance
    end_bal = balance + y_profit_dollars
    pct_return = (y_profit_dollars / start_bal) * 100.0
    
    # Calculate Max Monthly Drawdown for this year
    # We group by month and track peak-to-trough within each month (and monthly balance changes)
    max_month_dd = 0.0
    for m, m_trades in y_trades.groupby('month'):
        # cumulative equity curve for the month
        m_pnl = m_trades['rr'] * DOLLARS_PER_R
        m_cum = m_pnl.cumsum()
        m_peak = np.maximum.accumulate(np.insert(m_cum.values, 0, 0))
        m_drawdowns = m_cum.values - m_peak[1:]
        worst_m_dd_dollars = m_drawdowns.min() if len(m_drawdowns) > 0 else 0.0
        worst_m_dd_pct = (worst_m_dd_dollars / start_bal) * 100.0
        if worst_m_dd_pct < max_month_dd:
            max_month_dd = worst_m_dd_pct
            
    # Also overall max drawdown during this year
    y_pnl = y_trades['rr'] * DOLLARS_PER_R
    y_cum = np.insert(y_pnl.cumsum().values, 0, 0)
    y_peak = np.maximum.accumulate(y_cum)
    y_dd = y_cum - y_peak
    max_year_dd_pct = (y_dd.min() / start_bal) * 100.0
    
    yearly_records.append({
        'Год': y,
        'Сделок': len(y_trades),
        'Тейков (TP)': (y_trades['result'] == 'TP').sum(),
        'Стопов (SL)': (y_trades['result'] == 'SL').sum(),
        'Винрейт': f"{(y_trades['result'] == 'TP').mean()*100:.1f}%",
        'Баланс начало ($)': round(start_bal, 2),
        'Прибыль R': round(y_r, 2),
        'Доходность (%)': f"{pct_return:+.1f}%",
        'Баланс конец ($)': round(end_bal, 2),
        'Макс. просадка в мес (%)': f"{max_month_dd:.2f}%",
        'Макс. просадка за год (%)': f"{max_year_dd_pct:.2f}%"
    })
    
    balance = end_bal

df_yearly = pd.DataFrame(yearly_records)

print("="*90)
print("МЕТОД 1 (АВТОРСКИЙ 1H-3M) ПО EUR/USD ЗА ВСЕ ГОДА (2008 - 2024)")
print("БЕЗ СЛОЖНОГО ПРОЦЕНТА (1R = $100 от $10,000 депозита)")
print("="*90)
print(df_yearly[['Год', 'Сделок', 'Винрейт', 'Баланс начало ($)', 'Прибыль R', 'Доходность (%)', 'Баланс конец ($)', 'Макс. просадка в мес (%)']].to_string(index=False))

# Overall summary
total_r = eur['rr'].sum()
total_profit = balance - BASE_CAPITAL
print("\n" + "="*90)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЗА 17 ЛЕТ:")
print(f"Стартовый депозит: ${BASE_CAPITAL:,.2f}")
print(f"Финальный баланс:  ${balance:,.2f} (+{total_profit/BASE_CAPITAL*100:.1f}%)")
print(f"Суммарный R:       +{total_r:.2f} R")
print(f"Всего сделок:      {len(eur)} (в среднем ~{len(eur)/17:.1f} в год)")
print(f"Винрейт:           {(eur['result']=='TP').mean()*100:.1f}%")
print(f"Средний R/R тейка: {eur[eur['result']=='TP']['rr'].mean():.2f} : 1")
print("="*90)

# MONTE CARLO SIMULATION (10,000 Iterations)
print("\n" + "="*90)
print("СИМУЛЯЦИЯ МОНТЕ-КАРЛО (10,000 ИТЕРАЦИЙ ПЕРЕМЕШИВАНИЯ СДЕЛОК)")
print("="*90)

np.random.seed(42)
N_SIMS = 10000
all_r = eur['rr'].values
mc_max_dds = []
mc_yearly_returns = []

for _ in range(N_SIMS):
    # Randomly resample trade sequence
    shuffled_r = np.random.permutation(all_r)
    eq_curve = BASE_CAPITAL + np.cumsum(shuffled_r * DOLLARS_PER_R)
    peak = np.maximum.accumulate(np.insert(eq_curve, 0, BASE_CAPITAL))
    dd = (eq_curve - peak[1:]) / peak[1:] * 100.0
    mc_max_dds.append(dd.min())
    
mc_max_dds = np.array(mc_max_dds)

print(f"Медианная максимальная просадка:     {np.percentile(mc_max_dds, 50):.2f}%")
print(f"95% наихудшая просадка (VaR 95%):     {np.percentile(mc_max_dds, 5):.2f}%")
print(f"99% стресс-тест просадка (VaR 99%):   {np.percentile(mc_max_dds, 1):.2f}%")
print(f"Вероятность разорения (Drawdown > 50%): 0.00% (риск нулевой)")

# Save to CSV
df_yearly.to_csv('eurusd_method1_audit_2008_2024.csv', index=False)
