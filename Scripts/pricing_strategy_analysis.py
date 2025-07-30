def run_pricing_analysis(profit_df, interest_rate_df, step3_profits):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ========== INPUTS ==========
    # 1. profit_df: Profit for (alpha, LGD) after threshold
    # rows = alpha values; cols = LGDs
    profit_df = profit_df

    # 2. interest_rate_df: rows = loans, columns = alpha values (interest rates)
    # Simulated average rates (replace with real results if needed)
    interest_rate_df = interest_rate_df

    # 3. Threshold-only profits from Step 3
    step3_profits = step3_profits

    # ========== 1. Profit vs Alpha for each LGD ==========
    plt.figure(figsize=(10,6))
    for lgd in profit_df.columns:
        plt.plot(profit_df.index, profit_df[lgd], label=f'LGD = {lgd}', marker='o')

    plt.title('Profit vs Alpha for Different LGDs (Threshold = 0.75)')
    plt.xlabel('Alpha (Risk Premium)')
    plt.ylabel('Profit ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========== 2. Avg Interest Rate per Alpha ==========
    # Plot: Average Interest Rate vs Alpha
    import matplotlib.pyplot as plt

    # interest_rate_df should have columns like: 'interest_rate_alpha_0.5', ..., and same number of rows as test set
    interest_means = interest_rate_df.mean()

    # Clean column names for plot
    interest_means.index = interest_means.index.str.replace('interest_rate_alpha_', 'α = ')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(interest_means.index, interest_means.values, marker='o', color='purple')
    plt.title('Average Interest Rate Charged vs Alpha')
    plt.xlabel('Risk Premium Factor (α)')
    plt.ylabel('Average Interest Rate (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ========== 3. Step 3 vs Pricing Strategy Profits ==========
    # Pick LGD = 0.45 and α = 2.0 for fair comparison
    pricing_profit = profit_df.loc[2.0, 0.45]

    comparison_df = pd.DataFrame({
        'Strategy': ['Threshold-Only (Best @ 0.65)', 'Risk-Based Pricing (α=2.0, LGD=0.45)'],
        'Profit ($)': [step3_profits[0.65], pricing_profit]
    })

    plt.figure(figsize=(8,6))
    sns.barplot(data=comparison_df, x='Strategy', y='Profit ($)', palette='Set2')
    plt.title('Step 3 Thresholding vs Risk-Based Pricing')
    plt.ylabel('Profit ($)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # ========== 4. Profit at Thresholds: 0.65, 0.75, 0.80 ==========
    # Just for illustration, simulate you got same profit_df logic at other thresholds
    threshold_comp = pd.DataFrame({
        'Threshold = 0.65': [3.2e8, 5.6e8, 7.8e8, 9.5e8, 1.3e9],
        'Threshold = 0.75': profit_df[0.45].values,
        'Threshold = 0.80': [2.1e8, 4.3e8, 6.0e8, 7.5e8, 1.0e9]
    }, index=[0.5, 1.0, 1.5, 2.0, 3.0])

    threshold_comp.plot(figsize=(10,6), marker='o')
    plt.title('Profit vs Alpha at Different Thresholds')
    plt.xlabel('Alpha')
    plt.ylabel('Profit ($)')
    plt.legend(title='Threshold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========== 5. Risk-Return Frontier (Optional Bonus Plot) ==========
    # Simulated Return = avg rate, Risk = LGD
    frontier_df = pd.DataFrame({
        'Alpha': profit_df.index,
        'Return': interest_means.values[:5],
        'Risk (LGD)': [0.45, 0.65, 0.8, 0.8, 0.8],  # Simulate risk increasing
        'Profit': profit_df[0.45].values  # Simulate only LGD=0.45
    })

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=frontier_df, x='Risk (LGD)', y='Return', size='Profit', hue='Alpha', palette='viridis', sizes=(100, 1000))
    plt.title('Risk-Return Frontier (Bubble Size = Profit)')
    plt.xlabel('LGD (Risk)')
    plt.ylabel('Avg Interest Rate (Return)')
    plt.legend(title='Alpha')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
