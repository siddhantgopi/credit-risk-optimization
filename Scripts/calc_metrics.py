import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_lending_club_metrics(df):
    """
    Calculate actual loss per default and profit per loan from Lending Club data
    with robust error handling and data cleaning
    
    Parameters:
    df: DataFrame with Lending Club data
    
    Key columns needed:
    - loan_amnt: loan amount
    - int_rate: interest rate
    - term: loan term (36 months or 60 months)
    - loan_status: loan status (Fully Paid, Charged Off, etc.)
    - total_pymnt: total payment received
    - recoveries: recovery amount for charged off loans
    - collection_recovery_fee: collection fees
    """
    
    print("=== LENDING CLUB LOAN ANALYSIS ===")
    print(f"Original dataset: {len(df):,} loans")
    
    # Create a working copy
    df_work = df.copy()
    
    # Data cleaning and validation
    print("\n=== DATA CLEANING ===")
    
    # Clean interest rate column
    if 'int_rate' in df_work.columns:
        # Handle string values with % symbols
        df_work['int_rate'] = df_work['int_rate'].astype(str).str.replace('%', '').str.strip()
        df_work['int_rate'] = pd.to_numeric(df_work['int_rate'], errors='coerce')
        print(f"Interest rate cleaned: {df_work['int_rate'].isna().sum()} NaN values")
    
    # Clean loan amount
    if 'loan_amnt' in df_work.columns:
        df_work['loan_amnt'] = pd.to_numeric(df_work['loan_amnt'], errors='coerce')
        print(f"Loan amount cleaned: {df_work['loan_amnt'].isna().sum()} NaN values")
    
    # Clean total payments
    if 'total_pymnt' in df_work.columns:
        df_work['total_pymnt'] = pd.to_numeric(df_work['total_pymnt'], errors='coerce')
        df_work['total_pymnt'] = df_work['total_pymnt'].fillna(0)
    
    # Clean recoveries and fees
    if 'recoveries' in df_work.columns:
        df_work['recoveries'] = pd.to_numeric(df_work['recoveries'], errors='coerce')
        df_work['recoveries'] = df_work['recoveries'].fillna(0)
    
    if 'collection_recovery_fee' in df_work.columns:
        df_work['collection_recovery_fee'] = pd.to_numeric(df_work['collection_recovery_fee'], errors='coerce')
        df_work['collection_recovery_fee'] = df_work['collection_recovery_fee'].fillna(0)
    
    # Clean loan status
    df_work['loan_status_clean'] = df_work['loan_status'].astype(str).str.strip()
    
    # Remove rows with missing critical data
    critical_columns = ['loan_amnt', 'int_rate']
    initial_count = len(df_work)
    df_work = df_work.dropna(subset=critical_columns)
    dropped_count = initial_count - len(df_work)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing critical data")
    
    print(f"Final dataset: {len(df_work):,} loans")
    
    # Define loan outcomes
    good_loans = df_work[df_work['loan_status_clean'].isin(['Fully Paid'])]
    bad_loans = df_work[df_work['loan_status_clean'].isin(['Charged Off', 'Default'])]
    
    print(f"\nLoan Status Distribution:")
    print(f"Fully Paid loans: {len(good_loans):,} ({len(good_loans)/len(df_work)*100:.1f}%)")
    print(f"Charged Off/Default loans: {len(bad_loans):,} ({len(bad_loans)/len(df_work)*100:.1f}%)")
    
    # Initialize default values
    avg_profit_per_loan = 0
    median_profit_per_loan = 0
    avg_loss_per_default = 0
    median_loss_per_default = 0
    
    # Calculate metrics for good loans (Fully Paid)
    if len(good_loans) > 0:
        print("\n=== PROFIT ANALYSIS (FULLY PAID LOANS) ===")
        
        # Expected payments calculation
        good_loans = good_loans.copy()
        
        # Handle interest rate calculation safely
        good_loans['monthly_rate'] = good_loans['int_rate'] / 100 / 12
        
        # Extract term safely
        if 'term' in good_loans.columns:
            good_loans['num_payments'] = good_loans['term'].astype(str).str.extract('(\d+)').astype(float)
        else:
            print("Warning: 'term' column not found, using default 36 months")
            good_loans['num_payments'] = 36
        
        # Calculate expected monthly payment using loan formula
        # Handle zero interest rates
        mask_positive_rate = good_loans['monthly_rate'] > 0
        mask_zero_rate = good_loans['monthly_rate'] == 0
        
        # For positive interest rates
        if mask_positive_rate.any():
            good_loans.loc[mask_positive_rate, 'expected_payment'] = (
                good_loans.loc[mask_positive_rate, 'loan_amnt'] * 
                good_loans.loc[mask_positive_rate, 'monthly_rate'] * 
                (1 + good_loans.loc[mask_positive_rate, 'monthly_rate'])**good_loans.loc[mask_positive_rate, 'num_payments']
            ) / ((1 + good_loans.loc[mask_positive_rate, 'monthly_rate'])**good_loans.loc[mask_positive_rate, 'num_payments'] - 1)
        
        # For zero interest rates
        if mask_zero_rate.any():
            good_loans.loc[mask_zero_rate, 'expected_payment'] = (
                good_loans.loc[mask_zero_rate, 'loan_amnt'] / good_loans.loc[mask_zero_rate, 'num_payments']
            )
        
        good_loans['expected_total'] = good_loans['expected_payment'] * good_loans['num_payments']
        good_loans['expected_interest'] = good_loans['expected_total'] - good_loans['loan_amnt']
        
        # Actual profit (interest earned)
        good_loans['actual_profit'] = good_loans['total_pymnt'] - good_loans['loan_amnt']
        
        avg_profit_per_loan = good_loans['actual_profit'].mean()
        median_profit_per_loan = good_loans['actual_profit'].median()
        
        print(f"Average profit per fully paid loan: ${avg_profit_per_loan:,.2f}")
        print(f"Median profit per fully paid loan: ${median_profit_per_loan:,.2f}")
        print(f"Average expected interest: ${good_loans['expected_interest'].mean():,.2f}")
        print(f"Average actual interest earned: ${good_loans['actual_profit'].mean():,.2f}")
    
    # Calculate metrics for bad loans (Charged Off)
    if len(bad_loans) > 0:
        print("\n=== LOSS ANALYSIS (CHARGED OFF LOANS) ===")
        
        bad_loans = bad_loans.copy()
        
        # Calculate loss per default
        bad_loans['gross_loss'] = bad_loans['loan_amnt'] - bad_loans['total_pymnt']
        bad_loans['net_loss'] = bad_loans['gross_loss'] - bad_loans['recoveries']
        bad_loans['total_loss'] = bad_loans['net_loss'] + bad_loans['collection_recovery_fee']
        
        avg_loss_per_default = bad_loans['total_loss'].mean()
        median_loss_per_default = bad_loans['total_loss'].median()
        
        print(f"Average loss per default: ${avg_loss_per_default:,.2f}")
        print(f"Median loss per default: ${median_loss_per_default:,.2f}")
        print(f"Average gross loss (before recoveries): ${bad_loans['gross_loss'].mean():,.2f}")
        print(f"Average recoveries: ${bad_loans['recoveries'].mean():,.2f}")
        
        recovery_rate = bad_loans['recoveries'].sum() / bad_loans['gross_loss'].sum() * 100
        print(f"Recovery rate: {recovery_rate:.1f}%")
    
    # Overall portfolio analysis
    print("\n=== PORTFOLIO ANALYSIS ===")
    
    total_loans = len(df_work)
    total_loan_amount = df_work['loan_amnt'].sum()
    total_payments = df_work['total_pymnt'].sum()
    
    if len(good_loans) > 0:
        total_profit = good_loans['actual_profit'].sum()
    else:
        total_profit = 0
    
    if len(bad_loans) > 0:
        total_losses = bad_loans['total_loss'].sum()
    else:
        total_losses = 0
    
    net_portfolio_return = total_profit - total_losses
    
    print(f"Total loan amount: ${total_loan_amount:,.0f}")
    print(f"Total payments received: ${total_payments:,.0f}")
    print(f"Total profit from good loans: ${total_profit:,.0f}")
    print(f"Total losses from bad loans: ${total_losses:,.0f}")
    print(f"Net portfolio return: ${net_portfolio_return:,.0f}")
    
    portfolio_roi = net_portfolio_return/total_loan_amount if total_loan_amount > 0 else 0
    print(f"Portfolio ROI: {portfolio_roi*100:.2f}%")
    
    # Risk-adjusted metrics
    default_rate = len(bad_loans) / len(df_work) if len(df_work) > 0 else 0
    expected_loss = avg_loss_per_default * default_rate
    expected_profit = avg_profit_per_loan * (1 - default_rate)
    expected_net_profit = expected_profit - expected_loss
    
    print(f"\n=== RISK-ADJUSTED METRICS ===")
    print(f"Default rate: {default_rate*100:.2f}%")
    print(f"Expected loss per loan: ${expected_loss:,.2f}")
    print(f"Expected profit per loan: ${expected_profit:,.2f}")
    print(f"Expected net profit per loan: ${expected_net_profit:,.2f}")
    
    # Loan grade analysis
    if 'grade' in df_work.columns:
        print(f"\n=== ANALYSIS BY LOAN GRADE ===")
        try:
            grade_analysis = df_work.groupby('grade').agg({
                'loan_amnt': 'mean',
                'int_rate': 'mean',
                'loan_status_clean': lambda x: (x.isin(['Charged Off', 'Default'])).mean()
            }).round(3)
            grade_analysis.columns = ['Avg_Loan_Amount', 'Avg_Interest_Rate', 'Default_Rate']
            print(grade_analysis)
        except Exception as e:
            print(f"Could not generate grade analysis: {e}")
    
    return {
        'avg_profit_per_loan': avg_profit_per_loan,
        'avg_loss_per_default': avg_loss_per_default,
        'median_profit_per_loan': median_profit_per_loan,
        'median_loss_per_default': median_loss_per_default,
        'default_rate': default_rate,
        'expected_net_profit': expected_net_profit,
        'portfolio_roi': portfolio_roi,
        'total_loans': len(df_work),
        'data_quality': {
            'original_rows': len(df),
            'final_rows': len(df_work),
            'dropped_rows': len(df) - len(df_work)
        }
    }

def load_and_analyze_lending_club_data(file_path):
    """
    Safely load and analyze Lending Club data with comprehensive error handling
    """
    try:
        print(f"Loading data from: {file_path}")
        
        # Try different file formats
        if file_path.endswith('.gzip') or file_path.endswith('.gz'):
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)
        
        print(f"Successfully loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['loan_status', 'loan_amnt']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            return None
        
        # Run analysis
        metrics = calculate_lending_club_metrics(df)
        
        return metrics
        
    except Exception as e:
        print(f"Error loading or analyzing data: {str(e)}")
        return None

# Example usage with your file path
"""
file_path = 'C:/Users/siddh/AppData/Local/Microsoft/WindowsApps/files/credit-risk-optimization/credit-risk-optimization/data/raw/data/raw/Loan_status_2007-2020Q3.gzip'
metrics = load_and_analyze_lending_club_data(file_path)

if metrics:
    print("\n=== METRICS FOR OPTIMIZER ===")
    print(f"Average Loss per Default: ${metrics['avg_loss_per_default']:,.2f}")
    print(f"Average Profit per Loan: ${metrics['avg_profit_per_loan']:,.2f}")
    print(f"Default Rate: {metrics['default_rate']:.2%}")
    
    # Use in your optimizer
    optimizer.avg_loss_per_default = metrics['avg_loss_per_default']
    optimizer.avg_profit_per_loan = metrics['avg_profit_per_loan']
"""

print("Enhanced Lending Club Analysis Functions Created!")
print("\nKey improvements:")
print("- Robust data cleaning and type conversion")
print("- Handles missing values and edge cases")
print("- Comprehensive error handling")
print("- Detailed data quality reporting")
print("- Safe mathematical operations")