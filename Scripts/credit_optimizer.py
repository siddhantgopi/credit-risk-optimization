import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, fbeta_score
from sklearn.metrics import classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class CreditRiskOptimizer:
    def __init__(self, y_true, y_pred_proba, loan_amount=None, segments=None):
        """
        Initialize the credit risk optimizer
        
        Parameters:
        y_true: actual labels (0=no default, 1=default)
        y_pred_proba: predicted probabilities for default class
        loan_amount: loan amounts for each sample (optional)
        segments: customer segments for each sample (optional)
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.loan_amount = loan_amount if loan_amount is not None else np.ones(len(y_true))
        self.segments = segments
        
        # Business parameters (you can adjust these)
        self.avg_loss_per_default = 50000  # Average loss when loan defaults
        self.avg_profit_per_loan = 5000    # Average profit from good loan
        self.interest_rate = 0.12          # Annual interest rate
        
    def calculate_business_metrics(self, threshold):
        """Calculate business impact for a given threshold"""
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        # Business calculations
        # TP: Correctly identified defaults (saved losses)
        # FP: Rejected good customers (lost profits)
        # FN: Missed defaults (actual losses)
        # TN: Approved good customers (gained profits)
        
        saved_losses = tp * self.avg_loss_per_default
        lost_profits = fp * self.avg_profit_per_loan
        actual_losses = fn * self.avg_loss_per_default
        gained_profits = tn * self.avg_profit_per_loan
        
        net_benefit = gained_profits + saved_losses - actual_losses - lost_profits
        
        return {
            'threshold': threshold,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'saved_losses': saved_losses,
            'lost_profits': lost_profits,
            'actual_losses': actual_losses,
            'gained_profits': gained_profits,
            'net_benefit': net_benefit,
            'approval_rate': (tn + fn) / len(self.y_true)
        }
    
    def optimize_threshold_expected_value(self, thresholds=None):
        """1. Calculate expected value for different thresholds"""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        for threshold in thresholds:
            results.append(self.calculate_business_metrics(threshold))
        
        df_results = pd.DataFrame(results)
        
        # Find optimal threshold
        optimal_idx = df_results['net_benefit'].idxmax()
        optimal_threshold = df_results.loc[optimal_idx, 'threshold']
        
        print("=== EXPECTED VALUE ANALYSIS ===")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Maximum Net Benefit: ${df_results.loc[optimal_idx, 'net_benefit']:,.0f}")
        print(f"Precision: {df_results.loc[optimal_idx, 'precision']:.3f}")
        print(f"Recall: {df_results.loc[optimal_idx, 'recall']:.3f}")
        print(f"Approval Rate: {df_results.loc[optimal_idx, 'approval_rate']:.3f}")
        
        return df_results, optimal_threshold
    
    def analyze_fbeta_scores(self, beta_values=[0.5, 1.0, 1.5, 2.0]):
        """2. Use F-beta score analysis"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = {}
        print("\n=== F-BETA SCORE ANALYSIS ===")
        
        for beta in beta_values:
            best_threshold = 0
            best_fbeta = 0
            
            for threshold in thresholds:
                y_pred = (self.y_pred_proba >= threshold).astype(int)
                fbeta = fbeta_score(self.y_true, y_pred, beta=beta)
                
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_threshold = threshold
            
            results[beta] = {'threshold': best_threshold, 'fbeta': best_fbeta}
            
            # Get metrics for best threshold
            metrics = self.calculate_business_metrics(best_threshold)
            
            print(f"Beta = {beta} (recall weight = {beta}, precision weight = 1)")
            print(f"  Best Threshold: {best_threshold:.3f}")
            print(f"  F-beta Score: {best_fbeta:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"  Net Benefit: ${metrics['net_benefit']:,.0f}")
            print()
        
        return results
    
    def profit_based_optimization(self):
        """3. Profit-based metrics optimization"""
        print("=== PROFIT-BASED OPTIMIZATION ===")
        
        # Try different business scenarios
        scenarios = [
            {"name": "Conservative (High Loss Aversion)", "loss": 100000, "profit": 3000},
            {"name": "Balanced", "loss": 50000, "profit": 5000},
            {"name": "Aggressive (Growth Focus)", "loss": 30000, "profit": 8000}
        ]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        scenario_results = {}
        
        for scenario in scenarios:
            # Temporarily update business parameters
            original_loss = self.avg_loss_per_default
            original_profit = self.avg_profit_per_loan
            
            self.avg_loss_per_default = scenario["loss"]
            self.avg_profit_per_loan = scenario["profit"]
            
            best_threshold = 0
            best_net_benefit = float('-inf')
            
            for threshold in thresholds:
                metrics = self.calculate_business_metrics(threshold)
                if metrics['net_benefit'] > best_net_benefit:
                    best_net_benefit = metrics['net_benefit']
                    best_threshold = threshold
            
            scenario_results[scenario["name"]] = {
                'threshold': best_threshold,
                'net_benefit': best_net_benefit,
                'metrics': self.calculate_business_metrics(best_threshold)
            }
            
            print(f"{scenario['name']}:")
            print(f"  Loss per default: ${scenario['loss']:,}")
            print(f"  Profit per loan: ${scenario['profit']:,}")
            print(f"  Optimal threshold: {best_threshold:.3f}")
            print(f"  Net benefit: ${best_net_benefit:,.0f}")
            print(f"  Precision: {scenario_results[scenario['name']]['metrics']['precision']:.3f}")
            print(f"  Recall: {scenario_results[scenario['name']]['metrics']['recall']:.3f}")
            print()
            
            # Restore original parameters
            self.avg_loss_per_default = original_loss
            self.avg_profit_per_loan = original_profit
        
        return scenario_results
    
    def segment_analysis(self, segment_names=None):
        """4. Segment-based threshold optimization"""
        if self.segments is None:
            print("=== SEGMENT ANALYSIS ===")
            print("No segments provided. Creating example segments based on predicted probability quartiles.")
            
            # Create example segments based on risk quartiles
            quartiles = np.percentile(self.y_pred_proba, [25, 50, 75])
            segments = np.digitize(self.y_pred_proba, quartiles)
            segment_names = ['Low Risk', 'Medium-Low Risk', 'Medium-High Risk', 'High Risk']
            self.segments = segments
        
        if segment_names is None:
            segment_names = [f'Segment {i}' for i in range(len(np.unique(self.segments)))]
        
        print("=== SEGMENT ANALYSIS ===")
        
        segment_results = {}
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for segment_id in np.unique(self.segments):
            segment_mask = self.segments == segment_id
            segment_name = segment_names[segment_id] if segment_id < len(segment_names) else f'Segment {segment_id}'
            
            if np.sum(segment_mask) == 0:
                continue
            
            # Get segment data
            y_true_seg = self.y_true[segment_mask]
            y_pred_proba_seg = self.y_pred_proba[segment_mask]
            
            # Find optimal threshold for this segment
            best_threshold = 0
            best_net_benefit = float('-inf')
            
            for threshold in thresholds:
                y_pred_seg = (y_pred_proba_seg >= threshold).astype(int)
                
                if len(np.unique(y_pred_seg)) < 2:  # Skip if all predictions are the same
                    continue
                
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true_seg, y_pred_seg).ravel()
                    
                    # Calculate net benefit for segment
                    saved_losses = tp * self.avg_loss_per_default
                    lost_profits = fp * self.avg_profit_per_loan
                    actual_losses = fn * self.avg_loss_per_default
                    gained_profits = tn * self.avg_profit_per_loan
                    net_benefit = gained_profits + saved_losses - actual_losses - lost_profits
                    
                    if net_benefit > best_net_benefit:
                        best_net_benefit = net_benefit
                        best_threshold = threshold
                except:
                    continue
            
            # Calculate final metrics for best threshold
            y_pred_seg = (y_pred_proba_seg >= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_seg, y_pred_seg).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            default_rate = np.mean(y_true_seg)
            
            segment_results[segment_name] = {
                'threshold': best_threshold,
                'net_benefit': best_net_benefit,
                'precision': precision,
                'recall': recall,
                'default_rate': default_rate,
                'size': np.sum(segment_mask)
            }
            
            print(f"{segment_name} (n={np.sum(segment_mask):,}):")
            print(f"  Default rate: {default_rate:.3f}")
            print(f"  Optimal threshold: {best_threshold:.3f}")
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
            print(f"  Net benefit: ${best_net_benefit:,.0f}")
            print()
        
        return segment_results
    
    def plot_optimization_results(self, df_results):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Net Benefit vs Threshold
        axes[0, 0].plot(df_results['threshold'], df_results['net_benefit'] / 1000000, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Net Benefit ($ Millions)')
        axes[0, 0].set_title('Net Benefit vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Tradeoff
        axes[0, 1].plot(df_results['recall'], df_results['precision'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Tradeoff')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Approval Rate vs Threshold
        axes[1, 0].plot(df_results['threshold'], df_results['approval_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Approval Rate')
        axes[1, 0].set_title('Approval Rate vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Components of Net Benefit
        axes[1, 1].plot(df_results['threshold'], df_results['gained_profits'] / 1000000, 
                       label='Gained Profits', linewidth=2)
        axes[1, 1].plot(df_results['threshold'], df_results['saved_losses'] / 1000000, 
                       label='Saved Losses', linewidth=2)
        axes[1, 1].plot(df_results['threshold'], -df_results['actual_losses'] / 1000000, 
                       label='Actual Losses', linewidth=2)
        axes[1, 1].plot(df_results['threshold'], -df_results['lost_profits'] / 1000000, 
                       label='Lost Profits', linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Amount ($ Millions)')
        axes[1, 1].set_title('Components of Net Benefit')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run all four optimization approaches"""
        print("CREDIT RISK THRESHOLD OPTIMIZATION")
        print("=" * 50)
        
        # 1. Expected Value Analysis
        df_results, optimal_threshold = self.optimize_threshold_expected_value()
        
        # 2. F-beta Score Analysis
        fbeta_results = self.analyze_fbeta_scores()
        
        # 3. Profit-based Optimization
        profit_results = self.profit_based_optimization()
        
        # 4. Segment Analysis
        segment_results = self.segment_analysis()
        
        # Plot results
        self.plot_optimization_results(df_results)
        
        return {
            'expected_value': df_results,
            'fbeta_results': fbeta_results,
            'profit_scenarios': profit_results,
            'segment_results': segment_results,
            'optimal_threshold': optimal_threshold
        }

# Example usage with your data:
"""
# Assuming you have:
# y_true: actual labels from your test set
# y_pred_proba: predicted probabilities from your XGBoost model
# 
# Example:
# y_true = your_test_labels
# y_pred_proba = your_model.predict_proba(X_test)[:, 1]  # probabilities for class 1
# 
# optimizer = CreditRiskOptimizer(y_true, y_pred_proba)
# results = optimizer.run_complete_analysis()
"""

print("Credit Risk Optimizer created!")
print("\nTo use this with your XGBoost model:")
print("1. Get predicted probabilities: y_pred_proba = model.predict_proba(X_test)[:, 1]")
print("2. Create optimizer: optimizer = CreditRiskOptimizer(y_true, y_pred_proba)")
print("3. Run analysis: results = optimizer.run_complete_analysis()")
print("\nYou can also customize business parameters:")
print("- optimizer.avg_loss_per_default = 50000")
print("- optimizer.avg_profit_per_loan = 5000")