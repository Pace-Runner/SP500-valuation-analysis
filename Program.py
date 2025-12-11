import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia with proper headers"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tables = pd.read_html(StringIO(response.text))
    sp500_table = tables[0]
    return sp500_table[['Symbol', 'Security', 'GICS Sector']].values.tolist()

def calculate_valuation_metrics(ticker):
    """Calculate valuation metrics for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            'ticker': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'price': info.get('currentPrice', np.nan),
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
            'price_to_fcf': info.get('priceToFreeCashflow', np.nan),
            'market_cap': info.get('marketCap', np.nan),
            'earnings_growth': info.get('earningsGrowth', np.nan),
            'revenue_growth': info.get('revenueGrowth', np.nan)
        }
        
        return metrics
    except Exception as e:
        return None

def calculate_valuation_score(df):
    """Calculate composite valuation score (lower = more undervalued)"""
    
    score_components = []
    
    for metric in ['pe_ratio', 'forward_pe', 'peg_ratio', 'pb_ratio', 'ev_ebitda', 'price_to_fcf']:
        if metric in df.columns:
            # Group by sector and calculate percentile rank
            df[f'{metric}_pct'] = df.groupby('sector')[metric].rank(pct=True)
            score_components.append(f'{metric}_pct')
    
    # Average the percentile ranks (lower percentile = more undervalued)
    if score_components:
        df['valuation_score'] = df[score_components].mean(axis=1, skipna=True)
        
        # Classify stocks
        df['valuation_category'] = pd.cut(
            df['valuation_score'], 
            bins=[0, 0.25, 0.4, 0.6, 0.75, 1.0],
            labels=['Highly Undervalued', 'Undervalued', 'Fairly Valued', 'Overvalued', 'Highly Overvalued']
        )
        
        # Calculate deviation from sector median
        for metric in ['pe_ratio', 'peg_ratio', 'pb_ratio']:
            if metric in df.columns:
                sector_median = df.groupby('sector')[metric].transform('median')
                df[f'{metric}_vs_sector'] = ((df[metric] - sector_median) / sector_median * 100).round(2)
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations of valuation analysis"""
    
    # Filter out N/A sectors and clean data
    df_clean = df[df['sector'] != 'N/A'].copy()
    
    # Create figure with subplots - now 2x2 grid instead of 2x3
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Valuation Score Distribution
    ax1 = plt.subplot(2, 2, 1)
    valuation_counts = df_clean['valuation_category'].value_counts().sort_index()
    colors = ['#27ae60', '#95a5a6', '#f39c12', '#e67e22', '#c0392b']
    bars = ax1.bar(range(len(valuation_counts)), valuation_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Distribution of Valuation Categories', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Valuation Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(valuation_counts)))
    ax1.set_xticklabels(valuation_counts.index, rotation=0, ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, v) in enumerate(zip(bars, valuation_counts.values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(v)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Sector Valuation Heatmap - Only Valuation Score
    ax2 = plt.subplot(2, 2, 2)
    sector_val_score = df_clean.groupby('sector')['valuation_score'].mean().sort_values().to_frame()
    sector_val_score.columns = ['Avg Valuation Score']
    
    sns.heatmap(sector_val_score, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                ax=ax2, cbar_kws={'label': 'Score (0=Undervalued, 1=Overvalued)'}, 
                linewidths=2, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Average Valuation Score by Sector', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('')
    ax2.set_ylabel('Sector', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=10, rotation=0)
    
    # 3. Top 15 Undervalued Stocks (moved from position 4, now position 3)
    ax3 = plt.subplot(2, 2, 3)
    top_undervalued = df_clean.nsmallest(15, 'valuation_score')[['ticker', 'valuation_score']].iloc[::-1]
    bars4 = ax3.barh(range(len(top_undervalued)), top_undervalued['valuation_score'].values, 
                     color='#27ae60', edgecolor='black', linewidth=1.5)
    ax3.set_yticks(range(len(top_undervalued)))
    ax3.set_yticklabels(top_undervalued['ticker'].values, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Valuation Score (Lower = More Undervalued)', fontsize=12, fontweight='bold')
    ax3.set_title('Top 15 Most Undervalued Stocks', fontsize=15, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.set_xlim(0, max(top_undervalued['valuation_score'].values) * 1.2)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars4, top_undervalued['valuation_score'].values)):
        width = bar.get_width()
        ax3.text(width + 0.005, bar.get_y() + bar.get_height()/2., 
                f'{v:.3f}', va='center', ha='left', fontweight='bold', fontsize=10)
    
    # 4. Top 15 Overvalued Stocks (moved from position 5, now position 4)
    ax4 = plt.subplot(2, 2, 4)
    top_overvalued = df_clean.nlargest(15, 'valuation_score')[['ticker', 'valuation_score']].iloc[::-1]
    bars5 = ax4.barh(range(len(top_overvalued)), top_overvalued['valuation_score'].values, 
                     color='#c0392b', edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top_overvalued)))
    ax4.set_yticklabels(top_overvalued['ticker'].values, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Valuation Score (Higher = More Overvalued)', fontsize=12, fontweight='bold')
    ax4.set_title('Top 15 Most Overvalued Stocks', fontsize=15, fontweight='bold', pad=15)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars5, top_overvalued['valuation_score'].values)):
        width = bar.get_width()
        ax4.text(width - 0.015, bar.get_y() + bar.get_height()/2., 
                f'{v:.3f}', va='center', ha='right', fontweight='bold', fontsize=10, color='white')
    
    plt.tight_layout(pad=3)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'sp500_valuation_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualizations saved to: {filename}")
    
    plt.show()

def main():
    print("=" * 80)
    print("S&P 500 VALUATION ANALYZER")
    print("=" * 80)
    print(f"\nStarting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nStep 1: Fetching S&P 500 companies...")
    
    sp500_companies = get_sp500_tickers()
    print(f"Found {len(sp500_companies)} companies")
    
    print("\nStep 2: Collecting financial data (this may take several minutes)...")
    
    results = []
    total = len(sp500_companies)
    
    for i, (ticker, name, sector) in enumerate(sp500_companies, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{total} companies processed...")
        
        metrics = calculate_valuation_metrics(ticker)
        if metrics:
            results.append(metrics)
    
    print(f"\nStep 3: Analyzing valuation metrics for {len(results)} companies...")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Remove stocks with insufficient data
    required_cols = ['pe_ratio', 'forward_pe', 'peg_ratio']
    df = df.dropna(subset=required_cols, how='all')
    
    # Calculate valuation scores
    df = calculate_valuation_score(df)
    
    # Sort by valuation score
    df = df.sort_values('valuation_score')
    

    
    # Save to CSV
    output_file = f'sp500_valuation_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full data saved to: {output_file}")
    
    print("\nGenerating visualization...")
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()