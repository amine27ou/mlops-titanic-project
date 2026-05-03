"""
Exploratory Data Analysis for Titanic dataset.
Generates plots and saves to reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path so sibling packages like data can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_data import load_raw_data

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Create figures directory
Path("reports/figures").mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame):
    """Plot survival distribution."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    df['Survived'].value_counts().plot(
        kind='bar', 
        ax=ax[0], 
        color=['#e74c3c', '#2ecc71']
    )
    ax[0].set_title('Survival Count', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Survived (0=No, 1=Yes)')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(['No', 'Yes'], rotation=0)
    
    # Pie chart
    df['Survived'].value_counts(normalize=True).plot(
        kind='pie',
        ax=ax[1],
        autopct='%1.1f%%',
        colors=['#e74c3c', '#2ecc71'],
        labels=['Did Not Survive', 'Survived']
    )
    ax[1].set_title('Survival Rate', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('reports/figures/01_target_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 01_target_distribution.png")
    plt.close()


def plot_survival_by_sex(df: pd.DataFrame):
    """Plot survival by gender."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Sex', hue='Survived', palette=['#e74c3c', '#2ecc71'])
    plt.title('Survival by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('reports/figures/02_survival_by_sex.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 02_survival_by_sex.png")
    plt.close()


def plot_survival_by_class(df: pd.DataFrame):
    """Plot survival by passenger class."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette=['#e74c3c', '#2ecc71'])
    plt.title('Survival by Passenger Class', fontsize=14, fontweight='bold')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('reports/figures/03_survival_by_pclass.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 03_survival_by_pclass.png")
    plt.close()


def plot_age_distribution(df: pd.DataFrame):
    """Plot age distribution."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # All passengers
    df['Age'].hist(bins=30, ax=ax[0], color='skyblue', edgecolor='black')
    ax[0].set_title('Age Distribution (All Passengers)', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Frequency')
    
    # By survival
    df[df['Survived']==0]['Age'].hist(bins=30, alpha=0.5, label='Not Survived', ax=ax[1], color='#e74c3c')
    df[df['Survived']==1]['Age'].hist(bins=30, alpha=0.5, label='Survived', ax=ax[1], color='#2ecc71')
    ax[1].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('Age')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('reports/figures/04_age_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 04_age_distribution.png")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True, 
        linewidths=1
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 05_correlation_heatmap.png")
    plt.close()


def main():
    """Run all EDA plots."""
    print("="*60)
    print("GENERATING EDA PLOTS")
    print("="*60)
    
    # Load data
    df = load_raw_data()
    
    # Generate plots
    plot_target_distribution(df)
    plot_survival_by_sex(df)
    plot_survival_by_class(df)
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    
    print(f"\n{'='*60}")
    print("✅ ALL PLOTS GENERATED")
    print(f"{'='*60}")
    print(f"Location: reports/figures/")
    print(f"Total plots: 5")


if __name__ == "__main__":
    main()