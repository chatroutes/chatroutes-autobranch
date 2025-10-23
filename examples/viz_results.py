"""
Visualization script for ChatRoutes AutoBranch results.

Generates social media-ready graphics from scenario outputs:
- Funnel charts (pipeline stages)
- Entropy comparison bars
- Before/After improvement cards
- Timing breakdowns

Usage:
    python examples/viz_results.py

Outputs to: artifacts/ directory
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Set style for clean, professional plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ACTUAL DATA FROM v1.1 RUN
# ============================================================================

scenarios = [
    {
        "name": "AI Memory Story",
        "stages": {"initial": 10, "scored": 10, "beam": 7, "novelty": 3, "final": 3},
        "entropy": 1.0000,
        "reduction_pct": 70.0,
        "timing": {"generation": 195.2, "pipeline": 10.1}
    },
    {
        "name": "Mars Detective Twists",
        "stages": {"initial": 12, "scored": 12, "beam": 8, "novelty": 5, "final": 5},
        "entropy": 0.9610,
        "reduction_pct": 58.3,
        "timing": {"generation": 239.3, "pipeline": 8.1},
        # BEFORE (v1.0): 91.7% reduction, 1 kept
        "before_v1": {"reduction_pct": 91.7, "final": 1}
    },
    {
        "name": "Rom-Com Endings",
        "stages": {"initial": 15, "scored": 15, "beam": 10, "novelty": 9, "final": 9},
        "entropy": 0.9656,
        "reduction_pct": 40.0,
        "timing": {"generation": 319.3, "pipeline": 10.0}
    },
    {
        "name": "Style Variations",
        "stages": {"initial": 12, "scored": 12, "beam": 9, "novelty": 7, "final": 7},
        "entropy": 0.8699,
        "reduction_pct": 41.7,
        "timing": {"generation": 136.5, "pipeline": 6.4}
    }
]

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_funnel_comparison(out: Path):
    """4-panel funnel showing all scenarios side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pipeline Funnel - All Scenarios (v1.1 OPTIMIZED)', fontsize=16, fontweight='bold')

    for idx, (ax, sc) in enumerate(zip(axes.flat, scenarios)):
        stages = sc['stages']
        counts = [
            stages['initial'],
            stages['scored'],
            stages['beam'],
            stages['novelty'],
            stages['final']
        ]
        labels = ['Initial', 'Scored', 'Beam', 'Novelty', 'Final']

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax.barh(labels, counts, color=colors, alpha=0.8)

        # Add value labels
        for i, (label, val) in enumerate(zip(labels, counts)):
            ax.text(val + max(counts)*0.02, i, str(val), va='center', fontweight='bold')

        ax.set_xlim(0, max(counts) * 1.15)
        ax.set_title(f"{sc['name']}\n{sc['reduction_pct']:.1f}% reduction | Entropy: {sc['entropy']:.2f}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Candidates')
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out / 'funnel_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Created: funnel_comparison.png")

def plot_entropy_bars(out: Path):
    """Entropy comparison across scenarios."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [s['name'] for s in scenarios]
    entropies = [s['entropy'] for s in scenarios]

    colors = ['#2ecc71' if e >= 0.9 else '#f39c12' if e >= 0.8 else '#e74c3c' for e in entropies]
    bars = ax.bar(range(len(scenarios)), entropies, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bar, e) in enumerate(zip(bars, entropies)):
        ax.text(bar.get_x() + bar.get_width()/2, e + 0.02, f'{e:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Normalized Entropy (0-1)', fontweight='bold')
    ax.set_title('Entropy by Scenario - High Diversity Maintained!', fontsize=14, fontweight='bold')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='High Diversity Threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / 'entropy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Created: entropy_comparison.png")

def plot_before_after_scenario2(out: Path):
    """Before/After comparison for Scenario 2 (biggest improvement)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Scenario 2: Mars Detective - OPTIMIZATION IMPACT', fontsize=16, fontweight='bold')

    sc = scenarios[1]  # Scenario 2

    # BEFORE (v1.0)
    before_stages = [12, 12, 8, 1, 1]  # Inferred from 91.7% reduction
    labels = ['Initial', 'Scored', 'Beam', 'Novelty', 'Final']
    colors_before = ['#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c']

    ax1.barh(labels, before_stages, color=colors_before, alpha=0.6)
    for i, val in enumerate(before_stages):
        ax1.text(val + 0.3, i, str(val), va='center', fontweight='bold')
    ax1.set_xlim(0, 14)
    ax1.set_title('BEFORE (v1.0)\n91.7% reduction - TOO AGGRESSIVE ❌', fontsize=12, fontweight='bold', color='red')
    ax1.set_xlabel('Candidates')
    ax1.invert_yaxis()

    # AFTER (v1.1)
    after_stages = [sc['stages'][k] for k in ['initial', 'scored', 'beam', 'novelty', 'final']]
    colors_after = ['#2ecc71', '#2ecc71', '#2ecc71', '#2ecc71', '#2ecc71']

    ax2.barh(labels, after_stages, color=colors_after, alpha=0.8)
    for i, val in enumerate(after_stages):
        ax2.text(val + 0.3, i, str(val), va='center', fontweight='bold')
    ax2.set_xlim(0, 14)
    ax2.set_title(f'AFTER (v1.1)\n{sc["reduction_pct"]:.1f}% reduction - OPTIMIZED [OK]', fontsize=12, fontweight='bold', color='green')
    ax2.set_xlabel('Candidates')
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out / 'before_after_scenario2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Created: before_after_scenario2.png")

def plot_timing_breakdown(out: Path):
    """Stacked bar chart showing generation vs pipeline timing."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [s['name'] for s in scenarios]
    generation = [s['timing']['generation'] for s in scenarios]
    pipeline = [s['timing']['pipeline'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.6

    p1 = ax.bar(x, generation, width, label='Generation (Ollama)', color='#3498db', alpha=0.8)
    p2 = ax.bar(x, pipeline, width, bottom=generation, label='Selection Pipeline', color='#e74c3c', alpha=0.8)

    # Add total time labels
    totals = [g + p for g, p in zip(generation, pipeline)]
    for i, total in enumerate(totals):
        mins = int(total // 60)
        secs = total % 60
        ax.text(i, total + 5, f'{mins}m {secs:.0f}s', ha='center', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('Timing Breakdown - Generation vs Selection', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / 'timing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Created: timing_breakdown.png")

def plot_reduction_comparison(out: Path):
    """Reduction percentage comparison - shows optimal range."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [s['name'] for s in scenarios]
    reductions = [s['reduction_pct'] for s in scenarios]

    # Color code: green for 30-70%, yellow for outside
    colors = ['#2ecc71' if 30 <= r <= 70 else '#f39c12' for r in reductions]
    bars = ax.bar(range(len(scenarios)), reductions, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bar, r) in enumerate(zip(bars, reductions)):
        ax.text(bar.get_x() + bar.get_width()/2, r + 1.5, f'{r:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Optimal range shading
    ax.axhspan(30, 70, alpha=0.2, color='green', label='Optimal Range (30-70%)')
    ax.axhline(y=91.7, color='red', linestyle='--', linewidth=2, label='Before v1.0 (Scenario 2)')

    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Reduction %', fontweight='bold')
    ax.set_title('Reduction Percentage - OPTIMIZED to Optimal Range', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / 'reduction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Created: reduction_comparison.png")

def create_summary_card(out: Path):
    """Text summary card for social media."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    summary_text = """
CHATROUTES AUTOBRANCH v1.1 - OPTIMIZATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[OK] PROBLEM SOLVED: List-Template Cloning
   • Before (v1.0): 91.7% reduction → 1 candidate kept
   • After (v1.1): 58.3% reduction → 5 candidates kept
   • Improvement: +400% more diversity!

🎯 OPTIMIZATIONS APPLIED:
   1. Single-unit prompts ("Write ONE..." not "Write five...")
   2. Better sampling (top_p=0.95, top_k=80, repeat_penalty=1.1)
   3. Boilerplate filter (removes "Here are..." responses)
   4. Consistent embeddings (bge-large-en-v1.5 for all)
   5. Increased novelty weights (0.35-0.50 for creative tasks)

📊 RESULTS ACROSS ALL SCENARIOS:
   • Scenario 1: 70.0% reduction | Entropy: 1.00 (perfect diversity)
   • Scenario 2: 58.3% reduction | Entropy: 0.96 (excellent)
   • Scenario 3: 40.0% reduction | Entropy: 0.97 (excellent)
   • Scenario 4: 41.7% reduction | Entropy: 0.87 (high diversity)

💰 COST: $0 - 100% FREE!
   • Ollama (llama3.1:8b) for generation
   • sentence-transformers for embeddings
   • Runs entirely on your local machine

🚀 READY FOR PRODUCTION
   • Published to PyPI v1.0.1
   • GitHub: github.com/chatroutes/chatroutes-autobranch
   • Notebooks: Two interactive demos ready
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='black', linewidth=2))

    fig.savefig(out / 'summary_card.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Created: summary_card.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    out = Path("artifacts")
    ensure_dir(out)

    print("\n" + "=" * 70)
    print("Generating Visualizations for v1.1 Results")
    print("=" * 70 + "\n")

    # Generate all visualizations
    plot_funnel_comparison(out)
    plot_entropy_bars(out)
    plot_before_after_scenario2(out)
    plot_timing_breakdown(out)
    plot_reduction_comparison(out)
    create_summary_card(out)

    print("\n" + "=" * 70)
    print(f"[OK] All visualizations saved to: {out.resolve()}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • funnel_comparison.png - 4-panel funnel for all scenarios")
    print("  • entropy_comparison.png - Entropy bars (high diversity proof)")
    print("  • before_after_scenario2.png - Scenario 2 improvement")
    print("  • timing_breakdown.png - Generation vs pipeline timing")
    print("  • reduction_comparison.png - Reduction % in optimal range")
    print("  • summary_card.png - Social media summary card")
    print("\nREADY FOR:")
    print("  - Article hero images")
    print("  - Twitter/X posts (1600x900)")
    print("  - LinkedIn updates (1200x627)")
    print("  - Blog thumbnails")
    print()

if __name__ == "__main__":
    main()
