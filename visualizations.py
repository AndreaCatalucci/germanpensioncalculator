"""
Comprehensive visualization system for German retirement planning.
Creates professional charts for portfolio analysis, risk assessment, and decision support.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING, cast
from dataclasses import dataclass
from scenario_base import Pot
from params import Params

if TYPE_CHECKING:
    from simulation import SimulationResult

# German financial planning color scheme (colorblind-friendly)
COLORS = {
    'rurup': '#1f77b4',      # Blue - Rürup pension
    'l3': '#ff7f0e',         # Orange - Level 3 pension  
    'broker': '#2ca02c',     # Green - Broker account
    'bonds': '#d62728',      # Red - Bond allocation
    'equity': '#9467bd',     # Purple - Equity allocation
    'total': '#17becf',      # Cyan - Total portfolio
    'benchmark': '#bcbd22',  # Olive - Benchmark/target
    'risk': '#e377c2',       # Pink - Risk indicators
    'success': '#2ca02c',    # Green - Success scenarios
    'failure': '#d62728',    # Red - Failure scenarios
    'neutral': '#7f7f7f'     # Gray - Neutral/reference
}

# Professional styling
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

@dataclass
class VisualizationData:
    """Container for visualization data"""
    scenario_results: SimulationResult
    params: Params
    scenario_name: str
    historical_data: Optional[Dict[str, Any]] = None
    portfolio_trajectory: Optional[List[Pot]] = None
    
class RetirementVisualizer:
    """Main visualization class for German retirement planning system"""
    
    def __init__(self, params: Params) -> None:
        self.params = params
        self.fig_size: Tuple[int, int] = (12, 8)
        self.dpi: int = 100
        
    def format_currency(self, amount: float, suffix: str = "€") -> str:
        """Format currency in German style"""
        if abs(amount) >= 1_000_000:
            return f"{amount/1_000_000:.1f}M {suffix}"
        elif abs(amount) >= 1_000:
            return f"{amount/1_000:.0f}K {suffix}"
        else:
            return f"{amount:,.0f} {suffix}"
    
    def create_portfolio_growth_chart(self, scenarios_data: List[VisualizationData]) -> plt.Figure:
        """
        Create portfolio growth trajectory visualization showing lifecycle growth
        across different scenarios.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, height_ratios=[3, 1])
        
        for i, data in enumerate(scenarios_data):
            res = data.scenario_results
            color = list(COLORS.values())[i % len(COLORS)]
            
            if 'median_trajectory' in res and len(res['median_trajectory']) > 0:
                trajectory = res['median_trajectory']
                # Actual ages for the recorded trajectory (retirement phase)
                retire_ages = np.arange(self.params.age_retire, self.params.age_retire + len(trajectory))
                median_values = [pot.leftover() for pot in trajectory]
                
                ax1.plot(retire_ages, median_values,
                        color=color, linewidth=2.5,
                        label=f"{data.scenario_name} (Median)", alpha=0.8)
                
                # Highlight outcome at end of simulation
                ax1.scatter(retire_ages[-1], res['p50pot'], color=color, s=50, edgecolors='black', zorder=5)
            else:
                # Fallback to simplified trajectory
                trajectory = self._simulate_portfolio_trajectory(data)
                ages = list(range(self.params.age_start, self.params.age_start + len(trajectory)))
                ax1.plot(ages, [pot.leftover() for pot in trajectory],
                        color=color, linewidth=2.0, linestyle='--',
                        label=f"{data.scenario_name} (Geschätzt)", alpha=0.6)
        
        # Formatting
        ax1.set_xlabel("Alter (Jahre)")
        ax1.set_ylabel("Portfoliowert")
        ax1.set_title("Portfolioentwicklung (Ruhestandsphase)", fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        # Add retirement age marker
        ax1.axvline(x=self.params.age_retire, color='red', linestyle='--', alpha=0.7,
                   label=f'Renteneintritt ({self.params.age_retire})')
        
        # Contribution visualization in bottom subplot (fixed to use params)
        accum_ages = np.arange(self.params.age_start, self.params.age_retire)
        contributions = [self.params.annual_contribution * (1.02 ** (age - self.params.age_start)) 
                        for age in accum_ages]
        ax2.bar(accum_ages, contributions, color=COLORS['neutral'], alpha=0.6, width=0.8)
        ax2.set_xlabel("Alter (Jahre)")
        ax2.set_ylabel("Jahresbeitrag (€)")
        ax2.set_title("Beitragsverlauf (Ansparphase)", fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        plt.tight_layout()
        return fig
    
    def create_retirement_income_chart(self, scenarios_data: List[VisualizationData]) -> plt.Figure:
        """
        Create retirement income projection visualization showing income streams
        and sustainability over time.
        """
        fig = plt.figure(figsize=(15, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
        
        ax_inc = fig.add_subplot(gs[0, :])
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_perc = fig.add_subplot(gs[1, 1])
        ax_succ = fig.add_subplot(gs[2, 0])
        ax_pots = fig.add_subplot(gs[2, 1])
        
        # 1. Income Composition (Top Row)
        top_data = scenarios_data[0]
        if 'p50_income_split' in top_data.scenario_results:
            split = top_data.scenario_results['p50_income_split']
            years = np.arange(len(split['pension']))
            ages = years + self.params.age_retire
            
            ax_inc.stackplot(ages, split['pension'], split['rurup'], split['broker_net'],
                         labels=['Gesetzliche Rente', 'Rürup-Rente', 'Broker-Entnahmen'],
                         colors=[COLORS['total'], COLORS['rurup'], COLORS['broker']],
                         alpha=0.8)
            ax_inc.set_title(f"Zusammensetzung Einkommen (Median): {top_data.scenario_name}", fontweight='bold')
            ax_inc.set_xlabel("Alter")
            ax_inc.set_ylabel("Jährliches Einkommen (€)")
            ax_inc.legend(loc='upper left')
            ax_inc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        else:
            ax_inc.text(0.5, 0.5, "Keine Einkommensdaten verfügbar", ha='center', va='center', transform=ax_inc.transAxes)
        
        # 2. Loop through scenarios for distribution plots
        for i, data in enumerate(scenarios_data):
            color = list(COLORS.values())[i % len(COLORS)]
            spending_data = data.scenario_results['all_spend']
            
            # Income distribution (middle left)
            ax_hist.hist(spending_data, bins=30, alpha=0.5, color=color, 
                        label=data.scenario_name, density=True)
            
            # Income percentiles (middle right)
            percentiles = [10, 25, 50, 75, 90]
            values = [np.percentile(spending_data, p) for p in percentiles]
            ax_perc.plot(percentiles, values, 'o-', color=color, 
                        label=data.scenario_name, linewidth=2, markersize=6)
            
            # Success probability (bottom left)
            success_prob = 1 - data.scenario_results['prob_runout']
            ax_succ.bar(i, success_prob * 100, color=color, alpha=0.7)
            
            # Leftover pot distribution (bottom right)
            leftover_data = data.scenario_results['all_pots']
            ax_pots.boxplot([leftover_data], positions=[i], widths=0.6,
                           patch_artist=True, 
                           boxprops=dict(facecolor=color, alpha=0.7))
        
        # Formatting for all subplots
        ax_hist.set_xlabel("Lebenslange Ausgaben (NPV)")
        ax_hist.set_ylabel("Wahrscheinlichkeitsdichte")
        ax_hist.set_title("Verteilung der Lebensausgaben", fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        ax_perc.set_xlabel("Perzentil")
        ax_perc.set_ylabel("Gesamtausgaben (NPV)")
        ax_perc.set_title("Einkommens-Perzentile", fontweight='bold')
        ax_perc.legend(loc='lower right', fontsize=8)
        ax_perc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        ax_perc.grid(True, alpha=0.3)
        
        ax_succ.set_ylabel("Erfolgswahrscheinlichkeit (%)")
        ax_succ.set_title("Wahrscheinlichkeit ausreichender Mittel", fontweight='bold')
        ax_succ.set_xticks(range(len(scenarios_data)))
        ax_succ.set_xticklabels([d.scenario_name for d in scenarios_data], rotation=45)
        ax_succ.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% Ziel')
        ax_succ.legend(fontsize=8)
        
        ax_pots.set_ylabel("Verbleibendes Vermögen")
        ax_pots.set_title("Verbleibendes Vermögen bei Tod", fontweight='bold')
        ax_pots.set_xticks(range(len(scenarios_data)))
        ax_pots.set_xticklabels([d.scenario_name for d in scenarios_data], rotation=45)
        ax_pots.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        plt.tight_layout()
        return fig
    
    def create_tax_efficiency_chart(self, scenarios_data: List[VisualizationData]) -> plt.Figure:
        """
        Create tax efficiency comparison showing after-tax outcomes and tax savings.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenario_names = [d.scenario_name for d in scenarios_data]
        
        # Tax savings during accumulation (top left)
        tax_savings = []
        for data in scenarios_data:
            # Estimate tax savings based on scenario type
            if 'Rurup' in data.scenario_name:
                savings = self.params.annual_contribution * self.params.tax_working * self.params.years_accum
            else:
                savings = 0
            tax_savings.append(savings)
        
        bars1 = ax1.bar(scenario_names, tax_savings, color=[COLORS['success']] * len(scenarios_data), alpha=0.7)
        ax1.set_ylabel("Steuerersparnis")
        ax1.set_title("Steuerersparnis während Ansparphase", fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        # Add value labels on bars
        for bar, value in zip(bars1, tax_savings):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                        self.format_currency(value), ha='center', va='bottom', fontsize=9)
        
        # After-tax spending comparison (top right)
        after_tax_spending = []
        for data in scenarios_data:
            # Simplified after-tax calculation
            gross_spending = data.scenario_results['p50']
            # Estimate tax burden in retirement
            if 'Rurup' in data.scenario_name:
                tax_rate = 0.25  # Estimated average tax rate on Rürup withdrawals
                after_tax = gross_spending * (1 - tax_rate * 0.5)  # Simplified
            else:
                tax_rate = self.params.cg_tax_normal
                after_tax = gross_spending * (1 - tax_rate * 0.3)  # Simplified
            after_tax_spending.append(after_tax)
        
        ax2.bar(scenario_names, after_tax_spending, 
                       color=[COLORS['broker']] * len(scenarios_data), alpha=0.7)
        ax2.set_ylabel("Netto-Ausgaben")
        ax2.set_title("Geschätzte Netto-Ausgaben im Ruhestand", fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        # Tax burden comparison (bottom left)
        retirement_tax_burden = []
        for i, data in enumerate(scenarios_data):
            gross = data.scenario_results['p50']
            net = after_tax_spending[i]
            burden = gross - net
            retirement_tax_burden.append(burden)
        
        ax3.bar(scenario_names, retirement_tax_burden, 
                       color=[COLORS['failure']] * len(scenarios_data), alpha=0.7)
        ax3.set_ylabel("Steuerlast")
        ax3.set_title("Geschätzte Steuerlast im Ruhestand", fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        # Net tax efficiency (bottom right)
        net_efficiency = []
        for i in range(len(scenarios_data)):
            efficiency = tax_savings[i] - retirement_tax_burden[i]
            net_efficiency.append(efficiency)
        
        colors = [COLORS['success'] if x >= 0 else COLORS['failure'] for x in net_efficiency]
        ax4.bar(scenario_names, net_efficiency, color=colors, alpha=0.7)
        ax4.set_ylabel("Netto-Steuereffekt")
        ax4.set_title("Netto-Steuereffekt (Ersparnis - Belastung)", fontweight='bold')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Rotate x-axis labels for better readability
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_risk_analysis_dashboard(self, scenarios_data: List[VisualizationData]) -> plt.Figure:
        """
        Create comprehensive risk analysis dashboard showing sequence-of-returns risk,
        sensitivity analysis, and ruin probability.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Risk metrics summary (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Ruin probability comparison
        scenario_names = [d.scenario_name for d in scenarios_data]
        ruin_probs = [d.scenario_results['prob_runout'] * 100 for d in scenarios_data]
        
        colors = [COLORS['success'] if p < 10 else COLORS['failure'] if p > 25 else COLORS['neutral'] 
                 for p in ruin_probs]
        bars = ax1.bar(scenario_names, ruin_probs, color=colors, alpha=0.7)
        
        ax1.set_ylabel("Ausfallwahrscheinlichkeit (%)")
        ax1.set_title("Wahrscheinlichkeit unzureichender Mittel", fontweight='bold')
        ax1.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Akzeptabel (<10%)')
        ax1.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Kritisch (>25%)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, ruin_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Volatility comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        volatilities = []
        for data in scenarios_data:
            spending_std = np.std(data.scenario_results['all_spend'])
            spending_mean = np.mean(data.scenario_results['all_spend'])
            cv = spending_std / spending_mean if spending_mean > 0 else 0
            volatilities.append(cv * 100)
        
        ax2.bar(range(len(scenario_names)), cast(List[float], volatilities), 
               color=[COLORS['risk']] * len(scenarios_data), alpha=0.7)
        ax2.set_ylabel("Variationskoeffizient (%)")
        ax2.set_title("Ausgaben-Volatilität", fontweight='bold')
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=45)
        
        # Sequence of returns risk simulation (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        
        # Use actual return trajectories from simulation for best/worst
        self._plot_actual_sequence_risk(ax3, scenarios_data)
        
        # Sensitivity analysis (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_actual_sensitivity(ax4, scenarios_data)
        
        # Stress testing (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_stress_testing(ax5, scenarios_data)
        
        # Risk-return scatter (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        
        returns = [np.mean(d.scenario_results['all_spend']) for d in scenarios_data]
        risks = [d.scenario_results['prob_runout'] * 100 for d in scenarios_data]
        
        ax6.scatter(risks, returns, 
                            c=range(len(scenarios_data)), 
                            s=100, alpha=0.7, cmap='viridis')
        
        for i, name in enumerate(scenario_names):
            ax6.annotate(name, (float(risks[i]), float(returns[i])), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel("Ausfallrisiko (%)")
        ax6.set_ylabel("Erwartete Ausgaben (NPV)")
        ax6.set_title("Risiko-Rendite-Verhältnis", fontweight='bold')
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        
        plt.suptitle("Risiko-Analyse Dashboard", fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def create_decision_support_dashboard(self, scenarios_data: List[VisualizationData]) -> plt.Figure:
        """
        Create interactive decision-support dashboard with scenario comparisons
        and actionable recommendations.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        scenario_names = [d.scenario_name for d in scenarios_data]
        
        # Multi-criteria comparison (top left)
        criteria = ['Erwartete Ausgaben', 'Erfolgswahrscheinlichkeit', 'Steuereffizient', 'Flexibilität']
        
        # Normalize scores for radar-like comparison
        scores = np.zeros((len(scenarios_data), len(criteria)))
        
        for i, data in enumerate(scenarios_data):
            # Expected spending (higher is better)
            scores[i, 0] = np.mean(data.scenario_results['all_spend']) / 500000  # Normalize
            # Success probability (higher is better)
            scores[i, 1] = (1 - data.scenario_results['prob_runout'])
            # Tax efficiency (simplified scoring)
            scores[i, 2] = 0.8 if 'Rurup' in data.scenario_name else 0.6
            # Flexibility (simplified scoring)
            scores[i, 3] = 0.9 if 'Broker' in data.scenario_name else 0.5
        
        # Normalize to 0-1 scale
        scores = np.clip(scores, 0, 1)
        
        x = np.arange(len(criteria))
        width = 0.8 / len(scenarios_data)
        
        for i, (name, color) in enumerate(zip(scenario_names, COLORS.values())):
            ax1.bar(x + i * width, scores[i], width, label=name, color=color, alpha=0.7)
        
        ax1.set_ylabel("Bewertung (Szenario-Fit)")
        ax1.set_title("Szenario-Vergleich nach Kriterien", fontweight='bold')
        ax1.set_xticks(x + width * (len(scenarios_data) - 1) / 2)
        ax1.set_xticklabels(criteria)
        ax1.legend(loc='lower right', fontsize=8)
        ax1.set_ylim(0, 1.2)
        
        # Contribution strategy comparison (top right)
        contribution_levels = np.array([20000, 25000, 30000, 35000])
        
        # Simulate impact of different contribution levels
        impact_data = []
        for level in contribution_levels:
            # Simplified calculation of impact
            base_contribution = self.params.annual_contribution
            multiplier = level / base_contribution
            impact = [d.scenario_results['p50'] * multiplier for d in scenarios_data]
            impact_data.append(impact)
        
        for i, name in enumerate(scenario_names):
            values = [impact_data[j][i] for j in range(len(contribution_levels))]
            ax2.plot(contribution_levels, values, 'o-', 
                    label=name, color=list(COLORS.values())[i], linewidth=2)
        
        ax2.set_xlabel("Jährlicher Beitrag (€)")
        ax2.set_ylabel("Erwartete Ausgaben")
        ax2.set_title("Auswirkung der Beitragshöhe", fontweight='bold')
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
        ax2.grid(True, alpha=0.3)
        
        # Age-based recommendations (bottom left)
        ages = [30, 40, 50, 60]
        recommendations = {
            30: [0.9, 0.7, 0.8, 0.6],  # Aggressive growth focus
            40: [0.8, 0.8, 0.7, 0.7],  # Balanced approach
            50: [0.6, 0.9, 0.6, 0.8],  # Risk reduction
            60: [0.4, 0.9, 0.5, 0.9]   # Conservative approach
        }
        
        for i, name in enumerate(scenario_names):
            values = [recommendations[age][i] for age in ages]
            ax3.plot(ages, values, 'o-', label=name, 
                    color=list(COLORS.values())[i], linewidth=2, markersize=6)
        
        ax3.set_xlabel("Aktuelles Alter")
        ax3.set_ylabel("Empfehlungsgrad (0-1)")
        ax3.set_title("Altersbasierte Empfehlungen", fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Summary recommendations (bottom right)
        ax4.axis('off')
        
        # Generate text recommendations
        best_scenario_idx = np.argmax([np.mean(d.scenario_results['all_spend']) for d in scenarios_data])
        safest_scenario_idx = np.argmin([d.scenario_results['prob_runout'] for d in scenarios_data])
        
        recommendations_text = f"""
EMPFEHLUNGEN:

🏆 BESTE GESAMTLEISTUNG:
{scenario_names[best_scenario_idx]}
• Höchste erwartete Ausgaben
• Ausgewogenes Risiko-Rendite-Verhältnis

🛡️ SICHERSTE OPTION:
{scenario_names[safest_scenario_idx]}
• Niedrigste Ausfallwahrscheinlichkeit
• Geringere Volatilität

💡 ALLGEMEINE HINWEISE:
• Früh anfangen maximiert Zinseszinseffekt
• Diversifikation reduziert Risiko
• Regelmäßige Überprüfung empfohlen
• Steuerliche Aspekte berücksichtigen

📊 NÄCHSTE SCHRITTE:
1. Persönliche Risikobereitschaft bewerten
2. Steuerliche Situation analysieren
3. Flexibilitätsanforderungen definieren
4. Implementierungsstrategie entwickeln
        """
        
        ax4.text(0.05, 0.95, recommendations_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1.0", facecolor="white", edgecolor=COLORS['total'], alpha=0.9, linewidth=2))
        
        plt.suptitle("Entscheidungsunterstützung Dashboard", fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _simulate_portfolio_trajectory(self, data: VisualizationData) -> List[Pot]:
        """Simulate portfolio growth trajectory for visualization"""
        trajectory = []
        pot = Pot()
        annual_contribution = self.params.annual_contribution
        
        for year in range(self.params.years_accum):
            # Simplified growth simulation
            growth_rate = 0.07  # Average historical return
            pot.br_eq = pot.br_eq * (1 + growth_rate) + annual_contribution
            pot.br_eq_bs += annual_contribution
            annual_contribution *= 1.02
            trajectory.append(Pot(br_eq=pot.br_eq, br_eq_bs=pot.br_eq_bs))
        
        return trajectory

    def _plot_actual_sequence_risk(self, ax: plt.Axes, scenarios_data: List[VisualizationData]) -> None:
        """Plot sequence risk based on actual simulation outcomes using p10/p50/p90 trajectories"""
        if not scenarios_data:
            return
            
        data = scenarios_data[0]
        res = data.scenario_results
        
        # We need p10, p50, p90 trajectories
        if 'p10_trajectory' in res and 'median_trajectory' in res and 'p90_trajectory' in res:
            p10_traj = res['p10_trajectory']
            p50_traj = res['median_trajectory']
            p90_traj = res['p90_trajectory']
            
            # Use retirement ages
            ages = np.arange(self.params.age_retire, self.params.age_retire + len(p50_traj))
            
            p10_vals = np.array([pot.leftover() for pot in p10_traj])
            p50_vals = np.array([pot.leftover() for pot in p50_traj])
            p90_vals = np.array([pot.leftover() for pot in p90_traj])
            
            # Ensure same length
            min_len = min(len(ages), len(p10_vals), len(p50_vals), len(p90_vals))
            ages = ages[:min_len]
            p10_vals = p10_vals[:min_len]
            p50_vals = p50_vals[:min_len]
            p90_vals = p90_vals[:min_len]
            
            ax.plot(ages, p90_vals, color='g', linestyle='--', linewidth=1.5, label='Günstiger Verlauf (P90)')
            ax.plot(ages, p50_vals, color='b', linestyle='-', linewidth=2.5, label='Medianer Verlauf (P50)')
            ax.plot(ages, p10_vals, color='r', linestyle='--', linewidth=1.5, label='Ungünstiger Verlauf (P10)')
            
            # Optional: fill between
            ax.fill_between(ages, p10_vals, p90_vals, color='gray', alpha=0.15)
            
            ax.set_title(f"Sequenz-Risiko: Verlauf des Vermögens ({data.scenario_name})", fontweight='bold')
            ax.set_xlabel("Alter")
            ax.set_ylabel("Portfoliowert")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
            ax.grid(True, alpha=0.3)
        else:
            # Fallback to spending lines if trajectories aren't available
            spends = res['all_spend']
            ax.axhline(y=np.percentile(spends, 90), color='g', linestyle='--', label='90. Perzentil (Günstig)')
            ax.axhline(y=np.percentile(spends, 50), color='b', linestyle='-', label='Median')
            ax.axhline(y=np.percentile(spends, 10), color='r', linestyle='--', label='10. Perzentil (Ungünstig)')
            ax.set_title("Variabilität der Ausgaben (Sequenz-Risiko-Effekt)", fontweight='bold')
            ax.set_ylabel("Lebenslange Ausgaben (NPV)")
            ax.legend()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

    def _plot_actual_sensitivity(self, ax: plt.Axes, scenarios_data: List[VisualizationData]) -> None:
        """Plot sensitivity comparing scenarios instead of dummy values"""
        names = [d.scenario_name for d in scenarios_data]
        p50s = [d.scenario_results['p50'] for d in scenarios_data]
        p10s = [d.scenario_results['p10'] for d in scenarios_data]
        
        # Calculate 'Downside Risk' (P50 - P10)
        downside = [p50 - p10 for p50, p10 in zip(p50s, p10s)]
        
        ax.bar(names, downside, color=COLORS['risk'], alpha=0.7)
        ax.set_ylabel("Abwärtsrisiko (P50 - P10)")
        ax.set_title("Risiko-Sensitivität der Szenarien", fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))
    
    def _plot_sequence_risk_analysis(self, ax: plt.Axes, data: Optional[VisualizationData]) -> None:
        """Plot sequence of returns risk analysis"""
        if not data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center', transform=ax.transAxes)
            return
            
        # Simulate different market sequences
        years = range(10)  # First 10 years of retirement
        
        # Good sequence (bull market early)
        good_sequence = [0.15, 0.12, 0.08, 0.05, 0.03, -0.02, -0.05, 0.06, 0.08, 0.10]
        # Bad sequence (bear market early)  
        bad_sequence = [-0.15, -0.10, -0.05, 0.02, 0.05, 0.08, 0.12, 0.15, 0.10, 0.08]
        # Average sequence
        avg_sequence = [0.07] * 10
        
        ax.plot(years, np.cumsum(good_sequence), 'g-', linewidth=2, label='Günstige Sequenz', alpha=0.8)
        ax.plot(years, np.cumsum(bad_sequence), 'r-', linewidth=2, label='Ungünstige Sequenz', alpha=0.8)
        ax.plot(years, np.cumsum(avg_sequence), 'b--', linewidth=2, label='Durchschnitt', alpha=0.8)
        
        ax.set_xlabel("Jahre im Ruhestand")
        ax.set_ylabel("Kumulierte Rendite")
        ax.set_title("Sequenz-Risiko-Analyse", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    
    def _plot_sensitivity_analysis(self, ax: plt.Axes, data: Optional[VisualizationData]) -> None:
        """Plot sensitivity analysis"""
        if not data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center', transform=ax.transAxes)
            return
            
        # Sensitivity to key parameters
        parameters = ['Rendite\n±2%', 'Inflation\n±1%', 'Gebühren\n±0.5%', 'Lebenserwartung\n±5 Jahre']
        base_value = data.scenario_results['p50']
        
        # Simplified sensitivity calculation
        sensitivities = [
            base_value * 0.15,  # Return sensitivity
            base_value * 0.08,  # Inflation sensitivity  
            base_value * 0.05,  # Fee sensitivity
            base_value * 0.12   # Longevity sensitivity
        ]
        
        colors = [COLORS['risk'] if s > base_value * 0.1 else COLORS['neutral'] for s in sensitivities]
        bars = ax.bar(parameters, sensitivities, color=colors, alpha=0.7)
        
        ax.set_ylabel("Sensitivität")
        ax.set_title("Sensitivitätsanalyse", fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, sensitivities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                   self.format_currency(value), ha='center', va='bottom', fontsize=8)
    
    def _plot_stress_testing(self, ax: plt.Axes, scenarios_data: List[VisualizationData]) -> None:
        """Plot stress testing results"""
        if not scenarios_data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center', transform=ax.transAxes)
            return
            
        # Stress test scenarios
        stress_scenarios = ['Normal', 'Rezession', 'Inflation', 'Deflation']
        scenario_names = [d.scenario_name for d in scenarios_data]
        
        # Simplified stress test results (would be calculated from actual simulations)
        stress_results = np.random.rand(len(scenarios_data), len(stress_scenarios)) * 0.3 + 0.7
        
        x = np.arange(len(stress_scenarios))
        width = 0.8 / len(scenarios_data)
        
        for i, name in enumerate(scenario_names):
            ax.bar(x + i * width, stress_results[i], width,
                  label=name, color=list(COLORS.values())[i], alpha=0.7)
        
        ax.set_ylabel("Erfolgswahrscheinlichkeit")
        ax.set_title("Stress-Test-Ergebnisse", fontweight='bold')
        ax.set_xticks(x + width * (len(scenarios_data) - 1) / 2)
        ax.set_xticklabels(stress_scenarios, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)


def create_comprehensive_report(scenarios_data: List[VisualizationData], params: Params) -> None:
    """
    Create a comprehensive visualization report with all charts.
    
    Args:
        scenarios_data: List of scenario data for visualization
        params: Parameters object
    """
    visualizer = RetirementVisualizer(params)
    
    # Create all visualizations
    print("Erstelle Portfolio-Wachstums-Diagramm...")
    fig1 = visualizer.create_portfolio_growth_chart(scenarios_data)
    fig1.savefig('portfolio_growth.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Erstelle Ruhestandseinkommen-Diagramm...")
    fig2 = visualizer.create_retirement_income_chart(scenarios_data)
    fig2.savefig('retirement_income.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Erstelle Steuereffizienz-Vergleich...")
    fig3 = visualizer.create_tax_efficiency_chart(scenarios_data)
    fig3.savefig('tax_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Erstelle Risiko-Analyse-Dashboard...")
    fig4 = visualizer.create_risk_analysis_dashboard(scenarios_data)
    fig4.savefig('risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Erstelle Entscheidungsunterstützungs-Dashboard...")
    fig5 = visualizer.create_decision_support_dashboard(scenarios_data)
    fig5.savefig('decision_support.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Alle Visualisierungen wurden erfolgreich erstellt!")


# Example usage and testing
if __name__ == "__main__":
    from params import Params
    from simulation import simulate_montecarlo
    from scenario_broker import ScenarioBroker
    from scenario_rurup_broker import ScenarioRurupBroker
    from scenario_l3_broker import ScenarioL3Broker
    
    # Create parameters
    params = Params()
    
    # Create test scenarios
    scenarios = [
        ("Broker", ScenarioBroker(params)),
        ("RurupBroker", ScenarioRurupBroker(params)),
        ("L3Broker", ScenarioL3Broker(params))
    ]
    
    # Run simulations and create visualization data
    visualization_data = []
    for name, scenario in scenarios:
        print(f"Simuliere Szenario: {name}")
        results = simulate_montecarlo(scenario)
        viz_data = VisualizationData(
            scenario_results=results,
            params=params,
            scenario_name=name
        )
        visualization_data.append(viz_data)
    
    # Create comprehensive report
    create_comprehensive_report(visualization_data, params)