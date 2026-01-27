"""
Enhanced German retirement planning calculator with comprehensive visualizations.
Integrates Monte Carlo simulations with professional visualization system.
"""

from __future__ import annotations
from typing import Optional, Dict, Type, TYPE_CHECKING
from params import Params
from scenario_broker import ScenarioBroker
from scenario_enhanced_broker import ScenarioEnhancedBroker
from scenario_enhanced_l3_broker import ScenarioEnhancedL3Broker
from scenario_l3_broker import ScenarioL3Broker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_safe_spend import ScenarioSafeSpend
from simulation import simulate_montecarlo, SimulationResult
from visualizations import VisualizationData, RetirementVisualizer
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from scenario_base import Scenario


class EnhancedRetirementCalculator:
    """Enhanced retirement calculator with comprehensive visualization capabilities"""
    
    def __init__(self, params: Optional[Params] = None) -> None:
        self.params = params or Params()
        self.visualizer = RetirementVisualizer(self.params)
        self.scenarios: Dict[str, Scenario] = {}
        self.results: Dict[str, SimulationResult] = {}
        
    def add_scenario(self, name: str, scenario_class: Type[Scenario]) -> None:
        """Add a scenario to be analyzed"""
        self.scenarios[name] = scenario_class(self.params)
        
    def run_all_simulations(self):
        """Run Monte Carlo simulations for all scenarios"""
        print("🚀 Starte Monte Carlo Simulationen...")
        print(f"📊 Parameter: {self.params.num_sims:,} Simulationen, {self.params.years_accum} Jahre Ansparphase")
        print("-" * 60)
        
        for name, scenario in self.scenarios.items():
            print(f"⚡ Simuliere Szenario: {name}")
            result = simulate_montecarlo(scenario)
            self.results[name] = result
            
            # Print key metrics
            print(f"   💰 Median Ausgaben: {result['p50']:,.0f} €")
            print(f"   📈 P90 Ausgaben: {result['p90']:,.0f} €")
            print(f"   ⚠️  Ausfallrisiko: {result['prob_runout']*100:.1f}%")
            print(f"   🏦 Verbleibendes Vermögen: {result['p50pot']:,.0f} €")
            print()
            
        print("✅ Alle Simulationen abgeschlossen!")
        return self.results
    
    def create_visualization_data(self) -> list[VisualizationData]:
        """Convert simulation results to visualization data"""
        viz_data = []
        for name, result in self.results.items():
            data = VisualizationData(
                scenario_results=result,
                params=self.params,
                scenario_name=name
            )
            viz_data.append(data)
        return viz_data
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary of results"""
        if not self.results:
            print("❌ Keine Ergebnisse verfügbar. Führen Sie zuerst run_all_simulations() aus.")
            return
            
        print("\n" + "="*80)
        print("📋 ZUSAMMENFASSUNG DER RUHESTANDSPLANUNG")
        print("="*80)
        
        # Sort scenarios by median spending
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['p50'], reverse=True)
        
        print("\n🎯 SZENARIO-RANKING (nach medianen Ausgaben):")
        print("-" * 50)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            risk_level = "🟢 NIEDRIG" if result['prob_runout'] < 0.1 else "🟡 MITTEL" if result['prob_runout'] < 0.25 else "🔴 HOCH"
            
            print(f"{i}. {name}")
            print(f"   💰 Erwartete Ausgaben: {result['p50']:,.0f} € (P10: {result['p10']:,.0f} €, P90: {result['p90']:,.0f} €)")
            print(f"   ⚠️  Ausfallrisiko: {result['prob_runout']*100:.1f}% {risk_level}")
            print(f"   🏦 Verbleibendes Vermögen: {result['p50pot']:,.0f} €")
            print()
        
        # Best scenario analysis
        best_scenario = sorted_results[0]
        safest_scenario = min(self.results.items(), key=lambda x: x[1]['prob_runout'])
        
        print("🏆 EMPFEHLUNGEN:")
        print("-" * 30)
        print(f"💎 Beste Gesamtleistung: {best_scenario[0]}")
        print(f"   → Höchste erwartete Ausgaben: {best_scenario[1]['p50']:,.0f} €")
        print()
        print(f"🛡️  Sicherste Option: {safest_scenario[0]}")
        print(f"   → Niedrigstes Ausfallrisiko: {safest_scenario[1]['prob_runout']*100:.1f}%")
        print()
        
        # Risk analysis
        high_risk_scenarios = [name for name, result in self.results.items() 
                             if result['prob_runout'] > 0.25]
        
        if high_risk_scenarios:
            print("⚠️  RISIKO-WARNUNG:")
            print(f"   Folgende Szenarien haben hohes Ausfallrisiko (>25%): {', '.join(high_risk_scenarios)}")
            print()
        
        # Parameter sensitivity
        print("📊 PARAMETER-ANALYSE:")
        print("-" * 30)
        print(f"💼 Jährlicher Beitrag: {self.params.annual_contribution:,.0f} €")
        print(f"🎂 Renteneintrittsalter: {self.params.age_retire} Jahre")
        print(f"💰 Gewünschte Ausgaben: {self.params.desired_spend:,.0f} € (inflationsbereinigt)")
        print(f"🏛️  Gesetzliche Rente: {self.params.public_pension:,.0f} € monatlich")
        print()
        
        print("💡 ALLGEMEINE HINWEISE:")
        print("-" * 30)
        print("• Früher Beginn maximiert den Zinseszinseffekt")
        print("• Diversifikation zwischen verschiedenen Anlageformen reduziert Risiko")
        print("• Steuerliche Optimierung kann erhebliche Vorteile bringen")
        print("• Regelmäßige Überprüfung und Anpassung der Strategie empfohlen")
        print("• Berücksichtigung der deutschen Steuergesetzgebung wichtig")
        
    def create_all_visualizations(self, save_plots: bool = True):
        """Create all visualization charts"""
        if not self.results:
            print("❌ Keine Ergebnisse verfügbar. Führen Sie zuerst run_all_simulations() aus.")
            return
            
        viz_data = self.create_visualization_data()
        
        print("\n🎨 Erstelle Visualisierungen...")
        print("-" * 40)
        
        # Portfolio Growth Chart
        print("📈 Portfolio-Wachstums-Diagramm...")
        fig1 = self.visualizer.create_portfolio_growth_chart(viz_data)
        if save_plots:
            fig1.savefig('portfolio_growth.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Retirement Income Chart
        print("💰 Ruhestandseinkommen-Analyse...")
        fig2 = self.visualizer.create_retirement_income_chart(viz_data)
        if save_plots:
            fig2.savefig('retirement_income.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Tax Efficiency Chart
        print("💸 Steuereffizienz-Vergleich...")
        fig3 = self.visualizer.create_tax_efficiency_chart(viz_data)
        if save_plots:
            fig3.savefig('tax_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Risk Analysis Dashboard
        print("⚠️  Risiko-Analyse-Dashboard...")
        fig4 = self.visualizer.create_risk_analysis_dashboard(viz_data)
        if save_plots:
            fig4.savefig('risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Decision Support Dashboard
        print("🎯 Entscheidungsunterstützungs-Dashboard...")
        fig5 = self.visualizer.create_decision_support_dashboard(viz_data)
        if save_plots:
            fig5.savefig('decision_support.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if save_plots:
            print("💾 Alle Diagramme wurden als PNG-Dateien gespeichert!")
        
        print("✅ Visualisierungen abgeschlossen!")
    
    def run_comprehensive_analysis(self):
        """Run complete analysis with simulations, summary, and visualizations"""
        print("🚀 STARTE UMFASSENDE RUHESTANDSANALYSE")
        print("="*60)
        
        # Run simulations
        self.run_all_simulations()
        
        # Generate summary
        self.generate_summary_report()
        
        # Create visualizations
        self.create_all_visualizations()
        
        print("\n🎉 ANALYSE ABGESCHLOSSEN!")
        print("="*60)


def create_default_scenarios():
    """Create a comprehensive set of default scenarios for analysis"""
    scenarios = [
        ("Broker", ScenarioBroker),
        ("RurupBroker", ScenarioRurupBroker),
        ("L3Broker", ScenarioL3Broker),
        ("EnhancedBroker", ScenarioEnhancedBroker),
        ("EnhancedL3Broker", ScenarioEnhancedL3Broker),
        ("SafeSpend", ScenarioSafeSpend),
    ]
    return scenarios


def main():
    """Main function to run the enhanced retirement calculator"""
    print("🇩🇪 DEUTSCHER RUHESTANDSPLANER - ERWEITERTE VERSION")
    print("="*60)
    print("Professionelle Monte Carlo Simulation mit umfassenden Visualisierungen")
    print("Optimiert für deutsche Steuergesetze und Rentensysteme")
    print()
    
    # Create calculator with default parameters
    calculator = EnhancedRetirementCalculator()
    
    # Add scenarios
    scenarios = create_default_scenarios()
    for name, scenario_class in scenarios:
        calculator.add_scenario(name, scenario_class)
    
    print(f"📋 Konfigurierte Szenarien: {', '.join(calculator.scenarios.keys())}")
    print()
    
    # Run comprehensive analysis
    calculator.run_comprehensive_analysis()


if __name__ == "__main__":
    main()