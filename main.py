#!/usr/bin/env python3
"""
German Retirement Planning Analysis - Main Entry Point

This script provides a complete retirement planning analysis workflow:
1. Load parameters from params.py
2. Run Monte Carlo simulations for multiple scenarios
3. Generate comprehensive visualizations
4. Display formatted analysis results
5. Save charts as PNG files

Usage: poetry run python main.py
"""

from __future__ import annotations
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

# Import system components
from params import Params
from simulation import simulate_montecarlo, SimulationResult
# Core scenarios (simplified to 3 best strategies)
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_enhanced_l3_broker import ScenarioEnhancedL3Broker
from visualizations import VisualizationData, RetirementVisualizer

if TYPE_CHECKING:
    from scenario_base import Scenario

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retirement_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RetirementAnalyzer:
    """Main class for comprehensive German retirement planning analysis"""
    
    def __init__(self) -> None:
        self.params: Optional[Params] = None
        self.scenarios: List[Tuple[str, Scenario, str]] = []
        self.results: Dict[str, SimulationResult] = {}
        self.visualization_data: List[VisualizationData] = []
        
    def load_parameters(self) -> bool:
        """Load and validate parameters from params.py"""
        try:
            print("🔧 Lade Parameter aus params.py...")
            self.params = Params()
            
            # Display key parameters
            print(f"   📊 Startalter: {self.params.age_start} Jahre")
            print(f"   📊 Übergangsalter: {self.params.age_transition} Jahre") 
            print(f"   📊 Rentenalter: {self.params.age_retire} Jahre")
            print(f"   📊 Jährlicher Beitrag: €{self.params.annual_contribution:,}")
            print(f"   📊 Gewünschte Jahresausgaben: €{self.params.desired_spend:,}")
            print(f"   📊 Anzahl Simulationen: {self.params.num_sims:,}")
            print(f"   📊 Gesetzliche Rente: €{self.params.public_pension:,}/Jahr")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Parameter: {e}")
            logger.error(f"Parameter loading failed: {e}")
            return False
    
    def setup_scenarios(self) -> bool:
        """Setup retirement scenarios for comparison"""
        try:
            print("\n🎯 Richte Szenarien ein...")
            
            assert self.params is not None
            # Simplified to 3 best strategies covering all retirement pillars
            self.scenarios = [
                ("Broker", ScenarioBroker(self.params), "Reines Broker-Depot (Baseline)"),
                ("RurupBroker", ScenarioRurupBroker(self.params), "Rürup + Broker (Steueroptimiert)"),
                ("EnhancedL3Broker", ScenarioEnhancedL3Broker(self.params), "L3 + Broker Optimiert (Max Tax Efficiency)")
            ]
            
            for name, scenario, description in self.scenarios:
                print(f"   ✅ {name}: {description}")
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Einrichten der Szenarien: {e}")
            logger.error(f"Scenario setup failed: {e}")
            return False
    
    def run_simulations(self) -> bool:
        """Run Monte Carlo simulations for all scenarios"""
        try:
            assert self.params is not None
            print(f"\n🎲 Führe Monte Carlo Simulationen durch ({self.params.num_sims:,} Läufe pro Szenario)...")
            
            total_scenarios = len(self.scenarios)
            
            for i, (name, scenario, description) in enumerate(self.scenarios, 1):
                print(f"\n   📈 Simuliere Szenario {i}/{total_scenarios}: {name}")
                print(f"      {description}")
                
                # Show progress
                start_time = time.time()
                
                try:
                    # Run simulation
                    result = simulate_montecarlo(scenario)
                    
                    # Store results
                    self.results[name] = result
                    
                    # Create visualization data
                    viz_data = VisualizationData(
                        scenario_results=result,
                        params=self.params,
                        scenario_name=name
                    )
                    self.visualization_data.append(viz_data)
                    
                    # Show completion time
                    elapsed = time.time() - start_time
                    print(f"      ✅ Abgeschlossen in {elapsed:.1f}s")
                    
                    # Show key metrics
                    self._display_scenario_summary(name, result)
                    
                except Exception as e:
                    print(f"      ❌ Simulation fehlgeschlagen: {e}")
                    logger.error(f"Simulation failed for {name}: {e}")
                    return False
            
            print("\n✅ Alle Simulationen erfolgreich abgeschlossen!")
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei den Simulationen: {e}")
            logger.error(f"Simulation execution failed: {e}")
            return False
    
    def _display_scenario_summary(self, name: str, result: SimulationResult) -> None:
        """Display summary metrics for a scenario"""
        try:
            prob_runout = result['prob_runout'] * 100
            p50_spend = result['p50']
            p50_pot = result['p50pot']
            
            print(f"         💰 Median Ausgaben: €{p50_spend:,.0f}")
            print(f"         🏦 Median verbleibendes Vermögen: €{p50_pot:,.0f}")
            print(f"         ⚠️  Ausfallwahrscheinlichkeit: {prob_runout:.1f}%")
            
        except Exception as e:
            print(f"         ❌ Fehler bei Zusammenfassung: {e}")
    
    def generate_visualizations(self) -> bool:
        """Generate all visualization charts"""
        try:
            print("\n📊 Erstelle Visualisierungen...")
            
            if not self.visualization_data:
                print("❌ Keine Visualisierungsdaten verfügbar")
                return False
            
            assert self.params is not None
            # Create visualizer
            visualizer = RetirementVisualizer(self.params)
            
            # Generate 5 simplified, focused charts
            charts = [
                ("Vermögens-Trajektorien (P10/P50/P90)", "graph1_probability_bands.png", visualizer.create_probability_bands_chart),
                ("Erfolgswahrscheinlichkeit", "graph2_success_comparison.png", visualizer.create_success_comparison_chart),
                ("Einkommenszusammensetzung", "graph3_income_composition.png", visualizer.create_income_composition_chart),
                ("Szenario-Vergleich", "graph4_metrics_table.png", visualizer.create_metrics_table_chart),
                ("Risiko-Rendite", "graph5_risk_return.png", visualizer.create_risk_return_scatter_chart)
            ]
            
            for chart_name, filename, chart_function in charts:
                try:
                    print(f"   📈 Erstelle {chart_name}...")
                    fig = chart_function(self.visualization_data)
                    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"      ✅ Gespeichert als {filename}")
                    
                    # Close figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"      ❌ Fehler bei {chart_name}: {e}")
                    logger.error(f"Chart generation failed for {chart_name}: {e}")
            
            print("✅ Alle Visualisierungen erfolgreich erstellt!")
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei der Visualisierung: {e}")
            logger.error(f"Visualization generation failed: {e}")
            return False
    
    def display_comprehensive_analysis(self) -> None:
        """Display comprehensive analysis results"""
        try:
            print("\n" + "="*80)
            print("📋 UMFASSENDE RUHESTANDSPLANUNG - ANALYSEERGEBNISSE")
            print("="*80)
            
            # Analysis timestamp
            print(f"🕐 Analyse erstellt: {datetime.now().strftime('%d.%m.%Y um %H:%M:%S')}")
            print(f"🔢 Simulationen pro Szenario: {self.params.num_sims if self.params else 'N/A':,}")
            
            # Parameter summary
            self._display_parameter_summary()
            
            # Scenario comparison
            self._display_scenario_comparison()
            
            # Risk analysis
            self._display_risk_analysis()
            
            # Recommendations
            self._display_recommendations()
            
            # File outputs
            self._display_output_files()
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ Fehler bei der Anzeige der Analyse: {e}")
            logger.error(f"Analysis display failed: {e}")
    
    def _display_parameter_summary(self) -> None:
        """Display parameter summary"""
        if not self.params:
            print("❌ Keine Parameter geladen")
            return
            
        print("\n📊 PARAMETER-ÜBERSICHT:")
        print(f"   Alter: {self.params.age_start} → {self.params.age_transition} → {self.params.age_retire} Jahre")
        print(f"   Ansparphase: {self.params.years_accum} Jahre")
        print(f"   Übergangsphase: {self.params.years_transition} Jahre")
        print(f"   Jährlicher Beitrag: €{self.params.annual_contribution:,}")
        print(f"   Gewünschte Jahresausgaben: €{self.params.desired_spend:,}")
        print(f"   Gesetzliche Rente: €{self.params.public_pension:,}/Jahr")
        print(f"   Grenzsteuersatz (Erwerbsphase): {self.params.tax_working:.1%}")
        print(f"   Kapitalertragssteuer: {self.params.cg_tax_normal:.1%}")
    
    def _display_scenario_comparison(self) -> None:
        """Display detailed scenario comparison"""
        print("\n🎯 SZENARIO-VERGLEICH:")
        print(f"{'Szenario':<15} {'Median Ausgaben':<18} {'Verbl. Vermögen':<18} {'Ausfallrisiko':<15} {'Bewertung'}")
        print("-" * 80)
        
        # Sort scenarios by median spending (descending)
        sorted_scenarios = sorted(
            [(name, self.results[name]) for name, _, _ in self.scenarios],
            key=lambda x: x[1]['p50'],
            reverse=True
        )
        
        for name, result in sorted_scenarios:
            median_spend = result['p50']
            median_pot = result['p50pot']
            prob_runout = result['prob_runout'] * 100
            
            # Simple rating system
            if prob_runout < 5:
                rating = "🟢 Ausgezeichnet"
            elif prob_runout < 15:
                rating = "🟡 Gut"
            elif prob_runout < 25:
                rating = "🟠 Akzeptabel"
            else:
                rating = "🔴 Riskant"
            
            print(f"{name:<15} €{median_spend:>15,.0f} €{median_pot:>15,.0f} {prob_runout:>12.1f}% {rating}")
    
    def _display_risk_analysis(self) -> None:
        """Display risk analysis"""
        print("\n⚠️  RISIKO-ANALYSE:")
        
        for name, result in self.results.items():
            prob_runout = result['prob_runout'] * 100
            p10_spend = result['p10']
            p90_spend = result['p90']
            
            # Calculate coefficient of variation
            all_spend = result['all_spend']
            cv = np.std(all_spend) / np.mean(all_spend) * 100
            
            print(f"\n   📈 {name}:")
            print(f"      Ausfallwahrscheinlichkeit: {prob_runout:.1f}%")
            print(f"      10. Perzentil: €{p10_spend:,.0f}")
            print(f"      90. Perzentil: €{p90_spend:,.0f}")
            print(f"      Variationskoeffizient: {cv:.1f}%")
            
            # Risk assessment
            if prob_runout < 10:
                risk_level = "🟢 Niedrig"
            elif prob_runout < 20:
                risk_level = "🟡 Mittel"
            else:
                risk_level = "🔴 Hoch"
            
            print(f"      Risikobewertung: {risk_level}")
    
    def _display_recommendations(self) -> None:
        """Display actionable recommendations"""
        print("\n💡 EMPFEHLUNGEN:")
        
        # Find best and safest scenarios
        best_scenario = max(self.results.items(), key=lambda x: x[1]['p50'])
        safest_scenario = min(self.results.items(), key=lambda x: x[1]['prob_runout'])
        
        print(f"\n   🏆 BESTE GESAMTLEISTUNG: {best_scenario[0]}")
        print(f"      Höchste erwartete Ausgaben: €{best_scenario[1]['p50']:,.0f}")
        print(f"      Ausfallrisiko: {best_scenario[1]['prob_runout']*100:.1f}%")
        
        print(f"\n   🛡️  SICHERSTE OPTION: {safest_scenario[0]}")
        print(f"      Niedrigstes Ausfallrisiko: {safest_scenario[1]['prob_runout']*100:.1f}%")
        print(f"      Erwartete Ausgaben: €{safest_scenario[1]['p50']:,.0f}")
        
        # General recommendations
        print("\n   📋 ALLGEMEINE HINWEISE:")
        print("      • Früher Beginn maximiert den Zinseszinseffekt")
        print("      • Diversifikation über mehrere Schichten reduziert Risiko")
        print("      • Regelmäßige Überprüfung und Anpassung empfohlen")
        print("      • Steuerliche Optimierung kann Ergebnis erheblich verbessern")
        print("      • Inflationsschutz durch Sachwerte wichtig")
        
        # Specific recommendations based on results
        avg_runout = np.mean([r['prob_runout'] for r in self.results.values()])
        if avg_runout > 0.15:
            print("      ⚠️  WARNUNG: Hohes Ausfallrisiko - Beiträge erhöhen oder Ausgaben reduzieren!")
        
        # Age-specific advice
        assert self.params is not None
        current_age = self.params.age_start
        if current_age < 40:
            print(f"      🎯 Alter {current_age}: Fokus auf Wachstum, höhere Aktienquote vertretbar")
        elif current_age < 50:
            print(f"      🎯 Alter {current_age}: Ausgewogene Strategie, Risiko schrittweise reduzieren")
        else:
            print(f"      🎯 Alter {current_age}: Kapitalerhalt wichtiger, konservativere Allokation")
    
    def _display_output_files(self) -> None:
        """Display information about generated files"""
        print("\n📁 GENERIERTE DATEIEN:")
        
        output_files = [
            ("portfolio_growth.png", "Portfolio-Wachstum während Ansparphase"),
            ("retirement_income.png", "Ruhestandseinkommen und Erfolgswahrscheinlichkeit"),
            ("tax_efficiency.png", "Steuereffizienz-Vergleich der Szenarien"),
            ("risk_analysis.png", "Umfassende Risiko-Analyse"),
            ("decision_support.png", "Entscheidungsunterstützung mit Empfehlungen"),
            ("retirement_analysis.log", "Detailliertes Analyse-Protokoll")
        ]
        
        for filename, description in output_files:
            print(f"   📄 {filename:<25} - {description}")
        
        print("\n   💡 Tipp: Öffnen Sie die PNG-Dateien für detaillierte Visualisierungen!")
    
    def run_complete_analysis(self) -> bool:
        """Run the complete retirement planning analysis"""
        try:
            print("🚀 DEUTSCHE RUHESTANDSPLANUNG - VOLLSTÄNDIGE ANALYSE")
            print("="*60)
            
            # Step 1: Load parameters
            if not self.load_parameters():
                return False
            
            # Step 2: Setup scenarios
            if not self.setup_scenarios():
                return False
            
            # Step 3: Run simulations
            if not self.run_simulations():
                return False
            
            # Step 4: Generate visualizations
            if not self.generate_visualizations():
                return False
            
            # Step 5: Display comprehensive analysis
            self.display_comprehensive_analysis()
            
            print("\n🎉 ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
            print("   Überprüfen Sie die generierten PNG-Dateien für detaillierte Visualisierungen.")
            print("   Passen Sie params.py an und führen Sie die Analyse erneut aus, um verschiedene Szenarien zu testen.")
            
            return True
            
        except Exception as e:
            print(f"\n❌ KRITISCHER FEHLER: {e}")
            logger.error(f"Complete analysis failed: {e}")
            return False


def main():
    """Main entry point for the retirement planning analysis"""
    try:
        # Create analyzer instance
        analyzer = RetirementAnalyzer()
        
        # Run complete analysis
        success = analyzer.run_complete_analysis()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analyse durch Benutzer abgebrochen.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()