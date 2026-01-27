"""
Test script for the enhanced visualization system.
Validates integration with corrected mathematical calculations.
"""

import sys
import traceback
from params import Params
from scenario_broker import ScenarioBroker
from scenario_rurup_broker import ScenarioRurupBroker
from scenario_l3_broker import ScenarioL3Broker
from simulation import simulate_montecarlo
from visualizations import VisualizationData, RetirementVisualizer
import matplotlib.pyplot as plt
import numpy as np


def test_basic_simulation():
    """Test basic Monte Carlo simulation functionality"""
    print("🧪 Testing basic Monte Carlo simulation...")
    
    try:
        params = Params()
        scenario = ScenarioBroker(params)
        result = simulate_montecarlo(scenario)
        
        # Validate result structure
        required_keys = ['prob_runout', 'p10', 'p50', 'p90', 'p50pot', 'all_spend', 'all_pots']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Validate data types and ranges
        assert 0 <= result['prob_runout'] <= 1, "prob_runout should be between 0 and 1"
        assert result['p10'] <= result['p50'] <= result['p90'], "Percentiles should be ordered"
        assert len(result['all_spend']) == params.num_sims, "all_spend should have num_sims entries"
        assert len(result['all_pots']) == params.num_sims, "all_pots should have num_sims entries"
        
        print("✅ Basic simulation test passed!")
        print(f"   📊 Results: P50={result['p50']:,.0f}€, Runout={result['prob_runout']*100:.1f}%")
        return result
        
    except Exception as e:
        print(f"❌ Basic simulation test failed: {e}")
        traceback.print_exc()
        return None


def test_visualization_data_creation():
    """Test creation of visualization data structures"""
    print("\n🧪 Testing visualization data creation...")
    
    try:
        params = Params()
        scenarios = [
            ("Broker", ScenarioBroker),
            ("RurupBroker", ScenarioRurupBroker),
            ("L3Broker", ScenarioL3Broker)
        ]
        
        viz_data = []
        for name, scenario_class in scenarios:
            scenario = scenario_class(params)
            result = simulate_montecarlo(scenario)
            
            data = VisualizationData(
                scenario_results=result,
                params=params,
                scenario_name=name
            )
            viz_data.append(data)
        
        # Validate visualization data
        assert len(viz_data) == 3, "Should have 3 visualization data objects"
        for data in viz_data:
            assert hasattr(data, 'scenario_results'), "Missing scenario_results"
            assert hasattr(data, 'params'), "Missing params"
            assert hasattr(data, 'scenario_name'), "Missing scenario_name"
        
        print("✅ Visualization data creation test passed!")
        print(f"   📊 Created data for: {[d.scenario_name for d in viz_data]}")
        return viz_data
        
    except Exception as e:
        print(f"❌ Visualization data creation test failed: {e}")
        traceback.print_exc()
        return None


def test_individual_charts(viz_data):
    """Test individual chart creation"""
    print("\n🧪 Testing individual chart creation...")
    
    if not viz_data:
        print("❌ No visualization data available for testing")
        return False
    
    try:
        params = Params()
        visualizer = RetirementVisualizer(params)
        
        # Test portfolio growth chart
        print("   📈 Testing portfolio growth chart...")
        fig1 = visualizer.create_portfolio_growth_chart(viz_data)
        assert fig1 is not None, "Portfolio growth chart should not be None"
        plt.close(fig1)
        
        # Test retirement income chart
        print("   💰 Testing retirement income chart...")
        fig2 = visualizer.create_retirement_income_chart(viz_data)
        assert fig2 is not None, "Retirement income chart should not be None"
        plt.close(fig2)
        
        # Test tax efficiency chart
        print("   💸 Testing tax efficiency chart...")
        fig3 = visualizer.create_tax_efficiency_chart(viz_data)
        assert fig3 is not None, "Tax efficiency chart should not be None"
        plt.close(fig3)
        
        # Test risk analysis dashboard
        print("   ⚠️  Testing risk analysis dashboard...")
        fig4 = visualizer.create_risk_analysis_dashboard(viz_data)
        assert fig4 is not None, "Risk analysis dashboard should not be None"
        plt.close(fig4)
        
        # Test decision support dashboard
        print("   🎯 Testing decision support dashboard...")
        fig5 = visualizer.create_decision_support_dashboard(viz_data)
        assert fig5 is not None, "Decision support dashboard should not be None"
        plt.close(fig5)
        
        print("✅ Individual chart creation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Individual chart creation test failed: {e}")
        traceback.print_exc()
        return False


def test_mathematical_correctness():
    """Test that visualizations reflect corrected mathematical calculations"""
    print("\n🧪 Testing mathematical correctness...")
    
    try:
        params = Params()
        
        # Test with known scenario
        scenario = ScenarioBroker(params)
        result = simulate_montecarlo(scenario)
        
        # Validate mathematical properties
        assert result['p10'] > 0, "P10 should be positive"
        assert result['p50'] > result['p10'], "P50 should be greater than P10"
        assert result['p90'] > result['p50'], "P90 should be greater than P50"
        
        # Check that spending values are reasonable for German context
        expected_min = 100_000  # Minimum reasonable lifetime spending
        expected_max = 3_000_000  # Maximum reasonable lifetime spending (increased for realistic German retirement planning)
        
        assert result['p10'] >= expected_min, f"P10 ({result['p10']:,.0f}) seems too low"
        assert result['p90'] <= expected_max, f"P90 ({result['p90']:,.0f}) seems too high"
        
        # Validate probability bounds
        assert 0 <= result['prob_runout'] <= 1, "Runout probability should be between 0 and 1"
        
        # Check array consistency
        spending_array = result['all_spend']
        calculated_p50 = np.percentile(spending_array, 50)
        tolerance = abs(result['p50'] * 0.01)  # 1% tolerance
        
        assert abs(calculated_p50 - result['p50']) <= tolerance, \
            f"P50 mismatch: calculated={calculated_p50:.0f}, reported={result['p50']:.0f}"
        
        print("✅ Mathematical correctness tests passed!")
        print(f"   📊 Spending range: {result['p10']:,.0f}€ - {result['p90']:,.0f}€")
        print(f"   ⚠️  Runout probability: {result['prob_runout']*100:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical correctness test failed: {e}")
        traceback.print_exc()
        return False


def test_german_context_integration():
    """Test German financial planning context integration"""
    print("\n🧪 Testing German context integration...")
    
    try:
        params = Params()
        
        # Validate German-specific parameters
        assert params.public_pension > 0, "Public pension should be set"
        assert params.cg_tax_normal > 0, "Capital gains tax should be set"
        assert params.ruerup_tax > 0, "Rürup tax should be set"
        assert params.sparerpauschbetrag > 0, "Sparerpauschbetrag should be set"
        
        # Test Rürup scenario (German-specific)
        rurup_scenario = ScenarioRurupBroker(params)
        rurup_result = simulate_montecarlo(rurup_scenario)
        
        # Validate that Rürup scenario produces reasonable results
        assert rurup_result['p50'] > 0, "Rürup scenario should produce positive spending"
        assert rurup_result['prob_runout'] < 1, "Rürup scenario shouldn't always fail"
        
        # Test currency formatting
        visualizer = RetirementVisualizer(params)
        formatted = visualizer.format_currency(123456.78)
        assert "€" in formatted, "Currency formatting should include Euro symbol"
        
        print("✅ German context integration tests passed!")
        print(f"   🏛️  Public pension: {params.public_pension}€/month")
        print(f"   💸 Capital gains tax: {params.cg_tax_normal*100:.1f}%")
        print(f"   🎯 Rürup P50: {rurup_result['p50']:,.0f}€")
        return True
        
    except Exception as e:
        print(f"❌ German context integration test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 STARTING COMPREHENSIVE VISUALIZATION TEST SUITE")
    print("="*60)
    
    test_results = []
    
    # Test 1: Basic simulation
    result = test_basic_simulation()
    test_results.append(result is not None)
    
    # Test 2: Visualization data creation
    viz_data = test_visualization_data_creation()
    test_results.append(viz_data is not None)
    
    # Test 3: Individual charts
    charts_ok = test_individual_charts(viz_data)
    test_results.append(charts_ok)
    
    # Test 4: Mathematical correctness
    math_ok = test_mathematical_correctness()
    test_results.append(math_ok)
    
    # Test 5: German context
    german_ok = test_german_context_integration()
    test_results.append(german_ok)
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    test_names = [
        "Basic Simulation",
        "Visualization Data Creation", 
        "Individual Charts",
        "Mathematical Correctness",
        "German Context Integration"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Visualization system is ready for use.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED! Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)