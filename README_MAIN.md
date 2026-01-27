# German Retirement Planning - Main Analysis System

## Overview

The `main.py` file provides a complete, user-friendly workflow for German retirement planning analysis. It integrates all system components to deliver comprehensive analysis with professional visualizations in a single command.

## Quick Start

### 1. Edit Parameters
Customize your retirement scenario by editing [`params.py`](params.py):

```python
# Key parameters to adjust:
age_start = 38              # Your current age
annual_contribution = 12000  # Annual contribution in EUR
desired_spend = 36000       # Desired annual spending in retirement
num_sims = 1000            # Number of Monte Carlo simulations
```

### 2. Run Analysis
Execute the complete analysis:

```bash
poetry run python main.py
```

### 3. Review Results
- **Terminal Output**: Comprehensive analysis with key insights
- **PNG Files**: Professional visualizations saved automatically
- **Log File**: Detailed analysis log (`retirement_analysis.log`)

## What You Get

### 📊 Scenario Comparison
Compares three retirement strategies:
- **Broker**: Pure broker account strategy
- **RurupBroker**: Rürup pension + broker (tax refund invested)
- **L3Broker**: Level 3 pension + broker (40/60 split)

### 📈 Professional Visualizations
Five comprehensive charts automatically generated:

1. **`portfolio_growth.png`** - Portfolio development during accumulation phase
2. **`retirement_income.png`** - Retirement income and success probability
3. **`tax_efficiency.png`** - Tax efficiency comparison across scenarios
4. **`risk_analysis.png`** - Comprehensive risk analysis dashboard
5. **`decision_support.png`** - Decision support with actionable recommendations

### 📋 Comprehensive Analysis
- **Parameter Overview**: All key assumptions and inputs
- **Scenario Comparison**: Side-by-side performance metrics
- **Risk Analysis**: Detailed risk assessment for each strategy
- **Recommendations**: Actionable insights based on your specific situation
- **File Outputs**: Summary of all generated files

## Sample Output

```
🚀 DEUTSCHE RUHESTANDSPLANUNG - VOLLSTÄNDIGE ANALYSE
============================================================

🎯 SZENARIO-VERGLEICH:
Szenario        Median Ausgaben    Verbl. Vermögen    Ausfallrisiko   Bewertung
--------------------------------------------------------------------------------
Broker          €        619,263 €      8,401,405          2.3% 🟢 Ausgezeichnet
L3Broker        €        616,572 €      8,679,474          2.2% 🟢 Ausgezeichnet
RurupBroker     €        610,385 €      7,033,753          6.5% 🟡 Gut

💡 EMPFEHLUNGEN:
   🏆 BESTE GESAMTLEISTUNG: Broker
   🛡️  SICHERSTE OPTION: L3Broker
```

## Key Features

### ✅ Complete Integration
- Uses corrected mathematical calculations
- Leverages professional visualization system
- Integrates with realistic German retirement parameters
- Includes enhanced calculator functionality

### ✅ User-Friendly Workflow
- **Single Command**: Complete analysis in one execution
- **Progress Indicators**: Real-time feedback during analysis
- **Error Handling**: Clear error messages and graceful failure handling
- **Professional Output**: Both terminal and file-based results

### ✅ Comprehensive Analysis
- **Monte Carlo Simulations**: Robust statistical analysis
- **Multiple Scenarios**: Compare different retirement strategies
- **Risk Assessment**: Detailed probability analysis
- **German Context**: All calculations optimized for German tax and pension system

## Customization Options

### Parameter Adjustment
Edit [`params.py`](params.py) to customize:
- **Demographics**: Age, gender, retirement timeline
- **Financial**: Contributions, spending goals, existing assets
- **Tax Rates**: Current tax situation and assumptions
- **Simulation**: Number of Monte Carlo runs, historical data period

### Advanced Usage
For more detailed analysis or custom scenarios:
- Modify individual scenario files (`scenario_*.py`)
- Adjust visualization parameters in [`visualizations.py`](visualizations.py)
- Use individual components for targeted analysis

## Technical Details

### System Requirements
- Python 3.8+
- Poetry for dependency management
- All dependencies automatically managed via `pyproject.toml`

### Performance
- **Fast Execution**: ~1 second per 1,000 simulations per scenario
- **Memory Efficient**: Automatic cleanup of visualization objects
- **Scalable**: Tested with up to 100,000 simulations

### Error Handling
- **Parameter Validation**: Comprehensive input validation
- **Numerical Stability**: Overflow protection and bounds checking
- **Graceful Degradation**: Continues analysis even if individual components fail
- **Detailed Logging**: Complete audit trail in log files

## File Structure

```
├── main.py                    # Main analysis entry point
├── params.py                  # User-configurable parameters
├── simulation.py              # Monte Carlo simulation engine
├── scenario_*.py              # Individual retirement scenarios
├── visualizations.py          # Professional chart generation
├── *.png                      # Generated visualization files
└── retirement_analysis.log    # Detailed analysis log
```

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed:
```bash
poetry install
```

**Visualization Warnings**: Font warnings are cosmetic and don't affect functionality.

**Memory Issues**: Reduce `num_sims` in `params.py` for large-scale analysis.

### Getting Help
- Check `retirement_analysis.log` for detailed error information
- Verify parameter values are within realistic bounds
- Ensure historical data cache is accessible

## Next Steps

1. **Run Initial Analysis**: Use default parameters to understand the system
2. **Customize Parameters**: Adjust `params.py` for your specific situation
3. **Compare Scenarios**: Analyze different contribution levels and strategies
4. **Regular Updates**: Re-run analysis as your situation changes

The system is designed to be your comprehensive tool for German retirement planning, providing professional-grade analysis with minimal setup required.