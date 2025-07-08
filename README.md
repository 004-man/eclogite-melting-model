# Eclogite Melting Model: Non-Modal Batch Melting Calculator

A comprehensive Python tool for modeling REE behavior during non-modal batch melting of eclogites and related assemblages.

## 🔬 Scientific Purpose

This model calculates trace element (REE) concentrations in melts and residual solids during non-modal batch melting processes, following Shaw (1970) equations. Designed for eclogite melting research in high-pressure metamorphic environments.

## 📊 Model Scenarios

### 1. **Garnet Stability Field - Continuous Melting**
- Eclogite assemblage (Clinopyroxene + Garnet)
- Continuous melt extraction (1-20%)
- Single-stage melting process

### 2. **Garnet Stability Field - Stepwise Melting**  
- Eclogite assemblage (Clinopyroxene + Garnet)
- Stepwise melt extraction (5% per step, 6 steps)
- Modal composition evolution tracking

### 3. **Plagioclase Stability Field - Continuous Melting**
- Shallow assemblage (Olivine + Clinopyroxene + Plagioclase)
- Continuous melt extraction (1-20%)
- Lower pressure conditions

## 🎯 Key Features

- ✅ **Shaw (1970) batch melting equations** with mass balance verification
- ✅ **Non-modal melting calculations** (Bulk D ≠ Bulk P)
- ✅ **Modal composition evolution** in stepwise scenarios
- ✅ **REE chondrite normalization** (McDonough & Sun, 1995)
- ✅ **Dynamic Excel input system** - easily change parameters
- ✅ **Comprehensive outputs**: Excel files + REE spider plots
- ✅ **Publication-ready visualizations**

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
2. Prepare Input File

Use the provided Eclogite_melting_Input.xlsx template
Modify REE concentrations, Kd values, and modal compositions as needed
Save your input file

3. Run the Model
pythonfrom batch_melting_model import BatchMeltingModel

# Initialize with your input file
model = BatchMeltingModel('Eclogite_melting_Input.xlsx')

# Calculate all scenarios
results = model.run_all_scenarios()
4. View Results

Excel Output: Batch_Melting_Results.xlsx (3 sheets with detailed calculations)
Plots: REE_Spider_Plots.png (chondrite-normalized REE patterns)

📁 Input File Structure
The Excel input file contains three sheets:
Garnet Sheet

Starting REE concentrations (ppm)
Kd values for each mineral-melt pair
Modal proportions (Cpx: 50%, Grt: 50%)
Melting reaction (Cpx: 60%, Grt: 40%)
Stepwise parameters (5% per step)

Plagioclase Sheet

Starting REE concentrations (ppm)
Kd values including plagioclase
Modal proportions (Ol: 20%, Cpx: 30%, Plg: 50%)
Melting reaction (Cpx: 60%, Plg: 40%)

Chondrite Value Sheet

McDonough & Sun (1995) chondrite normalization values

📊 Output Files
Excel Results (Batch_Melting_Results.xlsx)
Three sheets containing:

Input parameters and constraints
Bulk partition coefficients (D and P)
Melt and residue concentrations
Chondrite-normalized values
Step-by-step modal evolution (stepwise scenario)

REE Spider Plots (REE_Spider_Plots.png)

2×3 subplot grid
Top row: Melt patterns (enriched)
Bottom row: Residue patterns (depleted)
Chondrite-normalized REE abundances

🔧 Customization
Change Model Parameters
Simply modify the Excel input file:

Starting compositions: Update Column B values
Kd values: Modify partition coefficients from literature
Modal assemblages: Change mineral percentages
Melting reactions: Adjust reaction stoichiometry
Step size: Modify stepwise extraction percentage

Add New Elements

Add rows for new trace elements
Include corresponding Kd values
Update chondrite normalization values if needed

📚 Scientific Background
Shaw (1970) Equations
Melt concentration:
C_melt = C_0 / [D + F(1-P)]
Residue concentration:
C_residue = (C_0 - F×C_melt) / (1-F)
Where:

C_0 = initial concentration
F = melt fraction
D = bulk partition coefficient (source assemblage)
P = bulk partition coefficient (melting assemblage)

Bulk Partition Coefficients
D = Σ(X_i × Kd_i)    # Source assemblage
P = Σ(p_i × Kd_i)    # Melting assemblage
Where:

X_i = modal proportion of mineral i
p_i = proportion of mineral i in melting reaction
Kd_i = mineral/melt partition coefficient

🎓 Citation
If you use this model in your research, please cite:
Sarbajit Dash (2025). "Eclogite Melting Model: Non-Modal Batch Melting Calculator." 
GitHub repository: https://github.com/004-man/eclogite-melting-model
📖 References

Shaw, D.M. (1970). Trace element fractionation during anatexis. Geochimica et Cosmochimica Acta, 34(2), 237-243.
McDonough, W.F. & Sun, S.S. (1995). The composition of the Earth. Chemical Geology, 120(3-4), 223-253.

🔬 Applications
This model is suitable for:

Eclogite melting studies in subduction zones
REE behavior during high-pressure metamorphism
Trace element modeling in crustal recycling
Melt-residue relationships in deep crustal processes

📞 Support
For questions about the model implementation or scientific applications, please open an issue in this repository.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Keywords: eclogite, batch melting, REE, trace elements, geochemistry, Shaw equations, non-modal melting, chondrite normalization
