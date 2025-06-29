# LLM-Designed Evolutionary Algorithm for GNBG Competition

This repository contains a comprehensive implementation of Large Language Model (LLM) designed Evolutionary Algorithms for the GNBG (Generalized Numerical Benchmark Generator) competition at GECCO 2025. The project demonstrates the innovative use of LLMs to design sophisticated evolutionary algorithms for box-constrained numerical global optimization problems.

## 🏆 Competition Overview

The first edition of the LLM-designed EA competition is being held at [GECCO 2025](https://gecco-2025.sigevo.org/HomePage/). This contest explores the potential of Large Language Models in creating sophisticated Evolutionary Algorithms that can tackle complex optimization problems.

### 📅 Important Dates
* **Submission deadline**: 30th June 2025
* **Notification of acceptance**: TBA
* **Early registration deadline**: TBA

## 🎯 Project Achievements

Our LLM-designed Enhanced Genetic Algorithm (EGA) achieved significant improvements over standard Differential Evolution:

- **17.09%** average reduction in mean error values across all 24 benchmark functions
- **6.7%** higher average success rate
- **7.2%** fewer function evaluations on average
- **Total Score**: 19.37 out of 24 (vs 16.86 for DE baseline)

## 📊 GNBG Benchmark

The benchmark consists of 24 test functions for box-constrained numerical global optimization with varying characteristics:

- **Dimensions**: 2D to 64D
- **Function Evaluations**: 500,000 (f1-f15) and 1,000,000 (f16-f24)
- **Features**: Multimodality, ill-conditioning, rotation, non-separability

### Reference
[A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global Optimization," arXiv preprint arXiv:2312.07034, 2023](https://arxiv.org/abs/2312.07034)

## 🔧 Available Implementations

The project provides GNBG benchmark implementations in three programming languages:

### 1. Python Implementation 🐍
- **Location**: [`codes/GNBG-Python/`](codes/GNBG-Python/)
- **Main File**: `GNBG_instances.py`
- **Features**: 
  - LLM-designed Enhanced Genetic Algorithm using OpenRouter API
  - Integration with DeepSeek R1 model
  - Comprehensive comparison with Differential Evolution
  - Automated result generation and visualization
  - Progress tracking and convergence analysis

#### Dependencies
```bash
pip install -r codes/GNBG-Python/requirements.txt
```

### 2. MATLAB Implementation 🔢
- **Location**: [`codes/GNBG-Matlab/`](codes/GNBG-Matlab/)
- **Main File**: `main.m`
- **Features**: 
  - Original MATLAB implementation
  - Differential Evolution optimizer example
  - Direct access to .mat parameter files

### 3. C++ Implementation ⚡
- **Location**: [`codes/GNBG-C/`](codes/GNBG-C/)
- **Main File**: `gnbg-c++.cpp`
- **Features**: 
  - High-performance C++ implementation
  - Differential Evolution with rand/1 strategy
  - Requires parameter conversion from .mat files

#### Compilation
```bash
cd codes/GNBG-C/
python convert.py  # Convert .mat files to .txt
g++ -std=c++11 -O3 gnbg-c++.cpp -o gnbg-c++
./gnbg-c++
```

## 🤖 LLM Integration

Our approach leverages Large Language Models for algorithm design:

### LLM Used
- **Model**: DeepSeek R1 (via OpenRouter API)
- **Temperature**: 0.1 (for consistent parameter recommendations)
- **API**: OpenRouter (https://openrouter.ai/)

### Design Process
1. **Initial Consultation**: LLM analyzes GNBG benchmark characteristics
2. **Algorithm Selection**: Recommendations for suitable EA variants
3. **Parameter Optimization**: LLM suggests optimal parameter configurations
4. **Feedback Loop**: Performance results guide parameter refinement

### Key LLM-Designed Features
- **Hybrid Algorithm**: Genetic Algorithm with Differential Evolution elements
- **Adaptive Mechanisms**: Success-based parameter adaptation
- **Operator Selection**: 
  - Blend crossover (BLX-α) with α = 0.5
  - Gaussian mutation with σ = 0.1
  - Tournament selection with size 7
  - Elitism with 5 individuals

## 🚀 Quick Start

### Running the Python Implementation

1. **Setup Environment**:
```bash
cd codes/GNBG-Python/
pip install -r requirements.txt
```

2. **Configure API** (Optional - will use defaults if not available):
   - Get API key from [OpenRouter](https://openrouter.ai/)
   - Update `OPENROUTER_API_TOKEN` in `GNBG_instances.py`

3. **Run the Algorithm**:
```bash
python GNBG_instances.py
```

### Expected Output
- Individual function results in `results/` directory
- Convergence plots for each function
- Comparison visualizations
- Performance statistics and LLM prompts

## 📁 Project Structure

```
LLMdesignedEA_comp-1/
├── README.md                    # This comprehensive guide
├── GNBG_Documentation.md        # Detailed methodology and results
├── f_x_value.txt               # Example result values format
├── f_x_params.txt              # Example parameters format
├── codes/
│   ├── GNBG-Python/            # Python implementation
│   │   ├── GNBG_instances.py   # Main LLM-designed GA implementation
│   │   ├── requirements.txt    # Python dependencies
│   │   ├── f1.mat - f24.mat   # Benchmark parameter files
│   │   ├── results/           # Generated results and plots
│   │   └── README.md          # Python-specific documentation
│   ├── GNBG-Matlab/           # MATLAB implementation
│   │   ├── main.m             # Main MATLAB script
│   │   ├── fitness.m          # Fitness evaluation function
│   │   ├── f1.mat - f24.mat  # Benchmark parameter files
│   │   └── Figures/           # Visualization plots
│   └── GNBG-C/               # C++ implementation
│       ├── gnbg-c++.cpp      # Main C++ implementation
│       ├── convert.py        # .mat to .txt converter
│       └── f1.mat - f24.mat  # Benchmark parameter files
```

## 📈 Performance Results

### Algorithm Comparison Summary

| Metric | LLM-designed EGA | Standard DE | Improvement |
|--------|------------------|-------------|-------------|
| Mean Error Reduction | - | - | **17.09%** |
| Success Rate | - | - | **+6.7%** |
| Function Evaluations | - | - | **-7.2%** |
| Total Score (out of 24) | **19.37** | 16.86 | **+14.9%** |

### Best Performance Functions
- **f10** (8D): 40.33% improvement
- **f3** (2D): 36.22% improvement  
- **f24** (64D): 25.62% improvement

## 🎯 Competition Participation

### Submission Requirements

To participate in the competition, submit through the [official form](https://forms.gle/HVStaicFG7GytrTN8):

1. **Algorithm Title**: "LLM-Designed Enhanced Genetic Algorithm"
2. **Results ZIP**: Generated automatically in `results/` directory
3. **LLM Details**: DeepSeek R1 via OpenRouter API, temperature = 0.1
4. **Algorithm Code**: Available in this repository
5. **Design Prompts**: Saved in `results/mistral_prompts.txt`

### Results Format

For each function f_x:
- **f_x_value.txt**: 31 best-found values (one per line)
- **f_x_params.txt**: 31 parameter vectors (comma-separated)

### Competition Rules ✅

✅ **LLM Integration**: DeepSeek R1 used for algorithm design  
✅ **Consistent Parameters**: Same parameters across all benchmark functions  
✅ **Benchmark Integrity**: No modifications to GNBG benchmark  
✅ **Blackbox Treatment**: Fitness function treated as blackbox  

## 🔬 Methodology Details

### LLM-Guided Design Process

1. **Problem Analysis**: LLM analyzes GNBG benchmark characteristics
2. **Algorithm Architecture**: Hybrid GA/DE approach recommended
3. **Operator Design**: Specific crossover/mutation operators selected
4. **Parameter Tuning**: LLM suggests optimal parameter ranges
5. **Adaptive Mechanisms**: Success-based adaptation strategies

### Key Technical Innovations

- **Opposition-based Learning**: Enhanced population initialization
- **Adaptive Crossover**: BLX-α with dynamic α adjustment
- **Gaussian Mutation**: Adaptive step size based on success rate
- **Elite Preservation**: Top 5 individuals maintained across generations
- **Early Stopping**: Convergence detection to save computational resources

## 📚 Documentation

- **[GNBG_Documentation.md](GNBG_Documentation.md)**: Comprehensive methodology, experimental setup, and detailed results analysis
- **Individual README files**: Specific instructions for each implementation
- **Code Comments**: Extensive inline documentation

## 🤝 Team

1. **Nihal Saran Das Duggirala** (Roll No: 2104394, Mechanical Engineering)
2. **N Shikhar** (Roll No: 2104166, Mechanical Engineering)  
3. **Nirakh Sattsangi** (Roll No: 2104253, Electrical Engineering)

## 🔗 Repository

- **GitHub**: https://github.com/nihalsaran/LLMdesignedEA_comp

## 📝 License

This project uses components under various licenses:
- GNBG benchmark implementations: GNU General Public License
- Our LLM-designed algorithm: Available for research and competition use

## 🙏 Acknowledgments

- GNBG benchmark authors for providing the test suite
- DeepSeek team for the LLM capabilities
- OpenRouter for API access
- GECCO 2025 competition organizers

---

*This project demonstrates the successful integration of Large Language Models in evolutionary algorithm design, achieving significant performance improvements over traditional approaches across complex optimization landscapes.*
