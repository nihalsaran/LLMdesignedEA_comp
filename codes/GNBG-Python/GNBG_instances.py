"""
**************************************GNBG with GA (Hugging Face API)**************************************
Fixed code for GECCO 2025 LLM-designed EA competition using Hugging Face Inference API.
Runs 31 independent GA runs for each of 24 GNBG test functions, saves results, and compares with DE.
"""

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import random
import time
import json
import re
import requests
import logging
import signal
import pickle
import tqdm  # Adding tqdm for progress indicators

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure OpenRouter API
OPENROUTER_API_TOKEN = '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'  # Replace with your actual token
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODEL = 'deepseek/deepseek-r1:free'  # Equivalent model on OpenRouter

# Global variables for timeout handling
TIMEOUT_SECONDS = 3600  # 1 hour default timeout
timeout_triggered = False

def extract_json_from_text(text):
    """Extract JSON from text response, handling various formats including markdown code blocks"""
    if not text or text.strip() == "":
        logger.warning("Received empty response from API")
        return None
        
    # Remove markdown code block delimiters (```json, ```)
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text)
    
    # Clean up any extra whitespace
    text = text.strip()
    
    # First try to parse the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # If that fails, try to find JSON object using regex
    json_pattern = r'\{[\s\S]*?\}'  # This pattern matches from { to } including newlines
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse the match
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                if 'pop_size' in parsed:  # For GA parameters
                    return parsed
                elif 'test' in parsed:  # For API test
                    return parsed
        except json.JSONDecodeError:
            continue
    
    # If we still couldn't parse any JSON, log the text and raise an error
    logger.error(f"Full response text that couldn't be parsed: {text}")
    raise ValueError(f"Could not extract valid JSON from API response")

def validate_ga_parameters(params):
    """Validate and fix GA parameters"""
    defaults = {
        'pop_size': 150,
        'cx_type': 'blx',
        'cx_prob': 0.85,
        'cx_alpha': 0.5,
        'mut_type': 'gaussian',
        'mut_prob': 0.15,
        'mut_sigma': 0.1,
        'select_type': 'tournament',
        'select_size': 7,
        'elitism': True,
        'elite_count': 5
    }
    
    # Ensure all required keys exist
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val
    
    # Validate ranges
    params['pop_size'] = max(50, min(200, int(params['pop_size'])))
    params['cx_prob'] = max(0.6, min(0.9, float(params['cx_prob'])))
    params['mut_prob'] = max(0.05, min(0.3, float(params['mut_prob'])))
    
    if params['cx_type'] == 'blx':
        params['cx_alpha'] = max(0.1, min(1.0, float(params['cx_alpha'])))
    
    if params['mut_type'] == 'gaussian':
        params['mut_sigma'] = max(0.05, min(0.5, float(params['mut_sigma'])))
    
    if params['select_type'] == 'tournament':
        params['select_size'] = max(3, min(10, int(params['select_size'])))
    
    if params['elitism']:
        params['elite_count'] = max(1, min(10, int(params['elite_count'])))
    
    return params

# Test API connectivity
def test_api():
    try:
        test_prompt = "Return ONLY this exact JSON with no additional text or formatting: {\"test\": \"success\"}"
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_TOKEN}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://optimizationapi.com",
            "X-Title": "GA Parameter Optimization Test"
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a JSON API that returns only valid JSON with no explanation or additional text."},
                {"role": "user", "content": test_prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}  # Request JSON-only response if supported
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"].strip()
        logger.info("API test response: %s", content)
        
        # Try to parse the response
        try:
            test_json = extract_json_from_text(content)
            if test_json and 'test' in test_json and test_json['test'] == 'success':
                logger.info("Successfully parsed test JSON: %s", test_json)
                return True
            else:
                logger.warning("API test response didn't contain expected JSON format")
                return False
        except Exception as e:
            logger.error("Failed to parse API test response: %s", str(e))
            return False
    except Exception as e:
        logger.error("API test failed: %s", str(e))
        return False

# Define the GNBG class
class GNBG:
    def __init__(self, MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition):
        self.MaxEvals = MaxEvals
        self.AcceptanceThreshold = AcceptanceThreshold
        self.Dimension = Dimension
        self.CompNum = CompNum
        self.MinCoordinate = MinCoordinate
        self.MaxCoordinate = MaxCoordinate
        self.CompMinPos = CompMinPos
        self.CompSigma = CompSigma
        self.CompH = CompH
        self.Mu = Mu
        self.Omega = Omega
        self.Lambda = Lambda
        self.RotationMatrix = RotationMatrix
        self.OptimumValue = OptimumValue
        self.OptimumPosition = OptimumPosition
        self.FEhistory = []
        self.FE = 0
        self.AcceptanceReachPoint = np.inf
        self.BestFoundResult = np.inf
        self.BestFoundPosition = None

    def fitness(self, X):
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        SolutionNumber = X.shape[0]
        result = np.nan * np.ones(SolutionNumber)
        for jj in range(SolutionNumber):
            x = X[jj, :].reshape(-1, 1)
            f = np.nan * np.ones(self.CompNum)
            for k in range(self.CompNum):
                if len(self.RotationMatrix.shape) == 3:
                    rotation_matrix = self.RotationMatrix[:, :, k]
                else:
                    rotation_matrix = self.RotationMatrix
                a = self.transform((x - self.CompMinPos[k, :].reshape(-1, 1)).T @ rotation_matrix.T, self.Mu[k, :], self.Omega[k, :])
                b = self.transform(rotation_matrix @ (x - self.CompMinPos[k, :].reshape(-1, 1)), self.Mu[k, :], self.Omega[k, :])
                # Fix for deprecation warning - extract scalar value from array
                temp_result = (a @ np.diag(self.CompH[k, :]) @ b)
                if isinstance(temp_result, np.ndarray):
                    temp_result = float(temp_result.item())
                
                # Extract scalar values from arrays to avoid deprecation warnings
                sigma_value = float(self.CompSigma[k].item()) if isinstance(self.CompSigma[k], np.ndarray) else self.CompSigma[k]
                lambda_value = float(self.Lambda[k].item()) if isinstance(self.Lambda[k], np.ndarray) else self.Lambda[k]
                
                f[k] = sigma_value + temp_result ** lambda_value
            result[jj] = np.min(f)
            if self.FE >= self.MaxEvals:
                return result
            self.FE += 1
            self.FEhistory.append(result[jj])
            if result[jj] < self.BestFoundResult:
                self.BestFoundResult = result[jj]
                self.BestFoundPosition = X[jj, :].copy()
            if abs(result[jj] - self.OptimumValue) < self.AcceptanceThreshold and np.isinf(self.AcceptanceReachPoint):
                self.AcceptanceReachPoint = self.FE
        return result

    def transform(self, X, Alpha, Beta):
        Y = X.copy()
        tmp = (X > 0)
        Y[tmp] = np.log(X[tmp])
        Y[tmp] = np.exp(Y[tmp] + Alpha[0] * (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
        tmp = (X < 0)
        Y[tmp] = np.log(-X[tmp])
        Y[tmp] = -np.exp(Y[tmp] + Alpha[1] * (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
        return Y

# Use OpenRouter API to optimize GA parameters
def get_ga_parameters(previous_results=None, retries=3, delay=5):
    prompt = """You are an expert in Genetic Algorithms for numerical optimization. 

IMPORTANT: Return ONLY a JSON object with the exact format shown below, without any explanation or additional text:
{
  "pop_size": 150,
  "cx_type": "blx",
  "cx_prob": 0.85,
  "cx_alpha": 0.5,
  "mut_type": "gaussian",
  "mut_prob": 0.15,
  "mut_sigma": 0.1,
  "select_type": "tournament",
  "select_size": 7,
  "elitism": true,
  "elite_count": 5
}

Design optimal GA parameters for the GNBG (Generalized Numerical Benchmark Generator) competition within these constraints:
- pop_size: 50-200
- cx_type: "blx" or "sbx"
- cx_prob: 0.6-0.9
- cx_alpha: 0.1-1.0 (only if cx_type is "blx")
- mut_type: "gaussian" or "polynomial"
- mut_prob: 0.05-0.3
- mut_sigma: 0.05-0.5 (only if mut_type is "gaussian")
- select_type: "tournament" or "roulette"
- select_size: 3-10 (only if select_type is "tournament")
- elitism: true or false
- elite_count: 1-10 (only if elitism is true)"""
    
    if previous_results:
        prompt += f"\n\nPrevious performance was poor. Suggest different parameters to improve optimization: {previous_results}"
    
    for attempt in range(retries):
        try:
            logger.info("Calling OpenRouter API (attempt %d/%d)", attempt + 1, retries)
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_TOKEN}",
                "Content-Type": "application/json",
                # Add HTTP header to request JSON-only response
                "HTTP-Referer": "https://optimizationapi.com",  # Using a custom referrer 
                "X-Title": "GA Parameter Optimization"
            }
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a JSON API that returns only valid JSON with no explanation or additional text."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}  # Request JSON-only response if supported
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"].strip()
            logger.info("Raw API response: %s", content)
            
            # Check if the response is empty
            if not content or content.strip() == "":
                raise ValueError("Received empty response from API")
                
            # Extract and validate JSON
            params = extract_json_from_text(content)
            
            if params is None:
                logger.warning("Failed to extract JSON from response, using default parameters")
                params = {
                    'pop_size': 150,
                    'cx_type': 'blx',
                    'cx_prob': 0.85,
                    'cx_alpha': 0.5,
                    'mut_type': 'gaussian',
                    'mut_prob': 0.15,
                    'mut_sigma': 0.1,
                    'select_type': 'tournament',
                    'select_size': 7,
                    'elitism': True,
                    'elite_count': 5
                }
            
            params = validate_ga_parameters(params)
            
            logger.info("Successfully parsed parameters: %s", params)
            return params, prompt
            
        except Exception as e:
            logger.error("API call failed on attempt %d: %s", attempt + 1, str(e))
            if attempt < retries - 1:
                logger.info("Retrying in %d seconds...", delay)
                time.sleep(delay)
            else:
                logger.warning("Max retries reached. Using default parameters.")
                params = {
                    'pop_size': 150,
                    'cx_type': 'blx',
                    'cx_prob': 0.85,
                    'cx_alpha': 0.5,
                    'mut_type': 'gaussian',
                    'mut_prob': 0.15,
                    'mut_sigma': 0.1,
                    'select_type': 'tournament',
                    'select_size': 7,
                    'elitism': True,
                    'elite_count': 5
                }
                return params, prompt

# GA setup with DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

def init_individual(dim, min_coord, max_coord):
    return creator.Individual(np.random.uniform(min_coord, max_coord, dim))

def cxBlend(ind1, ind2, alpha=0.5):
    for i in range(len(ind1)):
        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
        x1, x2 = ind1[i], ind2[i]
        ind1[i] = (1.0 - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1.0 - gamma) * x2
    return ind1, ind2

def mutGaussian(individual, mu=0, sigma=0.1, indpb=0.15):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            individual[i] = np.clip(individual[i], -100, 100)
    return individual,

def array_equal_wrapper(ind1, ind2):
    """Equality function for numpy arrays to use with DEAP's HallOfFame"""
    return np.array_equal(ind1, ind2)

# Run GA for one instance
def run_ga(gnbg, params):
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, dim=gnbg.Dimension, min_coord=gnbg.MinCoordinate, max_coord=gnbg.MaxCoordinate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (gnbg.fitness(np.array([ind]))[0],))
    
    if params['cx_type'] == 'blx':
        toolbox.register("mate", cxBlend, alpha=params['cx_alpha'])
    else:
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=gnbg.MinCoordinate, up=gnbg.MaxCoordinate)
    
    if params['mut_type'] == 'gaussian':
        toolbox.register("mutate", mutGaussian, mu=0, sigma=params['mut_sigma'], indpb=params['mut_prob'])
    else:
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=gnbg.MinCoordinate, up=gnbg.MaxCoordinate, indpb=params['mut_prob'])
    
    if params['select_type'] == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=params['select_size'])
    else:
        toolbox.register("select", tools.selRoulette)

    pop = toolbox.population(n=params['pop_size'])
    # Use custom similar function for numpy arrays
    hof = tools.HallOfFame(params['elite_count'] if params['elitism'] else 1, similar=array_equal_wrapper)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    # Initialize progress bar
    pbar = tqdm.tqdm(total=gnbg.MaxEvals//params['pop_size'], desc="GA Progress", unit="gen")
    
    # Early stopping criteria
    best_overall = np.inf
    no_improvement_gen = 0
    max_no_improvement = 50  # Stop if no improvement over 50 generations

    for gen in range(1, gnbg.MaxEvals//params['pop_size'] + 1):
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=params['cx_prob'], mutpb=params['mut_prob'], 
                                          ngen=1, stats=stats, halloffame=hof, verbose=False)
        
        # Check for improvement
        if hof.items[0].fitness.values[0] < best_overall:
            best_overall = hof.items[0].fitness.values[0]
            no_improvement_gen = 0  # Reset counter if there is an improvement
        else:
            no_improvement_gen += 1
        
        # Update progress bar
        pbar.update(1)
        
        # Early stopping
        if no_improvement_gen >= max_no_improvement:
            logger.info("Early stopping triggered after %d generations without improvement", max_no_improvement)
            break

    pbar.close()

    return hof.items[0], gnbg.BestFoundResult, gnbg.BestFoundPosition, gnbg.FEhistory

# Run DE for one instance
def run_de(gnbg, popsize=15, max_evals=500000):
    bounds = [(-100, 100)] * gnbg.Dimension
    results = differential_evolution(gnbg.fitness, bounds=bounds, disp=False, polish=False, popsize=popsize, maxiter=max_evals//popsize)
    return results.x, results.fun, gnbg.FEhistory

# Process convergence data
def process_convergence(history, optimum):
    convergence = []
    best_error = float('inf')
    for value in history:
        error = abs(value - optimum)
        if error < best_error:
            best_error = error
        convergence.append(best_error)
    return convergence

# Save results in competition format
def save_results(func_idx, values, params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"f_{func_idx}_value.txt"), "w") as f:
        for val in values:
            f.write(f"{val}\n")
    with open(os.path.join(output_dir, f"f_{func_idx}_params.txt"), "w") as f:
        for param in params:
            f.write(",".join(map(str, param)) + "\n")

# Main execution
if __name__ == "__main__":
    logger.info("Starting GNBG benchmark script")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "results")
    os.makedirs(folder_path, exist_ok=True)

    # Test API before starting
    api_working = test_api()
    if not api_working:
        logger.warning("API test failed. Will use default parameters for all functions.")

    # Parameters
    num_runs = 31  # Reduced from 31 for quick testing
    problem_indices = range(1, 25)  # Just using functions 1 and 2 for quick testing
    ga_results = {}
    de_results = {}
    all_prompts = []

    # Initial Hugging Face API call for GA parameters
    if api_working:
        ga_params, initial_prompt = get_ga_parameters()
        all_prompts.append(initial_prompt)
    else:
        ga_params = {
            'pop_size': 150,
            'cx_type': 'blx',
            'cx_prob': 0.85,
            'cx_alpha': 0.5,
            'mut_type': 'gaussian',
            'mut_prob': 0.15,
            'mut_sigma': 0.1,
            'select_type': 'tournament',
            'select_size': 7,
            'elitism': True,
            'elite_count': 5
        }
        initial_prompt = "Default parameters used due to API failure"
        all_prompts.append(initial_prompt)
    
    logger.info("Initial GA Parameters: %s", ga_params)

    for idx in problem_indices:
        logger.info("Processing function f%d", idx)
        # Load GNBG parameters from the existing .mat files in the workspace
        filename = f"f{idx}.mat"
        try:
            # Try to load from the current directory first (codes/GNBG-Python/)
            GNBG_tmp = loadmat(os.path.join(current_dir, filename))['GNBG']
            logger.info(f"Successfully loaded {filename} from {current_dir}")
        except FileNotFoundError:
            logger.error("f%d.mat not found. Skipping function f%d.", idx, idx)
            continue
            
        MaxEvals = 500000 if idx <= 15 else 1000000
        AcceptanceThreshold = 1e-8
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])

        # Initialize lists for results
        ga_values = []
        ga_params_list = []
        de_values = []
        de_params = []
        ga_convergences = []
        de_convergences = []

        # Run all 31 independent GA and DE runs as required
        test_runs = num_runs  # Use all 31 runs for the full experiment
        for run in range(test_runs):
            np.random.seed(run)
            random.seed(run)

            # Run GA
            gnbg_ga = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
            best_ind, best_val, best_pos, ga_history = run_ga(gnbg_ga, ga_params)
            ga_values.append(best_val)
            ga_params_list.append(best_pos)
            ga_convergences.append(process_convergence(ga_history, OptimumValue))

            # Run DE
            gnbg_de = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
            best_pos_de, best_val_de, de_history = run_de(gnbg_de, max_evals=MaxEvals)
            de_values.append(best_val_de)
            de_params.append(best_pos_de)
            de_convergences.append(process_convergence(de_history, OptimumValue))

        # Compute mean errors
        ga_mean = np.mean(ga_values)
        de_mean = np.mean(de_values)
        ga_results[idx] = ga_mean
        de_results[idx] = de_mean
        logger.info("Function f%d: GA mean error=%.4e, DE mean error=%.4e", idx, ga_mean, de_mean)

        # If GA underperforms DE and API is working, query for improved parameters
        if ga_mean > de_mean and api_working:
            feedback = f"Function f{idx}: GA mean error={ga_mean:.4e}, DE mean error={de_mean:.4e}. GA underperformed."
            new_params, new_prompt = get_ga_parameters(feedback)
            all_prompts.append(new_prompt)
            logger.info("Function f%d: Retrying with new parameters: %s", idx, new_params)

            # Rerun GA with new parameters
            ga_values = []
            ga_params_list = []
            ga_convergences = []
            for run in range(test_runs):
                np.random.seed(run)
                random.seed(run)
                gnbg_ga = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
                best_ind, best_val, best_pos, ga_history = run_ga(gnbg_ga, new_params)
                ga_values.append(best_val)
                ga_params_list.append(best_pos)
                ga_convergences.append(process_convergence(ga_history, OptimumValue))
            ga_results[idx] = np.mean(ga_values)
            ga_params = new_params

        # Save results
        save_results(idx, ga_values, ga_params_list, os.path.join(folder_path, f"g{idx}_ga"))
        save_results(idx, de_values, de_params, os.path.join(folder_path, f"g{idx}_de"))

        # Plot convergence
        plt.figure(figsize=(10, 6))
        for run in range(test_runs):        plt.plot(range(1, len(ga_convergences[run]) + 1), ga_convergences[run], 'b-', alpha=0.3, label="GA" if run == 0 else "")
        plt.plot(range(1, len(de_convergences[run]) + 1), de_convergences[run], 'r-', alpha=0.3, label="DE" if run == 0 else "")
        plt.xlabel('Function Evaluation Number (FE)')
        plt.ylabel('Error')
        plt.title(f'Convergence Plot for f{idx}')
        
        # Check if all values are positive before using log scale
        all_values = ga_convergences[run] + de_convergences[run]
        if min(all_values) > 0:
            plt.yscale('log')
        else:
            logger.info(f"Using linear scale for f{idx} convergence plot due to non-positive values")
            
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"g{idx}_convergence.png"))
        plt.close()

    # Compare mean performance
    ga_mean_scores = [ga_results[i] for i in problem_indices if i in ga_results]
    de_mean_scores = [de_results[i] for i in problem_indices if i in de_results]

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(list(ga_results.keys()), ga_mean_scores, 'bo-', label='GA Mean Error')
    plt.plot(list(de_results.keys()), de_mean_scores, 'ro-', label='DE Mean Error')
    plt.xlabel('Function Index')
    plt.ylabel('Mean Error')
    plt.title('GA vs DE Mean Performance Across GNBG Functions')
    
    # Check if all values are positive before using log scale
    all_values = ga_mean_scores + de_mean_scores
    if min(all_values) > 0:
        plt.yscale('log')
    else:
        logger.info("Using linear scale for comparison plot due to non-positive values")
        
    plt.legend()
    plt.savefig(os.path.join(folder_path, "ga_vs_de_comparison.png"))
    plt.close()

    # Calculate total scores
    ga_total_score = sum([1 - (ga_results[i] / max(ga_results[i], de_results[i], 1e-8)) for i in ga_results])
    de_total_score = sum([1 - (de_results[i] / max(ga_results[i], de_results[i], 1e-8)) for i in de_results])
    logger.info("GA Total Score: %.4f / %d", ga_total_score, len(problem_indices))
    logger.info("DE Total Score: %.4f / %d", de_total_score, len(problem_indices))

    # Save prompts and settings for submission
    with open(os.path.join(folder_path, "mistral_prompts.txt"), "w") as f:
        for i, prompt in enumerate(all_prompts):
            f.write(f"Prompt {i+1}:\n{prompt}\n\n")
    
    logger.info(f"LLM Used: {OPENROUTER_MODEL} (OpenRouter API)")
    logger.info("Final GA Parameters: %s", ga_params)
    logger.info("Prompts saved to: %s", os.path.join(folder_path, "mistral_prompts.txt"))
    logger.info("Script completed successfully!")



def debug_setup():
    """Function to debug the setup and identify missing dependencies or issues"""
    logger.info("Starting debug setup")
    
    # Check Python version
    import sys
    logger.info(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = ["numpy", "scipy", "matplotlib", "deap", "huggingface_hub"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"Package {package} is installed")
        except ImportError:
            logger.error(f"Package {package} is not installed")
            missing_packages.append(package)
    
    # Check if .mat files exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mat_files_found = False
    for idx in range(1, 25):  # Check for f1.mat through f24.mat
        mat_file = os.path.join(current_dir, f"f{idx}.mat")
        if os.path.exists(mat_file):
            mat_files_found = True
            logger.info(f"Found {mat_file}")
            break
    
    if not mat_files_found:
        logger.error("No .mat files found. Please download the GNBG benchmark files.")
    
    # Check OpenRouter API token
    if OPENROUTER_API_TOKEN == 'your_openrouter_api_key':
        logger.warning("Using default API token. Replace with your own token.")
    
    # Return installation instructions if needed
    if missing_packages:
        logger.info("To install missing packages, run:")
        logger.info(f"pip install {' '.join(missing_packages)}")
    
    if not mat_files_found:
        logger.info("To create test .mat files for initial testing, run the following function:")
        logger.info("create_test_mat_files()")
    
    return missing_packages, mat_files_found

def create_test_mat_files():
    """Create simple test .mat files for initial testing"""
    try:
        from scipy.io import savemat
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for idx in range(1, 3):  # Create f1.mat and f2.mat for testing
            test_data = {
                'GNBG': {
                    'Dimension': np.array([[2]]),
                    'o': np.array([[1]]),
                    'MinCoordinate': np.array([[-100]]),
                    'MaxCoordinate': np.array([[100]]),
                    'Component_MinimumPosition': np.array([[np.zeros((1, 2))]]),
                    'ComponentSigma': np.array([[np.zeros(1)]]),
                    'Component_H': np.array([[np.ones((1, 2))]]),
                    'Mu': np.array([[np.ones((1, 2))]]),
                    'Omega': np.array([[np.ones((1, 4))]]),
                    'lambda': np.array([[np.ones(1)]]),
                    'RotationMatrix': np.array([[np.eye(2)]]),
                    'OptimumValue': np.array([[0]]),
                    'OptimumPosition': np.array([[np.zeros(2)]])
                }
            }
            
            savemat(os.path.join(current_dir, f"f{idx}.mat"), test_data)
            logger.info(f"Created test file f{idx}.mat for initial testing")
        
        return True
    except Exception as e:
        logger.error(f"Failed to create test .mat files: {e}")
        return False

if __name__ == "__main__":
    # Add debug setup call at the beginning of the main execution
    debug_setup()