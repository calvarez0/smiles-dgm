#!/usr/bin/env python3
"""
dgm_openended_enhanced.py

A Darwin-Gödel–inspired open-ended molecular evolution loop with hierarchical mutations:
1. Genetic mutations every generation
2. LLM-assisted mutations every 10 generations  
3. LLM empirical assessment and prompt refinement every 50 generations

Uses Google Gemini as the mutation "agent" and RDKit for fitness & novelty evaluation.
Logs snapshots & JSON, and visualizes fitness & novelty every 10 generations.

Requires:
    - python3
    - rdkit
    - numpy
    - pandas
    - matplotlib
    - google-genai (Gemini SDK)

Install dependencies:
    pip install rdkit-pypi numpy pandas matplotlib google-genai

Set your Gemini API key in the environment:
    export GEMINI_API_KEY="YOUR_KEY_HERE"

Run:
    python3 dgm_openended_enhanced.py
"""

import os
import json
import time
import signal
import random
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs, rdMolDescriptors

# Gemini SDK imports
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai (Gemini SDK) not installed. Install with `pip install google-genai`.")
    exit(1)

# ------------- Configuration -------------
MODEL_NAME = "gemini-1.5-flash-latest"  # Gemini model for SMILES mutation
API_KEY = os.environ.get("GEMINI_API_KEY", None)
if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    exit(1)

POPULATION_SIZE = 10
ARCHIVE_FINGERPRINTS = []  # Will store fingerprints of all seen molecules
ARCHIVE_SMILES = []
NOVELTY_ARCHIVE = []  # Store highly novel molecules for diversity injection

# Evolution intervals and open-ended parameters
LLM_MUTATION_INTERVAL = 10
PROMPT_REFINEMENT_INTERVAL = 50
NOVELTY_PUSH_INTERVAL = 25  # Force novelty-only selection
RESTART_INTERVAL = 200      # Restart with diverse seeds if stagnant

# Open-ended evolution settings
PLATEAU_THRESHOLD = 30      # Generations without improvement = plateau
NOVELTY_ARCHIVE_SIZE = 1000 # Keep most novel molecules
MIN_NOVELTY_THRESHOLD = 0.7 # Minimum novelty to be considered "interesting"

# For logging and snapshots
BASE_OUTPUT_DIR = "dgm_evolution_runs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_ID}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subdirectories
SNAPSHOTS_DIR = os.path.join(OUTPUT_DIR, "snapshots")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PROMPTS_DIR = os.path.join(OUTPUT_DIR, "prompts")
MOLECULES_DIR = os.path.join(OUTPUT_DIR, "molecules")
SUMMARIES_DIR = os.path.join(OUTPUT_DIR, "summaries")

for dir_path in [SNAPSHOTS_DIR, PLOTS_DIR, PROMPTS_DIR, MOLECULES_DIR, SUMMARIES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Visualization settings
VISUALIZE_EVERY = 10  # generations
fitness_history = []  # This will track all-time best fitness (monotonic)
novelty_history = []
all_time_best_fitness = 0.0  # Global best fitness tracker

# Prompt management
current_prompt = """You are a chemistry assistant. Given the molecule SMILES '{smiles}', propose exactly one mutation (one atom/functional group change) and return only the new valid SMILES string. Do not include any extra text or explanation."""

best_prompt = current_prompt
best_molecule = {"smiles": "", "fitness": 0.0, "generation": 0}
best_prompt_info = {"prompt": current_prompt, "fitness_achieved": 0.0, "generation": 0}
last_prompt_performance = {"generations_since_improvement": 0, "fitness_before": 0.0, "fitness_after": 0.0}
top_molecules = []  # Track top 10 best molecules found

# Performance tracking for prompt refinement
prompt_performance_history = []

# Graceful shutdown
stop_requested = False


def signal_handler(sig, frame):
    global stop_requested
    print("\nReceived interrupt. Finishing current generation, then exiting...")
    stop_requested = True


signal.signal(signal.SIGINT, signal_handler)


# ------------- Genetic Mutation Functions -------------
def genetic_mutate_smiles(smiles: str) -> str:
    """
    Perform simple genetic mutations on SMILES string.
    Returns the original SMILES if mutation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles
    
    try:
        # Random mutation strategies
        mutation_type = random.choice([
            'add_atom', 'remove_atom', 'change_atom', 'add_bond', 'change_bond'
        ])
        
        if mutation_type == 'add_atom':
            # Add a random atom (C, N, O, S, F, Cl)
            atoms_to_add = ['C', 'N', 'O', 'S', 'F', 'Cl']
            new_atom = random.choice(atoms_to_add)
            
            # Simple approach: try to add to a random existing atom
            if mol.GetNumAtoms() > 0:
                rand_atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
                # This is a simplified approach - in practice you'd want more sophisticated chemistry
                mutated_smiles = smiles + new_atom
                test_mol = Chem.MolFromSmiles(mutated_smiles)
                if test_mol:
                    return Chem.MolToSmiles(test_mol)
        
        elif mutation_type == 'change_atom':
            # Change a random atom to another type
            if mol.GetNumAtoms() > 0:
                mw = Chem.RWMol(mol)
                rand_atom_idx = random.randint(0, mw.GetNumAtoms() - 1)
                atom = mw.GetAtomWithIdx(rand_atom_idx)
                
                # Change to a different common atom
                current_symbol = atom.GetSymbol()
                new_symbols = [s for s in ['C', 'N', 'O', 'S'] if s != current_symbol]
                if new_symbols:
                    new_symbol = random.choice(new_symbols)
                    atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
                    
                    try:
                        mutated = Chem.MolToSmiles(mw)
                        test_mol = Chem.MolFromSmiles(mutated)
                        if test_mol:
                            return Chem.MolToSmiles(test_mol)
                    except:
                        pass
        
        # If specific mutations fail, try a simple random substitution
        if len(smiles) > 1:
            pos = random.randint(0, len(smiles) - 1)
            if smiles[pos] in 'CNOSFcl':
                replacements = {'C': 'N', 'N': 'O', 'O': 'S', 'S': 'C', 'F': 'Cl', 'c': 'n', 'l': 'F'}
                new_char = replacements.get(smiles[pos], smiles[pos])
                mutated = smiles[:pos] + new_char + smiles[pos+1:]
                test_mol = Chem.MolFromSmiles(mutated)
                if test_mol:
                    return Chem.MolToSmiles(test_mol)
        
        return smiles  # Return original if all mutations fail
        
    except Exception as e:
        return smiles


# ------------- LLM Mutation Functions -------------
def mutate_smiles_with_gemini(smiles: str, prompt_template: str = None) -> str:
    """
    Ask Gemini to propose a single-step mutation of the input SMILES.
    If Gemini's response cannot be parsed into a valid SMILES, fallback to the original.
    """
    if prompt_template is None:
        prompt_template = current_prompt
    
    client = genai.Client(api_key=API_KEY)
    prompt = prompt_template.format(smiles=smiles)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    try:
        # Use streaming to get text chunks
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=config
        ):
            response_text += chunk.text
        candidate = response_text.strip()
        
        # Clean up the response - remove quotes, explanations, etc.
        candidate = candidate.strip('"').strip("'")
        
        # Extract only the SMILES part if there's extra text
        lines = candidate.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('The', 'Here', 'This', 'I', 'A', 'To')):
                # Try this line as a potential SMILES
                test_line = line.strip('"').strip("'").strip()
                try:
                    mol = Chem.MolFromSmiles(test_line)
                    if mol:
                        # Additional validation - check for reasonable atom counts and valences
                        num_atoms = mol.GetNumAtoms()
                        if 3 <= num_atoms <= 100:  # Reasonable size range
                            try:
                                # Try to sanitize the molecule to catch valence errors
                                Chem.SanitizeMol(mol)
                                return Chem.MolToSmiles(mol)
                            except:
                                continue  # Skip this candidate if sanitization fails
                except:
                    continue  # Skip this line if parsing fails
        
        # If no valid SMILES found in individual lines, try the whole candidate
        try:
            mol = Chem.MolFromSmiles(candidate)
            if mol:
                num_atoms = mol.GetNumAtoms()
                if 3 <= num_atoms <= 100:
                    try:
                        Chem.SanitizeMol(mol)
                        return Chem.MolToSmiles(mol)
                    except:
                        pass
        except:
            pass
            
        # All parsing attempts failed
        print(f"[LLM PARSE ERROR] Could not parse valid SMILES from '{candidate[:50]}...'")
        return smiles  # fallback
            
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return smiles


# ------------- Prompt Refinement Functions -------------
def analyze_performance_and_refine_prompt(generation: int) -> str:
    """
    Use LLM to analyze recent performance and suggest prompt improvements.
    Enhanced with full context and robust error handling.
    """
    global current_prompt, best_prompt, best_prompt_info, last_prompt_performance, all_time_best_fitness
    
    # Analyze recent performance
    recent_generations = min(PROMPT_REFINEMENT_INTERVAL, len(fitness_history))
    if recent_generations < 10:
        return current_prompt
    
    recent_fitness = fitness_history[-recent_generations:]
    recent_novelty = novelty_history[-recent_generations:]
    
    avg_fitness = np.mean(recent_fitness)
    fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
    avg_novelty = np.mean(recent_novelty)
    novelty_trend = np.polyfit(range(len(recent_novelty)), recent_novelty, 1)[0]
    
    # Calculate performance of last prompt
    generations_since_best = generation - best_molecule["generation"]
    
    # Create a comprehensive summary for the LLM
    evolution_summary = f"""MOLECULAR EVOLUTION STATUS - Generation {generation}

🎯 MISSION: Break through fitness plateau by evolving better drug-like molecules (QED fitness)

📊 CURRENT PERFORMANCE:
- All-time best fitness: {all_time_best_fitness:.4f} (QED drug-likeness score, 0-1 scale)
- Best molecule found: {best_molecule['smiles']} 
- Found in generation: {best_molecule['generation']}
- Generations since improvement: {generations_since_best}
- Plateau status: {"🚨 LONG PLATEAU" if generations_since_best > 30 else "✅ Recent progress"}

🧬 BEST MOLECULE DETAILS:
- SMILES: {best_molecule['smiles']}
- Fitness: {best_molecule['fitness']:.4f}
- Prompt that created it: "{best_prompt_info['prompt']}"

📈 RECENT TRENDS (last {recent_generations} generations):
- Average fitness: {avg_fitness:.4f}
- Fitness trend: {"📈 improving" if fitness_trend > 0 else "📉 declining" if fitness_trend < -0.001 else "➡️ flat"}
- Average novelty: {avg_novelty:.4f}
- Novelty trend: {"📈 increasing" if novelty_trend > 0 else "📉 declining"}

🔄 CURRENT STRATEGY:
- Current mutation prompt: "{current_prompt}"
- Prompt performance: No improvement for {generations_since_best} generations

🎲 OPEN-ENDED EVOLUTION PRINCIPLES:
- When stuck (30+ gen plateau): PRIORITIZE NOVELTY over optimization
- Seek "stepping stones" - unusual molecules that open new paths
- Break out of local optima with creative, unconventional mutations
- Explore completely different chemical families when severely stuck"""

    # Enhanced refinement prompt with full context
    refinement_prompt = f"""You are an expert molecular evolution strategist. Your job is to create a prompt that will help break through fitness plateaus.

{evolution_summary}

CURRENT SITUATION ANALYSIS:
Plateau length: {generations_since_best} generations

STRATEGY GUIDELINES:
🟢 SHORT PLATEAU (1-29 gens): Minor refinements to current approach
🟡 MEDIUM PLATEAU (30-49 gens): Moderate creativity, explore related chemical space  
🔴 LONG PLATEAU (50+ gens): RADICAL CREATIVITY - completely new chemical territories

EXAMPLE PROMPTS BY SITUATION:

NORMAL PROGRESS:
"Given molecule '{{smiles}}', make a small strategic modification to improve drug-likeness. Return only SMILES."

MEDIUM PLATEAU (your situation needs {"medium" if 15 <= generations_since_best < 50 else "radical" if generations_since_best >= 50 else "normal"} approach):
"Transform '{{smiles}}' with an unconventional but valid modification. Explore unusual functional groups or ring systems. Return only SMILES."

SEVERE PLATEAU:
"Completely reimagine '{{smiles}}' using bold chemistry. Try rare elements, exotic rings, or unconventional structures while maintaining validity. Explore uncharted chemical space. Return only SMILES."

YOUR TASK:
Create a NEW mutation prompt that will help break this {generations_since_best}-generation plateau.
- Consider what made the best prompt successful: "{best_prompt_info['prompt']}"
- The prompt should include '{{smiles}}' placeholder
- Be more creative/radical if plateau is longer
- Focus on valid SMILES output
- Encourage exploration of new chemical space

Return ONLY the new prompt text. No explanations or formatting."""

    print(f"\n[PROMPT REFINEMENT] Analyzing {generations_since_best}-gen plateau...")
    
    # Multiple attempts with error handling
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            client = genai.Client(api_key=API_KEY)
            contents = [
                types.Content(
                    role="user", 
                    parts=[types.Part.from_text(text=refinement_prompt)],
                )
            ]
            config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.7  # Add some creativity
            )
            
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME, contents=contents, config=config
            ):
                response_text += chunk.text
            
            refined_prompt = response_text.strip().strip('"').strip("'")
            
            # Validate the prompt
            if '{smiles}' in refined_prompt and len(refined_prompt) > 20:
                # Save the refined prompt with full context
                prompt_data = {
                    "generation": generation,
                    "old_prompt": current_prompt,
                    "new_prompt": refined_prompt,
                    "evolution_summary": evolution_summary,
                    "generations_since_best": generations_since_best,
                    "plateau_severity": "severe" if generations_since_best > 50 else "medium" if generations_since_best > 30 else "mild",
                    "best_molecule": best_molecule,
                    "best_prompt_info": best_prompt_info,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt + 1
                }
                
                with open(os.path.join(PROMPTS_DIR, f"prompt_refinement_gen_{generation}.json"), "w") as f:
                    json.dump(prompt_data, f, indent=2)
                
                print(f"[PROMPT REFINED] Attempt {attempt + 1}: {refined_prompt[:80]}...")
                return refined_prompt
            else:
                print(f"[PROMPT REFINEMENT] Attempt {attempt + 1} failed: Invalid prompt format")
                continue
                
        except Exception as e:
            print(f"[PROMPT REFINEMENT ERROR] Attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                print("Retrying...")
                time.sleep(2)  # Brief delay before retry
            continue
    
    # If all attempts failed, create a fallback prompt based on plateau length
    print("[PROMPT REFINEMENT] All attempts failed, using fallback strategy")
    if generations_since_best > 50:
        fallback_prompt = "Completely reimagine molecule '{smiles}' with radical but chemically valid transformations. Explore exotic chemistry. Return only SMILES."
    elif generations_since_best > 30:
        fallback_prompt = "Transform '{smiles}' with bold, unconventional mutations while maintaining chemical validity. Return only SMILES."
    else:
        fallback_prompt = "Modify molecule '{smiles}' with a creative but strategic change to improve drug-likeness. Return only SMILES."
    
    print(f"[FALLBACK PROMPT] {fallback_prompt}")
    return fallback_prompt


# ------------- Fitness & Novelty Functions -------------
def compute_fitness(smiles: str) -> float:
    """
    Compute fitness using QED (quantitative estimate of drug-likeness).
    Invalid SMILES get zero fitness.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0.0
    try:
        return QED.qed(mol)
    except:
        return 0.0


def fingerprint(smiles: str):
    """
    Return a Morgan fingerprint (radius=2, 2048 bits) for novelty computations.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    # Use the newer rdMolDescriptors to avoid deprecation warning
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_novelty(fp):
    """
    Novelty = average Tanimoto distance to all fingerprints in the archive.
    If archive is empty, novelty = 1.0 (max).
    """
    if not ARCHIVE_FINGERPRINTS:
        return 1.0
    similarities = []
    for ref_fp in ARCHIVE_FINGERPRINTS:
        sims = DataStructs.TanimotoSimilarity(fp, ref_fp)
        similarities.append(sims)
    if not similarities:
        return 1.0
    avg_sim = np.mean(similarities)
    return 1.0 - avg_sim  # distance


def update_novelty_archive(smiles: str, novelty: float):
    """
    Maintain an archive of highly novel molecules for diversity injection.
    Following Stanley's principle of preserving stepping stones.
    """
    global NOVELTY_ARCHIVE
    
    if novelty >= MIN_NOVELTY_THRESHOLD:
        NOVELTY_ARCHIVE.append({
            "smiles": smiles,
            "novelty": novelty,
            "fitness": compute_fitness(smiles)
        })
        
        # Keep only the most novel molecules
        if len(NOVELTY_ARCHIVE) > NOVELTY_ARCHIVE_SIZE:
            NOVELTY_ARCHIVE.sort(key=lambda x: x["novelty"], reverse=True)
            NOVELTY_ARCHIVE = NOVELTY_ARCHIVE[:NOVELTY_ARCHIVE_SIZE]


def get_diverse_seeds():
    """
    Generate diverse seed molecules from different chemical scaffolds.
    Stanley's principle: diversity of starting points leads to diverse outcomes.
    """
    diverse_seeds = [
        "c1ccccc1",           # benzene
        "C1CCCCC1",           # cyclohexane
        "c1ccc2ccccc2c1",     # naphthalene
        "c1cnccn1",           # pyrimidine
        "C1=CC=CC=C1N",       # aniline
        "c1csc2c1cccc2",      # benzothiophene
        "C1CCC2CCCCC2C1",     # decalin
        "c1cc2cc3ccccc3cc2cc1", # anthracene
        "C1=CC=C(C=C1)O",     # phenol
        "c1cc2[nH]c3ccccc3c2cc1", # carbazole
    ]
    
    # Add some from novelty archive if available
    if NOVELTY_ARCHIVE:
        archive_samples = random.sample(
            NOVELTY_ARCHIVE, 
            min(5, len(NOVELTY_ARCHIVE))
        )
        diverse_seeds.extend([mol["smiles"] for mol in archive_samples])
    
    return diverse_seeds


# ------------- Save Functions -------------
def save_best_molecule():
    """Save the best molecule found so far."""
    with open(os.path.join(MOLECULES_DIR, "best_molecule.json"), "w") as f:
        json.dump(best_molecule, f, indent=2)


def save_prompts():
    """Save current and best prompts."""
    prompt_data = {
        "current_prompt": current_prompt,
        "best_prompt": best_prompt,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(PROMPTS_DIR, "current_prompts.json"), "w") as f:
        json.dump(prompt_data, f, indent=2)


def update_top_molecules(smiles: str, fitness: float, generation: int, mutation_strategy: str):
    """
    Update the top 10 best molecules list.
    """
    global top_molecules
    
    # Create molecule entry
    molecule_entry = {
        "smiles": smiles,
        "fitness": fitness,
        "generation": generation,
        "mutation_strategy": mutation_strategy,
        "prompt_used": current_prompt,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add to list
    top_molecules.append(molecule_entry)
    
    # Sort by fitness (descending) and keep only top 10
    top_molecules.sort(key=lambda x: x["fitness"], reverse=True)
    top_molecules = top_molecules[:10]


def save_top_molecules():
    """Save the top 10 best molecules found so far."""
    top_molecules_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_molecules_evaluated": len(ARCHIVE_SMILES),
        "top_10_molecules": top_molecules
    }
    
    with open(os.path.join(MOLECULES_DIR, "top_10_molecules.json"), "w") as f:
        json.dump(top_molecules_data, f, indent=2)
    
    # Also save as a readable text file
    text_content = f"""TOP 10 BEST MOLECULES FOUND
==============================
Last Updated: {datetime.now(timezone.utc).isoformat()}
Total Molecules Evaluated: {len(ARCHIVE_SMILES)}

"""
    
    for i, mol in enumerate(top_molecules, 1):
        text_content += f"""
{i}. FITNESS: {mol['fitness']:.4f}
   SMILES: {mol['smiles']}
   Generation: {mol['generation']}
   Strategy: {mol['mutation_strategy']}
   Prompt: {mol['prompt_used'][:100]}{'...' if len(mol['prompt_used']) > 100 else ''}
   Found: {mol['timestamp']}
{"="*60}"""
    
    with open(os.path.join(MOLECULES_DIR, "top_10_molecules.txt"), "w") as f:
        f.write(text_content)


# ------------- Evolutionary Loop -------------
def evolve_forever():
    global current_prompt, best_prompt, best_molecule, all_time_best_fitness, best_prompt_info
    generation = 1

    # Initialize population with diverse seeds for open-ended evolution
    diverse_seeds = get_diverse_seeds()
    population = random.choices(diverse_seeds, k=POPULATION_SIZE)

    while True:
        print(f"\n=== Generation {generation} ===")
        
        # Determine mutation strategy for this generation (open-ended approach)
        mutation_strategy = "genetic"
        generations_since_best = generation - best_molecule["generation"]
        
        # Check for different intervention strategies
        if generation % RESTART_INTERVAL == 0 and generations_since_best > 50:
            mutation_strategy = "diversity_restart"
            print(f"DIVERSITY RESTART: Injecting diverse seeds after {generations_since_best} gen plateau")
            # Replace some population with diverse seeds
            new_seeds = get_diverse_seeds()
            num_replace = POPULATION_SIZE // 3
            population = population[:-num_replace] + random.choices(new_seeds, k=num_replace)
        elif generation % NOVELTY_PUSH_INTERVAL == 0 and generations_since_best > 15:
            mutation_strategy = "novelty_push"
            print(f"NOVELTY PUSH: Prioritizing exploration after {generations_since_best} gen plateau")
        elif generation % PROMPT_REFINEMENT_INTERVAL == 0:
            mutation_strategy = "prompt_refinement"
            old_prompt = current_prompt
            current_prompt = analyze_performance_and_refine_prompt(generation)
            print(f"PROMPT CHANGED: {'YES' if current_prompt != old_prompt else 'NO'}")
            if current_prompt != old_prompt:
                print(f"OLD: {old_prompt[:60]}...")
                print(f"NEW: {current_prompt[:60]}...")
        elif generation % LLM_MUTATION_INTERVAL == 0:
            mutation_strategy = "llm"
        
        print(f"Mutation strategy: {mutation_strategy}")
        
        gen_data = {
            "generation": generation,
            "mutation_strategy": mutation_strategy,
            "best_smiles": None,
            "best_fitness": 0.0,
            "avg_novelty": 0.0,
            "population": [],
        }

        new_population = []
        gen_novelties = []
        
        # For LLM generations, get one mutation guidance that applies to all
        llm_guidance = None
        if mutation_strategy in ["llm", "prompt_refinement"]:
            # Use the best molecule as guidance for LLM mutations
            guidance_molecule = best_molecule["smiles"] if best_molecule["smiles"] else population[0]
            print(f"   → Getting LLM guidance based on best molecule: {guidance_molecule}")
            llm_guidance = mutate_smiles_with_gemini(guidance_molecule, current_prompt)
            print(f"   → LLM suggests pattern: {llm_guidance}")
        
        for idx, parent in enumerate(population, start=1):
            fitness_parent = compute_fitness(parent)
            print(f" Parent #{idx}: SMILES={parent}  Fitness={fitness_parent:.4f}")
            
            # Apply appropriate mutation strategy
            if mutation_strategy == "genetic":
                print("   → Genetic mutation…", end="", flush=True)
                child = genetic_mutate_smiles(parent)
            elif mutation_strategy in ["diversity_restart", "novelty_push"]:
                # For open-ended strategies, use more creative mutations
                print("   → Creative/Novel mutation…", end="", flush=True)
                if random.random() < 0.7:  # 70% LLM for creativity
                    creative_prompt = "Create a bold, unconventional but chemically valid mutation of '{smiles}' that explores new chemical space. Return only SMILES."
                    child = mutate_smiles_with_gemini(parent, creative_prompt)
                else:
                    child = genetic_mutate_smiles(parent)
            else:  # llm or prompt_refinement - use the single LLM guidance
                print("   → Applying LLM pattern…", end="", flush=True)
                # Apply similar mutation pattern to this parent as suggested by LLM
                child = genetic_mutate_smiles(parent)  # Fallback to genetic if LLM pattern fails
                if llm_guidance and llm_guidance != guidance_molecule:
                    # Try to apply similar mutation pattern
                    child = mutate_smiles_with_gemini(parent, current_prompt)
            
            print(f"\r   → Mutated to {child}", flush=True)

            fitness_child = compute_fitness(child)
            fp_child = fingerprint(child)
            novelty_child = compute_novelty(fp_child) if fp_child else 0.0

            # Update archive
            if fp_child:
                ARCHIVE_FINGERPRINTS.append(fp_child)
                ARCHIVE_SMILES.append(child)
                
                # Update novelty archive (Stanley's stepping stones principle)
                update_novelty_archive(child, novelty_child)

            # Track best molecule (all-time) and update prompt tracking
            if fitness_child > best_molecule["fitness"]:
                old_best_fitness = best_molecule["fitness"]
                best_molecule = {
                    "smiles": child,
                    "fitness": fitness_child,
                    "generation": generation
                }
                
                # Update best prompt info if this improvement came from current prompt
                if mutation_strategy in ["llm", "prompt_refinement"]:
                    best_prompt_info = {
                        "prompt": current_prompt,
                        "fitness_achieved": fitness_child,
                        "generation": generation
                    }
                    best_prompt = current_prompt
                
                save_best_molecule()
                print(f"    *** NEW BEST MOLECULE! Fitness: {fitness_child:.4f} (was {old_best_fitness:.4f}) ***")

            # Update top 10 molecules list for any good molecule
            if fitness_child > 0.1:  # Only track molecules with reasonable fitness
                update_top_molecules(child, fitness_child, generation, mutation_strategy)

            new_population.append(child)
            gen_data["population"].append({
                "smiles": child,
                "fitness": fitness_child,
                "novelty": novelty_child
            })
            gen_novelties.append(novelty_child)

            print(f"    → Child Fitness={fitness_child:.4f}  Novelty={novelty_child:.4f}")

            if stop_requested:
                break

        # Evaluate generation data
        fitnesses = [p["fitness"] for p in gen_data["population"]]
        novelties = [p["novelty"] for p in gen_data["population"]]
        best_idx = int(np.argmax(fitnesses))
        gen_data["best_smiles"] = gen_data["population"][best_idx]["smiles"]
        gen_data["best_fitness"] = fitnesses[best_idx]
        gen_data["avg_novelty"] = float(np.mean(novelties)) if novelties else 0.0

        # Update all-time best fitness (monotonic)
        all_time_best_fitness = max(all_time_best_fitness, best_molecule["fitness"])

        print(f" Best fitness SMILES: {gen_data['best_smiles']}  (Fitness: {gen_data['best_fitness']:.4f})")
        print(f" Avg novelty this gen: {gen_data['avg_novelty']:.4f}")
        print(f" Overall best molecule: {best_molecule['smiles']} (Fitness: {best_molecule['fitness']:.4f}, Gen: {best_molecule['generation']})")
        print(f" Plateau length: {generations_since_best} generations")

        # Log JSON snapshot (less frequently to reduce file clutter)
        if generation % 50 == 0 or generation <= 10:  # Every 50 gens, plus first 10 for debugging
            timestamp = datetime.now(timezone.utc).isoformat()
            snapshot = {
                "timestamp": timestamp,
                "current_prompt": current_prompt,
                "all_time_best_fitness": all_time_best_fitness,
                "generations_since_best": generations_since_best,
                "novelty_archive_size": len(NOVELTY_ARCHIVE),
                **gen_data
            }
            with open(os.path.join(SNAPSHOTS_DIR, f"gen_{generation:03d}.json"), "w") as f:
                json.dump(snapshot, f, indent=2)

        # Save prompts and top molecules
        save_prompts()
        save_top_molecules()

        # Record history (fitness_history tracks all-time best, making it monotonic)
        fitness_history.append(all_time_best_fitness)
        novelty_history.append(gen_data["avg_novelty"])

        # Every VISUALIZE_EVERY generations, plot fitness & novelty
        if generation % VISUALIZE_EVERY == 0:
            plot_metrics(generation)
            save_summary_text(generation)

        # OPEN-ENDED SELECTION STRATEGY
        df = pd.DataFrame(gen_data["population"])
        df["rank_fitness"] = df["fitness"].rank(method="first", ascending=False)
        df["rank_novelty"] = df["novelty"].rank(method="first", ascending=False)

        # Adaptive selection based on plateau length
        survivors = set()
        
        if mutation_strategy == "novelty_push" or generations_since_best > PLATEAU_THRESHOLD:
            # During plateaus: PRIORITIZE NOVELTY (Stanley's principle)
            print("   → Selection: NOVELTY-FOCUSED (breaking plateau)")
            top_novelty = df.nsmallest(int(POPULATION_SIZE * 0.7), "rank_novelty")["smiles"]
            top_fitness = df.nsmallest(int(POPULATION_SIZE * 0.3), "rank_fitness")["smiles"]
        else:
            # Normal times: balanced selection
            top_fitness = df.nsmallest(POPULATION_SIZE // 2, "rank_fitness")["smiles"]
            top_novelty = df.nsmallest(POPULATION_SIZE // 2, "rank_novelty")["smiles"]
        
        for smi in list(top_fitness) + list(top_novelty):
            survivors.add(smi)

        # Fill next generation by repeating survivors (to maintain population size)
        next_pop = list(survivors)
        while len(next_pop) < POPULATION_SIZE:
            next_pop.append(np.random.choice(list(survivors)))
        population = next_pop[:POPULATION_SIZE]

        generation += 1
        if stop_requested:
            print("Evolution halted by user.")
            break


# ------------- Visualization -------------
def plot_metrics(generation):
    generations = list(range(1, len(fitness_history) + 1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fitness (now monotonic all-time best)
    ax1.plot(generations, fitness_history, label="All-Time Best Fitness", marker="o", color='blue')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (QED)")
    ax1.set_title("Evolution of All-Time Best Fitness (Monotonic)")
    ax1.grid(True)
    ax1.legend()
    
    # Add vertical lines for LLM mutations and prompt refinements
    for g in range(LLM_MUTATION_INTERVAL, generation + 1, LLM_MUTATION_INTERVAL):
        if g % PROMPT_REFINEMENT_INTERVAL != 0:  # Don't double-mark refinement generations
            ax1.axvline(x=g, color='orange', linestyle='--', alpha=0.7, label='LLM Mutation' if g == LLM_MUTATION_INTERVAL else "")
    
    for g in range(PROMPT_REFINEMENT_INTERVAL, generation + 1, PROMPT_REFINEMENT_INTERVAL):
        ax1.axvline(x=g, color='red', linestyle='-', alpha=0.7, label='Prompt Refinement' if g == PROMPT_REFINEMENT_INTERVAL else "")
    
    # Plot novelty
    ax2.plot(generations, novelty_history, label="Avg Novelty", marker="x", color='green')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Novelty")
    ax2.set_title("Evolution of Average Novelty")
    ax2.grid(True)
    ax2.legend()
    
    # Add vertical lines for LLM mutations and prompt refinements
    for g in range(LLM_MUTATION_INTERVAL, generation + 1, LLM_MUTATION_INTERVAL):
        if g % PROMPT_REFINEMENT_INTERVAL != 0:
            ax2.axvline(x=g, color='orange', linestyle='--', alpha=0.7)
    
    for g in range(PROMPT_REFINEMENT_INTERVAL, generation + 1, PROMPT_REFINEMENT_INTERVAL):
        ax2.axvline(x=g, color='red', linestyle='-', alpha=0.7)
    
    plt.tight_layout()
    png_path = os.path.join(PLOTS_DIR, f"metrics_gen_{generation}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [Metrics plot saved to {png_path}]")


def save_summary_text(generation):
    """Save a text summary instead of a PNG."""
    generations_since_best = generation - best_molecule["generation"]
    
    summary_content = f"""Evolution Summary - Generation {generation}
==========================================

Run ID: {RUN_ID}
Generation: {generation}
Timestamp: {datetime.now(timezone.utc).isoformat()}

BEST MOLECULE FOUND:
- SMILES: {best_molecule['smiles']}
- Fitness: {best_molecule['fitness']:.4f}
- Found in Generation: {best_molecule['generation']}
- Generations since found: {generations_since_best}

ALL-TIME BEST FITNESS: {all_time_best_fitness:.4f}

OPEN-ENDED EVOLUTION STATUS:
- Plateau length: {generations_since_best} generations
- Plateau threshold: {PLATEAU_THRESHOLD} generations
- Status: {"🚨 LONG PLATEAU - Seeking novelty" if generations_since_best > PLATEAU_THRESHOLD else "✅ Recent progress"}
- Novelty archive size: {len(NOVELTY_ARCHIVE)} stepping stones
- Highly novel molecules preserved: {len([m for m in NOVELTY_ARCHIVE if m['novelty'] >= MIN_NOVELTY_THRESHOLD])}

CURRENT EVOLUTION PARAMETERS:
- Population Size: {POPULATION_SIZE}
- LLM Mutation Interval: {LLM_MUTATION_INTERVAL}
- Prompt Refinement Interval: {PROMPT_REFINEMENT_INTERVAL}
- Novelty Push Interval: {NOVELTY_PUSH_INTERVAL}
- Diversity Restart Interval: {RESTART_INTERVAL}

CURRENT PROMPT:
{current_prompt}

BEST PERFORMING PROMPT (achieved fitness {best_prompt_info['fitness_achieved']:.4f}):
{best_prompt_info['prompt']}

PERFORMANCE HISTORY (last 10 generations):
- All-time best fitness: {fitness_history[-10:] if len(fitness_history) >= 10 else fitness_history}
- Average novelty: {novelty_history[-10:] if len(novelty_history) >= 10 else novelty_history}

ARCHIVE STATISTICS:
- Total molecules explored: {len(ARCHIVE_SMILES)}
- Novel molecules (novelty ≥ {MIN_NOVELTY_THRESHOLD}): {len(NOVELTY_ARCHIVE)}
- Top molecules found: {len(top_molecules)}
- Chemical diversity maintained: {"High" if len(NOVELTY_ARCHIVE) > 100 else "Medium" if len(NOVELTY_ARCHIVE) > 50 else "Growing"}

TOP 5 MOLECULES FOUND SO FAR:
{chr(10).join([f"{i+1}. {mol['smiles']} (fitness: {mol['fitness']:.4f}, gen: {mol['generation']})" for i, mol in enumerate(top_molecules[:5])])}

STANLEY'S OPEN-ENDED PRINCIPLES IN ACTION:
✓ Novelty preservation (stepping stones archive)
✓ Diversity injection during plateaus
✓ Adaptive selection pressure
✓ Creative exploration when stuck
✓ Multiple starting points for divergent evolution
"""
    
    summary_path = os.path.join(SUMMARIES_DIR, f"summary_gen_{generation}.txt")
    with open(summary_path, "w") as f:
        f.write(summary_content)
    print(f" [Summary text saved to {summary_path}]")


if __name__ == "__main__":
    print(f"Starting evolution run: {RUN_ID}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mutation intervals: LLM every {LLM_MUTATION_INTERVAL} gen, Prompt refinement every {PROMPT_REFINEMENT_INTERVAL} gen")
    print(f"Open-ended intervals: Novelty push every {NOVELTY_PUSH_INTERVAL} gen, Diversity restart every {RESTART_INTERVAL} gen")
    print(f"Plateau threshold: {PLATEAU_THRESHOLD} generations")
    evolve_forever()