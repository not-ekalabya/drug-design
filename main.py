import os
import json
import numpy as np
import random
from typing import List, Dict, Tuple
import google.generativeai as genai
from rdkit import Chem
from rdkit.Chem import Descriptors
import time

# Original imports from your code
from admet_ai import ADMETModel
from keras.models import load_model
import tensorflow as tf

# ---------------------------
# 1. Gemini API Setup
# ---------------------------
class GeminiDrugDesigner:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Memory for reinforcement learning
        self.memory = []
        self.best_molecules = []
        self.poor_molecules = []  # Track poorly performing molecules
        self.generation = 0
        
    def generate_molecules(self, n_molecules: int = 10, feedback_context: str = "") -> List[str]:
        """Generate new molecules using Gemini with RL feedback"""
        
        # Base prompt for molecular generation with detailed drug function context
        base_prompt = f"""
        You are an AI drug designer creating novel small molecules for pharmaceutical applications.
        
        DRUG FUNCTION GOAL:
        Design molecules that can bind to protein targets (specifically bacterial enzymes involved in cell wall synthesis) 
        to act as potential antibiotics. The ideal drug should:
        - Bind strongly to the target protein (binding affinity around 5.0)
        - Be moderately soluble in water (logS around -3.7) for proper distribution
        - Have high intestinal absorption (>90%) for oral delivery
        - Show good membrane permeability for cellular uptake
        - Maintain high oral bioavailability (>70%) for effective dosing
        
        DESIGN REQUIREMENTS:
        - Create structurally diverse, complex molecules (NOT simple alcohols like CCO or CC(C)C(=O)O)
        - Include aromatic rings, heterocycles, and functional groups typical of antibiotics
        - Target molecular weight: 200-500 Da (drug-like range)
        - Include nitrogen-containing rings or amide groups for protein binding
        - Consider beta-lactam rings, quinolone structures, or macrolide-like scaffolds
        - Avoid overly simple molecules that lack pharmaceutical complexity
        
        {feedback_context}
        
        Generate exactly {n_molecules} unique, complex SMILES strings for potential antibiotic compounds.
        Focus on molecules with multiple rings and functional groups suitable for protein binding.
        Return only the SMILES strings, one per line, with no additional text.
        """
        
        try:
            response = self.model.generate_content(base_prompt)
            smiles_list = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Validate SMILES and filter invalid ones
            valid_smiles = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # Clean up the SMILES
                    clean_smi = Chem.MolToSmiles(mol)
                    valid_smiles.append(clean_smi)
            
            return valid_smiles[:n_molecules]
            
        except Exception as e:
            print(f"Error generating molecules: {e}")
            # Fallback to more complex drug-like structures
            return [
                "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)N",  # More complex aromatic
                "C1CCC(CC1)NC(=O)C2=CC=CC=C2N"  # Cyclohexyl amide
            ]
    
    def update_memory(self, molecules: List[Dict]):
        """Update the agent's memory with new experiences"""
        self.memory.extend(molecules)
        
        # Separate good and poor performing molecules
        self.memory = sorted(self.memory, key=lambda x: x['reward'], reverse=True)
        if len(self.memory) > 200:  # Increased memory size
            self.memory = self.memory[:200]
            
        # Update best and worst molecules for learning
        self.best_molecules = self.memory[:15]  # Top performers
        self.poor_molecules = self.memory[-10:] if len(self.memory) > 10 else []  # Poor performers
        
    def generate_feedback_context(self) -> str:
        """Generate detailed context for the next generation based on performance"""
        if not self.memory:
            return ""
            
        feedback = f"\n=== GENERATION {self.generation} DETAILED FEEDBACK ===\n"
        
        # Analyze best performing molecules
        if self.best_molecules:
            feedback += "🎯 TOP PERFORMING MOLECULES (Learn from these patterns):\n"
            for i, mol in enumerate(self.best_molecules[:3]):
                reward_analysis = self.analyze_reward_components(mol)
                feedback += f"{i+1}. {mol['smiles']} (Overall Reward: {mol['reward']:.4f})\n"
                feedback += f"   {reward_analysis}\n"
        
        # Analyze poor performing molecules to avoid
        if self.poor_molecules:
            feedback += "\n❌ POOR PERFORMING MOLECULES (Avoid these patterns):\n"
            for i, mol in enumerate(self.poor_molecules[:3]):
                reward_analysis = self.analyze_reward_components(mol)
                feedback += f"{i+1}. {mol['smiles']} (Overall Reward: {mol['reward']:.4f})\n"
                feedback += f"   {reward_analysis}\n"
        
        # Strategic insights
        feedback += "\n📊 STRATEGIC INSIGHTS FOR ANTIBIOTIC DESIGN:\n"
        feedback += self.generate_strategic_insights()
        
        # Specific improvements needed
        feedback += "\n🎯 SPECIFIC IMPROVEMENTS NEEDED:\n"
        feedback += self.generate_improvement_suggestions()
        
        return feedback
    
    def analyze_reward_components(self, mol: Dict) -> str:
        """Provide natural language analysis of reward components"""
        diag = mol.get('diagnostics', {})
        analysis = []
        
        # Binding affinity analysis
        bind_score = diag.get('s_bind', 0)
        if bind_score > 0.8:
            analysis.append("✅ Excellent protein binding")
        elif bind_score > 0.5:
            analysis.append("⚠️ Moderate protein binding - needs improvement")
        else:
            analysis.append("❌ Poor protein binding - major issue")
            
        # Solubility analysis
        sol_score = diag.get('s_sol', 0)
        if sol_score > 0.8:
            analysis.append("✅ Good solubility")
        elif sol_score > 0.5:
            analysis.append("⚠️ Moderate solubility")
        else:
            analysis.append("❌ Poor solubility - will not dissolve properly")
            
        # Absorption analysis
        abs_score = diag.get('s_abs', 0)
        if abs_score > 0.8:
            analysis.append("✅ High intestinal absorption")
        elif abs_score > 0.5:
            analysis.append("⚠️ Moderate absorption")
        else:
            analysis.append("❌ Poor absorption - won't be absorbed by gut")
            
        # Permeability analysis
        pampa_score = diag.get('s_pampa', 0)
        if pampa_score > 0.8:
            analysis.append("✅ Good membrane permeability")
        elif pampa_score > 0.5:
            analysis.append("⚠️ Moderate permeability")
        else:
            analysis.append("❌ Poor permeability - can't cross cell membranes")
            
        # Bioavailability analysis
        bio_score = diag.get('s_bio', 0)
        if bio_score > 0.8:
            analysis.append("✅ High oral bioavailability")
        elif bio_score > 0.5:
            analysis.append("⚠️ Moderate bioavailability")
        else:
            analysis.append("❌ Low bioavailability - drug won't be effective orally")
            
        return " | ".join(analysis)
    
    def generate_strategic_insights(self) -> str:
        """Generate strategic insights based on performance patterns"""
        if not self.best_molecules:
            return "- No successful molecules yet - focus on more complex, drug-like structures\n"
            
        insights = []
        
        # Analyze successful patterns
        best_smiles = [mol['smiles'] for mol in self.best_molecules[:5]]
        
        # Check for common structural features
        has_aromatic = any('c' in smi or 'C1=C' in smi for smi in best_smiles)
        has_nitrogen = any('N' in smi for smi in best_smiles)
        has_oxygen = any('O' in smi for smi in best_smiles)
        has_rings = any('1' in smi or '2' in smi for smi in best_smiles)
        
        if has_aromatic:
            insights.append("- Aromatic rings appear beneficial for binding")
        if has_nitrogen:
            insights.append("- Nitrogen atoms improve drug-like properties")
        if has_oxygen:
            insights.append("- Oxygen functionalities help with solubility")
        if has_rings:
            insights.append("- Cyclic structures provide better binding geometry")
            
        # Avoid simple molecules
        insights.append("- CRITICAL: Avoid simple molecules like CCO, CC(C)O - they lack drug complexity")
        insights.append("- Target molecules should have 15-40 heavy atoms for proper drug-like properties")
        insights.append("- Include multiple functional groups for specific protein interactions")
        
        return "\n".join(insights) + "\n"
    
    def generate_improvement_suggestions(self) -> str:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        if not self.best_molecules or self.best_molecules[0]['reward'] < 0.3:
            suggestions.append("- URGENT: Current molecules are too simple - add aromatic rings and heterocycles")
            suggestions.append("- Include amide groups (C(=O)N) for hydrogen bonding with proteins")
            suggestions.append("- Add nitrogen-containing rings (pyridine, pyrimidine, imidazole)")
            
        if self.best_molecules and self.best_molecules[0]['reward'] < 0.5:
            suggestions.append("- Optimize binding: Add more rigid ring systems")
            suggestions.append("- Improve solubility: Include polar functional groups")
            suggestions.append("- Enhance absorption: Balance lipophilic and hydrophilic regions")
            
        suggestions.append("- Ensure molecular complexity: Use at least 2-3 ring systems")
        suggestions.append("- Target antibiotic scaffolds: Consider quinolone, macrolide, or beta-lactam inspired structures")
        
        return "\n".join(suggestions) + "\n"

# ---------------------------
# 2. Load existing models (from your original code)
# ---------------------------
admet_model = ADMETModel()

def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.linalg.band_part(tf.cast(f, tf.float32), -1, 0)
    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    return tf.where(tf.equal(g, 0), 0.0, g / f)

# Load binding affinity model
try:
    affinity_model = load_model(
        "DL4H/pretrained_models/combined_davis.h5",
        custom_objects={"cindex_score": cindex_score}
    )
except:
    print("Warning: Could not load affinity model. Using placeholder.")
    affinity_model = None

# ---------------------------
# 3. Protein sequence and encoding (from your original code)
# ---------------------------
protein_seq = "MKFVKYFLILAVCCILLGAGSIYGLYRYIEPQLPDVATLKDVRLQIPMQIYSADGELIAQYGEKRRIPVTLDQIPPEMVKAFIATEDSRFYEHHGVDPVGIFRAASVALFSGHASQGASTITQQLARNFFLSPERTLMRKIKEVFLAIRIEQLLTKDEILELYLNKIYLGYRAYGVGAAAQVYFGKTVDQLTLNEMAVIAGLPKAPSTFNPLYSMDRAVARRNVVLSRMLDEGYITQQQFDQTRTEAINANYHAPEIAFSAPYLSEMVRQEMYNRYGESAYEDGYRIYTTITRKVQQAAQQAVRNNVLDYDMRHGYRGPANVLWKVGESAWDNNKITDTLKALPTYGPLLPAAVTSANPQQATAMLADGSTVALSMEGVRWARPYRSDTQQGPTPRKVTDVLQTGQQIWVRQVGDAWWLAQVPEVNSALVSINPQNGAVMALVGGFDFNQSKFNRATQALRQVGSNIKPFLYTAAMDKGLTLASMLNDVPISRWDASAGSDWQPKNSPPQYAGPIRLRQGLGQSKNVVMVRAMRAMGVDYAAEYLQRFGFPAQNIVHTESLALGSASFTPMQVARGYAVMANGGFLVDPWFISKIENDQGGVIFEAKPKVACPECDIPVIYGDTQKSNVLENNDVEDVAISREQQNVSVPMPQLEQANQALVAKTGAQEYAPHVINTPLAFLIKSALNTNIFGEPGWQGTGWRAGRDLQRRDIGGKTGTTNSSKDAWFSGYGPGVVTSVWIGFDDHRRNLGHTTASGAIKDQISGYEGGAKSAQPAWDAYMKAVLEGVPEQPLTPPPGIVTVNIDRSTGQLANGGNSREEYFIEGTQPTQQAVHEVGTTIIDNGEAQELF"

smiles_vocab = list("CNOPSH123456789-=()@[]")
smiles_to_int = {ch: i + 1 for i, ch in enumerate(smiles_vocab)}
aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {aa: i + 1 for i, aa in enumerate(aa_alphabet)}

def encode_smiles(smi, max_len=100):
    seq = [smiles_to_int.get(ch, 0) for ch in smi]
    seq += [0] * max(0, max_len - len(seq))
    return np.array(seq[:max_len])

def encode_protein(prot, max_len=1000):
    seq = [aa_to_int.get(ch, 0) for ch in prot]
    seq += [0] * max(0, max_len - len(seq))
    return np.array(seq[:max_len])

# ---------------------------
# 4. Reward function (ADMET + Binding Affinity) - ORIGINAL VERSION WITHOUT DEEPCHEM
# ---------------------------
TARGET_PROPERTIES = {
    "binding_affinity": 5.0,     # strong binding
    "Solubility_AqSolDB": -3.7,  # moderately soluble
    "HIA_Hou": 0.97,             # high intestinal absorption
    "PAMPA_NCATS": 0.62,         # target permeability
    "Bioavailability_Ma": 0.74   # target oral bioavailability
}

def calculate_reward(smiles_list):
    results = []

    for smi in smiles_list:
        try:
            # --- ADMET predictions ---
            admet_preds = admet_model.predict(smiles=smi)

            # --- Binding affinity (your pretrained keras model) ---
            if affinity_model:
                try:
                    X_smiles = np.expand_dims(encode_smiles(smi), axis=0)
                    X_protein = np.expand_dims(encode_protein(protein_seq), axis=0)
                    pred_affinity = affinity_model.predict([X_smiles, X_protein])[0][0]
                except Exception as e:
                    print(f"Error predicting affinity for {smi}: {e}")
                    pred_affinity = 3.0  # Default value
            else:
                pred_affinity = 3.0

            # --- Compute similarity scores ---
            s_bind = np.exp(-abs(pred_affinity - TARGET_PROPERTIES["binding_affinity"]))
            s_sol = np.exp(-abs(admet_preds.get("Solubility_AqSolDB", 0) - TARGET_PROPERTIES["Solubility_AqSolDB"]))
            s_abs = np.exp(-abs(admet_preds.get("HIA_Hou", 0) - TARGET_PROPERTIES["HIA_Hou"]))
            s_pampa = np.exp(-abs(admet_preds.get("PAMPA_NCATS", 0) - TARGET_PROPERTIES["PAMPA_NCATS"]))
            s_bio = np.exp(-abs(admet_preds.get("Bioavailability_Ma", 0) - TARGET_PROPERTIES["Bioavailability_Ma"]))

            final_reward = np.mean([s_bind, s_sol, s_abs, s_pampa, s_bio])

            results.append({
                "smiles": smi,
                "reward": final_reward,
                "admet_ai": admet_preds,
                "binding_affinity": float(pred_affinity),
                "deepchem_pred": None,  # Removed DeepChem prediction
                "diagnostics": {
                    "s_bind": s_bind,
                    "s_sol": s_sol,
                    "s_abs": s_abs,
                    "s_pampa": s_pampa,
                    "s_bio": s_bio
                }
            })
            
        except Exception as e:
            print(f"Error processing {smi}: {e}")
            continue

    results = sorted(results, key=lambda x: x["reward"], reverse=True)
    return results

# ---------------------------
# 5. Main reinforcement learning loop
# ---------------------------
def run_drug_design_rl(api_key: str, n_generations: int = 20, n_molecules_per_gen: int = 15):
    """Main RL loop for drug design"""
    
    designer = GeminiDrugDesigner(api_key)
    all_results = []
    
    print("Starting LLM-based drug design with reinforcement learning...")
    print(f"Target: Discovering novel drug candidates through optimization")
    print(f"Generations: {n_generations}, Molecules per generation: {n_molecules_per_gen}")
    print("=" * 80)
    
    for generation in range(n_generations):
        print(f"\n--- Generation {generation + 1} ---")
        designer.generation = generation
        
        # Generate feedback context from previous results
        feedback_context = designer.generate_feedback_context()
        
        # Generate new molecules
        print("Generating molecules...")
        molecules = designer.generate_molecules(n_molecules_per_gen, feedback_context)
        
        if not molecules:
            print("No valid molecules generated, skipping generation")
            continue
            
        print(f"Generated {len(molecules)} valid molecules")
        
        # Evaluate molecules
        print("Evaluating molecules...")
        results = calculate_reward(molecules)
        
        if not results:
            print("No molecules could be evaluated, skipping generation")
            continue
        
        # Update agent's memory
        designer.update_memory(results)
        all_results.extend(results)
        
        # Print top results for this generation
        print(f"\nTop 3 molecules from generation {generation + 1}:")
        for i, res in enumerate(results[:3]):
            print(f"{i+1}. SMILES: {res['smiles']}")
            print(f"   Reward: {res['reward']:.4f}")
            print(f"   Binding Affinity: {res['binding_affinity']:.3f}")
        
        # Small delay to avoid API rate limits
        time.sleep(1)
    
    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Top 10 discovered molecules:")
    print("=" * 80)
    
    final_best = sorted(all_results, key=lambda x: x["reward"], reverse=True)[:10]
    
    for i, res in enumerate(final_best):
        print(f"\nRank {i+1}:")
        print(f"SMILES: {res['smiles']}")
        print(f"Reward: {res['reward']:.4f}")
        print(f"Binding Affinity: {res['binding_affinity']:.3f}")
        print(f"DeepChem Prediction: {res['deepchem_pred']}")
        print(f"ADMET Properties: {res['admet_ai']}")
        print("-" * 40)
    
    return final_best

# ---------------------------
# 6. Example usage
# ---------------------------
if __name__ == "__main__":
    # Set your Gemini API key
    GEMINI_API_KEY = "AIzaSyBpOg1ccrnhEXyx62sLhZLXBvF3X_bY_TE"  # Set this environment variable
    
    if not GEMINI_API_KEY:
        print("Please set your GEMINI_API_KEY environment variable")
        print("You can get an API key from: https://makersuite.google.com/app/apikey")
        # For testing, you can also set it directly here (not recommended for production):
        # GEMINI_API_KEY = "your-api-key-here"
    else:
        # Run the drug design process
        best_molecules = run_drug_design_rl(
            api_key=GEMINI_API_KEY,
            n_generations=10,  # Start with fewer generations for testing
            n_molecules_per_gen=10
        )
        
        # Save results
        with open("drug_design_results.json", "w") as f:
            json.dump(best_molecules, f, indent=2, default=str)
        
        print(f"\nResults saved to 'drug_design_results.json'")
        print("Experiment completed!")