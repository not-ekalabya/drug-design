import os
import json
import numpy as np
import random
from typing import List, Dict, Tuple
import google.generativeai as genai
from rdkit import Chem
from rdkit.Chem import Descriptors
import time
import re

# Original imports from your code
from admet_ai import ADMETModel
from keras.models import load_model
import tensorflow as tf

# ---------------------------
# 1. Enhanced Gemini API Setup with Structured Reasoning
# ---------------------------
class GeminiDrugDesigner:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Enhanced memory for reinforcement learning
        self.memory = []
        self.best_molecules = []
        self.poor_molecules = []
        self.generation_history = []  # Track generation-by-generation performance
        self.structural_insights = []  # Track what structures work/don't work
        self.generation = 0
        
        # Track exploration diversity
        self.explored_scaffolds = set()
        self.successful_patterns = []
        self.failed_patterns = []
        
    def extract_smiles_from_response(self, response_text: str) -> List[str]:
        """Enhanced SMILES extraction with multiple fallback methods"""
        smiles_list = []
        
        # Method 1: Look for explicit SMILES sections
        smiles_pattern = r'SMILES:\s*([A-Za-z0-9@\[\]()=\-#+\\/.%]+)'
        matches = re.findall(smiles_pattern, response_text)
        smiles_list.extend(matches)
        
        # Method 2: Look for numbered lists with SMILES
        numbered_pattern = r'\d+\.\s*([A-Za-z0-9@\[\]()=\-#+\\/.%]+)'
        matches = re.findall(numbered_pattern, response_text)
        smiles_list.extend(matches)
        
        # Method 3: Extract lines that look like SMILES (fallback)
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove any prefixes like "1.", "SMILES:", etc.
            cleaned_line = re.sub(r'^[\d\.\s\-\*]+', '', line)
            cleaned_line = re.sub(r'^SMILES:\s*', '', cleaned_line, flags=re.IGNORECASE)
            cleaned_line = cleaned_line.strip()
            
            # Check if it looks like a SMILES string
            if (len(cleaned_line) > 5 and 
                any(c in cleaned_line for c in 'CNOPSc()[]') and
                not any(word in cleaned_line.lower() for word in ['the', 'this', 'molecule', 'compound', 'structure'])):
                smiles_list.append(cleaned_line)
        
        # Validate and clean SMILES
        valid_smiles = []
        for smi in smiles_list:
            # Clean up common formatting issues
            smi = smi.strip().rstrip('.,;')
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # Canonicalize the SMILES
                    clean_smi = Chem.MolToSmiles(mol)
                    if clean_smi not in valid_smiles:  # Avoid duplicates
                        valid_smiles.append(clean_smi)
            except:
                continue
                
        return valid_smiles

    # SCAFFOLD LIBRARY (choose from these when designing molecules, use at least 3 different scaffolds across the set):
    # - Beta-lactam (penicillin/cephalosporin-like fused azetidinone)
    # - Carbapenem (bicyclic beta-lactam)
    # - Monobactam (standalone beta-lactam)
    # - Quinolone (4-oxo-1,4-dihydroquinoline)
    # - Sulfonamide (aryl-SO2-NH motif)
    # - Oxazolidinone (linezolid-like heterocycle)
    # - Thiazole/thiadiazole (S/N heterocycles)
    # - Benzoxazole (O/N heteroaromatic ring)
    # - Aminopyrimidine/aminopyridine (heteroaryl with basic nitrogen)
    # - Macrocyclic (if MW ≤ 500 Da)

    def generate_molecules(self, n_molecules: int = 10, feedback_context: str = "") -> List[str]:
        """Generate new molecules using Gemini with enhanced reasoning and feedback"""
        
        # Enhanced prompt with step-by-step reasoning requirement
        base_prompt = f"""
TASK: Design novel small-molecule inhibitors of plasma kallikrein (KLKB1) 
for oral on-demand therapy in hereditary angioedema (HAE).

BACKGROUND ON PREVIOUS DRUG CLASSES:
- Ecallantide: a peptide-based inhibitor. It is potent but not orally bioavailable. 
  Large peptides tend to have poor absorption, so avoid peptide-like scaffolds.
- Lanadelumab: a monoclonal antibody. Highly effective for prophylaxis but 
  unsuitable as a small-molecule oral drug.
- Earlier small-molecule inhibitors often relied on basic amidine or guanidine 
  scaffolds that bound strongly to the protease active site. These showed potency 
  but poor selectivity across serine proteases and weak oral bioavailability.

WHAT THIS MEANS FOR DESIGN:
- We need new scaffolds beyond peptides and antibodies. 
- Amidine- or guanidine-based scaffolds are acceptable starting points, 
  but the model should also explore alternative heteroaromatic or amide-linked 
  scaffolds that can provide potency without compromising oral drug-like properties.

DESIRED PROPERTY PROFILE:
- Potency: nanomolar inhibition (target Ki or IC50 ≈ 1–10 nM).
- Molecular weight: 350–550 Da (drug-like range)
- LogP: ~1.5–3.5 (balance solubility and permeability).
- Polar surface area (TPSA): < 120 Å² for absorption.
- Solubility and permeability: sufficient for fast onset.
- Bioavailability: > 50%.
- Low CYP450 inhibition.
- High selectivity versus other serine proteases.

REQUIREMENTS:
1. Propose at least 3 different scaffold families (e.g., amidine-like, heteroaromatic 
   cores, amide-linked frameworks).
2. For each scaffold family, explain in 2–3 sentences how its chemistry could 
   improve potency, selectivity, or PK compared to past approaches.
3. For each scaffold, generate 2–3 example molecules as valid SMILES strings.
4. Ensure molecules are synthetically plausible and drug-like.

OUTPUT FORMAT:
- Section per scaffold family.
- Short reasoning paragraph.
- Then exactly N SMILES lines, each formatted as: SMILES: (smiles string), do not include any quotation marks

        """
        
        try:
            response = self.model.generate_content(base_prompt)
            print(f"\n--- Gemini Reasoning (Generation {self.generation + 1}) ---")
            print(response.text)
            print("--- End Reasoning ---\n")
            
            # Extract SMILES using enhanced method
            valid_smiles = self.extract_smiles_from_response(response.text)
            
            if len(valid_smiles) < n_molecules:
                print(f"Warning: Only extracted {len(valid_smiles)} valid SMILES out of {n_molecules} requested")
                
            return valid_smiles[:n_molecules]
            
        except Exception as e:
            print(f"Error generating molecules: {e}")
            # Enhanced fallback with more complex drug-like structures
            return [
                
            ]
    
    def update_memory(self, molecules: List[Dict]):
        """Enhanced memory update with structural analysis"""
        self.memory.extend(molecules)
        
        # Sort and trim memory
        self.memory = sorted(self.memory, key=lambda x: x['reward'], reverse=True)
        if len(self.memory) > 300:  # Increased memory size
            self.memory = self.memory[:300]
            
        # Update performance categories
        self.best_molecules = self.memory[:20]  # Top performers
        self.poor_molecules = self.memory[-15:] if len(self.memory) > 15 else []
        
        # Track generation performance
        gen_avg_reward = np.mean([mol['reward'] for mol in molecules])
        self.generation_history.append({
            'generation': self.generation,
            'avg_reward': gen_avg_reward,
            'best_reward': max(mol['reward'] for mol in molecules),
            'molecules': len(molecules)
        })
        
        # Analyze structural patterns
        self._analyze_structural_patterns(molecules)
        
    def _analyze_structural_patterns(self, molecules: List[Dict]):
        """Analyze what structural patterns are working or failing"""
        for mol in molecules:
            try:
                m = Chem.MolFromSmiles(mol['smiles'])
                if m is None:
                    continue
                    
                # Extract basic structural features
                features = {
                    'aromatic_rings': Chem.rdMolDescriptors.CalcNumAromaticRings(m),
                    'rings': Chem.rdMolDescriptors.CalcNumRings(m),
                    'hbd': Chem.rdMolDescriptors.CalcNumHBD(m),
                    'hba': Chem.rdMolDescriptors.CalcNumHBA(m),
                    'mol_weight': Descriptors.MolWt(m),
                    'has_nitrogen': 'N' in mol['smiles'],
                    'has_sulfur': 'S' in mol['smiles'],
                    'has_fluorine': 'F' in mol['smiles']
                }
                
                # Categorize performance
                if mol['reward'] > 0.4:  # Good performance
                    self.successful_patterns.append(features)
                elif mol['reward'] < 0.2:  # Poor performance
                    self.failed_patterns.append(features)
                    
            except Exception as e:
                continue
                
        # Keep only recent patterns
        if len(self.successful_patterns) > 50:
            self.successful_patterns = self.successful_patterns[-50:]
        if len(self.failed_patterns) > 50:
            self.failed_patterns = self.failed_patterns[-50:]
    
    def generate_feedback_context(self) -> str:
        """Generate comprehensive feedback context with natural language insights"""
        if not self.memory:
            return ""
            
        feedback = f"\n{'='*20} GENERATION {self.generation} COMPREHENSIVE FEEDBACK {'='*20}\n"
        
        # Performance trend analysis
        if len(self.generation_history) > 1:
            recent_trend = self.generation_history[-3:] if len(self.generation_history) >= 3 else self.generation_history
            avg_rewards = [g['avg_reward'] for g in recent_trend]
            
            if len(avg_rewards) > 1:
                if avg_rewards[-1] > avg_rewards[-2]:
                    feedback += "📈 POSITIVE TREND: Performance is improving! Continue current strategy.\n"
                else:
                    feedback += "📉 PERFORMANCE DIP: Need to try new approaches and diversify structures.\n"
        
        # Detailed analysis of top performers
        if self.best_molecules:
            feedback += "\n🏆 TOP PERFORMING MOLECULES (Learn from these successes):\n"
            for i, mol in enumerate(self.best_molecules[:5]):
                reward_analysis = self._detailed_property_analysis(mol)
                molecular_insights = self._analyze_molecular_features(mol['smiles'])
                
                feedback += f"\n{i+1}. SMILES: {mol['smiles']}\n"
                feedback += f"   Overall Reward: {mol['reward']:.4f}\n"
                feedback += f"   Property Analysis: {reward_analysis}\n"
                feedback += f"   Structural Features: {molecular_insights}\n"
                feedback += f"   Key Success Factors: {self._identify_success_factors(mol)}\n"
        
        # Analysis of poor performers with specific warnings
        if self.poor_molecules:
            feedback += "\n❌ POOR PERFORMING MOLECULES (Avoid these patterns):\n"
            for i, mol in enumerate(self.poor_molecules[:3]):
                feedback += f"\n{i+1}. SMILES: {mol['smiles']} (Reward: {mol['reward']:.4f})\n"
                feedback += f"   Problems: {self._identify_failure_modes(mol)}\n"
                feedback += f"   Avoid: {self._extract_problematic_features(mol['smiles'])}\n"
        
        # Strategic exploration guidance
        feedback += "\n🎯 STRATEGIC EXPLORATION GUIDANCE:\n"
        feedback += self._generate_exploration_strategy()
        
        # Specific molecular design recommendations
        feedback += "\n💡 SPECIFIC DESIGN RECOMMENDATIONS:\n"
        feedback += self._generate_design_recommendations()
        
        # Diversity encouragement
        feedback += "\n🌟 DIVERSITY CHALLENGE:\n"
        feedback += self._generate_diversity_challenge()
        
        return feedback
    
    def _detailed_property_analysis(self, mol: Dict) -> str:
        """Provide detailed analysis of each property"""
        diag = mol.get('diagnostics', {})
        analysis = []
        
        bind_score = diag.get('s_bind', 0)

        # DeepDTA binding scores are pKd or KIBA scores
        # Adjust thresholds accordingly (example for pKd)
        if bind_score >= 8:  # Strong binding (nM or better)
            analysis.append(f"✅ Excellent binding (pKd={bind_score:.2f})")
        elif bind_score >= 6:  # Micromolar range
            analysis.append(f"⚠️ Moderate binding (pKd={bind_score:.2f} - needs optimization)")
        else:  # Weak binding
            analysis.append(f"❌ Poor binding (pKd={bind_score:.2f} - major issue)")

            
        sol_score = diag.get('s_sol', 0)
        if sol_score > 0.8:
            analysis.append("✅ Optimal solubility")
        elif sol_score > 0.5:
            analysis.append(f"⚠️ Moderate solubility ({sol_score:.3f})")
        else:
            analysis.append(f"❌ Solubility too low ({sol_score:.3f})")
            
        return " | ".join(analysis)
    
    def _analyze_molecular_features(self, smiles: str) -> str:
        """Analyze key molecular features"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid structure"
                
            features = []
            
            # Ring analysis
            num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
            num_aromatic = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
            if num_aromatic > 0:
                features.append(f"{num_aromatic} aromatic rings")
            if num_rings - num_aromatic > 0:
                features.append(f"{num_rings - num_aromatic} non-aromatic rings")
                
            # Heteroatom analysis
            if 'N' in smiles:
                features.append("nitrogen present")
            if 'O' in smiles and '=O' in smiles:
                features.append("carbonyl groups")
            if 'S' in smiles:
                features.append("sulfur functionality")
                
            # Size analysis
            mol_weight = Descriptors.MolWt(mol)
            features.append(f"MW: {mol_weight:.1f}")
            
            return ", ".join(features)
            
        except Exception as e:
            return "Analysis failed"
    
    def _identify_success_factors(self, mol: Dict) -> str:
        """Identify why a molecule performed well"""
        factors = []
        diag = mol.get('diagnostics', {})
        
        if diag.get('s_bind', 0) > 0.7:
            factors.append("strong protein binding")
        if diag.get('s_sol', 0) > 0.7:
            factors.append("good solubility balance")
        if diag.get('s_abs', 0) > 0.7:
            factors.append("high absorption")
            
        # Structural analysis
        smiles = mol['smiles']
        if 'c1c' in smiles or 'C1=C' in smiles:
            factors.append("aromatic rings for binding")
        if 'N' in smiles and 'C(=O)' in smiles:
            factors.append("amide functionality")
            
        return ", ".join(factors) if factors else "unclear success pattern"
    
    def _identify_failure_modes(self, mol: Dict) -> str:
        """Identify why a molecule performed poorly"""
        problems = []
        diag = mol.get('diagnostics', {})
        
        if diag.get('s_bind', 0) < 0.3:
            problems.append("very poor binding affinity")
        if diag.get('s_sol', 0) < 0.3:
            problems.append("solubility issues")
        if diag.get('s_abs', 0) < 0.3:
            problems.append("poor absorption")
            
        return ", ".join(problems) if problems else "multiple property failures"
    
    def _extract_problematic_features(self, smiles: str) -> str:
        """Extract features to avoid based on poor performance"""
        avoid = []
        
        # Check for overly simple structures
        if len(smiles) < 15:
            avoid.append("overly simple structure")
        
        # Check for lack of rings
        if '1' not in smiles and '2' not in smiles:
            avoid.append("no ring systems")
            
        # Check for lack of functional groups
        if 'N' not in smiles and 'O' not in smiles:
            avoid.append("no heteroatoms")
            
        return ", ".join(avoid) if avoid else "pattern unclear"
    
    def _generate_exploration_strategy(self) -> str:
        """Generate strategy for exploring new chemical space"""
        strategy = []
        
        # Analyze what's been tried
        if len(self.best_molecules) > 0:
            best_reward = self.best_molecules[0]['reward']
            if best_reward < 0.3:
                strategy.append("- URGENT: Current designs too simple - try complex multi-ring systems")
            elif best_reward < 0.5:
                strategy.append("- Moderate progress - refine current successful patterns")
                strategy.append("- Add more polar groups to top performers")
                strategy.append("- Explore bioisosteric replacements")
            else:
                strategy.append("- Good progress - explore variations of successful scaffolds")
                strategy.append("- Try larger ring systems and fused aromatics")
        
        strategy.append("- Ensure each molecule targets different chemical space")
        strategy.append("- Balance innovation with learning from successes")
        
        return "\n".join(strategy) + "\n"
    
    def _generate_design_recommendations(self) -> str:
        """Generate specific molecular design recommendations"""
        recommendations = []
        
        # Based on successful patterns
        if self.successful_patterns:
            avg_aromatic = np.mean([p['aromatic_rings'] for p in self.successful_patterns[-10:]])
            if avg_aromatic > 1.5:
                recommendations.append("- Include 2+ aromatic rings (successful pattern)")
            
        recommendations.extend([
            "- Target molecular weight 250-450 Da for optimal properties",
            "- Include nitrogen in rings for binding (quinoline, pyrimidine, imidazole)",
            "- Add hydrogen bond donors/acceptors (NH, OH, C=O)",
            "- Consider sulfonamide groups for antibacterial activity",
            "- Use fluorine sparingly for metabolic stability",
            "- Ensure at least 2 ring systems for rigidity"
        ])
        
        return "\n".join(recommendations) + "\n"
    
    def _generate_diversity_challenge(self) -> str:
        """Generate challenges to promote structural diversity"""
        challenges = [
            "- Try a quinolone-inspired scaffold you haven't used before",
            "- Design a macrocyclic structure (large ring >12 atoms)",
            "- Explore spiro compounds (molecules with shared ring atoms)",
            "- Include a heterocycle you haven't tried (thiazole, oxazole, pyrazole)",
            "- Design a molecule with a sulfonamide group",
            "- Try a bicyclic system with both aromatic and saturated rings",
            "- Explore molecules with chiral centers for selectivity",
            "- Design a prodrug concept with cleavable groups"
        ]
        
        # Randomly select 3-4 challenges
        selected = random.sample(challenges, min(4, len(challenges)))
        return "\n".join(selected) + "\n"

# Rest of the code remains the same (ADMET model loading, reward calculation, etc.)
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
def read_fasta(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    seq_lines = [line.strip() for line in lines if not line.startswith(">") and line.strip()]
    return "".join(seq_lines)

protein_seq = "MILFKQATYFISLFATVSCGCLTQLYENAFFRGGDVASMYTPNAQYCQMRCTFHPRCLLFSFLPASSINDMEKRFGCFLKDSVTGTLPKVHRTGAVSGHSLKQCGHQISACHRDIYKGVDMRGVNFNVSKVSSVEECQKRCTSNIRCQFFSYATQTFHKAEYRNNCLLKYSPGGTPTAIKVLSNVESGFSLKPCALSEIGCHMNIFQHLAFSDVDVARVLTPDAFVCRTICTYHPNCLFFTFYTNVWKIESQRNVCLLKTSESGTPSSSTPQENTISGYSLLTCKRTLPEPCHSKIYPGVDFGGEELNVTFVKGVNVCQETCTKMIRCQFFTYSLLPEDCKEEKCKCFLRLSMDGSPTRIAYGTQGSSGYSLRLCNTGDNSVCTTKTSTRIVGGTNSSWGEWPWQVSLQVKLTAQRHLCGGSLIGHQWVLTAAHCFDGLPLQDVWRIYSGILNLSDITKDTPFSQIKEIIIHQNYKVSEGNHDIALIKLQAPLNYTEFQKPICLPSKGDTSTIYTNCWVTGWGFSKEKGEIQNILQKVNIPLVTNEECQKRYQDYKITQRMVCAGYKEGGKDACKGDSGGPLVCKHNGMWRLVGITSWGEGCARREQPGVYTKVAEYMDWILEKTQSSDGKAQMQSPA"

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
# 4. Reward function (ADMET + Binding Affinity)
# ---------------------------
TARGET_PROPERTIES = {
    "binding_affinity": 8.0,  # Strong binding (nM range) - higher pKd value
    "Solubility_AqSolDB": -4.5,  # Moderate solubility for oral dosing
    "HIA_Hou": 0.85,  # Good intestinal absorption
    "PAMPA_NCATS": 0.75,  # Good BBB penetration potential
    "Bioavailability_Ma": 0.65,  # Reasonable oral bioavailability
    "BBB_Martins": 0.8,  # High blood-brain barrier penetration
    "CYP3A4_Substrate_CarbonMangels": 0.3,  # Low CYP3A4 interaction
}

def calculate_reward(smiles_list):
    results = []

    for smi in smiles_list:
        try:
            # --- ADMET predictions ---
            admet_preds = admet_model.predict(smiles=smi)

            # --- Binding affinity ---
            if affinity_model:
                try:
                    X_smiles = np.expand_dims(encode_smiles(smi), axis=0)
                    X_protein = np.expand_dims(encode_protein(protein_seq), axis=0)
                    pred_affinity = affinity_model.predict([X_smiles, X_protein])[0][0]
                except Exception as e:
                    print(f"Error predicting affinity for {smi}: {e}")
                    pred_affinity = 3.0
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
                "deepchem_pred": None,
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
# 5. Enhanced main reinforcement learning loop
# ---------------------------
def run_drug_design_rl(api_key: str, n_generations: int = 10, n_molecules_per_gen: int = 15):
    """Enhanced RL loop for drug design with better feedback"""
    
    designer = GeminiDrugDesigner(api_key)
    all_results = []
    
    print("🧬 Starting Enhanced LLM-based Drug Design with Structured Reasoning")
    print(f"🎯 Target: Novel antibiotic candidates with optimized ADMET properties")
    print(f"📊 Parameters: {n_generations} generations, {n_molecules_per_gen} molecules per generation")
    print(f"🤖 Using Gemini with step-by-step reasoning and enhanced feedback")
    print("=" * 90)
    
    for generation in range(n_generations):
        print(f"\n{'='*20} GENERATION {generation + 1} {'='*20}")
        designer.generation = generation
        
        # Generate comprehensive feedback context
        feedback_context = designer.generate_feedback_context()
        
        # Generate new molecules with reasoning
        print("🧠 Generating molecules with structured reasoning...")
        molecules = designer.generate_molecules(n_molecules_per_gen, feedback_context)
        
        if not molecules:
            print("❌ No valid molecules generated, skipping generation")
            continue
            
        print(f"✅ Generated {len(molecules)} valid molecules")
        print(f"📝 Molecules: {molecules}")
        
        # Evaluate molecules
        print("🔬 Evaluating molecular properties...")
        results = calculate_reward(molecules)
        
        if not results:
            print("❌ No molecules could be evaluated, skipping generation")
            continue
        
        # Update agent's memory with enhanced analysis
        designer.update_memory(results)
        all_results.extend(results)
        
        # Print detailed results for this generation
        print(f"\n📈 GENERATION {generation + 1} RESULTS:")
        print("-" * 60)
        for i, res in enumerate(results[:5]):  # Show top 5
            print(f"{i+1:2d}. SMILES: {res['smiles']}")
            print(f"    Reward: {res['reward']:.4f} | Binding: {res['binding_affinity']:.3f}")
            
            # Show property breakdown
            diag = res['diagnostics']
            print(f"    Properties: Bind={diag['s_bind']:.3f} | Sol={diag['s_sol']:.3f} | Abs={diag['s_abs']:.3f}")
            print()
        
        # Show generation statistics
        rewards = [r['reward'] for r in results]
        print(f"📊 Generation Stats: Avg={np.mean(rewards):.4f} | Max={np.max(rewards):.4f} | Min={np.min(rewards):.4f}")
        
        # Small delay to avoid API rate limits
        time.sleep(2)
    
    # Enhanced final analysis
    print("\n" + "=" * 90)
    print("🏆 FINAL RESULTS - BEST DISCOVERED MOLECULES")
    print("=" * 90)
    
    final_best = sorted(all_results, key=lambda x: x["reward"], reverse=True)[:15]
    
    print("\n📋 TOP 15 MOLECULES WITH DETAILED ANALYSIS:")
    print("-" * 90)
    
    for i, res in enumerate(final_best):
        print(f"\n🥇 RANK {i+1}:")
        print(f"   SMILES: {res['smiles']}")
        print(f"   Overall Reward: {res['reward']:.4f}")
        print(f"   Binding Affinity: {res['binding_affinity']:.3f}")
        
        # Property breakdown
        diag = res['diagnostics']
        print(f"   Property Scores:")
        print(f"     - Binding: {diag['s_bind']:.4f} (target: strong protein binding)")
        print(f"     - Solubility: {diag['s_sol']:.4f} (target: moderate water solubility)")
        print(f"     - Absorption: {diag['s_abs']:.4f} (target: high intestinal absorption)")
        print(f"     - Permeability: {diag['s_pampa']:.4f} (target: good membrane permeability)")
        print(f"     - Bioavailability: {diag['s_bio']:.4f} (target: high oral bioavailability)")
        
        # Molecular analysis
        try:
            mol = Chem.MolFromSmiles(res['smiles'])
            if mol:
                mw = Descriptors.MolWt(mol)
                rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                aromatic = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
                hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
                hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
                print(f"   Molecular Features:")
                print(f"     - Molecular Weight: {mw:.1f} Da")
                print(f"     - Rings: {rings} total ({aromatic} aromatic)")
                print(f"     - H-bond donors: {hbd}, H-bond acceptors: {hba}")
        except:
            print(f"   Molecular analysis failed")
            
        print(f"   Raw ADMET: {res['admet_ai']}")
        print("-" * 60)
    
    # Performance trend analysis
    if len(designer.generation_history) > 1:
        print(f"\n📈 LEARNING PROGRESS ANALYSIS:")
        print("-" * 40)
        for i, gen_data in enumerate(designer.generation_history):
            print(f"Generation {gen_data['generation']+1:2d}: Avg Reward = {gen_data['avg_reward']:.4f} | Best = {gen_data['best_reward']:.4f}")
        
        # Calculate improvement
        first_avg = designer.generation_history[0]['avg_reward']
        last_avg = designer.generation_history[-1]['avg_reward']
        improvement = ((last_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        print(f"\n🎯 Overall Improvement: {improvement:+.1f}%")
    
    # Success rate analysis
    excellent_molecules = sum(1 for r in final_best if r['reward'] > 0.9)
    good_molecules = sum(1 for r in final_best if r['reward'] > 0.8)
    
    print(f"\n🎉 SUCCESS METRICS:")
    print(f"   - Excellent molecules (reward > 0.9): {excellent_molecules}")
    print(f"   - Good molecules (reward > 0.8): {good_molecules}")
    print(f"   - Total molecules evaluated: {len(all_results)}")
    
    return final_best

# ---------------------------
# 6. Enhanced example usage with better error handling
# ---------------------------
if __name__ == "__main__":
    # Set your Gemini API key
    GEMINI_API_KEY = "YOUR_KEY_HERE"
    
    if not GEMINI_API_KEY:
        print("❌ Please set your GEMINI_API_KEY")
        print("🔗 Get an API key from: https://makersuite.google.com/app/apikey")
    else:
        try:
            print("🚀 Initializing Enhanced Drug Design System...")
            
            # Run the enhanced drug design process
            best_molecules = run_drug_design_rl(
                api_key=GEMINI_API_KEY,
                n_generations=4,  # Reasonable number for testing
                n_molecules_per_gen=12  # Manageable batch size
            )
            
            # Enhanced result saving
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_drug_design_results_{timestamp}.json"
            
            # Prepare data for JSON serialization
            json_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "total_molecules": len(best_molecules),
                    "generations": 3,
                    "molecules_per_generation": 8
                },
                "best_molecules": best_molecules
            }
            
            with open(filename, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to '{filename}'")
            print("🎊 Enhanced Drug Design Experiment Completed Successfully!")
            
            # Print summary of best result
            if best_molecules:
                best = best_molecules[0]
                print(f"\n🏆 BEST MOLECULE DISCOVERED:")
                print(f"   SMILES: {best['smiles']}")
                print(f"   Reward: {best['reward']:.4f}")
                print(f"   This molecule shows promise as a potential drug candidate!")
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()