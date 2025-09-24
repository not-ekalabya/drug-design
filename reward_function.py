import numpy as np
import sys
from typing import Dict, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors

# Import the required models
from admet_ai import ADMETModel
from keras.models import load_model
import tensorflow as tf

class DrugRewardEvaluator:
    """
    A standalone evaluator for calculating drug reward scores based on ADMET properties,
    binding affinity predictions, and toxicity penalties.
    """
    
    def __init__(self):
        """Initialize the evaluator with pre-trained models"""
        print("🔬 Initializing Drug Reward Evaluator...")
        
        # Load ADMET model
        try:
            self.admet_model = ADMETModel()
            print("✅ ADMET model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading ADMET model: {e}")
            self.admet_model = None
        
        # Custom metric for binding affinity model
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
            self.affinity_model = load_model(
                "DL4H/pretrained_models/combined_davis.h5",
                custom_objects={"cindex_score": cindex_score}
            )
            print("✅ Binding affinity model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load affinity model: {e}")
            print("   Using placeholder predictions for binding affinity")
            self.affinity_model = None
        
        # Target protein sequence (plasma kallikrein - KLKB1)
        self.protein_seq = "MILFKQATYFISLFATVSCGCLTQLYENAFFRGGDVASMYTPNAQYCQMRCTFHPRCLLFSFLPASSINDMEKRFGCFLKDSVTGTLPKVHRTGAVSGHSLKQCGHQISACHRDIYKGVDMRGVNFNVSKVSSVEECQKRCTSNIRCQFFSYATQTFHKAEYRNNCLLKYSPGGTPTAIKVLSNVESGFSLKPCALSEIGCHMNIFQHLAFSDVDVARVLTPDAFVCRTICTYHPNCLFFTFYTNVWKIESQRNVCLLKTSESGTPSSSTPQENTISGYSLLTCKRTLPEPCHSKIYPGVDFGGEELNVTFVKGVNVCQETCTKMIRCQFFTYSLLPEDCKEEKCKCFLRLSMDGSPTRIAYGTQGSSGYSLRLCNTGDNSVCTTKTSTRIVGGTNSSWGEWPWQVSLQVKLTAQRHLCGGSLIGHQWVLTAAHCFDGLPLQDVWRIYSGILNLSDITKDTPFSQIKEIIIHQNYKVSEGNHDIALIKLQAPLNYTEFQKPICLPSKGDTSTIYTNCWVTGWGFSKEKGEIQNILQKVNIPLVTNEECQKRYQDYKITQRMVCAGYKEGGKDACKGDSGGPLVCKHNGMWRLVGITSWGEGCARREQPGVYTKVAEYMDWILEKTQSSDGKAQMQSPA"
        
        # Encoding vocabularies
        self.smiles_vocab = list("CNOPSH123456789-=()@[]")
        self.smiles_to_int = {ch: i + 1 for i, ch in enumerate(self.smiles_vocab)}
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i + 1 for i, aa in enumerate(self.aa_alphabet)}
        
        # Target properties for reward calculation
        self.target_properties = {
            "binding_affinity": 8.0,  # Strong binding (nM range) - higher pKd value
            "Solubility_AqSolDB": -4.5,  # Moderate solubility for oral dosing
            "HIA_Hou": 0.85,  # Good intestinal absorption
            "PAMPA_NCATS": 0.75,  # Good permeability
            "Bioavailability_Ma": 0.65,  # Reasonable oral bioavailability
            "ClinTox": 0.2,  # Low toxicity risk
        }
        
        print("🎯 Target properties configured for plasma kallikrein inhibition with toxicity consideration")
        print("✅ Drug Reward Evaluator initialized successfully\n")
    
    def encode_smiles(self, smi: str, max_len: int = 100) -> np.ndarray:
        """Encode SMILES string to numerical representation"""
        seq = [self.smiles_to_int.get(ch, 0) for ch in smi]
        seq += [0] * max(0, max_len - len(seq))
        return np.array(seq[:max_len])
    
    def encode_protein(self, prot: str, max_len: int = 1000) -> np.ndarray:
        """Encode protein sequence to numerical representation"""
        seq = [self.aa_to_int.get(ch, 0) for ch in prot]
        seq += [0] * max(0, max_len - len(seq))
        return np.array(seq[:max_len])
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate if SMILES string is chemically valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_molecular_properties(self, smiles: str) -> Dict:
        """Get basic molecular properties from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
                "hbd": Chem.rdMolDescriptors.CalcNumHBD(mol),
                "hba": Chem.rdMolDescriptors.CalcNumHBA(mol),
                "tpsa": Descriptors.TPSA(mol),
                "logp": Descriptors.MolLogP(mol)
            }
        except Exception as e:
            print(f"⚠️ Error calculating molecular properties: {e}")
            return {}
    
    def predict_binding_affinity(self, smiles: str) -> float:
        """Predict binding affinity for the target protein"""
        if not self.affinity_model:
            # Return placeholder value if model not available
            return 5.0
        
        try:
            X_smiles = np.expand_dims(self.encode_smiles(smiles), axis=0)
            X_protein = np.expand_dims(self.encode_protein(self.protein_seq), axis=0)
            pred_affinity = self.affinity_model.predict([X_smiles, X_protein], verbose=0)[0][0]
            return float(pred_affinity)
        except Exception as e:
            print(f"⚠️ Error predicting binding affinity: {e}")
            return 5.0
    
    def predict_admet_properties(self, smiles: str) -> Dict:
        """Predict ADMET properties including ClinTox using ADMET-AI"""
        if not self.admet_model:
            # Return placeholder values if model not available
            return {
                "Solubility_AqSolDB": -5.0,
                "HIA_Hou": 0.7,
                "PAMPA_NCATS": 0.6,
                "Bioavailability_Ma": 0.5,
                "ClinTox": 0.5
            }
        
        try:
            predictions = self.admet_model.predict(smiles=smiles)
            return predictions
        except Exception as e:
            print(f"⚠️ Error predicting ADMET properties: {e}")
            return {
                "Solubility_AqSolDB": -5.0,
                "HIA_Hou": 0.7,
                "PAMPA_NCATS": 0.6,
                "Bioavailability_Ma": 0.5,
                "ClinTox": 0.5
            }
    
    def calculate_reward(self, smiles: str, verbose: bool = True) -> Dict:
        """
        Calculate the overall reward score for a given SMILES string with toxicity penalties
        
        Args:
            smiles (str): SMILES string of the molecule
            verbose (bool): Whether to print detailed information
            
        Returns:
            Dict: Dictionary containing reward score and detailed breakdown
        """
        
        if verbose:
            print(f"🧪 Evaluating molecule: {smiles}")
            print("-" * 60)
        
        # Validate SMILES
        if not self.validate_smiles(smiles):
            if verbose:
                print("❌ Invalid SMILES string!")
            return {
                "smiles": smiles,
                "valid": False,
                "reward": 0.0,
                "error": "Invalid SMILES"
            }
        
        try:
            # Get molecular properties
            mol_props = self.get_molecular_properties(smiles)
            if verbose and mol_props:
                print(f"⚛️  Molecular Weight: {mol_props.get('molecular_weight', 'N/A'):.1f} Da")
                print(f"💍 Rings: {mol_props.get('num_rings', 'N/A')} total ({mol_props.get('num_aromatic_rings', 'N/A')} aromatic)")
                print(f"🔗 H-bonds: {mol_props.get('hbd', 'N/A')} donors, {mol_props.get('hba', 'N/A')} acceptors")
                print(f"🌊 LogP: {mol_props.get('logp', 'N/A'):.2f}")
                print()
            
            # Predict binding affinity
            if verbose:
                print("🎯 Predicting binding affinity...")
            binding_affinity = self.predict_binding_affinity(smiles)
            
            # Predict ADMET properties
            if verbose:
                print("💊 Predicting ADMET properties...")
            admet_props = self.predict_admet_properties(smiles)
            
            # Calculate similarity scores (closer to target = higher score)
            s_bind = np.exp(-abs(binding_affinity - self.target_properties["binding_affinity"]))
            s_sol = np.exp(-abs(admet_props.get("Solubility_AqSolDB", 0) - self.target_properties["Solubility_AqSolDB"]))
            s_abs = np.exp(-abs(admet_props.get("HIA_Hou", 0) - self.target_properties["HIA_Hou"]))
            s_pampa = np.exp(-abs(admet_props.get("PAMPA_NCATS", 0) - self.target_properties["PAMPA_NCATS"]))
            s_bio = np.exp(-abs(admet_props.get("Bioavailability_Ma", 0) - self.target_properties["Bioavailability_Ma"]))
            s_tox = np.exp(-abs(admet_props.get("ClinTox", 0) - self.target_properties["ClinTox"]))
            
            # Calculate toxicity penalty (strong penalty if ClinTox > 0.5)
            tox_penalty = 1.0
            clin_tox = admet_props.get("ClinTox")
            if clin_tox > 0.02:
                tox_penalty = 0.5  # Reduce reward by 50% if high toxicity risk
            
            # Calculate final reward (weighted average with toxicity penalty)
            final_reward = np.mean([s_bind, s_sol, s_abs, s_pampa, s_bio, s_tox]) * tox_penalty
            
            # Detailed results
            result = {
                "smiles": smiles,
                "valid": True,
                "reward": float(final_reward),
                "binding_affinity": binding_affinity,
                "admet_properties": admet_props,
                "molecular_properties": mol_props,
                "property_scores": {
                    "binding_score": float(s_bind),
                    "solubility_score": float(s_sol),
                    "absorption_score": float(s_abs),
                    "permeability_score": float(s_pampa),
                    "bioavailability_score": float(s_bio),
                    "toxicity_score": float(s_tox)
                },
                "toxicity_penalty": tox_penalty,
                "targets": self.target_properties
            }
            
            if verbose:
                print(f"\n📊 EVALUATION RESULTS:")
                print(f"{'='*60}")
                print(f"🎯 Binding Affinity: {binding_affinity:.3f} (target: {self.target_properties['binding_affinity']:.1f})")
                print(f"   Score: {s_bind:.4f}")
                print()
                print(f"💧 Solubility: {admet_props.get('Solubility_AqSolDB', 'N/A'):.3f} (target: {self.target_properties['Solubility_AqSolDB']:.1f})")
                print(f"   Score: {s_sol:.4f}")
                print()
                print(f"🔄 Absorption: {admet_props.get('HIA_Hou', 'N/A'):.3f} (target: {self.target_properties['HIA_Hou']:.2f})")
                print(f"   Score: {s_abs:.4f}")
                print()
                print(f"🚪 Permeability: {admet_props.get('PAMPA_NCATS', 'N/A'):.3f} (target: {self.target_properties['PAMPA_NCATS']:.2f})")
                print(f"   Score: {s_pampa:.4f}")
                print()
                print(f"💊 Bioavailability: {admet_props.get('Bioavailability_Ma', 'N/A'):.3f} (target: {self.target_properties['Bioavailability_Ma']:.2f})")
                print(f"   Score: {s_bio:.4f}")
                print()
                print(f"☠️ Toxicity (ClinTox): {clin_tox:.3f} (target: {self.target_properties['ClinTox']:.2f})")
                print(f"   Score: {s_tox:.4f}")
                print(f"   Toxicity Multiplier: {tox_penalty:.2f}")
                print()
                print(f"🏆 OVERALL REWARD: {final_reward:.4f}")
                print(f"{'='*60}")
                
                # Interpretation
                if final_reward >= 0.8:
                    print("🌟 EXCELLENT: This molecule shows very promising drug-like properties!")
                elif final_reward >= 0.6:
                    print("👍 GOOD: This molecule has good potential with some optimization needed.")
                elif final_reward >= 0.4:
                    print("⚠️  MODERATE: This molecule needs significant improvement.")
                else:
                    print("❌ POOR: This molecule has major issues as a drug candidate.")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during evaluation: {e}"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                "smiles": smiles,
                "valid": False,
                "reward": 0.0,
                "error": error_msg
            }

def main():
    """Main function for command-line usage"""
    evaluator = DrugRewardEvaluator()
    
    if len(sys.argv) > 1:
        # Command line usage
        smiles = sys.argv[1]
        result = evaluator.calculate_reward(smiles, verbose=True)
        
        # Print just the reward score for easy parsing
        print(f"\nFINAL REWARD: {result['reward']:.6f}")
        
    else:
        # Interactive usage
        print("🧬 Drug Reward Evaluator - Interactive Mode")
        print("Enter SMILES strings to evaluate (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            try:
                smiles = input("\nEnter SMILES: ").strip()
                
                if smiles.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not smiles:
                    print("Please enter a valid SMILES string.")
                    continue
                
                result = evaluator.calculate_reward(smiles, verbose=True)
                print(f"\n🎯 Quick Result: {result['reward']:.4f}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()