from admet_ai import ADMETModel
from keras.models import load_model
import tensorflow as tf
import numpy as np
import deepchem as dc

# ---------------------------
# 1. Load ADMET prediction model
# ---------------------------
admet_model = ADMETModel()

# ---------------------------
# 2. Load pretrained binding affinity model
# ---------------------------
def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.linalg.band_part(tf.cast(f, tf.float32), -1, 0)
    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    return tf.where(tf.equal(g, 0), 0.0, g / f)

affinity_model = load_model(
    "DL4H/pretrained_models/combined_davis.h5",
    custom_objects={"cindex_score": cindex_score}
)

# ---------------------------
# 3. Protein sequence preparation
# ---------------------------
def read_fasta(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    seq_lines = [line.strip() for line in lines if not line.startswith(">") and line.strip()]
    return "".join(seq_lines)

protein_seq = "MKFVKYFLILAVCCILLGAGSIYGLYRYIEPQLPDVATLKDVRLQIPMQIYSADGELIAQYGEKRRIPVTLDQIPPEMVKAFIATEDSRFYEHHGVDPVGIFRAASVALFSGHASQGASTITQQLARNFFLSPERTLMRKIKEVFLAIRIEQLLTKDEILELYLNKIYLGYRAYGVGAAAQVYFGKTVDQLTLNEMAVIAGLPKAPSTFNPLYSMDRAVARRNVVLSRMLDEGYITQQQFDQTRTEAINANYHAPEIAFSAPYLSEMVRQEMYNRYGESAYEDGYRIYTTITRKVQQAAQQAVRNNVLDYDMRHGYRGPANVLWKVGESAWDNNKITDTLKALPTYGPLLPAAVTSANPQQATAMLADGSTVALSMEGVRWARPYRSDTQQGPTPRKVTDVLQTGQQIWVRQVGDAWWLAQVPEVNSALVSINPQNGAVMALVGGFDFNQSKFNRATQALRQVGSNIKPFLYTAAMDKGLTLASMLNDVPISRWDASAGSDWQPKNSPPQYAGPIRLRQGLGQSKNVVMVRAMRAMGVDYAAEYLQRFGFPAQNIVHTESLALGSASFTPMQVARGYAVMANGGFLVDPWFISKIENDQGGVIFEAKPKVACPECDIPVIYGDTQKSNVLENNDVEDVAISREQQNVSVPMPQLEQANQALVAKTGAQEYAPHVINTPLAFLIKSALNTNIFGEPGWQGTGWRAGRDLQRRDIGGKTGTTNSSKDAWFSGYGPGVVTSVWIGFDDHRRNLGHTTASGAIKDQISGYEGGAKSAQPAWDAYMKAVLEGVPEQPLTPPPGIVTVNIDRSTGQLANGGNSREEYFIEGTQPTQQAVHEVGTTIIDNGEAQELF"

# ---------------------------
# 4. Encoding utilities
# ---------------------------
smiles_vocab = list("CNOPSH123456789-=()@[]")  # example vocab; replace with training vocab
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
# 5. Reward function (ADMET + DeepChem comparison)
# ---------------------------
TARGET_PROPERTIES = {
    "binding_affinity": 5.0,     # strong binding
    "Solubility_AqSolDB": -3.7,  # moderately soluble
    "HIA_Hou": 0.97,             # high intestinal absorption
    "PAMPA_NCATS": 0.62,         # target permeability
    "Bioavailability_Ma": 0.74   # target oral bioavailability
}

# DeepChem featurizer + a baseline model
featurizer = dc.feat.MolGraphConvFeaturizer()
deepchem_model = dc.models.GraphConvModel(n_tasks=1, mode="regression")

def calculate_reward(smiles_list):
    results = []

    for smi in smiles_list:
        # --- ADMET predictions ---
        admet_preds = admet_model.predict(smiles=smi)

        # --- Binding affinity (your pretrained keras model) ---
        X_smiles = np.expand_dims(encode_smiles(smi), axis=0)
        X_protein = np.expand_dims(encode_protein(protein_seq), axis=0)
        pred_affinity = affinity_model.predict([X_smiles, X_protein])[0][0]

        # --- DeepChem predictions (example: solubility-like task) ---
        try:
            mol = dc.utils.mol_from_smiles(smi)
            if mol is not None:
                feat = featurizer.featurize([mol])
                deepchem_pred = deepchem_model.predict_on_batch(feat)[0][0]
            else:
                deepchem_pred = None
        except Exception:
            deepchem_pred = None

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
            "deepchem_pred": None if deepchem_pred is None else float(deepchem_pred),
            "diagnostics": {
                "s_bind": s_bind,
                "s_sol": s_sol,
                "s_abs": s_abs,
                "s_pampa": s_pampa,
                "s_bio": s_bio
            }
        })

    results = sorted(results, key=lambda x: x["reward"], reverse=True)
    return results

# ---------------------------
# 6. Example usage
# ---------------------------
smiles_candidates = [
    "CC1(C)S[C@@H]2[C@@H](NC(=O)Cc3ccccc3)C(=O)N2C1=O",  # Penicillin G
    "CC(=O)OC1=CC=CC=C1C(=O)O",                         # Aspirin
]

ranked_results = calculate_reward(smiles_candidates)

for res in ranked_results:
    print(f"SMILES: {res['smiles']}")
    print(f"Reward: {res['reward']:.3f}")
    print(f"ADMET AI: {res['admet_ai']}")
    print(f"Binding Affinity: {res['binding_affinity']:.3f}")
    print(f"DeepChem Prediction: {res['deepchem_pred']}")
    print("Diagnostics:", res["diagnostics"])
    print("------------------------------------------------------------")
