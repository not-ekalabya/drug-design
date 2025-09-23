# Computing Tanimoto similarity between two SMILES strings.
# We'll try to use RDKit (preferred). If RDKit is unavailable, we'll fall back to an approximate
# SMILES-based Jaccard (n-gram) similarity as a proxy and clearly label it.
smiles_user = "Cc1c(NC(=O)C2CCN(C)C2)cccc1C(=O)NCc1ccc(Cl)cc1"
smiles_daridorexant = "COCC1=NN(C=C1C(=O)NCC2=NC=CC(=C2F)OC)CC3=CC=C(C=C3)CN4C=CC=CC4=O"

result = {"smiles_user": smiles_user, "smiles_daridorexant": smiles_daridorexant}

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    # Create molecules
    mol1 = Chem.MolFromSmiles(smiles_user)
    mol2 = Chem.MolFromSmiles(smiles_daridorexant)
    if mol1 is None or mol2 is None:
        raise ValueError("RDKit failed to parse one of the SMILES strings.")
    # Compute Morgan fingerprints (ECFP4: radius=2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
    result.update({"method": "rdkit_morgan_ecfp4", "tanimoto": float(tanimoto)})
    print("RDKit detected. Computed Morgan (ECFP4) Tanimoto similarity:")
    print(f"Tanimoto = {tanimoto:.4f}")
except Exception as e:
    # Fallback: approximate similarity using SMILES n-gram Jaccard index
    import re, itertools, math
    def tokenize_smiles(smiles):
        # basic tokenization: atoms in brackets, two-letter atoms, single letters, digits, bonds, parentheses, stereochem, etc.
        pattern = r"\[.*?\]|Cl|Br|@@|@|=|#|-|\(|\)|[A-Za-z]|\d+|%\d{2}"
        tokens = re.findall(pattern, smiles)
        return tokens
    def ngram_set(tokens, n_max=3):
        s = set()
        for n in range(1, n_max+1):
            for i in range(len(tokens)-n+1):
                s.add(tuple(tokens[i:i+n]))
        return s
    t1 = tokenize_smiles(smiles_user)
    t2 = tokenize_smiles(smiles_daridorexant)
    set1 = ngram_set(t1, n_max=3)
    set2 = ngram_set(t2, n_max=3)
    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    # Also provide a hashed-bit fingerprint approximation (2048 bits) using Python's hash to mimic ECFP bits
    def hashed_bits_from_ngrams(ngrams, n_bits=2048):
        bits = [0]*n_bits
        for ng in ngrams:
            h = abs(hash(ng)) % n_bits
            bits[h] = 1
        return bits
    bits1 = hashed_bits_from_ngrams(set1)
    bits2 = hashed_bits_from_ngrams(set2)
    # compute Tanimoto on these bit lists
    intersect_bits = sum(1 for a,b in zip(bits1,bits2) if a==1 and b==1)
    union_bits = sum(1 for a,b in zip(bits1,bits2) if a==1 or b==1)
    tanimoto_bits = intersect_bits / union_bits if union_bits>0 else 0.0
    result.update({
        "method": "approx_smiles_jaccard_and_hashed_bits",
        "jaccard_ngrams": float(jaccard),
        "tanimoto_hashed_bits": float(tanimoto_bits),
        "tokens_user": t1,
        "tokens_daridorexant": t2,
        "common_ngrams_sample": list(itertools.islice(intersection, 10))
    })
    print("RDKit not available or failed. Returned approximate similarities (n-gram Jaccard and hashed-bit Tanimoto):")
    print(f"SMILES n-gram Jaccard = {jaccard:.4f}")
    print(f"Hashed-bit (2048) Tanimoto approximation = {tanimoto_bits:.4f}")

# Print result dict for transparency
print("\nResult summary:")
for k,v in result.items():
    print(f"{k}: {v}")

result

