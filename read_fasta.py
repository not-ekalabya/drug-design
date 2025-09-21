def read_fasta(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    seq_lines = [line.strip() for line in lines if not line.startswith(">") and line.strip()]
    return "".join(seq_lines)

protein_seq = read_fasta("sequences/P02918.fa")

print(protein_seq)