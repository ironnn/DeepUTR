import torch
from colonmodel import jambaregression
from colontokenrizer import colonTokenizer

tokenizer = colonTokenizer(model_max_length=300)

# Example 5' UTR
seqs = [
    "GAGAGAACCCACCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGG",    # example sequence 1
    "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC",    # example sequence 2
]

tpm = [1, 1,] #example rna expression values

cell_type = [10, 10]   # cell 293T 

# Tokenize and pad/truncate
input_ids, attention_masks = [], []
for seq in seqs:
    toks = tokenizer(seq)
    ids = torch.tensor(toks["input_ids"], dtype=torch.long)
    mask = torch.tensor(toks["attention_mask"], dtype=torch.bool)
    input_ids.append(ids)
    attention_masks.append(mask)

device = "cuda:0"
input_ids = torch.stack(input_ids).to(device)          
attention_mask = torch.stack(attention_masks).to(device)
tpm_tensor = torch.tensor(tpm, dtype=torch.float32, device=device)            
cell_type_tensor = torch.tensor(cell_type, dtype=torch.long, device=device)   

model = jambaregression(
    d_model=16,
    num_mamba_blocks=3,
    d_intermediate=32,
    vocab_size=5,
    max_seq_len=300,
)
state_dict = torch.load("pure_weights_colon/colon_model.pt", map_location="cuda")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# inference
with torch.no_grad():
    output = model(input_ids, attention_mask=attention_mask, tpm=tpm_tensor, cell_type=cell_type_tensor)
print("Inference outputs:", output)