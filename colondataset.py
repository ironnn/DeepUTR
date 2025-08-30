from Bio import SeqIO
import numpy as np
import torch

class FastaDataset(object):
    def __init__(self, sequence_labels, sequence_strs, tokenizer=None):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, fasta_file, tokenizer=None):
        sequence_labels, sequence_strs = [], []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_labels.append(record.description)  
            sequence_strs.append(str(record.seq))
        return cls(sequence_labels, sequence_strs, tokenizer=tokenizer)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        header = self.sequence_labels[idx]
        sequence = self.sequence_strs[idx]

        parts = header.split("|")
        if parts[0].startswith(">"):
            parts[0] = parts[0][1:]

        d = {
            'gene_id': parts[0],
            'cell_line_merged': parts[1],
            'cell_line_index': int(parts[2]),
            'rna': float(parts[3]),
            'rpf': float(parts[4]),
            'sequence': sequence,
        }

        if self.tokenizer:
            tokenized = self.tokenizer(sequence)
            d['input_ids'] = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            d['attention_mask'] = torch.tensor(tokenized["attention_mask"], dtype=torch.bool)
        else:
            d['input_ids'] = None
            d['attention_mask'] = None

        d['cell_line_index'] = torch.tensor(d['cell_line_index'], dtype=torch.long)
        d['rna'] = torch.tensor(d['rna'], dtype=torch.float32)
        d['rpf'] = torch.tensor(d['rpf'], dtype=torch.float32)

        return d
