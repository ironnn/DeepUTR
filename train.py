from colontokenrizer import colonTokenizer
from colondataset import FastaDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from colonmodule import colonmodule
from pytorch_lightning.strategies import DDPStrategy

train_path = "data/colonmodeldata0830_train.fasta"
val_path = "data/colonmodeldata0830_val.fasta"
# test_path = "data/colonmodeldata0830_test.fasta"


tokenizer = colonTokenizer(model_max_length=300)

train_dataset = FastaDataset.from_file(train_path, tokenizer=tokenizer)
val_dataset = FastaDataset.from_file(val_path, tokenizer=tokenizer)
# test_dataset = FastaDataset.from_file(test_path, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)
# test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

model = colonmodule(
    d_model=16,            
    num_mamba_blocks=3,      
    d_intermediate=32,     
    vocab_size=tokenizer.vocab_size,  
    max_seq_len=300,
    lr=1e-3,
)


trainer = pl.Trainer(
    max_epochs=20,
    accelerator='gpu', 
    devices=4,
    strategy=DDPStrategy(find_unused_parameters=True),
    precision=32,
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

