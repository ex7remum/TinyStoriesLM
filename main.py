import json
from dataset import TextDataset
from model import LanguageModel
from train import run_train
from torch.utils.data import DataLoader
import torch 
import wandb

if __name__ == "__main__":
    wandb.login(key='bbe60953ed99662c4459f461386ecd58a2f2ee3a')
    
    run = wandb.init(
        project="TinyStoriesLM"
    )
    
    train_set = TextDataset(data_file='train.json', 
                            tokenizer_path='bpe.model',
                            train=True, 
                            max_length=256)
    valid_set = TextDataset(data_file='val.json', 
                            tokenizer_path='bpe.model',
                            train=False, 
                            max_length=256)
            
    model = LanguageModel(train_set, 4, 256, 4, train_set.vocab_size, 512, 256, 0.1)

    train_loader = DataLoader(train_set, batch_size=1280, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=1280, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = 100
    steps_per_epoch = 1000

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=num_epochs*steps_per_epoch, 
                                                    epochs=num_epochs, steps_per_epoch=steps_per_epoch,
                                                    pct_start=0.01, final_div_factor=20)
    run_train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, steps_per_epoch)
