import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_train(model, optimizer, scheduler, train_loader, val_loader, num_epochs, num_steps):
    
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i, (indices, lengths) in tqdm(enumerate(train_loader), desc=f'Training {epoch}/{num_epochs}'):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                indices = indices[:, :lengths.max()].to(device)
                logits = model(indices[:, :-1])        
                loss = criterion(logits.transpose(1, 2), indices[:, 1:])
                wandb.log({"train_loss": loss.item()})
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            if i + 1 >= num_steps:
                break
        

        if epoch % 10 == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for indices, lengths in tqdm(val_loader, desc=f'Validating {epoch}/{num_epochs}'):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        indices = indices[:, :lengths.max()].to(device)
                        logits = model(indices[:, :-1]) 
                        loss = criterion(logits.transpose(1, 2), indices[:, 1:])
                        val_loss += loss.item() * indices.shape[0]

                val_loss /= len(val_loader.dataset)
                wandb.log({"val_loss": val_loss})

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                gen_text = model.inference()
            wandb.log({"Generated text": wandb.Html(gen_text)})
        
        if epoch % 50 == 0:
            name = 'checkpoint_epoch_{}'.format(epoch)
            torch.save(model.state_dict(), name)
