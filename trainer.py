import model
import torch
import torch.nn as nn
import numpy as np
def train_one_epoch(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    # pos_weight = torch.FloatTensor([2,3,2,3,1]).to(device)
    # pos_weight.to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        # target=torch.Tensor(target)
        target = target.to(device).float()
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        
        # get the prediction label and target label
        # output = model(data)
        # preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        output_sigmoid = torch.sigmoid(output)
        with torch.no_grad():            
            preds = (output_sigmoid.cpu().numpy()>0.5).astype(float)
        labels = target.cpu().numpy()
        # calculate-the-batch-loss
      
        loss = criterion(output,target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        # calculate training metrics
        train_metrics.step(labels, preds)

    # validate-the-model
    model.eval()
    all_labels = []
    all_preds = []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device).float()
        with torch.no_grad():
            output = model(data)
            output_sigmoid = torch.sigmoid(output)
            preds = (output_sigmoid.cpu().numpy() >0.5).astype(float)
        labels = target.cpu().numpy()
        # preds = torch.argmax(output, axis=1).tolist()
            
        # labels = target.tolist()
       
        # all_labels.extend(np.array(labels,dtype="float32"))
        # all_preds.extend(preds)
        loss = criterion(output, target)
        
        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)
        val_metrics.step(labels, preds)
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    return (
        train_loss,
        valid_loss,
        train_metrics.epoch(),
        val_metrics.epoch(),
    )
