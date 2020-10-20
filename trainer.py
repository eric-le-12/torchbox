import model
import torch
import torch.nn as nn
import numpy as np
import time

def train_one_epoch(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    train_metrics,
    val_metrics,
    num_of_class
):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    # pos_weight = torch.FloatTensor([2,3,2,3,1]).to(device)
    # pos_weight.to(device)
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    print("LR : {} \n".format(optimizer.param_groups[0]['lr']))
    for data, abnormal, target in train_loader:
        model.train()
        # move-tensors-to-GPU
        # print(data)
        # print((data[0].shape))
        # print(data[0].un)
        # data = data.to(device)
        # print(abnormal,target)
        # remember to unlock for 2 branch
        # abnormal=abnormal.to(device)
        bs = len(data)
        # target=torch.Tensor(target)
        target = target.to(device).long()
        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        Y_pred = torch.empty((0,num_of_class)).to(device)
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        for i in range(0,bs):
            input_ecg = data[i].unsqueeze(0).to(device)
            abnormal_3 = abnormal[i].unsqueeze(0).to(device)
            # print(input_ecg.shape)
            # print("shape:",abnormal.shape)
            # print(abnormal[[i]].unsqueeze(0).shape)
            # print(abnormal[[i]].unsqueeze(0).unsqueeze(0))
            # unblock the below cmt to back to the 2 branch mode
            # preds = model(input_ecg,abnormal[[i]].unsqueeze(0))
            preds = model(input_ecg,abnormal_3)
            # preds = model(input_ecg)
            Y_pred = torch.cat((Y_pred,preds))
        # output = model(data)
        # get the prediction label and target label
        # output = model(data)
        # preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        # output_sigmoid = torch.sigmoid(output)
        # with torch.no_grad():            
            # preds = (output_sigmoid.cpu().numpy()>0.5).astype(float)
        labels = target.cpu().numpy()
        # calculate-the-batch-loss
        loss = criterion(Y_pred,target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # if(loss.item()>100):
        #     print(torch.softmax(Y_pred,dim=-1),target)
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        
        # update-training-loss
        train_loss += loss.item() * bs
        # calculate training metrics
        preds = torch.softmax(Y_pred,dim=-1).cpu().detach().numpy()
        preds = np.argmax(preds,axis=-1)
        train_metrics.step(labels, preds)

    # validate-the-model
    model.eval()
    all_labels = []
    all_preds = []
    for data, abnormal, target in test_loader:
        # data = data.to(device)
        # remember to unlock for 2 branch
        # abnormal = abnormal.to(device)
        bs = len(data)
        target = target.to(device).long()
        with torch.no_grad():
            Y_pred = torch.empty((0,num_of_class)).to(device)
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            for i in range(0,bs):
                input_ecg = data[i].unsqueeze(0).to(device)
                abnormal_3 = abnormal[i].unsqueeze(0).to(device)
                # print(input_ecg.shape)
                # print("shape:",abnormal.shape)
                # print(abnormal[[i]].unsqueeze(0).shape)
                # preds = model(input_ecg,abnormal[[i]].unsqueeze(0))
                preds = model(input_ecg,abnormal_3)
                # unblock below comment for 2 branch
                # preds = model(input_ecg)
                Y_pred = torch.cat((Y_pred,preds))
            # output_sigmoid = torch.sigmoid(output)
            # preds = (output_sigmoid.cpu().numpy() >0.5).astype(float)
        labels = target.cpu().numpy()
        # preds = torch.argmax(output, axis=1).tolist()
            
        # labels = target.tolist()
       

        loss = criterion(Y_pred, target)
        
        # update-average-validation-loss
        valid_loss += loss.item() * bs
        preds = torch.softmax(Y_pred,dim=-1).cpu().detach().numpy()
        preds = np.argmax(preds,axis=-1)
        all_labels.extend(np.array(labels,dtype="int32"))
        all_preds.extend(preds)
    val_metrics.step(all_labels, all_preds)
    # print(all_preds)
    # print(all_labels)
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(test_loader.dataset)

    return (
        train_loss,
        valid_loss,
        train_metrics.epoch(),
        val_metrics.last_step_metrics(),
    )
