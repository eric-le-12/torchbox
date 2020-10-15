import torch

class collocation():

    def adaptive_collate(batch):
        # data = [item[0].permute(1,0) for item in batch]
        # abnormal = [item[1] for item in batch]
        # update with variable input size
        data = [item[0] for item in batch]
        abnormal = torch.stack([item[2].squeeze(0) for item in batch])
        # data = torch.nn.utils.rnn.pad_sequence(data,batch_first=False )
        # data = data.permute(1,2,0)
        target = [item[2] for item in batch]
        target = torch.LongTensor(target)
        return [data, abnormal, target]
    
    def multi_lead_collate(batch):
        # data = [item[0].permute(1,0) for item in batch]
        abnormal = [item[1] for item in batch]
        # update with variable input size
        data = [item[0] for item in batch]
        # abnormal = torch.stack([item[2].squeeze(0) for item in batch])
        # data = torch.nn.utils.rnn.pad_sequence(data,batch_first=False )
        # data = data.permute(1,2,0)
        target = [item[2] for item in batch]
        target = torch.LongTensor(target)
        return [data, abnormal, target]

