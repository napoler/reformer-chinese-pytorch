from transformers import BertTokenizer
import torch
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split,TensorDataset
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    maxLen=512
    data=[]
    with open("data/data.txt",'r') as f:
        for line in tqdm(f):
            # print(line)
            data.append(str(line))
            # break
        textTensor=tokenizer(data, padding="max_length",max_length=maxLen,  truncation=True,return_tensors="pt")
        
        
        fl=len(textTensor["input_ids"])
        trainl=int(fl*0.7)
        testl=int(fl*0.15)
        vall=fl-trainl-testl
        train_set,val_set,test_set=random_split(textTensor["input_ids"], [trainl,vall,testl])

        torch.save(train_set,"data/train.pkt")
        torch.save(val_set,"data/val.pkt")
        torch.save(test_set,"data/test.pkt")
        
    pass

if __name__ == '__main__':
    main()