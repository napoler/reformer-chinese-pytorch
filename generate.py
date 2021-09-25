from transformers import BertTokenizer
from model.reformercn import LitGPT
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# print(dir(tokenizer))

# 生成内容示例

model=LitGPT.load_from_checkpoint("/mnt/data/dev/github/reformer-chinese-pytorch/litGPT/envso3r2/checkpoints/last.ckpt")

# print(model)

# when evaluating, just use the generate function, which will default to top_k sampling with temperature of 1.
initial = torch.tensor([[0]]).long() # assume 0 is start token
sample = model.model.generate(initial, 100, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
print(sample.shape) # (1, <=100) toke

# print(sample.tolist()[0]) #
ids=sample.tolist()[0]

out=tokenizer.batch_decode(ids)

print(out) #