from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(dir(tokenizer))