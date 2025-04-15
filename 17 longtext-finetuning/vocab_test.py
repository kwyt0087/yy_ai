#字典自定义操作
from transformers import BertTokenizer

token = BertTokenizer.from_pretrained(r"bert-base-chinese")


#编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=["阳光洒在大地上"],
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    max_length=20,
    return_tensors=None
)

print(token.decode(out["input_ids"][0]))

vocab = token.vocab
print(len(vocab))
#print(vocab)
print("阳" in vocab)
print("光" in vocab)
print("阳光" in vocab)
#
#自定义内容添加到vocab中
token.add_tokens(new_tokens=["阳光"])
vocab = token.get_vocab()
print(len(vocab))
print("阳光" in vocab)

out = token.batch_encode_plus(
    batch_text_or_text_pairs=["阳光洒在大地上"],
    add_special_tokens=True,
    truncation=True,
    padding="max_length",
    max_length=20,
    return_tensors=None
)

print(token.decode(out["input_ids"][0]))
print(vocab)