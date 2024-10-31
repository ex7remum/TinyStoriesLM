import os
import json
from tqdm import tqdm
import torch
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

if __name__ == "__main__":
    n_tokenizer = 500000

    data_path = './TinyStories'

    with open("data.txt", "w") as f:
        for filename in os.listdir(data_path):
            curfile = open(os.path.join(data_path, filename))
            data = json.load(curfile)
            curfile.close()
            os.remove(os.path.join(data_path, filename))
            for i in tqdm(range(len(data))):
                f.write(data[i]['story'].replace("\n", " "))
                f.write('\n')
    
    with open("data.txt") as file:
        texts = file.readlines()[:n_tokenizer]
        with open("tokenizer_data.txt", 'w') as tkn:
            for s in texts:
                tkn.write(s)

    vocab_size = 3000
    model_type = 'bpe'

    SentencePieceTrainer.train(
                     input="tokenizer_data.txt", vocab_size=vocab_size,
                     model_type=model_type, model_prefix='bpe',
                     normalization_rule_name='nmt_nfkc_cf',
                     pad_id=0, unk_id=1, bos_id=2, eos_id=3
                 )
    
    VAL_RATIO = 0.1
    os.remove('tokenizer_data.txt')

    sp_model = SentencePieceProcessor(model_file="bpe.model")

    with open('data.txt') as file:
        texts = file.readlines()
        
    os.remove('data.txt')

    test_size = int(len(texts) * VAL_RATIO)
    train_size = len(texts) - test_size
    train_texts, val_texts = torch.utils.data.random_split(texts, [train_size, test_size],
                                                       torch.Generator().manual_seed(54))
        
    indices_train = [sp_model.encode(text) for text in tqdm(train_texts)]

    res = {}
    for i, token in tqdm(enumerate(indices_train)):
        res[i] = {}
        res[i]['tokens'] = token
    with open("train.json", "w") as outfile:
        json.dump(res, outfile)
    indices_train = []

    indices_val = [sp_model.encode(text) for text in tqdm(val_texts)]

    res = {}
    for i, token in tqdm(enumerate(indices_val)):
        res[i] = {}
        res[i]['tokens'] = token
    with open("val.json", "w") as outfile:
        json.dump(res, outfile)
