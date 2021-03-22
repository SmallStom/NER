from utils import tokenize
with open('./test1.txt', 'r', encoding='utf-8') as fi, open('./test_for_eval.txt','w',encoding='utf-8') as fw:
    for line in fi:
        if line == '': continue
        splits = line.strip().split('|')
        query_tokens = ' '.join(tokenize(splits[0]))
        splits.insert(0, query_tokens)
        fw.write('{}\n'.format('\t'.join(splits)))
