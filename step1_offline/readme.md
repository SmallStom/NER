1、依赖的pytorch版本为1.7.1
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

2、依赖的embedding文件在
pre_embed_path = '/home/nlu/chrism/pre_embeddings'

yangjie_rich_pretrain_unigram_path = pre_embed_path + '/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_bigram_path = pre_embed_path + '/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_word_path = pre_embed_path + '/ctb.50d.vec'
yangjie_rich_pretrain_char_and_word_path = pre_embed_path + '/yangjie_word_char_mix.txt'

ip_step1_path = '/home/nlu/chrism/ner_data/step1'

3、数据处理需要依赖fastnlp，版本为0.6.0
pip install fastNLP