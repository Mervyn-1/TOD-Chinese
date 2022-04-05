from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 不需要分词
src = '我是好人。'
tgt = '我是好人你。'


score = corpus_bleu([src], tgt)

print(score)





