from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input = "glove.6B.100d.txt"
word2vec_output = "glove.6B.100d.word2vec"
glove2word2vec(glove_input, word2vec_output)

model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)
# print(model['istanbul'])
# Kümeliyor verileri  data sonra istediğimize göre getiriyor.
# print(model.most_similar("istanbul"))

print(model.most_similar(positive=["woman", "king"], negative=["man"], topn=1))
# Kelimeler arası ilişki kuruyor ve bize isteğimizi getiriyor.

print(model.most_similar(positive=["teach", "doctor"], negative=["treat"], topn=1))
# doktordan tedavi etmeyi çıkarınca meslek olduğu anlaşılıyo.Burada da öğretmek kelimesi var ve model bunun meslek karşılğını getiriyor.

# Kelime vektörlerini kullanarak eğitilmiş bir model üzerinde çalışıldı.
