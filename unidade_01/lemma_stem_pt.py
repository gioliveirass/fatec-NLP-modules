import spacy
from nltk.stem.snowball import SnowballStemmer

frase = "O cão correu na chuva"

# ===========================================
# == Lemmatization ==========================
#============================================

nlp = spacy.load("pt_core_news_sm")
doc = nlp(frase)
print("Lemmatization: ", [token.lemma_ for token in doc])

# ===========================================
# == Stemming ===============================
#============================================

stemmer = SnowballStemmer("portuguese")
result = []

for palavra in frase.split(" "):
    result.append(stemmer.stem(palavra))

print("Stemming: ", result)