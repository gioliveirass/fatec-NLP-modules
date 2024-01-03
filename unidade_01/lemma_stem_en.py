import spacy
from nltk.stem.snowball import SnowballStemmer

frase = "I listen to music"

# ===========================================
# == Lemmatization ==========================
#============================================

nlp = spacy.load("en_core_web_sm")
doc = nlp(frase)
print("Lemmatization: ", [token.lemma_ for token in doc])

# ===========================================
# == Stemming ===============================
#============================================

stemmer = SnowballStemmer("english")
result = []

for palavra in frase.split(" "):
    result.append(stemmer.stem(palavra))

print("Stemming: ", result)