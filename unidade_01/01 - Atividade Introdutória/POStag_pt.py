import nltk
import spacy
import numpy as np
import pandas as pd
import copy as cp
import joblib

sentence = "A minha idéia, depois de tantas cabriolas, constituíra-se idéia fixa. Deus te livre, leitor, de uma idéia fixa; antes um argueiro, antes uma trave no olho. Vê o Cavour; foi a idéia fixa da unidade italiana que o matou. Verdade é que Bismarck não morreu; mas cumpre advertir que a natureza é uma grande caprichosa e a história uma eterna loureira. Por exemplo, Suetônio deu-nos um Cláudio, que era um simplório, — ou “uma abóbora” como lhe chamou Sêneca, e um Tito, que mereceu ser as delícias de Roma. Veio modernamente um professor e achou meio de demonstrar que dos dois césares, o delicioso, o verdadeiro delicioso, foi o “abóbora” de Sêneca. E tu, madama Lucrécia, flor dos Bórgias, se um poeta te pintou como a Messalina católica, apareceu um Gregorovius incrédulo que te apagou muito essa qualidade, e, se não vieste a lírio, também não ficaste pântano. Eu deixo-me estar entre o poeta e o sábio. Viva pois a história, a volúvel história que dá para tudo; e, tornando à idéia fixa, direi que é ela a que faz os varões fortes e os doidos; a idéia móbil, vaga ou furta-cor é a que faz os Cláudios, — fórmula Suetônio. Era fixa a minha idéia, fixa como... Não me ocorre nada que seja assaz fixo nesse mundo: talvez a lua, talvez as pirâmides do Egito, talvez a finada dieta germânica. Veja o leitor a comparação que melhor lhe quadrar, veja-a e não esteja daí a torcer-me o nariz, só porque ainda não chegamos à parte narrativa destas memórias. Lá iremos. Creio que prefere a anedota à reflexão, como os outros leitores, seus confrades, e acho que faz muito bem. Pois lá iremos. Todavia, importa dizer que este livro é escrito com pachorra, com a pachorra de um homem já desafrontado da brevidade do século, obra supinamente filosófica, de uma filosofia desigual, agora austera, logo brincalhona, coisa que não edifica nem destrói, não inflama nem regala, e é todavia mais do que passatempo e menos do que apostolado. Vamos lá; retifique o seu nariz, e tornemos ao emplasto. Deixemos a história com os seus caprichos de dama elegante. Nenhum de nós pelejou a batalha de Salamina, nenhum escreveu a confissão de Augsburgo; pela minha parte, se alguma vez me lembro de Cromwell, é só pela idéia de que Sua Alteza, com a mesma mão que trancara o parlamento, teria imposto aos ingleses o emplasto Brás Cubas. Não se riam dessa vitória comum da farmácia e do puritanismo. Quem não sabe que ao pé de cada bandeira grande, pública, ostensiva, há muitas vezes várias outras bandeiras modestamente particulares, que se hasteiam e flutuam à sombra daquela, e não poucas vezes lhe sobrevivem? Mal comparando, é como a arraia-miúda, que se acolhia à sombra do castelo feudal; caiu este e a arraia ficou. Verdade é que se fez graúda e castelã... Não, a comparação não presta."
words = sentence.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print(bag_of_words)

# Annotated words:
# nltk.download('popular')
# Using natural language toolkit
print("Usando o natural language toolkit:")
# Use lang with ISO 639 code of the language
#pos_tags = nltk.pos_tag(sentence.split(), lang="pt") # not implemented.

# Reference to get the trained model:
# https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
trained_data_folder = 'data/'
portuguese_tagger = joblib.load(trained_data_folder+'POS_tagger_brill.pkl')
pos_tags = portuguese_tagger.tag(nltk.word_tokenize(sentence))
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("Usando o Spacy para se obter as partes do discurso.")
## https://spacy.io/models/pt
model_spacy = spacy.load('pt_core_news_sm')
pos_tags_2 = [ (word, word.pos_) for word in model_spacy(sentence)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)