import nltk
import spacy
import numpy as np
import pandas as pd
import copy as cp

sentence = """My idea, after so many pranks, had become a fixed idea. God forbid you, reader, from a fixed idea; rather a mote, rather a beam in the eye. See Cavour; it was the fixed idea of Italian unity that killed him. It is true that Bismarck did not die; but it must be warned that nature is a great capricious and history an eternal laurel tree. For example, Suetonius gave us a Claudius, who was a simpleton, - or "a pumpkin" as Seneca called him, and a Titus, who deserved to be the delights of Rome. A modern professor came along and found a way to demonstrate that of the two Caesars, the delicious one, the truly delicious one, was Seneca's pumpkin. And you, Madame Lucrezia, flower of the Borgias, if a poet painted you like the Catholic Messalina, an incredulous Gregorovius appeared who erased that quality a lot from you, and if you didn't turn out to be a lily, you didn't become a swamp either. I let myself be between the poet and the sage. So long live history, the voluble history that goes for everything; and, returning to the fixed idea, I will say that it is what makes strong men and madmen; the mobile, vague or iridescent idea is what makes the Claudius, — formula Suetonius. My idea was fixed, fixed like... I can't think of anything that is quite fixed in this world: maybe the moon, maybe the pyramids of Egypt, maybe the late Germanic diet. Let the reader see the comparison that best suits him, see it and don't turn my nose up there, just because we haven't yet reached the narrative part of these memoirs. There we will go. I think he prefers anecdotes to reflections, like the other readers, his confreres, and I think he does it very well. Well, there we go. However, it is important to say that this book is written with composure, with the composure of a man who is no longer confronted with the brevity of the century, a supremely philosophical work, of an unequal philosophy, now austere, then playful, something that neither builds nor destroys, neither inflames nor it delights, and yet it is more than a pastime and less than an apostolate. Let's go; straighten your nose, and let's go back to the poultice. Let us leave history to its elegant lady's whims. None of us fought the battle of Salamis, none wrote the Augsburg confession; for my part, if I ever remember Cromwell, it is only for the idea that his Highness, with the same hand that had locked up parliament, would have imposed the Brás Cubas poultice on the English. Do not laugh at this common victory of pharmacy and Puritanism. Who doesn't know that at the foot of each large, public, ostentatious flag, there are often several other modestly private flags, which hoist and float in the shadow of that one, and not infrequently survive it? Barely comparing, it's like the small fry, which took shelter in the shadow of the feudal castle; this one fell and the ray stayed. It's true that she grew up to be tall and noble... No, the comparison sucks."""
words = sentence.split()
bag_of_words = cp.deepcopy(words)
np.random.shuffle(bag_of_words)
# Bag of words:
print(bag_of_words)

# Annotated words:
# nltk.download('popular')
# Using natural language toolkit
print("Using natural language toolkit:")
pos_tags = nltk.pos_tag(sentence.split())
print(pos_tags)
pos_tags_df = pd.DataFrame(pos_tags).T
print(pos_tags_df)

print("Using spacy to get parts of speech tags.")
nlp = spacy.load('en_core_web_sm')
pos_tags_2 = [ (word, word.tag_,  word.pos_) for word in nlp(sentence)]
pos_tags_2_df = pd.DataFrame(pos_tags_2).T
print(pos_tags_2)
print(pos_tags_2_df)


