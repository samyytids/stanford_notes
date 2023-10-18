import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

plt.style.use("ggplot")

from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import KeyedVectors

model = api.load("glove-wiki-gigaword-100")
print(model.most_similar("usa"))
"""
Gensim is good for looking at some word vectors, but it isn't used much in deep learning
so is just being used for examples in this lecture
"""

result = model.most_similar(positive=["woman", "king"], negative=["man"])
print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative = [x1])
    return result[0][0]

"""
What this is basically doing is taking a vector and subtracting a vector from it and then seeing if we add a different vector to it
we end up at the right word
Idea being
if we subtract man from king and add woman to the result we should get queen.
king = man = regal 
regal + woman = queen
"""

print(analogy("man", "king", "woman"))
print(analogy("king", "man", "queen"))
print(model.doesnt_match(list("breakfast cereal dinner lunch".split())))
