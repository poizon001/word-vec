import os
import pickle
import numpy as np
from scipy import spatial
import sys


model_path = './models/'
loss_model = 'cross_entropy'
if len(sys.argv) > 1:
    if sys.argv[1] == 'nce':
      loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

input_file = open('word_analogy_dev.txt', 'r')
output_file = open('word_analogy_'+ loss_model+'.txt', 'w')
print('word_analogy_'+ loss_model+'.txt')
result = ""

def getCosineSimilarity(v1, v2):
	return 1 - spatial.distance.cosine(v1, v2)

def getSimilarityForPairs(pairsList):
	similarity = []
	for pair in pairsList:
		first_embedding_vector = embeddings[dictionary[pair[0]]]
		second_embedding_vector = embeddings[dictionary[pair[1]]]
		similarity.append(getCosineSimilarity(first_embedding_vector, second_embedding_vector))

	return similarity

def getAvgOfList(numList):
	return sum(numList)/len(numList)

line_result = ""

for lines in input_file:
	lines.strip()
	parts = lines.split("||")

	left_word_pairs = parts[0]
	left_word_pairs = left_word_pairs.strip().split(",")
	left_word_pairs = [(x.lstrip("\"").rstrip("\"").split(":")) for x in left_word_pairs]

	right_word_pairs = parts[1]
	right_word_pairs = right_word_pairs.strip().split(",")
	line_result += ' '.join(right_word_pairs)

	right_word_pairs = [(x.lstrip("\"").rstrip("\"").split(":")) for x in right_word_pairs]

	left_cosine_similarity = getSimilarityForPairs(left_word_pairs)
	left_cosine_similarity_avg = getAvgOfList(left_cosine_similarity)

	# print("left sim avg", left_cosine_similarity_avg)
	right_cosine_similarity = getSimilarityForPairs(right_word_pairs)
	# print("right_cosine_similarity", right_cosine_similarity)


	right_cosine_similarity = [x - left_cosine_similarity_avg for x in right_cosine_similarity]
	# print("right_cosine_similarity", right_cosine_similarity)

	min_dis_pair = right_word_pairs[right_cosine_similarity.index(min(right_cosine_similarity))]
	max_dis_pair = right_word_pairs[right_cosine_similarity.index(max(right_cosine_similarity))]

	line_result += " \"" + min_dis_pair[0]+":"+min_dis_pair[1] + "\" \"" + max_dis_pair[0]+":"+max_dis_pair[1] + "\""+"\n"

print(line_result)
output_file.write(line_result)
output_file.close()
