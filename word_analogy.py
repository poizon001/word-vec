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

def getCosineSimilarity(word1, word2):
	return 1 - spatial.distance.cosine(word1, word2)


line_result = ""

for lines in input_file:
	lines.strip()
	parts = lines.split("||")

	example_word_pairs = parts[0]
	example_word_pairs = example_word_pairs.strip().split(",")
	example_word_pairs = [(x.lstrip("\"").rstrip("\"").split(":")) for x in example_word_pairs]

	choices_word_pairs = parts[1]
	choices_word_pairs = choices_word_pairs.strip().split(",")
	line_result += ' '.join(choices_word_pairs)

	choices_word_pairs = [(x.lstrip("\"").rstrip("\"").split(":")) for x in choices_word_pairs]

	print(choices_word_pairs)
	print(example_word_pairs)

	example_rel = np.zeros(embeddings.shape[1])
	for pair in example_word_pairs:
		word1_embed = embeddings[dictionary[pair[0]]]
		word2_embed = embeddings[dictionary[pair[1]]]
		
		rel = word1_embed - word2_embed
		example_rel = example_rel + rel
	
	avg_rel = example_rel/len(example_rel)
	print(avg_rel)


	choices_similarity=[]
	for pair in choices_word_pairs:
		word1_embed = embeddings[dictionary[pair[0]]]
		word2_embed = embeddings[dictionary[pair[1]]]

		rel = word1_embed - word2_embed
		choices_similarity.append(getCosineSimilarity(avg_rel, rel))

	print(choices_similarity)

	max_dis_pair = choices_word_pairs[choices_similarity.index(max(choices_similarity))]
	min_dis_pair = choices_word_pairs[choices_similarity.index(min(choices_similarity))]

	line_result += " \"" + min_dis_pair[0]+":"+min_dis_pair[1] + "\" \"" + max_dis_pair[0]+":"+max_dis_pair[1] + "\""+"\n"

print(line_result)
output_file.write(line_result)
output_file.close()


