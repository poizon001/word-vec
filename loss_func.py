import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A = log(exp(true_w)^T inputs)


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B = log(\sum{exp({true_w}^T inputs)})

    ==========================================================================
    """
    
    A = tf.log(tf.exp(tf.reduce_sum(tf.multiply(true_w, inputs), axis=1)))

    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(inputs, true_w, transpose_b=True)), axis=1))

    return tf.subtract(B,A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):

    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    input_shape = inputs.get_shape()
    embedding_size = input_shape[1]
    batch_size = input_shape[0]
    
    #Target word vector (batch_sizexembedding_size)
    target_words = tf.nn.embedding_lookup(weights, labels)
    target_words = tf.reshape(target_words, [-1,embedding_size])

    #Context word vector (batch_sizexembedding_size)
    context_words = inputs
    # print("Context word vector")
    # print(context_words)

    #Bias vector specific to target word vector
    bias_for_target_words = tf.nn.embedding_lookup(biases, labels)
    # print("Bias vector specific to target word vector")
    # print(bias_for_target_words)

    #Convert to tensors as unigram_prob is scalar
    target_words_unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    #Unigram probabilities for target word vectors
    target_words_unigram_prob = tf.nn.embedding_lookup(target_words_unigram_prob, labels)
    # print("Unigram probabilities for target word vectors")
    # print(target_words_unigram_prob)

    
    #Calculating dot product of context_words and target_words
    context_target_dot_product = tf.reshape(tf.reduce_sum(tf.multiply(context_words, target_words), axis=1), [batch_size, 1])
    context_target_dot_product = tf.add(context_target_dot_product, bias_for_target_words)
    # print("Calculating dot product of context_target_dot_product")
    # print(context_target_dot_product)

    
    sample_size = len(sample)
    #Convert to tensor as it is np array
    sample = tf.convert_to_tensor(sample, dtype=tf.int32)
    
    #Calculating unigram probability for k samples
    #Creating smoothing array for target_words_unigram_prob so when taking log it does not go to NAN
    target_words_unigram_prob_add = tf.fill([batch_size, 1], 0.0000000001)
    target_words_unigram_prob = tf.log(tf.scalar_mul(sample_size, target_words_unigram_prob))
    target_words_unigram_prob = tf.subtract(context_target_dot_product, target_words_unigram_prob)
    target_words_unigram_prob = tf.log(tf.add(tf.sigmoid(target_words_unigram_prob), target_words_unigram_prob_add))
    # print("target_words_unigram_prob")
    # print(target_words_unigram_prob)

    #Creating negative target words by using sample (sample_sizexembedding_size)
    neg_target_words = tf.nn.embedding_lookup(weights, sample)
    # print("neg_target_words")
    # print(neg_target_words)

    neg_unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    neg_unigram_prob = tf.reshape(tf.nn.embedding_lookup(neg_unigram_prob, sample), [sample_size, 1])
    # print("Neg unigram prob")
    # print(neg_unigram_prob)
    #Negative unigram prob has size (sample_sizex1), Need to change it to dim (batch_sizexsample_size)
    neg_target_word_unigram_probs = tf.transpose(neg_unigram_prob)
    neg_target_word_unigram_probs = tf.tile(neg_target_word_unigram_probs, [batch_size, 1])

    #Negative word vector bias
    #(sample_sizex1)
    bias_for_neg_target_words = tf.reshape(tf.nn.embedding_lookup(biases, sample), [sample_size, 1])
    #(1xsample_size)
    bias_for_neg_target_words = tf.transpose(bias_for_neg_target_words)
    #(batch_sizexsample_size)
    bias_for_neg_target_words = tf.tile(bias_for_neg_target_words, [batch_size, 1])
    # print("Negative bias")
    # print(bias_for_neg_target_words)

    #Calculating dot product of context_words and neg_target_words (batch_sizexsample_size)
    # print("Calculating dot product of context_words and neg_target_words")
    context_neg_target_dot_product = tf.matmul(context_words, neg_target_words, transpose_b=True)
    # print("Neg dot product")
    # print(context_neg_target_dot_product)
    
    #Addng bias to the dot product (batch_sizexsample_size)
    context_neg_target_dot_product = tf.add(context_neg_target_dot_product, bias_for_neg_target_words)
    # print("Neg dot product with bias")
    # print(context_neg_target_dot_product)


    #scalar multipication with sample size
    neg_target_word_unigram_probs = tf.log(tf.scalar_mul(sample_size, neg_target_word_unigram_probs))

    # print("neg_target_word_unigram_probs")
    # print(neg_target_word_unigram_probs)

    neg_target_word_unigram_probs = tf.subtract(context_neg_target_dot_product, neg_target_word_unigram_probs)
    # print("subtract context_neg_target_dot_product and neg_target_word_unigram_probs")
    # print(neg_target_word_unigram_probs)


    total_prob = tf.fill([batch_size, sample_size], 1.0)
    # print("Total prob")
    # print(total_prob)
    neg_target_word_unigram_probs = tf.subtract(total_prob, tf.sigmoid(neg_target_word_unigram_probs))
    # print ("neg_target_word_unigram_probs after log and sigmoid")
    # print(neg_target_word_unigram_probs)

    #Creating matrix for adding to neg_unigram_prob so it doesn't got to NAN
    neg_target_word_unigram_probs_add = tf.fill([batch_size, sample_size], 0.0000000001)
    neg_target_word_unigram_probs = tf.reduce_sum(tf.log(tf.add(neg_target_word_unigram_probs, neg_target_word_unigram_probs_add)), axis=1)
    neg_target_word_unigram_probs = tf.reshape(neg_target_word_unigram_probs, [batch_size, 1])
    # print("neg_target_word_unigram_probs")
    # print(neg_target_word_unigram_probs)

    batch_probs = tf.scalar_mul(-1, tf.add(target_words_unigram_prob, neg_target_word_unigram_probs))
    
    return batch_probs

