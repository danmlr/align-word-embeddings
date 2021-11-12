import numpy as np

def eval_rot(l1, l2, l1_l2_dict, W):
    """ As input we use a list of translation pairs (same format as seed) and a rotation matrix W 
        Output : percentage of success P@10 
    """
    success = 0
    for l1_word,l2_word in l1_l2_dict: 
        x = l1.word_vec(l1_word)
        translates = l2.similar_by_vector(W@x)
        for transword,_ in translates:
            if transword==l2_word:
                success+=1 
                break 
    return 100*success/len(l1_l2_dict)

def eval_perm(l1, l2, l1_l2_dict, P):
    """P[i, j] donne la probabilité que le mot i de l1 soit traduit par le mot j de l2"""

    nb_tested_words = 0
    success = 0

    for l1_word, l2_word in l1_l2_dict:

        i = l1.vocab[l1_word].index
        
        if i >= P.shape[0]:
            break
        
        nb_tested_words += 1

        indices = np.argsort(P[i])[-10:]
        translates = [l2.index2word[j] for j in indices]

        for transword in translates:
            if transword == l2_word:
                success += 1
                break

    return 100 * success / nb_tested_words
