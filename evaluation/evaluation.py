import ssdeep
from collections import defaultdict
import progressbar
import numpy
import os

def load_dict(path):
    file = open(path, 'r')
    lines = file.readlines()

    processed_lines = []
    for line in lines:
        processed_lines.append(tuple(line.strip().split(' ')))

    fairseq_dict = dict(processed_lines)
    return fairseq_dict


def load_generated(filename):

    generated = {}
    lines = open(filename, 'r').readlines()

    for line in lines:
        split = line.strip().split('\t')
        id = split[0]
        text = split[1:]
        id_num = id[2:]

        if id.startswith('S-'):  # Prompt
            generated[id_num] = [' '.join(text), None, None]
        elif id.startswith('T-'):  # Target
            generated[id_num][1] = ' '.join(text)
        elif id.startswith('H-'):  # Generated
            generated[id_num][2] = ' '.join(text[1:])
        else:
            continue

    return generated


def fuzzy_hash_compare(source, generated):

    # print(source)
    # print(generated)
    hash1 = ssdeep.hash(source)
    hash2 = ssdeep.hash(generated)

    return ssdeep.compare(hash1, hash2)


# if __name__ == '__main__':
#     gen = load_generated('../data/generated/ner_pro_no_fusion.txt')
#
#     # for key, [a, b, c] in gen.items():
#     #     print(id)
#     #     print('s',a)
#     #     print('t', b)
#     #     print(c + '\n')
#
#     fuzzy = {}
#     p = progressbar.ProgressBar(term_width=80)
#
#     for id_1, [s_1, t_1, g_1] in p(gen.items()):
#         for id_2, [s_2, t_2, g_2] in gen.items():
#             if id_1 != id_2:
#                 fuzzy[(id_1, id_2)] = fuzzy_hash_compare(t_1, g_2)
#
#     for k, v in fuzzy.items():
#         if v != 0:
#             print(k)


def perplexity(text, unigram_prob):
    tokens = text.split(' ')

    if len(tokens) < 2:
        return 0

    probs = [unigram_prob[word] for word in tokens]
    probs = [prob for prob in probs if prob > 0.001]
    perp = numpy.prod([(1 / prob) for prob in probs])

    N = len(probs)

    return pow(perp, 1/ float(N)) if N > 1 else 0


def lang_model(text):
    tokens = text.split(' ')
    unigram_prob = defaultdict(float)  # Smoothing

    for t in tokens:
        if unigram_prob[t] < 1:
            unigram_prob[t] = 1.0
        else:
            unigram_prob[t] += 1.0

    total_weight = float(sum(unigram_prob.values()))

    # Normalize
    for word in unigram_prob:
        unigram_prob[word] = unigram_prob[word] / total_weight

    return unigram_prob


if __name__ == '__main__':
    path = '../data/generated/'
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)

        generated_text = load_generated(filename)

        all_text = ' '.join([gen for key, [source, target, gen] in generated_text.items()])

        lang = lang_model(all_text)

        perplexities = [perplexity(gen, lang) for [source, target, gen] in generated_text.values()]
        perplexities = [p for p in perplexities if p != float('Inf') and p != 0]

        (med, mean, max_p, min_p) = numpy.median(perplexities), numpy.mean(perplexities), max(perplexities), min(perplexities)
        print(filename, 'Perplexity: Med {}, Mean {}, Max {}, Min {}'.format(med, mean, max_p, min_p))