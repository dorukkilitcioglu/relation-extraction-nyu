import numpy as np

vocab = set()

"""
def get_word_embeddings():
    embeddings_index = {}
    all_words = []
    with open('../data/glove.6B.300d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            all_words.append(word)
    return all_words, embeddings_index
"""
_all_words = None
embeddings = None
word_to_id = None
id_to_word = None
num_words = None


def get_all_words():
    all_words = set()

    def get_words_in_file(filename):
        with open(filename) as f:
            for line in f:
                word_infos = line.split(word_delim)
                info_splits = [info.split(info_delim) for info in word_infos[1:]]
                words = [info[0] for info in info_splits]
                all_words.update(words)

    suf = '7' if relation_detail == 'basic' else ('19' if relation_detail == 'subtype' else '37')
    get_words_in_file('training%s.txt' % suf)
    get_words_in_file('test%s.txt' % suf)
    return list(all_words)


def load_word_embeddings():
    global _all_words, embeddings, word_to_id, id_to_word, num_words
    embeddings = []
    _all_words = get_all_words()
    embed_words = []
    with open('../data/glove.6B.300d.txt') as f:
        for line in f:
            values = line.split()
            embeddings.append(values[1:])
            embed_words.append(values[0])
    combined = [(embed_words[i], embeddings[i]) for i in range(len(embed_words)) if embed_words[i] in _all_words]
    _all_words = [t[0] for t in combined]
    embeddings = np.asarray([t[1] for t in combined], dtype='float32')
    word_to_id = {w: i for i, w in enumerate(_all_words)}
    id_to_word = {v: k for k, v in word_to_id.items()}
    num_words = len(_all_words) + 1


def get_word_index(word):
    try:
        return word_to_id[word]
    except TypeError:
        load_word_embeddings()
        return get_word_index(word)
    except KeyError:
        return num_words - 1

cluster_len = 17


def get_brown_clusters():
    cluster_index = {}
    with open('../data/brownClusters10-2014.txt') as f:
        for line in f:
            values = line.split()
            cluster = values[0]
            cluster += (cluster_len - len(cluster)) * '0'
            cluster_index[values[1]] = np.array([float(i) for i in cluster])
    return cluster_index

_cluster_index = None


def load_brown_clusters():
    global _cluster_index
    _cluster_index = get_brown_clusters()


basic_labels = ["OTHER", "GEN-AFF", "ORG-AFF", "PART-WHOLE", "PER-SOC", "PHYS", "ART"]

subtype_labels = ["OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity", "ORG-AFF:Employment", "PART-WHOLE:Subsidiary", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business", "GEN-AFF:Org-Location", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder", "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near", "PER-SOC:Family", "PART-WHOLE:Artifact", "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Sports-Affiliation"]

subtype_order_labels = ["OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity", "ORG-AFF:Employment-1", "ORG-AFF:Employment", "GEN-AFF:Citizen-Resident-Religion-Ethnicity-1", "PART-WHOLE:Subsidiary-1", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business", "GEN-AFF:Org-Location", "PHYS:Located-1", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder-1", "ORG-AFF:Membership-1", "PART-WHOLE:Geographical-1", "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near", "ART:User-Owner-Inventor-Manufacturer-1", "PART-WHOLE:Subsidiary", "PHYS:Near-1", "PER-SOC:Family", "GEN-AFF:Org-Location-1", "PER-SOC:Family-1", "PART-WHOLE:Artifact-1", "PER-SOC:Business-1", "PART-WHOLE:Artifact", "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum-1", "ORG-AFF:Founder", "ORG-AFF:Student-Alum", "ORG-AFF:Ownership-1", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Investor-Shareholder-1", "ORG-AFF:Sports-Affiliation", "ORG-AFF:Sports-Affiliation-1", "PER-SOC:Lasting-Personal-1"]

_pos_tags = ['NNS', 'PRP', 'VB', 'null', 'NNP', 'JJ', 'SYM', ',', 'VBP', ':', 'IN', "''", 'RBS', 'LS', 'DT', 'NNPS', 'NN', 'RBR', 'UH', 'RP', 'RB', 'EX', '(', 'JJR', 'CC', 'FW', 'PRP$', 'MD', 'VBD', 'VBN', '.', 'TO', 'PDT', 'POS', 'WP', 'CD', ')', '$', '``', 'WRB', 'JJS', 'WP$', 'VBG', 'VBZ', 'WDT']

pos_tag_to_id = {pt: i for i, pt in enumerate(_pos_tags)}
id_to_pos_tag = {v: k for k, v in pos_tag_to_id.items()}
num_pos_tags = len(_pos_tags) + 1


def get_pos_tag_index(pos_tag):
    try:
        return pos_tag_to_id[pos_tag]
    except KeyError:
        return len(_pos_tags)

labels = None
label_to_id = None
id_to_label = None
relation_detail = None


# basic -> only the 6 base types + OTHER
# subtype -> include subtypes
# subtype_with_order -> include ordering for subtypes
def set_relation_detail(rel_detail = 'basic'):
    global labels, label_to_id, id_to_label, relation_detail
    if rel_detail == 'basic':
        labels = basic_labels
    elif rel_detail == 'subtype':
        labels = subtype_labels
    elif rel_detail == 'subtype_with_order':
        labels = subtype_order_labels
    else:
        labels = None
    relation_detail = rel_detail
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {v: k for k, v in label_to_id.items()}


def get_cluster(word):
    try:
        return _cluster_index[word]
    except TypeError:
        load_brown_clusters()
        return get_cluster(word)
    except KeyError:
        return np.zeros((cluster_len), np.float32)

word_delim = '|||'
info_delim = '}}}'


def get_info(line):
    words = line.split(word_delim)
    label = label_to_id[words[0]]
    info_splits = [info.split(info_delim) for info in words[1:]]
    words = [get_word_index(info[0]) for info in info_splits]
    brown_clusters = [get_cluster(info[0]) for info in info_splits]
    pos_tags = [get_pos_tag_index(info[1]) for info in info_splits]
    positions = [[float(info[2]), float(info[3])] for info in info_splits]
    return words, brown_clusters, pos_tags, positions, label


def load_examples(filename):
    with open(filename, "r") as fp:
        words = []
        brown_clusters = []
        pos_tags = []
        positions = []
        labels = []
        for line in fp:
            line_words, line_brown_clusters, line_pos_tags, line_positions, label = get_info(line)
            words.append(line_words)
            brown_clusters.append(line_brown_clusters)
            pos_tags.append(line_pos_tags)
            positions.append(line_positions)
            labels.append(label)
        x_words = np.stack(words, axis = 0)
        x_brown_clusters = np.stack(brown_clusters, axis = 0)
        x_pos_tags = np.stack(pos_tags, axis = 0)
        x_positions = np.stack(positions, axis = 0)
        y = np.vstack(labels)
        return x_words, x_brown_clusters, x_pos_tags, x_positions, y


def load_training():
    suf = '7' if relation_detail == 'basic' else ('19' if relation_detail == 'subtype' else '37')
    filename = 'training%s.txt' % suf
    return load_examples(filename)


def load_test():
    suf = '7' if relation_detail == 'basic' else ('19' if relation_detail == 'subtype' else '37')
    filename = 'test%s.txt' % suf
    return load_examples(filename)
