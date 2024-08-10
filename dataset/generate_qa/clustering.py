from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import random
from collections import defaultdict


def cluster_sequence(model, num_clusters, sent, indices):
    """
    Cluster paragraph sequence using embedding model and kmeans
    input:
        sent (list of str): list of sentences in article
        indices (2-d list with dimension of ((# paragraphs) * 2): [starting index, ending index (+1)] for each paragraph

    output:
        selected item: return randomly selected paragraphs for each clusters

    NOTE: for articles with less than 20 paragraphs, num_clusters will be set as half of paragraphs.
    """
    closest_items = []
    num_paragraphs = len(indices)
    if num_paragraphs < 2:  # don't generate qas at all
        return closest_items

    paragraphs = [
        " ".join([sent[i] for i in range(*indices[paragraph_idx])])
        for paragraph_idx in range(num_paragraphs)
    ]

    embeddings = model.encode(paragraphs)

    # num_clusters will be less than half of entities
    if num_paragraphs < 20:
        n_clusters = num_paragraphs // 2
    else:
        n_clusters = num_clusters

    clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustering_model.fit(embeddings)

    labels = clustering_model.labels_
    clustered_items = defaultdict(list)

    for i, item in zip(labels, indices):
        clustered_items[i].append(item)

    selected_items = [random.choice(items) for _, items in clustered_items.items()]
    return selected_items


def adjacent_cluster_sequence(model, num_clusters, sent, indices):
    """
    Cluster SENTENCES using embedding model (SENTBERT) and HAC with custom linkage
    input:
        sent (list of str): list of sentences in article
        indices (list of int): list of consecutive indices

    output:
        out(list of list of int): return adjacent clusters
    """
    input_sent = [sent[i] for i in indices]
    embeddings = model.encode(input_sent)
    cosine_scores = util.cos_sim(embeddings[:-1], embeddings[1:]).diagonal()

    # Find the indices of the minimum n elements in a list
    sorted_indices = sorted(range(len(cosine_scores)), key=lambda x: cosine_scores[x])
    min_indices = sorted(sorted_indices[: num_clusters - 1])

    # Cluster sequence according to the splitter
    out = []
    for i in range(num_clusters):
        if i == 0:
            out.append(indices[: min_indices[i] + 1])
        elif i == num_clusters - 1:
            out.append(indices[min_indices[i - 1] + 1 :])
        else:
            out.append(indices[min_indices[i - 1] + 1 : min_indices[i] + 1])
    out = [x for x in out if x != []]
    return out