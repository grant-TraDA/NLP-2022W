from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import os
import numpy as np
import csv


def compute_similarity_scores(embeddings1, embeddings2):
  cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
  manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
  euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
  dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

  return cosine_scores, manhattan_distances, euclidean_distances, dot_products

def get_pair_guid(samples):
  return [pair.guid for pair in samples]

def create_csv_similarity_file(file_path, cosine, manhattan, euclidean, dot, guids):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['pair_id', 'cosine_score', 'manhattan_distance',
                     'euclidean_distance', 'dot_product'])
    for i in range(len(guids)):
      line = [guids[i], cosine[i], manhattan[i], euclidean[i], dot[i]]
      writer.writerow(line)

def compute_and_save_similarity_scores(output_path, samples, embeddings1, embedding2):
  cosine, manhattan, euclidean, dot = compute_similarity_scores(embeddings1, embedding2)
  guids = get_pair_guid(samples)
  create_csv_similarity_file(output_path, cosine, manhattan, euclidean, dot, guids)