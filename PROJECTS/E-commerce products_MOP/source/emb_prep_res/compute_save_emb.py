import os
import csv

def get_embeddings_pairs(samples, model, batch_size):
  sentences1 = [pair.texts[0] for pair in samples]
  sentences2 = [pair.texts[1] for pair in samples]

  embeddings1 = model.encode(sentences1, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
  embeddings2 = model.encode(sentences2, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

  return embeddings1, embeddings2


def get_offer_pair_records_from_sample(input_sample, embedding_right, embedding_left):
    ids = input_sample.guid.split("_")
    id_right = ids[0]
    id_left = ids[1]
    return [
        {
            "id": id_right,
            "embedding": ', '.join([str(x) for x in embedding_right.tolist()])
        },
        {
            "id": id_left,
            "embedding": ', '.join([str(x) for x in embedding_left.tolist()])
        }
    ]


def get_embedding_records(samples, embeddings1, embeddings2):
    embedding_records = []
    for i in range(0, len(samples)):
        pair_records = get_offer_pair_records_from_sample(samples[i], embeddings1[i], embeddings2[i])
        embedding_records.extend(pair_records)
    return embedding_records


def create_csv_file(file_path, field_names, data):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=field_names,
                            delimiter=";", extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
  