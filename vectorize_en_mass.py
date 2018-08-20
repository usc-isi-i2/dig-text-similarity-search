import os
import json

from time import time
from typing import List

from doc_loader import load_docs
from query_index_handler import Collector


def save_sent_dicts(save_these: List[dict], save_in: os.path, fname: str) -> None:

    if not fname.endswith(".json"):
        fname += ".json"

    try:
        with open(os.path.join(save_in, fname), "w") as fsd:
            for doc in save_these:
                doc_dict = json.dumps(doc) + "\n"
                fsd.write(doc_dict)

    except FileExistsError:
        print("{} already exists".format(os.path.join(save_in, fname)))


def main(start=0, end=-1):
    timing = list()
    t_start = time()
    timing.append(t_start)

    cwd = os.getcwd()
    doc_dir = os.path.join(cwd, "data/sage_news_split_sentences")

    dummy_index_dir = os.path.join(cwd, "saved_indexes")
    dummy_index_name = "IVF4096Flat.index"

    model_path = "https://tfhub.dev/google/universal-sentence-encoder/2"

    Q = Collector(path_to_index_dir=dummy_index_dir,
                  base_index_name=dummy_index_name,
                  grand_index_name=dummy_index_name,
                  path_to_model=model_path)

    files = list()
    for (_, _, filenames) in os.walk(doc_dir):
        files.extend(filenames)
        break

    files.sort(reverse=True)
    print("Files to vectorize {}".format(files[start:end]))

    # Execute
    n_sent = list()
    for f in files[start:end]:
        t0 = time()
        timing.append(t0)
        print("\n  Vectorizing {}".format(f))

        docs = load_docs(os.path.join(doc_dir, f))

        partial_sent_dicts = Q.initialize_sent_dicts(docs)
        n_sent_one_day = len(partial_sent_dicts)
        print("  Number of sentences to be vectorized: {}".format(n_sent_one_day))
        n_sent.append(n_sent_one_day)
        del docs

        tensor_string_iter = Q.batch_to_dataset(partial_sent_dicts)
        new_vector_tensors = Q.make_vectors(tensor_string_iter, print_activity=True)

        sent_dicts_w_vects = Q.tensors_to_list(embedding_tensors=new_vector_tensors,
                                               partial_sent_dicts=partial_sent_dicts)

        save_name = "sent_dicts_from_" + f.split(".")[0] + ".json"
        save_path = os.path.join(cwd, "data/saved_sent-vector_dicts")
        save_sent_dicts(save_these=sent_dicts_w_vects,
                        save_in=save_path,
                        fname=save_name)

        print("  Vectorization of {} completed in {}s".format(f, time()-t0))
        print("  Sentence dicts with vectors saved in "
              "{}".format(os.path.join(save_path, save_name)))

    timing.append(time())
    print("\n\nProcess completed")
    n_sent_tot = sum(n_sent)
    print("Number of sentences vectorized: {}".format(n_sent_tot))
    avg_n_sent = n_sent_tot/len(n_sent)
    print("Avg number of sentences in one day's worth of news: {}".format(avg_n_sent))
    print("Total runtime: {}s".format(time()-t_start))
    avg_time = (timing[-1] - timing[1])/len(timing[1:])
    print("Avg time to vectorize one day's worth of news: {}s".format(avg_time))


if __name__ == "__main__":
    main()
