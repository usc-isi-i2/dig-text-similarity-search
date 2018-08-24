from digtextsimilaritysearch.vectorizer.sentence_vectorizer \
    import SentenceVectorizer
from digtextsimilaritysearch.process_documents.document_processor \
    import DocumentProcessor

import os
import sys
import json

from time import time


def get_docs(file_loc, size_of_minibatch):
    doc_minibatch = list()
    num_minibatch = 0

    with open(file_loc, 'r') as fp:
        for i_count, _ in enumerate(fp):
            pass
    print('There are {} docs in {}'.format(i_count, file_loc))

    with open(file_loc, 'r') as fp:
        for line in fp:
            doc_minibatch.append(json.loads(line))

            if len(doc_minibatch) >= size_of_minibatch:
                print('\n Yielding minibatch {} with {} '
                      'docs'.format(num_minibatch, len(doc_minibatch)))
                yield doc_minibatch
                doc_minibatch = list()
                num_minibatch += 1

    print('\n Yielding last minibatch with {} docs'.format(len(doc_minibatch)))
    yield doc_minibatch


t_start = time()
sentence_vectorizer = SentenceVectorizer()
dp = DocumentProcessor(None, sentence_vectorizer, None)
t_init = time()
print('System initialized in {:0.3f}s'.format(t_init-t_start))

doc_col_loc = sys.argv[1]
save_dir = sys.argv[2]
minibatch_size = int(sys.argv[3]) if sys.argv[3] else 2500
doc_getter = get_docs(file_loc=doc_col_loc, size_of_minibatch=minibatch_size)

g_count = 0
runtimes = list()
doc_col_name = doc_col_loc.split('/')[-1].split('.')[0]
for j, minibatch in enumerate(doc_getter):

    # Check if vectors already exist on disk
    save_name = 'vectorized_' + doc_col_name + '_' + str(j) + '.npz'
    save_loc = os.path.join(save_dir, save_name)

    if not os.path.exists(save_loc):
        t_0 = time()

        # Preprocess and Deallocate
        sentences = dp.preprocess_documents(minibatch)
        minibatch = None
        t_1 = time()
        print('  Preprocessed {} sentences in {:0.3f}s'.format(len(sentences), t_1-t_0))

        # Vectorize
        text = [s[1] for s in sentences]
        embeddings = dp.vectorizer.make_vectors(text)
        t_2 = time()
        print('  Created {} embeddings in {:0.3f}s'.format(len(embeddings), t_2-t_1))

        # Save Vectors and Text
        dp.vectorizer.save_vectors(embeddings, sentences, save_loc)
        print('  Saved {} in {:0.3f}s'.format(save_name, time()-t_2))

        runtimes.append(time()-t_0)
        m, s = divmod(runtimes[-1], 60)
        print('  Preprocessed {} docs in {}m:{:0.3f}s'.format(minibatch_size, int(m), s))
        g_count += 1

        # Occasionally Reset TF Graph
        if g_count % 5 == 0:
            print('\nRefreshing TF Session...')
            dp.vectorizer.close_session()
            dp.vectorizer.start_session()
            t_2 = time()
            print('Resuming vectorization... \n')

    else:
        minibatch = None
        print(' File {} already exists'.format(save_name))

m, s = divmod(time()-t_init, 60)
print('Processing completed in {}m:{:0.3f}s'.format(int(m), s))

avg_runtime = sum(runtimes[:-1]) / len(runtimes[:-1])
m, s = divmod(avg_runtime, 60)
print('Average runtime per minibatch: {}m:{:0.3f}s'.format(int(m), s))
