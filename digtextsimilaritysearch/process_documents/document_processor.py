import pickle
import numpy as np
from time import time, sleep
from collections import OrderedDict

_SENTENCE_ID = 'sentence_id'
_SENTENCE_TEXT = 'sentence_text'


class DocumentProcessor(object):
    def __init__(self, indexer, vectorizer, storage_adapter, index_builder=None,
                 table_name='dig', vector_save_path='/tmp/saved_vectors.npz', save_vectors=False,
                 index_save_path='/tmp/faiss_index.index'):

        self.indexer = indexer
        self.vectorizer = vectorizer
        self.storage_adapter = storage_adapter
        self.index_builder = index_builder
        self.table_name = table_name
        if self.storage_adapter:
            self._configure()

        self.vector_save_path = vector_save_path
        self.save_vectors = save_vectors

        self.index_save_path = index_save_path

    def _configure(self):
        self.storage_adapter.create_table(self.table_name)

    @staticmethod
    def preprocess_documents(cdr_docs):
        """
        Preprocess cdr docs into a list of (<sid>, <sentence>) tuples
        :param cdr_docs: documents with split sentences
        :return: a list of (<sid>, <sentence>) tuples
        """
        if not cdr_docs:
            return []

        if not isinstance(cdr_docs, list):
            cdr_docs = [cdr_docs]

        sentences = []

        for cdr_doc in cdr_docs:
            doc_id = cdr_doc['doc_id']
            split_sentences = cdr_doc.get('split_sentences', [])
            ssl = len(split_sentences)
            if ssl > 0:
                for i in range(0, ssl):
                    sentences.append(('{}_{}'.format(doc_id, i), split_sentences[i]))
        return sentences

    def create_vectors(self, sentence_tuples):
        """
        This function does the following steps:
          1. call vectorize_sentences to get the corresponding vector ids
          2. adds the sids and vector ids to the hbase table
        :param sentence_tuples: list of tuples: (<sid>, <sentence>)
        :return: just does its job
        """
        sentences = [s[1] for s in sentence_tuples]
        vectors = self.vectorizer.make_vectors(sentences)
        if self.save_vectors:
            self.vectorizer.save_vectors(embeddings=vectors, sentences=sentence_tuples,
                                         file_path=self.vector_save_path)
        return vectors

    def query_text(self, str_query, k=3, fetch_sentences=False,
                   rerank_by_doc=False, start=0, end=-1, debug=False):
        """
        :param str_query: The actual text for querying.
        :param k: Number of results required
        :param fetch_sentences: Bool to fetch actual sentence text from the storage adapter
            and send them back in the response json
        :param rerank_by_doc: Bool to return results ranked by docs or individual sentences
        :param start: TODO: merge shards by date range
        :param end: TODO: merge shards by date range
        :param debug: Bool to toggle prints

        :return: List of top k results, where each result corresponds to a sentence that matched the query
            - If fetch_docs == True, we call elasticsearch and retrieve the text of the documents that matched
            - If False, we only return the ids and the scores of the sentence and document

            By Sentence: [ {'score': diff_score <float32>,          # Lower is better
                            'sentence_id': str(<int64>)
                            } ]
                  Where: doc_id <int64>, sent_id <int64> = divmod(int('sentence_id'), 10000)

            By Document: [ {'score': doc_relevance <float32>,       # Higher is better
                            'doc_id': str(doc_id),
                            'sentence_id': [ str(faiss_id) ],
                            'sentence_scores': [ diff_score <float32> ]
                            } ]
        """

        if not isinstance(str_query, list):
            str_query = [str_query]
        t_0 = time()
        query_vector = self.vectorizer.make_vectors(str_query)
        t_vector = time() - t_0
        print('  TF vectorization time: {:0.6f}s'.format(t_vector))

        if isinstance(query_vector[0], np.ndarray):
            query_vector = query_vector[0]
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype=np.float32)

        if rerank_by_doc:
            k_search = max(500, k * 100)
        else:
            k_search = max(50, k * 10)

        # TODO: change start/end params to be dates (not list indices)
        t_1 = time()
        if self.indexer.dynamic:
            scores, faiss_ids = self.indexer.search(query_vector, k_search,
                                                    start=start, end=end)
        else:
            scores, faiss_ids = self.indexer.search(query_vector, k_search)
        t_search = time() - t_1
        print('  Faiss search time: {:0.6f}s'.format(t_search))

        scores, faiss_ids = self.consistent(scores, faiss_ids)
        if rerank_by_doc:
            return self.rerank(scores=scores[0], faiss_ids=faiss_ids[0], k=k, norm_sents=5)
        else:
            return self.rerank_by_sents(scores=scores[0], faiss_ids=faiss_ids[0], k=k)

    @staticmethod
    def rerank(scores, faiss_ids, k, norm_sents=1):
        docs = dict()
        for score, faiss_id in zip(scores, faiss_ids):
            doc_id, sent_id = divmod(faiss_id, 10000)
            doc_id = str(doc_id)
            if doc_id not in docs:
                docs[doc_id] = dict()
                docs[doc_id]['sentence_id'] = list()
                docs[doc_id]['sentence_scores'] = list()
                docs[doc_id]['unique_scores'] = set()
                docs[doc_id]['sent_ids'] = list()
            docs[doc_id]['sentence_id'].append(str(sent_id))
            docs[doc_id]['sentence_scores'].append(score)
            docs[doc_id]['sent_ids'].append(faiss_id)
            if score not in docs[doc_id]['unique_scores']:
                docs[doc_id]['unique_scores'].add(score)

        similar_docs = list()
        unique_doc_scores = set()

        for doc_id, ids_and_scores in docs.items():
            if len(similar_docs) >= k:
                break

            doc_hits = pickle.dumps(ids_and_scores['unique_scores'])
            if doc_hits in unique_doc_scores:
                pass
            else:
                unique_doc_scores.add(doc_hits)
                out = dict()
                out['doc_id'] = doc_id
                out['sentence_id'] = ids_and_scores['sentence_id']

                # New score
                top_scores = sorted(list(ids_and_scores['unique_scores']))
                norm_scores = [min(1/sc, 1/0.125)*(0.25 + 0.75*(1/i)) for sc, i in
                               zip(top_scores[:norm_sents], range(1, norm_sents + 1))]
                new_score = sum(norm_scores)

                # Assign score
                out['score'] = new_score
                similar_docs.append(out)

        similar_docs.sort(key=lambda ds: ds['score'], reverse=True)
        return similar_docs

    def rerank_by_sents(self, scores, faiss_ids, k):
        similar_docs = self.rerank(scores, faiss_ids, k, norm_sents=2)

        old_payload = list()
        for doc in similar_docs:
            if len(old_payload) >= k:
                break
            out = dict()
            out['score'] = 1/doc['score']
            out['sentence_id'] = doc['doc_id'] + '{:04d}'.format(doc['sentence_id'][0])
            old_payload.append(out)

        return old_payload

    def index_documents(self, cdr_docs=None, load_vectors=False, column_family='dig',
                        save_faiss_index=False, batch_mode=False, batch_size=1000):

        vectors = None
        sentence_tuples = None
        if load_vectors:
            vectors, sentence_tuples = self.vectorizer.load_vectors(self.vector_save_path)
            print('Total sentences loaded from file: {}'.format(len(sentence_tuples)))
        else:
            if cdr_docs:
                print('Total cdr docs to be processed: {}'.format(len(cdr_docs)))
                sentence_tuples = self.preprocess_documents(cdr_docs)
                vectors = self.create_vectors(sentence_tuples)

        record_batches = list()
        if vectors.any() and len(sentence_tuples):
            faiss_ids = self.indexer.index_embeddings(vectors)
            del vectors  # Free up memory

            print('Adding {} faiss_ids to database sequentially...'.format(len(sentence_tuples)))
            # ASSUMPTION: returned vector ids are in the same order as the initial sentence order
            # TODO: iron out repeated code
            for s, f in zip(sentence_tuples, faiss_ids):
                data = dict()
                data[_SENTENCE_ID] = s[0]
                data[_SENTENCE_TEXT] = s[1]

                data['{}:{}'.format(column_family, _SENTENCE_ID)] = s[0]
                data['{}:{}'.format(column_family, _SENTENCE_TEXT)] = s[1]
                if batch_mode:
                    record_batches.append((str(f), data))
                else:
                    self.storage_adapter.insert_record(str(f), data, self.table_name)

            # TODO: depreciate or keep batch_mode
            if batch_mode:
                self.insert_bulk_records(record_batches, self.table_name, batch_size)
            if save_faiss_index:
                print('Saving faiss index...')
                self.indexer.save_index(self.index_save_path)
        else:
            print('Either provide cdr docs or file path to load vectors')

    def insert_bulk_records(self, records, table_name, batch_size):
        num_records = len(records)
        if num_records <= batch_size:
            self.storage_adapter.insert_records_batch(records, table_name)
        else:
            count = 0
            while count <= num_records:
                self.storage_adapter.insert_records_batch(records[count:count + batch_size], table_name)
                count += batch_size
                sleep(0.1)

    def add_to_db(self, sentence_tuples, faiss_ids, column_family='dig', batch_mode=False):
        # ASSUMPTION: vector ids are in the same order as the initial sentence order
        records = list()
        for s, f in zip(sentence_tuples, faiss_ids):
            data = dict()
            data[_SENTENCE_ID] = s[0]
            data[_SENTENCE_TEXT] = s[1]
            data['{}:{}'.format(column_family, _SENTENCE_ID)] = s[0]
            data['{}:{}'.format(column_family, _SENTENCE_TEXT)] = s[1]
            if not batch_mode:
                self.storage_adapter.insert_record(str(f), data, self.table_name)
            else:
                data = self.storage_adapter.prepare_record(str(f), data)
                records.append(data)
        if batch_mode:
            self.storage_adapter.insert_records_batch(self.table_name, records)

    def index_docs_on_disk(self, path_to_npz, path_to_invlist=None):
        if not path_to_invlist:
            path_to_invlist = 'invl_' + path_to_npz.replace('.npz', '.index')

        if self.index_builder:
            vectors, sentences, sent_ids = self.vectorizer.load_with_ids(path_to_npz)
            assert sent_ids.shape[0] == vectors.shape[0], \
                'Found {} sent_ids and {} vectors'.format(sent_ids.shape[0], vectors.shape[0])
            self.index_builder.generate_invlist(path_to_invlist, sent_ids, vectors)
        else:
            raise Exception('Cannot index on disk without an index_builder')

    def index_embeddings_on_disk(self, embeddings, sent_ids, path_to_invlist):
        if self.index_builder:
            assert sent_ids.shape[0] == embeddings.shape[0], \
                'Found {} sent_ids and {} vectors'.format(sent_ids.shape[0], vectors.shape[0])
            self.index_builder.generate_invlist(path_to_invlist, sent_ids, embeddings)
        else:
            raise Exception('Cannot index on disk without an index_builder')

    def build_index_on_disk(self, merged_ivfs_path, merged_index_path) -> int:
        if self.index_builder:
            ntotal = self.index_builder.build_disk_index(merged_ivfs_path, merged_index_path)
            return ntotal
        else:
            raise Exception('Cannot build index on disk without an index_builder')

    @staticmethod
    def consistent(scores, ids):
        results = dict()
        for score, sent_id in zip(scores[0], ids[0]):
            if score not in results:
                results[score] = list()
            results[score].append(sent_id)

        consistent_results = OrderedDict(sorted(results.items()))
        for score in consistent_results:
            consistent_results[score].sort()

        con_scores = list()
        con_ids = list()
        for score, sids in consistent_results.items():
            for sent_id in sids:
                con_scores.append(score)
                con_ids.append(sent_id)

        return [con_scores], [con_ids]
