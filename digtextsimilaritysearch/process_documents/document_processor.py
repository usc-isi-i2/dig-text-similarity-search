import numpy as np
from time import sleep, time

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

    def query_text(self, str_query, k=3, fetch_docs=False):
        """

        :param str_query: The actual text for querying.
        :param k: number of results required
        :param fetch_docs: whether or not to fetch actual documents and sentences and send them back in the response json
        :return: a list of top k results where each result is a sentence that matched and its id, score, doc id etc.
        If fetch_docs is True, we call elasticsearch and retrieve the text of the documents that matched and find the sentence
        that matched. If it is set to false, we only return the ids and the scores of the sentence and document.
        """
        similar_docs = list()
        if not isinstance(str_query, list):
            str_query = [str_query]
        t_0 = time()
        query_vector = self.vectorizer.make_vectors(str_query)
        t_1 = time()
        scores, faiss_ids = self.indexer.search(query_vector, k*5)
        t_vector = t_1 - t_0
        t_search = time() - t_1
        print('  TF vectorization time: {:0.6f}s'.format(t_vector))
        print('  Faiss search time: {:0.6f}s'.format(t_search))
        t_es = 0
        unique_sentences = set()
        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            if len(similar_docs) >= k:
                break
            doc_id, sent_id = divmod(faiss_id, 10000)
            sentence_info = None
            if fetch_docs:
                t_start = time()
                sentence_info = self.storage_adapter.get_record(str(doc_id), self.table_name)
                t_end = time()
                t_es = t_end - t_start
            if isinstance(sentence_info, list) and len(sentence_info) >= 1:
                sentence_info = sentence_info[0]
            out = dict()
            out['doc_id'] = str(doc_id)
            out['score'] = float(score)
            out['sentence_id'] = str(sent_id)
            out['vectorizer_time_taken'] = t_vector
            out['faiss_query_time'] = t_search
            if sentence_info and fetch_docs:
                out['es_query_time'] = t_es
                if sent_id == 0:
                    out['sentence'] = sentence_info['lexisnexis']['doc_title']
                else:
                    out['sentence'] = sentence_info['split_sentences'][sent_id-1]
                #
                if out['sentence'] not in unique_sentences:
                    similar_docs.append(out)
                    unique_sentences.add(str(out['sentence']))
                else:
                    pass
                # TODO: rerank by docs with multiple sentence hits
        return similar_docs

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
        records = []
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

    def index_docs_on_disk(self, offset, path_to_npz, path_to_invlist=None):
        if not path_to_invlist:
            path_to_invlist = 'invl_' + path_to_npz.replace('.npz', '.index')

        if self.index_builder:
            vectors, sent_tups = self.vectorizer.load_vectors(path_to_npz)
            # faiss_ids = self.index_builder.generate_faiss_ids(path_to_npz, vectors, sent_tups)
            faiss_ids = np.arange(start=0, stop=len(vectors), dtype=np.int64) + offset
            self.index_builder.generate_invlist(path_to_invlist, faiss_ids, vectors)
            self.add_to_db(sent_tups, faiss_ids)
        else:
            raise Exception('Cannot index on disk without an index_builder')

    def build_index_on_disk(self, merged_ivfs_path, merged_index_path) -> int:
        if self.index_builder:
            ntotal = self.index_builder.build_disk_index(merged_ivfs_path, merged_index_path)
            return ntotal
        else:
            raise Exception('Cannot build index on disk without an index_builder')
