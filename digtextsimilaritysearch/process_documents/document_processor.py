_SENTENCE_ID = 'sentence_id'
_SENTENCE_TEXT = 'sentence_text'


class DocumentProcessor(object):
    def __init__(self, indexer, vectorizer, storage_adapter,
                 table_name='dig', vector_save_path='/tmp/saved_vectors.npz', save_vectors=False,
                 index_save_path='/tmp/faiss_index.index'):

        self.indexer = indexer
        self.vectorizer = vectorizer
        self.storage_adapter = storage_adapter

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
        Proprocess cdr docs into a list of (<sid>, <sentence>) tuples
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

    def query_text(self, str_query, k=3):
        similar_docs = []
        if not isinstance(str_query, list):
            str_query = [str_query]
        query_vector = self.vectorizer.make_vectors(str_query)
        scores, faiss_ids = self.indexer.search(query_vector, k)

        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            sentence_info = self.storage_adapter.get_record(str(faiss_id), self.table_name)
            if sentence_info:
                out = dict()
                out['doc_id'] = sentence_info[_SENTENCE_ID].split('_')[0]
                out['score'] = float(score)
                out['sentence'] = sentence_info[_SENTENCE_TEXT]
                similar_docs.append(out)
        return similar_docs

    def index_documents(self, cdr_docs=None, load_vectors=False, column_family='dig'):

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

        if vectors.any() and sentence_tuples:
            faiss_ids = self.indexer.index_embeddings(vectors)
            # ASSUMPTION: returned vector ids are in the same order as the initial sentence order
            for s, f in zip(sentence_tuples, faiss_ids):
                data = dict()
                data['{}:{}'.format(column_family, _SENTENCE_ID)] = s[0]
                data['{}:{}'.format(column_family, _SENTENCE_TEXT)] = s[1]
                # self.add_record_hbase(str(f), data)
                self.storage_adapter.insert_record(str(f), data, self.table_name)
            print('saving faiss index')
            self.indexer.save_index(self.index_save_path)
        else:
            print('Either provide cdr docs or file path to load vectors')
