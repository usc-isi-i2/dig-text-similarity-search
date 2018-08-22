_SENTENCE_ID = 'sentence_id'
_SENTENCE_TEXT = 'sentence_text'


class DocumentProcessor(object):
    def __init__(self, indexer, batch_vectorizer, hbase_adapter, hbase_table='dig', hbase_column_family='dig',
                 save_vectors=False, vector_save_path='/tmp/saved_vectors.npz'):
        self.indexer = indexer
        self.batch_vectorizer = batch_vectorizer
        self.hbase_adapter = hbase_adapter
        self.hbase_table = hbase_table
        self.hbase_column_family = hbase_column_family
        if self.hbase_adapter:
            self._configure()
        self.save_vectors = save_vectors
        self.vector_save_path = vector_save_path

    def _configure(self):
        # create hbase table if it doesn't exist
        if not bytes(self.hbase_table, encoding='utf-8') in self.hbase_adapter.tables():
            self.hbase_adapter.create_table(self.hbase_table, family_name=self.hbase_column_family)

    def add_record_hbase(self, id, data):
        """
        Function to add id and value into hbase
        :param id: id at which the value will be added in the hbase table
        :param value: value to be added in hbase
        :param column_name: column name in hbase table where value should be interested
        :return:
        """

        self.hbase_adapter.insert_record_data(self.hbase_table, id, data)

    def get_record_hbase(self, id, column_names=[_SENTENCE_ID, _SENTENCE_TEXT]):
        """
        Function to return the sentence id from hbase
        :param id: input id to hbase
        :param column_names: selected column names
        :return: sentence_id, there should be only one
        """
        record = self.hbase_adapter.get_record(id, self.hbase_table)
        if record:
            result = {}
            for column_name in column_names:
                family_column = '{}:{}'.format(self.hbase_column_family, column_name).encode('utf-8')
                result[column_name] = record.get(family_column, '').decode('utf-8')
            return result
        return None

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
        vectors = self.batch_vectorizer.make_vectors(sentences)
        if self.save_vectors:
            self.batch_vectorizer.save_vectors(embeddings=vectors, sentences=sentence_tuples,
                                               file_path=self.vector_save_path)
        return vectors

    def query_text(self, str_query, k=3):
        similar_docs = []
        if not isinstance(str_query, list):
            str_query = [str_query]
        query_vector = self.batch_vectorizer.make_vectors(str_query)
        scores, faiss_ids = self.indexer.search(query_vector, k)

        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            sentence_info = self.get_record_hbase(str(faiss_id))
            if sentence_info:
                out = dict()
                out['doc_id'] = sentence_info[_SENTENCE_ID].split('_')[0]
                out['score'] = score
                out['sentence'] = sentence_info[_SENTENCE_TEXT]
                similar_docs.append(out)
        return similar_docs

    def index_documents(self, cdr_docs=None, load_vectors=False):

        vectors = None
        sentence_tuples = None
        if load_vectors:
            vectors, sentence_tuples = self.batch_vectorizer.load_vectors(self.vector_save_path)
            print('Total sentences loaded from file: {}'.format(len(sentence_tuples)))
        else:
            if cdr_docs:
                print('Total cdr docs to be processed: {}'.format(len(cdr_docs)))
                sentence_tuples = self.preprocess_documents(cdr_docs)
                vectors = self.create_vectors(sentence_tuples)

        if vectors and sentence_tuples:
            faiss_ids = self.indexer.index_embeddings(vectors)
            # ASSUMPTION: returned vector ids are in the same order as the initial sentence order
            for s, f in zip(sentence_tuples, faiss_ids):
                data = {}
                data['{}:{}'.format(self.hbase_column_family, _SENTENCE_ID)] = s[0]
                data['{}:{}'.format(self.hbase_column_family, _SENTENCE_TEXT)] = s[1]
                self.add_record_hbase(str(f), data)
            print('saving faiss index')
            self.indexer.save_index('/tmp/faiss_index')
        else:
            print('Either provide cdr docs or file path to load vectors')
