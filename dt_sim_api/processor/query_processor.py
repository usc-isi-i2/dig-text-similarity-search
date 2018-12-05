import os
import sys
import traceback
from time import time
from typing import Dict, List, Tuple, Union

import numpy as np

from .base_processor import BaseProcessor, QueryReturn
from .processor_cache import Memoized
from dt_sim_api.vectorizer.sentence_vectorizer import DockerVectorizer

__all__ = ['QueryProcessor']

DiffScores = List[List[np.float32]]
VectIDList = List[List[np.int64]]
SentHitList = List[Tuple[np.float32, np.int64]]
DocPayload = Dict[str, SentHitList]

FinalOutput = List[Dict[str, str]]


class QueryProcessor(BaseProcessor):

    def __init__(self, index_handler: object, query_vectorizer: object = None):
        BaseProcessor.__init__(self)
        if not query_vectorizer:
            query_vectorizer = DockerVectorizer()

        self.indexer = index_handler
        self.vectorizer = query_vectorizer

    @Memoized
    def query_corpus(self, query_str: str, k: int = 5, score_type: int = None,
                     verbose: bool = True) -> FinalOutput:
        """
        Vectorize query -> Search faiss index handler -> Format doc payload
        Expects to receive only one query per call.
        :param query_str: Query to vectorize
        :param k: Number of nearest neighboring documents to return
        :param score_type: TESTING
        :param verbose: Prints time spent on each step
        :return: k sorted document hits
        """
        # Vectorize
        t_v = time()
        query_vector = self.vectorize(query_str)

        # Search            # TODO: date-range search
        t_s = time()
        k_search = max(500, 10*k)
        scores, faiss_ids = self.indexer.search(query_vector, k=k_search)

        # Aggregate hits into docs -> rerank (soon) -> format
        t_p = time()
        doc_hits = self.aggregate_docs(scores, faiss_ids)
        if score_type:
            similar_docs = self.title_rerank(doc_hits, score_type)
        else:
            similar_docs = self.format_payload(doc_hits)

        t_r = time()
        if verbose:
            print('  Query vectorized in --- {:0.4f}s'.format(t_s - t_v))
            print('  Index searched in ----- {:0.4f}s'.format(t_p - t_s))
            print('  Payload formatted in -- {:0.4f}s'.format(t_r - t_p))

        return similar_docs[:k]

    def vectorize(self, query: Union[str, List[str]]) -> QueryReturn:
        """
        Use DockerVectorizer for fast Query Vectorization.
        :param query: Text to vectorize
        :return: Formatted query embedding
        """
        if not isinstance(query, list):
            query = [query]
        if len(query) > 1:
            query = query[:1]

        query_vector = self.vectorizer.make_vectors(query)

        if isinstance(query_vector[0], list):
            query_vector = np.array(query_vector, dtype=np.float32)
        return query_vector

    @staticmethod
    def aggregate_docs(scores: DiffScores, faiss_ids: VectIDList
                       ) -> DocPayload:
        """
        Collects outputs from faiss search into document entities.
        :param scores: Faiss query/hit vector L2 distances
        :param faiss_ids: Faiss vector ids
        :return: Dict of docs (key: document id, val: doc with sentence hits)
        """
        def min_diff_cutoff(diff_score, cutoff=0.01):
            return max(diff_score, cutoff)
        
        docs = dict()
        for score, faiss_id in zip(scores[0], faiss_ids[0]):
            if faiss_id > 0:
                doc_id, sent_id = divmod(faiss_id, 10000)
                doc_id = str(doc_id)
                if doc_id not in docs:
                    docs[doc_id] = list()
                docs[doc_id].append((min_diff_cutoff(score), faiss_id))
        return docs

    @staticmethod
    def format_payload(doc_hits: DocPayload) -> FinalOutput:
        """
        TMP payload formatting for current sandpaper implementation

        Old payload structure:
            [ { 'score': str(faiss_diff), 'sentence_id': str(faiss_id) } ]
        """
        payload = list()
        for doc_id, faiss_diff_ids in doc_hits.items():
            for faiss_diff, faiss_id in faiss_diff_ids:
                out = dict()
                out['score'] = str(faiss_diff)
                out['sentence_id'] = str(faiss_id)
                payload.append(out)
        return sorted(payload, key=lambda sc_id: sc_id['score'])

    @staticmethod
    def title_rerank(doc_hits: DocPayload, new_score_type: int) -> FinalOutput:
        titles = dict()
        for doc_id, faiss_diff_ids in doc_hits.items():
            for i, (_, faiss_id) in enumerate(faiss_diff_ids):
                if str(faiss_id).endswith('0000'):
                    faiss_diff_ids.append(faiss_diff_ids.pop(i))
                    titles[doc_id] = faiss_diff_ids
                    break

        def hard_score(score, hits: SentHitList):
            boosted_score = score / min(5, len(hits))
            return boosted_score

        def soft_score2(title_score, hits: SentHitList):
            avg_score = (title_score + hits[0][0])/2
            boosted_score = avg_score / min(5, len(hits))
            return boosted_score

        def soft_score(all_hits: SentHitList):
            avg_score = sum(all_hits[j][0] for j in range(len(all_hits)))/len(all_hits)
            boosted_score = avg_score / min(5, len(all_hits))
            return boosted_score

        reranked = list()
        for doc_id, faiss_diff_ids in titles.items():
            title_diff, title_id = faiss_diff_ids.pop(-1)
            faiss_diff_ids.sort()

            out = dict()
            # One match
            if not len(faiss_diff_ids):
                out['score'] = title_diff
                out['sentence_id'] = str(title_id)
            # By title
            elif new_score_type == 0:
                out['score'] = hard_score(title_diff, faiss_diff_ids)
                out['sentence_id'] = str(title_id)
            # By content
            elif new_score_type == 1:
                faiss_diff_ids.append((title_diff, title_id))
                faiss_diff_ids.sort()
                best_sent = faiss_diff_ids[0][0]
                out['score'] = hard_score(best_sent, faiss_diff_ids)
            # Avg all content
            elif new_score_type == 2:
                faiss_diff_ids.append((title_diff, title_id))
                out['score'] = soft_score(faiss_diff_ids)
            # Avg title+best
            elif new_score_type == 3:
                out['score'] = soft_score2(title_diff, faiss_diff_ids)
            # Default
            else:
                faiss_diff_ids.append((title_diff, title_id))
                faiss_diff_ids.sort()
                out['score'] = faiss_diff_ids[0][0]
            if 'sentence_id' not in out:
                out['sentence_id'] = str(faiss_diff_ids[0][1])
            reranked.append(out)

        return sorted(reranked, key=lambda sc_id: sc_id['score'])

    def add_shard(self, shard_path: str):
        """
         Attempts to deploy new shard on current index handler.
        :param shard_path: /full/path/to/shard.index
        """
        if os.path.isfile(shard_path) and shard_path.endswith('.index'):
            try:
                self.indexer.add_shard(shard_path)
            except NameError as e:
                exc_type, exc_val, exc_trace = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_val, exc_trace)
                print(''.join(lines))
                print(e)
                print('Could not add shard: {}'.format(shard_path))
        elif not os.path.isfile(shard_path):
            print('Error: Path does not specify a file: {}'.format(shard_path))
        elif not shard_path.endswith('.index'):
            print('Error: Path does not lead to .index: {}'.format(shard_path))
        else:
            print('Error: Unexpected input: {}'.format(shard_path))

    def print_shards(self):
        n_shards = len(self.indexer.paths_to_shards)
        print('Faiss Index Shards Deployed: {}'.format(n_shards))
        for i, shard_path in enumerate(self.indexer.paths_to_shards, start=1):
            print(' {:3d}/{}: {}'.format(i, n_shards, shard_path))
