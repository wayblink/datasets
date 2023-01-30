import os
import tempfile
from functools import partial
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest

from datasets.arrow_dataset import Dataset
from datasets.search import ElasticSearchIndex, FaissIndex, MissingIndex, MilvusIndex

from .utils import require_elasticsearch, require_faiss, require_milvus

pytestmark = pytest.mark.integration


@require_faiss
class IndexableDatasetTest(TestCase):
    def _create_dummy_dataset(self):
        dset = Dataset.from_dict({"filename": ["my_name-train" + "_" + str(x) for x in np.arange(30).tolist()]})
        return dset

    def test_add_faiss_index(self):
        import faiss

        dset: Dataset = self._create_dummy_dataset()
        dset = dset.map(
            lambda ex, i: {"vecs": i * np.ones(5, dtype=np.float32)}, with_indices=True, keep_in_memory=True
        )
        dset = dset.add_faiss_index("vecs", batch_size=100, metric_type=faiss.METRIC_INNER_PRODUCT)
        scores, examples = dset.get_nearest_examples("vecs", np.ones(5, dtype=np.float32))
        self.assertEqual(examples["filename"][0], "my_name-train_29")
        dset.drop_index("vecs")

    def test_add_faiss_index_from_external_arrays(self):
        import faiss

        dset: Dataset = self._create_dummy_dataset()
        dset.add_faiss_index_from_external_arrays(
            external_arrays=np.ones((30, 5)) * np.arange(30).reshape(-1, 1),
            index_name="vecs",
            batch_size=100,
            metric_type=faiss.METRIC_INNER_PRODUCT,
        )
        scores, examples = dset.get_nearest_examples("vecs", np.ones(5, dtype=np.float32))
        self.assertEqual(examples["filename"][0], "my_name-train_29")

    def test_serialization(self):
        import faiss

        dset: Dataset = self._create_dummy_dataset()
        dset.add_faiss_index_from_external_arrays(
            external_arrays=np.ones((30, 5)) * np.arange(30).reshape(-1, 1),
            index_name="vecs",
            metric_type=faiss.METRIC_INNER_PRODUCT,
        )

        # Setting delete=False and unlinking manually is not pretty... but it is required on Windows to
        # ensure somewhat stable behaviour. If we don't, we get PermissionErrors. This is an age-old issue.
        # see https://bugs.python.org/issue14243 and
        # https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file/23212515
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            dset.save_faiss_index("vecs", tmp_file.name)
            dset.load_faiss_index("vecs2", tmp_file.name)
        os.unlink(tmp_file.name)

        scores, examples = dset.get_nearest_examples("vecs2", np.ones(5, dtype=np.float32))
        self.assertEqual(examples["filename"][0], "my_name-train_29")

    def test_drop_index(self):
        dset: Dataset = self._create_dummy_dataset()
        dset.add_faiss_index_from_external_arrays(
            external_arrays=np.ones((30, 5)) * np.arange(30).reshape(-1, 1), index_name="vecs"
        )
        dset.drop_index("vecs")
        self.assertRaises(MissingIndex, partial(dset.get_nearest_examples, "vecs2", np.ones(5, dtype=np.float32)))

    def test_add_elasticsearch_index(self):
        from elasticsearch import Elasticsearch

        dset: Dataset = self._create_dummy_dataset()
        with patch("elasticsearch.Elasticsearch.search") as mocked_search, patch(
            "elasticsearch.client.IndicesClient.create"
        ) as mocked_index_create, patch("elasticsearch.helpers.streaming_bulk") as mocked_bulk:
            mocked_index_create.return_value = {"acknowledged": True}
            mocked_bulk.return_value([(True, None)] * 30)
            mocked_search.return_value = {"hits": {"hits": [{"_score": 1, "_id": 29}]}}
            es_client = Elasticsearch()

            dset.add_elasticsearch_index("filename", es_client=es_client)
            scores, examples = dset.get_nearest_examples("filename", "my_name-train_29")
            self.assertEqual(examples["filename"][0], "my_name-train_29")


@require_faiss
class FaissIndexTest(TestCase):
    def test_flat_ip(self):
        import faiss

        index = FaissIndex(metric_type=faiss.METRIC_INNER_PRODUCT)

        # add vectors
        index.add_vectors(np.eye(5, dtype=np.float32))
        self.assertIsNotNone(index.faiss_index)
        self.assertEqual(index.faiss_index.ntotal, 5)
        index.add_vectors(np.zeros((5, 5), dtype=np.float32))
        self.assertEqual(index.faiss_index.ntotal, 10)

        # single query
        query = np.zeros(5, dtype=np.float32)
        query[1] = 1
        scores, indices = index.search(query)
        self.assertRaises(ValueError, index.search, query.reshape(-1, 1))
        self.assertGreater(scores[0], 0)
        self.assertEqual(indices[0], 1)

        # batched queries
        queries = np.eye(5, dtype=np.float32)[::-1]
        total_scores, total_indices = index.search_batch(queries)
        self.assertRaises(ValueError, index.search_batch, queries[0])
        best_scores = [scores[0] for scores in total_scores]
        best_indices = [indices[0] for indices in total_indices]
        self.assertGreater(np.min(best_scores), 0)
        self.assertListEqual([4, 3, 2, 1, 0], best_indices)

    def test_factory(self):
        import faiss

        index = FaissIndex(string_factory="Flat")
        index.add_vectors(np.eye(5, dtype=np.float32))
        self.assertIsInstance(index.faiss_index, faiss.IndexFlat)
        index = FaissIndex(string_factory="LSH")
        index.add_vectors(np.eye(5, dtype=np.float32))
        self.assertIsInstance(index.faiss_index, faiss.IndexLSH)
        with self.assertRaises(ValueError):
            _ = FaissIndex(string_factory="Flat", custom_index=faiss.IndexFlat(5))

    def test_custom(self):
        import faiss

        custom_index = faiss.IndexFlat(5)
        index = FaissIndex(custom_index=custom_index)
        index.add_vectors(np.eye(5, dtype=np.float32))
        self.assertIsInstance(index.faiss_index, faiss.IndexFlat)

    def test_serialization(self):
        import faiss

        index = FaissIndex(metric_type=faiss.METRIC_INNER_PRODUCT)
        index.add_vectors(np.eye(5, dtype=np.float32))

        # Setting delete=False and unlinking manually is not pretty... but it is required on Windows to
        # ensure somewhat stable behaviour. If we don't, we get PermissionErrors. This is an age-old issue.
        # see https://bugs.python.org/issue14243 and
        # https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file/23212515
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            index.save(tmp_file.name)
            index = FaissIndex.load(tmp_file.name)
        os.unlink(tmp_file.name)

        query = np.zeros(5, dtype=np.float32)
        query[1] = 1
        scores, indices = index.search(query)
        self.assertGreater(scores[0], 0)
        self.assertEqual(indices[0], 1)


@require_elasticsearch
class ElasticSearchIndexTest(TestCase):
    def test_elasticsearch(self):
        from elasticsearch import Elasticsearch

        with patch("elasticsearch.Elasticsearch.search") as mocked_search, patch(
            "elasticsearch.client.IndicesClient.create"
        ) as mocked_index_create, patch("elasticsearch.helpers.streaming_bulk") as mocked_bulk:
            es_client = Elasticsearch()
            mocked_index_create.return_value = {"acknowledged": True}
            index = ElasticSearchIndex(es_client=es_client)
            mocked_bulk.return_value([(True, None)] * 3)
            index.add_documents(["foo", "bar", "foobar"])

            # single query
            query = "foo"
            mocked_search.return_value = {"hits": {"hits": [{"_score": 1, "_id": 0}]}}
            scores, indices = index.search(query)
            self.assertEqual(scores[0], 1)
            self.assertEqual(indices[0], 0)

            # batched queries
            queries = ["foo", "bar", "foobar"]
            mocked_search.return_value = {"hits": {"hits": [{"_score": 1, "_id": 1}]}}
            total_scores, total_indices = index.search_batch(queries)
            best_scores = [scores[0] for scores in total_scores]
            best_indices = [indices[0] for indices in total_indices]
            self.assertGreater(np.min(best_scores), 0)
            self.assertListEqual([1, 1, 1], best_indices)


@require_milvus
class MilvusIndexTest(TestCase):
    def test_simple_index(self):
        index = MilvusIndex()

        rng = np.random.default_rng(seed=19530)
        vectors = rng.random((102, 128))

        index.insert(vectors)
        self.assertIsNotNone(index.collection)
        self.assertEqual(index.collection.num_entities, 102)

        print("Start searching based on vector similarity")
        vector_to_search = vectors[0:]
        scores, indices = index.search(vector_to_search, 3)
        self.assertEqual(scores[0], 0)
        self.assertEqual(indices[0], 0)

        print("Start batch searching based on vector similarity")

        vectors_to_search = vectors[0:2,:]
        scores, indices = index.search_batch(vectors_to_search, 3)
        self.assertEqual(scores[0][0], 0)
        self.assertEqual(indices[0][0], 0)
        self.assertEqual(scores[1][0], 0)
        self.assertEqual(indices[1][0], 1)

        index.remove()

    def test_complex_index(self):
        index = MilvusIndex(
            milvus_collection_name="milvus_index_" + os.path.basename(tempfile.NamedTemporaryFile().name),
            milvus_collection_schema=[
                {
                    "name": "id",
                    "type": "VARCHAR",
                    "description": "VARCHAR field",
                    "is_primary": True,
                    "auto_id": False,
                    "params": {
                        "max_length": 100
                    }
                },
                {
                    "name": "random",
                    "type": "DOUBLE",
                    "description": "DOUBLE field",
                },
                {
                    "name": "vec",
                    "type": "FLOAT_VECTOR",
                    "description": "FLOAT_VECTOR field",
                    "params": {
                        "dim": 8
                    }
                }
            ],
            milvus_index_params={
                "field": "vec",
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            })

        dim = 8
        num_entities = 100
        rng = np.random.default_rng(seed=19530)
        entities = [
            # provide the pk field because `auto_id` is set to False
            [str(i) for i in range(num_entities)],
            rng.random(num_entities).tolist(),  # field random, only supports list
            rng.random((num_entities, dim)),  # field embeddings, supports numpy.ndarray and list
        ]

        index.insert(entities)
        self.assertIsNotNone(index.collection)
        self.assertEqual(index.collection.num_entities, 100)

        print("Start searching based on vector similarity")
        vector_to_search = entities[2][0:1]
        scores, indices = index.search(vector_to_search, 3)
        self.assertEqual(scores[0], 0)
        self.assertEqual(indices[0], '0')

        print("Start batch searching based on vector similarity")

        vectors_to_search = entities[2][0:2]
        scores, indices = index.search_batch(vectors_to_search, 3)
        self.assertEqual(scores[0][0], 0)
        self.assertEqual(indices[0][0], '0')
        self.assertEqual(scores[1][0], 0)
        self.assertEqual(indices[1][0], '1')

        index.remove()

