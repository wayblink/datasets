import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union

import numpy as np

from .utils import logging


if TYPE_CHECKING:
    from .arrow_dataset import Dataset  # noqa: F401

    try:
        from elasticsearch import Elasticsearch  # noqa: F401

    except ImportError:
        pass
    try:
        import faiss  # noqa: F401

    except ImportError:
        pass

    try:
        import pymilvus

    except ImportError:
        pass

_has_elasticsearch = importlib.util.find_spec("elasticsearch") is not None
_has_faiss = importlib.util.find_spec("faiss") is not None
_has_milvus = importlib.util.find_spec("pymilvus") is not None


logger = logging.get_logger(__name__)


class MissingIndex(Exception):
    pass


class SearchResults(NamedTuple):
    scores: List[float]
    indices: List[int]


class BatchedSearchResults(NamedTuple):
    total_scores: List[List[float]]
    total_indices: List[List[int]]


class NearestExamplesResults(NamedTuple):
    scores: List[float]
    examples: dict


class BatchedNearestExamplesResults(NamedTuple):
    total_scores: List[List[float]]
    total_examples: List[dict]


class BaseIndex:
    """Base class for indexing"""

    def search(self, query, k: int = 10) -> SearchResults:
        """
        To implement.
        This method has to return the scores and the indices of the retrieved examples given a certain query.
        """
        raise NotImplementedError

    def search_batch(self, queries, k: int = 10) -> BatchedSearchResults:
        """Find the nearest examples indices to the query.

        Args:
            queries (`Union[List[str], np.ndarray]`): The queries as a list of strings if `column` is a text index or as a numpy array if `column` is a vector index.
            k (`int`): The number of examples to retrieve per query.

        Ouput:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.search(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return BatchedSearchResults(total_scores, total_indices)

    def save(self, file: Union[str, PurePath]):
        """Serialize the index on disk"""
        raise NotImplementedError

    @classmethod
    def load(cls, file: Union[str, PurePath]) -> "BaseIndex":
        """Deserialize the index from disk"""
        raise NotImplementedError


class ElasticSearchIndex(BaseIndex):
    """
    Sparse index using Elasticsearch. It is used to index text and run queries based on BM25 similarity.
    An Elasticsearch server needs to be accessible, and a python client is declared with
    ```
    es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    ```
    for example.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_name: Optional[str] = None,
        es_index_config: Optional[dict] = None,
    ):
        if not _has_elasticsearch:
            raise ImportError(
                "You must install ElasticSearch to use ElasticSearchIndex. To do so you can run `pip install elasticsearch==7.7.1 for example`"
            )
        if es_client is not None and (host is not None or port is not None):
            raise ValueError("Please specify either `es_client` or `(host, port)`, but not both.")
        host = host or "localhost"
        port = port or 9200

        import elasticsearch.helpers  # noqa: need this to properly load all the es features
        from elasticsearch import Elasticsearch  # noqa: F811

        self.es_client = es_client if es_client is not None else Elasticsearch([{"host": host, "port": str(port)}])
        self.es_index_name = (
            es_index_name
            if es_index_name is not None
            else "huggingface_datasets_" + os.path.basename(tempfile.NamedTemporaryFile().name)
        )
        self.es_index_config = (
            es_index_config
            if es_index_config is not None
            else {
                "settings": {
                    "number_of_shards": 1,
                    "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                },
                "mappings": {"properties": {"text": {"type": "text", "analyzer": "standard", "similarity": "BM25"}}},
            }
        )

    def add_documents(self, documents: Union[List[str], "Dataset"], column: Optional[str] = None):
        """
        Add documents to the index.
        If the documents are inside a certain column, you can specify it using the `column` argument.
        """
        index_name = self.es_index_name
        index_config = self.es_index_config
        self.es_client.indices.create(index=index_name, body=index_config)
        number_of_docs = len(documents)
        progress = logging.tqdm(unit="docs", total=number_of_docs, disable=not logging.is_progress_bar_enabled())
        successes = 0

        def passage_generator():
            if column is not None:
                for i, example in enumerate(documents):
                    yield {"text": example[column], "_id": i}
            else:
                for i, example in enumerate(documents):
                    yield {"text": example, "_id": i}

        # create the ES index
        import elasticsearch as es

        for ok, action in es.helpers.streaming_bulk(
            client=self.es_client,
            index=index_name,
            actions=passage_generator(),
        ):
            progress.update(1)
            successes += ok
        if successes != len(documents):
            logger.warning(
                f"Some documents failed to be added to ElasticSearch. Failures: {len(documents)-successes}/{len(documents)}"
            )
        logger.info(f"Indexed {successes:d} documents")

    def search(self, query: str, k=10) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`str`): The query as a string.
            k (`int`): The number of examples to retrieve.

        Ouput:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        response = self.es_client.search(
            index=self.es_index_name,
            body={"query": {"multi_match": {"query": query, "fields": ["text"], "type": "cross_fields"}}, "size": k},
        )
        hits = response["hits"]["hits"]
        return SearchResults([hit["_score"] for hit in hits], [int(hit["_id"]) for hit in hits])

    def search_batch(self, queries, k: int = 10, max_workers=10) -> BatchedSearchResults:
        import concurrent.futures

        total_scores, total_indices = [None] * len(queries), [None] * len(queries)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.search, query, k): i for i, query in enumerate(queries)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results: SearchResults = future.result()
                total_scores[index] = results.scores
                total_indices[index] = results.indices
        return BatchedSearchResults(total_indices=total_indices, total_scores=total_scores)


class FaissIndex(BaseIndex):
    """
    Dense index using Faiss. It is used to index vectors.
    Faiss is a library for efficient similarity search and clustering of dense vectors.
    It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
    You can find more information about Faiss here:
    - For index types and the string factory: https://github.com/facebookresearch/faiss/wiki/The-index-factory
    - For GPU settings: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
    """

    def __init__(
        self,
        device: Optional[Union[int, List[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
    ):
        """
        Create a Dense index using Faiss. You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory
        """
        if string_factory is not None and custom_index is not None:
            raise ValueError("Please specify either `string_factory` or `custom_index` but not both.")
        if device is not None and custom_index is not None:
            raise ValueError(
                "Cannot pass both 'custom_index' and 'device'. "
                "Pass 'custom_index' already transferred to the target device instead."
            )
        self.device = device
        self.string_factory = string_factory
        self.metric_type = metric_type
        self.faiss_index = custom_index
        if not _has_faiss:
            raise ImportError(
                "You must install Faiss to use FaissIndex. To do so you can run `conda install -c pytorch faiss-cpu` or `conda install -c pytorch faiss-gpu`. "
                "A community supported package is also available on pypi: `pip install faiss-cpu` or `pip install faiss-gpu`. "
                "Note that pip may not have the latest version of FAISS, and thus, some of the latest features and bug fixes may not be available."
            )

    def add_vectors(
        self,
        vectors: Union[np.array, "Dataset"],
        column: Optional[str] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: Optional[bool] = None,
    ):
        """
        Add vectors to the index.
        If the arrays are inside a certain column, you can specify it using the `column` argument.
        """
        import faiss  # noqa: F811

        # Create index
        if self.faiss_index is None:
            size = len(vectors[0]) if column is None else len(vectors[0][column])
            if self.string_factory is not None:
                if self.metric_type is None:
                    index = faiss.index_factory(size, self.string_factory)
                else:
                    index = faiss.index_factory(size, self.string_factory, self.metric_type)
            else:
                if self.metric_type is None:
                    index = faiss.IndexFlat(size)
                else:
                    index = faiss.IndexFlat(size, self.metric_type)

            self.faiss_index = self._faiss_index_to_device(index, self.device)
            logger.info(f"Created faiss index of type {type(self.faiss_index)}")

        # Set verbosity level
        if faiss_verbose is not None:
            self.faiss_index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "index") and self.faiss_index.index is not None:
                self.faiss_index.index.verbose = faiss_verbose
            if hasattr(self.faiss_index, "quantizer") and self.faiss_index.quantizer is not None:
                self.faiss_index.quantizer.verbose = faiss_verbose
            if hasattr(self.faiss_index, "clustering_index") and self.faiss_index.clustering_index is not None:
                self.faiss_index.clustering_index.verbose = faiss_verbose

        # Train
        if train_size is not None:
            train_vecs = vectors[:train_size] if column is None else vectors[:train_size][column]
            logger.info(f"Training the index with the first {len(train_vecs)} vectors")
            self.faiss_index.train(train_vecs)
        else:
            logger.info("Ignored the training step of the faiss index as `train_size` is None.")

        # Add vectors
        logger.info(f"Adding {len(vectors)} vectors to the faiss index")
        for i in logging.tqdm(range(0, len(vectors), batch_size), disable=not logging.is_progress_bar_enabled()):
            vecs = vectors[i : i + batch_size] if column is None else vectors[i : i + batch_size][column]
            self.faiss_index.add(vecs)

    @staticmethod
    def _faiss_index_to_device(index: "faiss.Index", device: Optional[Union[int, List[int]]] = None) -> "faiss.Index":
        """
        Sends a faiss index to a device.
        A device can either be a positive integer (GPU id), a negative integer (all GPUs),
            or a list of positive integers (select GPUs to use), or `None` for CPU.
        """

        # If device is not specified, then it runs on CPU.
        if device is None:
            return index

        import faiss  # noqa: F811

        # If the device id is given as an integer
        if isinstance(device, int):
            # Positive integers are directly mapped to GPU ids
            if device > -1:
                faiss_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(faiss_res, device, index)
            # And negative integers mean using all GPUs
            else:
                index = faiss.index_cpu_to_all_gpus(index)
        # Device ids given as a list mean mapping to those devices specified.
        elif isinstance(device, (list, tuple)):
            index = faiss.index_cpu_to_gpus_list(index, gpus=list(device))
        else:
            raise TypeError(
                f"The argument type: {type(device)} is not expected. "
                + "Please pass in either nothing, a positive int, a negative int, or a list of positive ints."
            )

        return index

    def search(self, query: np.array, k=10) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`np.array`): The query as a numpy array.
            k (`int`): The number of examples to retrieve.

        Ouput:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        if len(query.shape) != 1 and (len(query.shape) != 2 or query.shape[0] != 1):
            raise ValueError("Shape of query is incorrect, it has to be either a 1D array or 2D (1, N)")

        queries = query.reshape(1, -1)
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k)
        return SearchResults(scores[0], indices[0].astype(int))

    def search_batch(self, queries: np.array, k=10) -> BatchedSearchResults:
        """Find the nearest examples indices to the queries.

        Args:
            queries (`np.array`): The queries as a numpy array.
            k (`int`): The number of examples to retrieve.

        Ouput:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        if len(queries.shape) != 2:
            raise ValueError("Shape of query must be 2D")
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        scores, indices = self.faiss_index.search(queries, k)
        return BatchedSearchResults(scores, indices.astype(int))

    def save(self, file: Union[str, PurePath]):
        """Serialize the FaissIndex on disk"""
        import faiss  # noqa: F811

        if self.device is not None and isinstance(self.device, (int, list, tuple)):
            index = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            index = self.faiss_index

        faiss.write_index(index, str(file))

    @classmethod
    def load(
        cls,
        file: Union[str, PurePath],
        device: Optional[Union[int, List[int]]] = None,
    ) -> "FaissIndex":
        """Deserialize the FaissIndex from disk"""
        import faiss  # noqa: F811

        # Instances of FaissIndex is essentially just a wrapper for faiss indices.
        faiss_index = cls(device=device)
        index = faiss.read_index(str(file))
        faiss_index.faiss_index = faiss_index._faiss_index_to_device(index, faiss_index.device)
        return faiss_index


class MilvusIndex(BaseIndex):
    """
    Milvus is a vector database focus on store, index and manage massive embedding vectors
    generated by deep neural networks and other machine learning (ML) models.

    Similar to Faiss, Milvus supports efficient similarity search of dense vector.
    Besides, Milvus provides plenty of database features to empower data/ML apps.
    https://milvus.io/
    """

    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None,
            milvus_collection_name: Optional[str] = None,
            milvus_collection_schema: Optional[list] = None,
            milvus_index_params: Optional[dict] = None,
    ):
        if not _has_milvus:
            raise ImportError(
                "You must install Milvus to use MilvusIndex. "
                "To do so you can run `pip install pymilvus==2.2.0 for example`"
            )

        host = host or "localhost"
        port = port or 19530

        from pymilvus import (
            connections,
            FieldSchema, CollectionSchema, DataType,
            Collection,
        )
        connections.connect("default", host=host, port=port)
        logger.info(f"create connection to milvus {host}:{port}")
        self.milvus_collection_name = (
            milvus_collection_name
            if milvus_collection_name is not None
            else "huggingface_datasets_" + os.path.basename(tempfile.NamedTemporaryFile().name)
        )

        self.auto_id = False
        if milvus_collection_schema is None:
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            self.is_default_schema = True
        else:
            fields = []
            for fieldDict in milvus_collection_schema:
                field = self._parse_field(fieldDict)
                fields.append(field)
            self.is_default_schema = False
        schema = CollectionSchema(fields)
        milvus_collection = Collection(self.milvus_collection_name, schema)
        self.schema = schema
        self.collection = milvus_collection
        self.collection_size = milvus_collection.num_entities
        logger.info(f"get or create milvus collection {milvus_collection_name} size: {self.collection_size}")

        self.milvus_index_params = (
            milvus_index_params
            if milvus_index_params is not None
            else {
                "field": "embeddings",
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 768},
            }
        )
        self.milvus_vector_field = self.milvus_index_params["field"]
        if not milvus_collection.has_index():
            milvus_collection.release()
            milvus_collection.create_index(self.milvus_vector_field, self.milvus_index_params)
            logger.info(f"create index on milvus collection {milvus_collection_name} "
                        f"index_params: {self.milvus_index_params}")

    def _parse_field(self, params: dict):
        from pymilvus import DataType, FieldSchema
        if params.get("auto_id", None) is not None:
            self.auto_id = params["auto_id"]
        dtype = params['type']
        if dtype == "BOOL":
            dtype = DataType.BOOL
        if dtype == "INT8":
            dtype = DataType.INT8
        if dtype == "INT16":
            dtype = DataType.INT16
        if dtype == "INT32":
            dtype = DataType.INT32
        if dtype == "INT64":
            dtype = DataType.INT64
        if dtype == "FLOAT":
            dtype = DataType.FLOAT
        if dtype == "DOUBLE":
            dtype = DataType.DOUBLE
        if dtype == "STRING":
            dtype = DataType.STRING
        if dtype == "VARCHAR":
            dtype = DataType.VARCHAR
        if dtype == "BINARY_VECTOR":
            dtype = DataType.BINARY_VECTOR
        if dtype == "FLOAT_VECTOR":
            dtype = DataType.FLOAT_VECTOR
        params['type'] = dtype
        params['description'] = params.get("description", "")
        return FieldSchema.construct_from_dict(params)

    def column_names(self):
        columns = []
        for i in range(0, len(self.schema)):
            if self.schema.fields[i].is_primary and self.schema.auto_id:
                continue
            columns.append(self.schema.fields[i].name)
        return columns

    def insert(self, entities: Union[np.array, "Dataset"], batch_size: int = 1000):
        if entities is None:
            return
        if self.is_default_schema:
            indice = [i + self.collection_size for i in range(len(entities))]
            entities = [indice, entities]

        columns = self.column_names()
        inserted_size = 0
        need_insert_size = len(entities)
        while inserted_size < need_insert_size:
            if inserted_size + batch_size <= need_insert_size:
                this_batch_size = batch_size
            else:
                this_batch_size = need_insert_size - inserted_size
            batch = []
            pks = []
            for id in range(0, this_batch_size):
                pks.append(id + self.collection_size)
            batch.append(pks)
            for i in range(0, len(columns)):
                if columns[i] == "pk":
                    continue
                column = []
                for j in range(inserted_size, inserted_size + this_batch_size):
                    column.append(entities[j][columns[i]])
                batch.append(column)
            insertResult = self.collection.insert(batch)
            inserted_size = inserted_size + this_batch_size
            self.collection_size = self.collection_size + this_batch_size

        self.collection.flush()

    def search(self, query: np.array, k=10, expr: Optional[str] = None) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`np.array`): The query as a numpy array.
            k (`int`): The number of examples to retrieve.
            expr (Optional `str`):
                Boolean expression that filter the scalar fields or the primary key field,
                you can limit your search with certain conditions.
                For example: pk >= 10

        Output:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        self.collection.load()
        queries = query.reshape(1, -1)
        if not queries.flags.c_contiguous:
            queries = np.asarray(queries, order="C")
        search_result = self.collection.search(queries,
                                               self.milvus_vector_field,
                                               self.milvus_index_params,
                                               k,
                                               expr)
        return SearchResults(search_result[0].distances, search_result[0].ids)

    def search_batch(self, queries: np.array, k=10, expr: Optional[str] = None) -> BatchedSearchResults:
        """Find the nearest examples indices to the queries.

        Args:
            queries (`np.array`): The queries as a numpy array.
            k (`int`): The number of examples to retrieve.
            expr (Optional `str`):
                Boolean expression that filter the scalar fields or the primary key field,
                you can limit your search with certain conditions.
                For example: pk >= 10

        Output:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """

        self.collection.load()
        search_result = self.collection.search(queries, self.milvus_vector_field, self.milvus_index_params, k, expr)
        distances = []
        ids = []
        for i in (0, len(search_result)-1):
            distances.append(search_result[i].distances)
            ids.append(search_result[i].ids)
        return BatchedSearchResults(distances, ids)

    def remove(self):
        from pymilvus import utility
        utility.drop_collection(self.collection.name)


class IndexableMixin:
    """Add indexing features to `datasets.Dataset`"""

    def __init__(self):
        self._indexes: Dict[str, BaseIndex] = {}

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def is_index_initialized(self, index_name: str) -> bool:
        return index_name in self._indexes

    def _check_index_is_initialized(self, index_name: str):
        if not self.is_index_initialized(index_name):
            raise MissingIndex(
                f"Index with index_name '{index_name}' not initialized yet. Please make sure that you call "
                f"`add_faiss_index` or `add_elasticsearch_index` or `add_milvus_index` first."
            )

    def list_indexes(self) -> List[str]:
        """List the `colindex_nameumns`/identifiers of all the attached indexes."""
        return list(self._indexes)

    def get_index(self, index_name: str) -> BaseIndex:
        """List the `index_name`/identifiers of all the attached indexes.

        Args:
            index_name (`str`): Index name.

        Returns:
            [`BaseIndex`]
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name]

    def add_faiss_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        device: Optional[Union[int, List[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of the specified column.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index, see more below).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            column (`str`): The column of the vectors to add to the index.
            index_name (Optional `str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
                By default it corresponds to `column`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        index_name = index_name if index_name is not None else column
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            self, column=column, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def add_faiss_index_from_external_arrays(
        self,
        external_arrays: np.array,
        index_name: str,
        device: Optional[Union[int, List[int]]] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of `external_arrays`.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:
        - For `string factory`: https://github.com/facebookresearch/faiss/wiki/The-index-factory

        Args:
            external_arrays (`np.array`): If you want to use arrays from outside the lib for the index, you can set `external_arrays`.
                It will use `external_arrays` to create the Faiss index instead of the arrays in the given `column`.
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (Optional `str`): This is passed to the index factory of Faiss to create the index. Default index class is IndexFlatIP.
            metric_type (Optional `int`): Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (Optional `faiss.Index`): Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (Optional `int`): If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False): Enable the verbosity of the Faiss index.
        """
        faiss_index = FaissIndex(
            device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index
        )
        faiss_index.add_vectors(
            external_arrays, column=None, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose
        )
        self._indexes[index_name] = faiss_index

    def save_faiss_index(self, index_name: str, file: Union[str, PurePath]):
        """Save a FaissIndex on disk.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk.
        """
        index = self.get_index(index_name)
        if not isinstance(index, FaissIndex):
            raise ValueError(f"Index '{index_name}' is not a FaissIndex but a '{type(index)}'")
        index.save(file)
        logger.info(f"Saved FaissIndex {index_name} at {file}")

    def load_faiss_index(
        self,
        index_name: str,
        file: Union[str, PurePath],
        device: Optional[Union[int, List[int]]] = None,
    ):
        """Load a FaissIndex from disk.

        If you want to do additional configurations, you can have access to the faiss index object by doing
        `.get_index(index_name).faiss_index` to make it fit your needs.

        Args:
            index_name (`str`): The index_name/identifier of the index. This is the index_name that is used to
                call `.get_nearest` or `.search`.
            file (`str`): The path to the serialized faiss index on disk.
            device (Optional `Union[int, List[int]]`): If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
        """
        index = FaissIndex.load(file, device=device)
        if index.faiss_index.ntotal != len(self):
            raise ValueError(
                f"Index size should match Dataset size, but Index '{index_name}' at {file} has {index.faiss_index.ntotal} elements while the dataset has {len(self)} examples."
            )
        self._indexes[index_name] = index
        logger.info(f"Loaded FaissIndex {index_name} from {file}")

    def add_elasticsearch_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_name: Optional[str] = None,
        es_index_config: Optional[dict] = None,
    ):
        """Add a text index using ElasticSearch for fast retrieval.

        Args:
            column (`str`): The column of the documents to add to the index.
            index_name (Optional `str`): The index_name/identifier of the index. This is the index name that is used to call `.get_nearest` or `.search`.
                By default it corresponds to `column`.
            host (Optional `str`, defaults to localhost):
                host of where ElasticSearch is running
            port (Optional `str`, defaults to 9200):
                port of where ElasticSearch is running
            es_client (Optional `elasticsearch.Elasticsearch`):
                The elasticsearch client used to create the index if host and port are None.
            es_index_name (Optional `str`): The elasticsearch index name used to create the index.
            es_index_config (Optional `dict`):
                The configuration of the elasticsearch index.
                Default config is:

        Config::

            {
                "settings": {
                    "number_of_shards": 1,
                    "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "similarity": "BM25"
                        },
                    }
                },
            }
        """
        index_name = index_name if index_name is not None else column
        es_index = ElasticSearchIndex(
            host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config
        )
        es_index.add_documents(self, column=column)
        self._indexes[index_name] = es_index

    def load_elasticsearch_index(
        self,
        index_name: str,
        es_index_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["Elasticsearch"] = None,
        es_index_config: Optional[dict] = None,
    ):
        """Load an existing text index using ElasticSearch for fast retrieval.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index. This is the index name that is used to call `get_nearest` or `search`.
            es_index_name (`str`):
                The name of elasticsearch index to load.
            host (`str`, *optional*, defaults to `localhost`):
                Host of where ElasticSearch is running.
            port (`str`, *optional*, defaults to `9200`):
                Port of where ElasticSearch is running.
            es_client (`elasticsearch.Elasticsearch`, *optional*):
                The elasticsearch client used to create the index if host and port are `None`.
            es_index_config (`dict`, *optional*):
                The configuration of the elasticsearch index.
                Default config is:
                    ```
                    {
                        "settings": {
                            "number_of_shards": 1,
                            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                        },
                        "mappings": {
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard",
                                    "similarity": "BM25"
                                },
                            }
                        },
                    }
                    ```
        """
        self._indexes[index_name] = ElasticSearchIndex(
            host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config
        )

    def add_milvus_index(
        self,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        milvus_collection_schema: Optional[list] = None,
        milvus_index_params: Optional[dict] = None,
        batch_size: int = 1000,
    ):
        """Add an index using Milvus for fast retrieval.

        Args:
            index_name (Optional `str`): The index_name/identifier of the index.
                Will create a collection with this name in Milvus and build index on it.
            host (Optional `str`, defaults to localhost):
                host of where ElasticSearch is running
            port (Optional `str`, defaults to 19530):
                port of where ElasticSearch is running
            milvus_collection_schema (Optional `List`):
                The schema of the Milvus collection.
                Milvus support multi-columns schema of various datatype and hybrid search with attribute filtering.
                The collection to create must contain a primary key field and a vector field.
                INT64 and String are supported data type on primary key field.
                Default schema is:
                {
                    "name": "pk",
                    "type": "INT64",
                    "is_primary": True,
                    "auto_id": False,
                },
                {
                    "name": "embeddings",
                    "type": "FLOAT_VECTOR",
                    "params": {
                        "dim": 128
                    }
                }
            milvus_index_params (Optional `dict`):
                The configuration of the Milvus index.
                Default config is:
                {
                    "field": "embeddings",
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128},
                }
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the Milvus. Default is 1000.
        """
        milvus_index = MilvusIndex(
            host=host,
            port=port,
            milvus_collection_name=index_name,
            milvus_collection_schema=milvus_collection_schema,
            milvus_index_params=milvus_index_params
        )

        milvus_index.insert(self, batch_size=batch_size)
        self._indexes[index_name] = milvus_index

    def add_milvus_index_from_external_arrays(
        self,
        external_arrays: Optional[np.array] = None,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        milvus_collection_schema: Optional[list] = None,
        milvus_index_params: Optional[dict] = None,
        batch_size: int = 1000,
        dtype=np.float32,
    ):
        """Add an index using Milvus for fast retrieval.

        Args:
            external_arrays (`np.array`):
                Data arrays to insert into the index.
                You can also insert by calling milvus_index.insert(arrays) after initialize
            index_name (Optional `str`): The index_name/identifier of the index.
                Will create a collection with this name in Milvus and build index on it.
            host (Optional `str`, defaults to localhost):
                host of where ElasticSearch is running
            port (Optional `str`, defaults to 19530):
                port of where ElasticSearch is running
            milvus_collection_schema (Optional `List`):
                The schema of the Milvus collection.
                Milvus support multi-columns schema of various datatype and hybrid search with attribute filtering.
                The collection to create must contain a primary key field and a vector field.
                INT64 and String are supported data type on primary key field.
                Default schema is:
                {
                    "name": "pk",
                    "type": "INT64",
                    "is_primary": True,
                    "auto_id": False,
                },
                {
                    "name": "embeddings",
                    "type": "FLOAT_VECTOR",
                    "params": {
                        "dim": 128
                    }
                }
            milvus_index_params (Optional `dict`):
                The configuration of the Milvus index.
                Default config is:
                {
                    "field": "embeddings",
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128},
                }
            batch_size (Optional `int`): Size of the batch to use while adding vectors to the Milvus. Default is 1000.
            dtype (`numpy.dtype`):
                The dtype of the numpy arrays that are indexed. Default is np.float32.
        """
        milvus_index = MilvusIndex(
            host=host,
            port=port,
            milvus_collection_name=index_name,
            milvus_collection_schema=milvus_collection_schema,
            milvus_index_params=milvus_index_params
        )
        if external_arrays is not None:
            milvus_index.insert(external_arrays.astype(dtype), batch_size=batch_size)
        self._indexes[index_name] = milvus_index

    def drop_index(self, index_name: str):
        """Drop the index with the specified column.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
        """
        del self._indexes[index_name]

    def search(self, index_name: str, query: Union[str, np.array], k: int = 10) -> SearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            - scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            - indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search(query, k)

    def search_batch(self, index_name: str, queries: Union[List[str], np.array], k: int = 10) -> BatchedSearchResults:
        """Find the nearest examples indices in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            - total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            - total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        self._check_index_is_initialized(index_name)
        return self._indexes[index_name].search_batch(queries, k)

    def get_nearest_examples(
        self, index_name: str, query: Union[str, np.array], k: int = 10
    ) -> NearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The index_name/identifier of the index.
            query (`Union[str, np.ndarray]`):
                The query as a string if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve.

        Returns:
            - scores (`List[float]`): The retrieval scores of the retrieved examples.
            - examples (`dict`): The retrieved examples.
        """
        self._check_index_is_initialized(index_name)
        scores, indices = self.search(index_name, query, k)
        top_indices = [i for i in indices if i >= 0]
        return NearestExamplesResults(scores[: len(top_indices)], self[top_indices])

    def get_nearest_examples_batch(
        self, index_name: str, queries: Union[List[str], np.array], k: int = 10
    ) -> BatchedNearestExamplesResults:
        """Find the nearest examples in the dataset to the query.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index.
            queries (`Union[List[str], np.ndarray]`):
                The queries as a list of strings if `index_name` is a text index or as a numpy array if `index_name` is a vector index.
            k (`int`):
                The number of examples to retrieve per query.

        Returns:
            - total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            - total_examples (`List[dict]`): The retrieved examples per query.
        """
        self._check_index_is_initialized(index_name)
        total_scores, total_indices = self.search_batch(index_name, queries, k)
        total_scores = [
            scores_i[: len([i for i in indices_i if i >= 0])]
            for scores_i, indices_i in zip(total_scores, total_indices)
        ]
        total_samples = [self[[i for i in indices if i >= 0]] for indices in total_indices]
        return BatchedNearestExamplesResults(total_scores, total_samples)
