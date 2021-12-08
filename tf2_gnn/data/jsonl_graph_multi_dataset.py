"""General dataset class for datasets with numeric properties stored as JSONLines files."""
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import GraphBatchTFDataDescription, GraphSample, GraphSampleType
from .jsonl_graph_dataset import JsonLGraphDataset


class GraphWithPropertiesSample(GraphSample):
    """Data structure holding a single graph with a multiple numeric properties."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: List[np.ndarray],
        target_value: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._target_value = target_value

    @property
    def target_value(self) -> np.ndarray:
        """Target value of the regression task."""
        return self._target_value

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"Target_value:   {self._target_value}"
        )


GraphWithPropertiesSampleType = TypeVar(
    "GraphWithPropertiesSampleType", bound=GraphWithPropertiesSample
)


class JsonLGraphPropertiesDataset(JsonLGraphDataset[GraphWithPropertiesSampleType]):
    """
    General class representing pre-split datasets in JSONLines format.
    Concretely, this class expects the following:
    * In the data directory, files "train.jsonl.gz", "valid.jsonl.gz" and
      "test.jsonl.gz" are used to store the train/valid/test datasets.
    * Each of the files is gzipped text file in which each line is a valid
      JSON dictionary with a "graph" key, which in turn points to a
      dictionary with keys
       - "node_features" (list of numerical initial node labels)
       - "adjacency_lists" (list of list of directed edge pairs)
      Addtionally, the dictionary has to contain a "Property" key with a
      vector of floating point values.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            # If None, the data-provided property is used; otherwise, a floating point
            # value is expected and property values greater than this value will be
            # encoded as 1.0 and smaller values will be encoded as 0.0.
            "threshold_for_classification": None,
            "add_self_loop_edges": False,
            "tie_fwd_bkwd_edges": False,
            "num_fwd_edge_types": 8
        }
        super_hypers.update(this_hypers)
        return super_hypers

    def __init__(
        self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ):
        super().__init__(params, metadata=metadata, **kwargs)
        self._threshold_for_classification = params["threshold_for_classification"]
        self._label_size = None
        self._one_counts = None
        self._total_counts = 0

    def _process_raw_datapoint(
        self, datapoint: Dict[str, Any]
    ) -> GraphWithPropertiesSampleType:
        node_features = datapoint["graph"]["node_features"]
        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists=datapoint["graph"]["adjacency_lists"],
            num_nodes=len(node_features),
        )

        target_value = np.array(datapoint["Property"], dtype='float64')
        if self._label_size is None:
            self._label_size = target_value.shape[0]
            self._one_counts = np.zeros((self._label_size, ), dtype='float64')
        else:
            assert self._label_size == target_value.shape[0]
        if self._threshold_for_classification is not None:
            target_value = np.array(
                target_value > self._threshold_for_classification, dtype='float64')
        self._one_counts += target_value
        self._total_counts += 1

        return GraphWithPropertiesSample(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
            node_features=node_features,
            target_value=target_value,
        )

    def _begin_load_data(self):
        if self._label_size is not None:
            self._one_counts = np.zeros((self._label_size, ), dtype='float64')
            self._total_counts = 0

    def _end_load_data(self):
        print("Label statistics: {}".format(self._one_counts / self._total_counts))

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["target_value"] = []
        return new_batch

    def _add_graph_to_batch(
        self, raw_batch: Dict[str, Any], graph_sample: GraphWithPropertiesSampleType
    ) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["target_value"].append(graph_sample.target_value)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        return batch_features, {"target_value": raw_batch["target_value"]}

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        # calls to graph_dataset.py:get_batch_tf_data_description
        data_description = super().get_batch_tf_data_description()
        assert self._label_size is not None
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "target_value": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "target_value": (None, self._label_size)},
        )
