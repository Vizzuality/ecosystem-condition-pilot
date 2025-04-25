
from kedro.pipeline import Pipeline, node

from .nodes import download_stacks, query_stac_items, filter_s2_stacks, encode_s2_stacks, concat_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=query_stac_items,
                name="query_stac_items_s2",
                inputs="predicts_sites_usa",
                outputs="s2_stac_items",
            ),
            node(
                func=download_stacks,
                name="download_stacks_s2",
                inputs=["predicts_sites_usa", "s2_stac_items"],
                outputs="s2_stacks",
            ),
            node(
                func=filter_s2_stacks,
                name="filter_s2_stacks",
                inputs=["predicts_sites_usa", "s2_stacks"],
                outputs="s2_stacks_filtered",
            ),
            node(
                func=encode_s2_stacks,
                name="encode_s2_stacks",
                inputs=["predicts_sites_usa", "s2_stacks_filtered"],
                outputs="s2_chips_encoded",
            ),
            node(
                func=concat_dataset,
                name="concat_dataset_s2",
                inputs="s2_chips_encoded",
                outputs="s2_encoded_chips",
            ),
        ]
    )
