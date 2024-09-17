
from kedro.pipeline import Pipeline, node

from .nodes import download_stacks, query_stac_items, filter_naip_stacks, encode_naip_stacks, concat_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=query_stac_items,
                name="query_stac_items_naip",
                inputs="predicts_sites_usa",
                outputs="naip_stac_items",
            ),
            node(
                func=download_stacks,
                name="download_stacks_naip",
                inputs=["predicts_sites_usa", "naip_stac_items"],
                outputs="naip_stacks",
            ),
            node(
                func=filter_naip_stacks,
                name="filter_naip_stacks",
                inputs=["predicts_sites_usa", "naip_stacks"],
                outputs="naip_stacks_filtered",
            ),
            node(
                func=encode_naip_stacks,
                name="encode_naip_stacks",
                inputs=["predicts_sites_usa", "naip_stacks_filtered"],
                outputs="naip_chips_encoded",
            ),
            node(
                func=concat_dataset,
                name="concat_dataset_naip",
                inputs="naip_chips_encoded",
                outputs="naip_encoded_chips",
            ),
        ]
    )
