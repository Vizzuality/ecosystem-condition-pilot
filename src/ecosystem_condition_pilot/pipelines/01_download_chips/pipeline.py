
from uu import encode
from kedro.pipeline import Pipeline, node
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from .nodes import download_l7_stacks, query_stac_items

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=query_stac_items,
                name="query_stac_items",
                inputs="predicts_sites",
                outputs="landsat7_stac_items",
            ),
            node(
                func=download_l7_stacks,
                name="download_l7",
                inputs=["predicts_sites", "landsat7_stac_items"],
                outputs=["landsat7_stacks", "landsat7_qa_stacks"],
            ),
        ]
    )
