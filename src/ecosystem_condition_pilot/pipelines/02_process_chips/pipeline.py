
from kedro.pipeline import Pipeline, node
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from .nodes import encode_l7_stacks, fill_l7_stacks, concat_dataset

def create_pipeline(**kwargs) -> Pipeline:
    params = OmegaConfigLoader(settings.CONF_SOURCE)["parameters"]
    return Pipeline(
        [
            node(
                func=fill_l7_stacks,
                name="fill_l7_stacks",
                inputs=["predicts_sites", "landsat7_stacks", "landsat7_qa_stacks"],
                outputs="landsat7_chips_filled",
            ),
            node(
                func=encode_l7_stacks,
                name="encode_l7_stacks",
                inputs="landsat7_chips_filled",
                outputs="landsat7_chips_encoded",
            ),
            node(
                concat_dataset,
                name="concat_dataset",
                inputs="landsat7_chips_encoded",
                outputs="landsat7_encoded_chips",
            )
        ]
    )
