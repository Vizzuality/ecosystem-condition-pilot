from typing import Callable

import pandas as pd
import planetary_computer
import pyproj
import pystac
import pystac_client
import stackstac
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1/"
COLLECTION = "landsat-c2-l2"
BANDS = ["blue", "green", "red", "nir08", "swir16"]
SEARCH_DAYS = 90
CHIP_SIZE = 256
GSD = 30
CLOUDMAX = 40
DATA_THRESHOLD = 0.35

DEVICE = "mps"  # Replace with "cuda" if you have a GPU
ENCODER = "data/06_models/clay_encoder_mps.pt"
PLATFORM_METADATA = "conf/clay/metadata.yaml"
PLATFORM = "landsat-c2l2-sr"

def _search_stac_items(lat, lon, date):
    # Search the catalogue
    start = (pd.to_datetime(date) - pd.Timedelta(days=SEARCH_DAYS/2)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(date) + pd.Timedelta(days=SEARCH_DAYS/2)).strftime("%Y-%m-%d")
    stac_api_io = StacApiIO(max_retries=Retry(total=5, backoff_factor=5))
    catalog = pystac_client.Client.open(STAC_API, stac_io=stac_api_io)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": CLOUDMAX}, "platform": {"eq": "landsat-7"}},
    )
    items = search.item_collection()

    return items

def _download_l7_stac(item, bounds):
    crs = item.properties["proj:epsg"]
    item = planetary_computer.sign(item)
    stack = stackstac.stack(
        item,
        bounds=bounds,
        snap_bounds=False,
        epsg=crs,
        resolution=GSD,
        dtype="float32", # type: ignore
        rescale=False,
        fill_value=0,
        assets=BANDS,
    )

    stack.compute()
    stack = stack.isel(time=0)
    stack.attrs.update({k.replace(":","_"):v for k,v in item.properties.items()})
    return stack

def _download_qa_stac(item, bounds):
    crs = item.properties["proj:epsg"]
    item = planetary_computer.sign(item)
    stack = stackstac.stack(
        item,
        bounds=bounds,
        snap_bounds=False,
        epsg=crs,
        resolution=GSD,
        dtype="uint16", # type: ignore
        rescale=False,
        fill_value=1,
        assets=["qa_pixel"],
    )
    stack.compute()
    stack = stack.isel(time=0)
    stack.attrs.update({k.replace(":","_"):v for k,v in item.properties.items()})
    stack.rio.write_nodata(1, inplace=True)
    return stack



def query_stac_items(points):
    """Query the STAC API for items."""
    return {
        point.SSBS: lambda p=point: _search_stac_items(
            p.Latitude, p.Longitude, p.Sample_midpoint).to_dict()
        for point in points.itertuples()
    }


def download_l7_stacks(
        points: pd.DataFrame,
        stac_items: dict[str, Callable[[], dict]],
    ):
    """Download Landsat 7 chips for each point in the input DataFrame."""
    item_args = {}
    for point in points.itertuples():
        stac_dict = stac_items.get(point.SSBS, lambda: {})()  # type: ignore
        if not stac_dict:
            continue
        items = pystac.ItemCollection.from_dict(stac_dict)
        for item in items:
            partition_id = f"{point.SSBS}..{item.properties['datetime']}"
            crs = item.properties["proj:epsg"]
            x, y = pyproj.Proj(crs).transform(point.Longitude, point.Latitude)
            buffer = (CHIP_SIZE * GSD) // 2
            bounds = (x - buffer, y - buffer, x + buffer, y + buffer)

            item_args[partition_id] = (item, bounds)

    l7_chips = {
        partition_id: lambda x=args: _download_l7_stac(*x)
        for partition_id, args in item_args.items()
    }
    l7_qa = {
        partition_id: lambda x=args: _download_qa_stac(*x)
        for partition_id, args in item_args.items()
    }

    return l7_chips, l7_qa

