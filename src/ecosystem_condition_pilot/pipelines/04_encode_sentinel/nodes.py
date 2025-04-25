import math
from typing import Callable, Mapping

import stackstac
import numpy as np
import pandas as pd
import planetary_computer
import pyproj
import pystac
import pystac_client
import torch
import xarray as xr
import yaml
from box import Box
from pystac_client.stac_api_io import StacApiIO
from rasterio import transform
from stacchip.processors.prechip import normalize_timestamp
from torchvision.transforms import v2
from urllib3 import Retry

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1/"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A"]
BAND_NAMES = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08"]

CHIP_SIZE = 256
GSD = 10

DEVICE = "mps"  # Replace with "cuda" if you have a GPU
ENCODER = "data/06_models/clay_encoder_mps.pt"
PLATFORM_METADATA = "conf/clay/metadata.yaml"
PLATFORM = "sentinel-2-l2a"
CLOUDMAX = 10
SEARCH_DAYS = 60


def _search_stac_items(lat, lon, date):
    # Search the catalogue
    start = (pd.to_datetime(date).replace(year=2017) - pd.Timedelta(days=SEARCH_DAYS/2)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(date).replace(year=2017) + pd.Timedelta(days=SEARCH_DAYS/2)).strftime("%Y-%m-%d")
    stac_api_io = StacApiIO(max_retries=Retry(total=5, backoff_factor=5))
    catalog = pystac_client.Client.open(STAC_API, stac_io=stac_api_io)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": CLOUDMAX}},
    )
    items = search.item_collection()
    items = sorted(items, key=lambda item: item.properties.get("eo:cloud_cover", float("inf")))
    items = pystac.ItemCollection(items)
    return items


def _download_stac(item, bounds):
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


def query_stac_items(points):
    """Query the STAC API for items."""
    return {
        point.SSBS: lambda p=point: _search_stac_items(p.Latitude, p.Longitude, p.Sample_midpoint).to_dict()
        for point in points.itertuples()
    }


def download_stacks(
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

    chips = {
        partition_id: lambda x=args: _download_stac(*x)
        for partition_id, args in item_args.items()
    }

    return chips



def _normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def _prep_datacube(stack: xr.DataArray, time: str):
    metadata = Box(yaml.safe_load(open(PLATFORM_METADATA)))
    x1, y1, x2, y2 = stack.rio.transform_bounds('epsg:4326')
    lon, lat = (x1+x2)/2, (y1+y2)/2

    # Normalize pixels
    mean = []
    std = []
    waves = []
    for band in BAND_NAMES:
        mean.append(metadata[PLATFORM].bands.mean[str(band)])
        std.append(metadata[PLATFORM].bands.std[str(band)])
        waves.append(metadata[PLATFORM].bands.wavelength[str(band)])
    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])
    pixels = torch.from_numpy(stack.expand_dims('time', 0).to_numpy().astype(np.float32))
    pixels = transform(pixels)

    # Prep datetimes embedding using a normalization function from the model code.
    times = [normalize_timestamp(pd.to_datetime(time))]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    latlons = [_normalize_latlon(lat, lon)]
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    gsd = abs(stack.rio.resolution()[0])

    return {
        "pixels": pixels.to(DEVICE),
        "time": torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=DEVICE),
        "latlon": torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=DEVICE),
        "wavelengths": torch.tensor(waves, device=DEVICE),
        "gsd": torch.tensor([gsd], device=DEVICE),
    }



def filter_s2_stacks(
        points: pd.DataFrame,
        chip_loadfuncs: Mapping[str, Callable[[], xr.DataArray]]
    ):
    data = {}
    for point in points.itertuples():
        partition_ids = [k for k in chip_loadfuncs.keys() if k.startswith(point.SSBS)]
        if partition_ids:
            i = partition_ids[-1] # take most recent chips
            data[point.SSBS] = lambda f=chip_loadfuncs[i]: f()

    return data


def encode_s2_stacks(
        points: pd.DataFrame,
        chip_loadfuncs: Mapping[str, Callable[[], xr.DataArray]]
    ):
    def _encode(name, stack):
        if stack.shape[0] < 4:
            return pd.DataFrame({name: [np.nan]}).transpose()
        encoder = torch.export.load(ENCODER).module()
        datacube = _prep_datacube(stack, stack.attrs['datetime'])

        with torch.inference_mode():
            results = encoder(
                datacube["pixels"],
                datacube["time"],
                datacube["latlon"],
                datacube["wavelengths"],
                datacube["gsd"],
            )
        embeddings = results[0][:, 0, :].cpu().numpy().flatten()
        df = pd.DataFrame({name: embeddings}).transpose()
        return df

    data = {}
    for point in points.itertuples():
        partition_ids = [k for k in chip_loadfuncs.keys() if k.startswith(point.SSBS)]
        if partition_ids:
            i = partition_ids[-1] # take most recent chips
            data[point.SSBS] = lambda k=point.SSBS, f=chip_loadfuncs[i]: _encode(k, f())

    return data

def concat_dataset(
    encoded_chips: Mapping[str, Callable[[], pd.DataFrame]]
):
    return pd.concat([v() for v in encoded_chips.values()])