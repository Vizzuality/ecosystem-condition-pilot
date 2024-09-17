from typing import Callable, Mapping
import pandas as pd
import pystac_client
import stackstac
import xarray as xr
import planetary_computer
import cv2
import numpy as np
import torch
import math
import yaml
import pyproj
from box import Box
from torchvision.transforms import v2
from stacchip.processors.prechip import normalize_timestamp



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


def _post_filter_clouds(stack: xr.DataArray, qa_stack: xr.DataArray):
    """Filter scenes with too much cloud cover (or nodata) within the image chip."""
    # bits 1, 3, and 4 are dilated cloud, cloud, and cloud shadow
    # 0b..10000 cloud shadow
    # 0b..01000 cloud
    # 0b..00010 dilated cloud
    # 0b..00001 data fill
    mask = (qa_stack & 0b0000000000011011) > 0
    data_ratio = mask.mean(dim=["band", "x", "y"])
    stack.attrs["data_ratio"] = data_ratio
    return stack[data_ratio < DATA_THRESHOLD]



def _inpaint_l7(arr):
    data = arr.data.reshape(arr.shape[-1], arr.shape[-2])
    mask = (data==0).astype(np.uint8)
    data = cv2.inpaint(data, mask, 3, cv2.INPAINT_TELEA)
    arr.data = data.reshape(arr.shape)
    return arr


def fill_l7_stacks(
        points: pd.DataFrame,
        chip_loadfuncs: Mapping[str, Callable[[], xr.DataArray]],
        qa_chip_loadfuncs: Mapping[str, Callable[[], xr.DataArray]]
    ):
    """Fill missing values in the Landsat 7 stacks using image inpainting."""

    def _process_chip(partition_ids):
        qa_chips = [qa_chip_loadfuncs[i]() for i in partition_ids]
        qa_chip_stack = xr.concat(qa_chips, dim="time", join="override")
        mask = (qa_chip_stack & 0b0000000000011011) > 0
        data_ratio = mask.mean(dim=["band", "x", "y"])
        chip_id = partition_ids[int(data_ratio.argmin())]

        chip = chip_loadfuncs[chip_id]()
        chip.attrs["data_ratio"] = data_ratio.min()
        chunked = chip.chunk({"band":1})
        chip = chunked.map_blocks(_inpaint_l7, template=chunked)

        return chip

    data = {}
    for point in points.itertuples():
        partition_ids = [i for i in qa_chip_loadfuncs.keys() if i.split("..")[0] == point.SSBS]
        if len(partition_ids):
            data[point.SSBS] = lambda ids=partition_ids: _process_chip(ids)
    return data


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
    for band in BANDS:
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


def encode_l7_stacks(chip_loadfuncs: Mapping[str, Callable[[], xr.DataArray]]):
    def _encode(name, stack):
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

    return {
        k: lambda k=k, v=v: _encode(k, v())
        for k, v in chip_loadfuncs.items()
    }

def concat_dataset(
    encoded_chips: Mapping[str, Callable[[], pd.DataFrame]]
):
    return pd.concat([v() for v in encoded_chips.values()])