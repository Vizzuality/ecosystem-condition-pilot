import xarray
from kedro.io.core import get_filepath_str
from kedro_datasets_experimental.rioxarray import geotiff_dataset


class GeoTIFFDataset(geotiff_dataset.GeoTIFFDataset):
    def _save(self, data: xarray.DataArray) -> None:
        self._sanity_check(data)
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        data.rio.to_raster(save_path, **self._save_args)
        self._fs.invalidate_cache(save_path)
