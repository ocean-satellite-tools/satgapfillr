import xarray as xr
import fsspec

def read_data(path):
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        return xr.open_zarr(fs.get_mapper(path))
    elif path.endswith(".zarr"):
        return xr.open_zarr(path)
    else:
        return xr.open_dataset(path)

def save_data(ds, path):
    ds.to_zarr(path)
