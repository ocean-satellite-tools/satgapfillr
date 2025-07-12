import satgapfillr as sgf
import xarray as xr

def test_read_data():
    # Create a dummy dataset and save as NetCDF
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    ds.to_netcdf("test.nc")

    # Use your package to read it
    loaded = sgf.read_data("test.nc")
    assert "a" in loaded

def test_unet_class():
    model = sgf.UNetGapFiller(input_shape=(64, 64, 4))
    assert model.model is not None