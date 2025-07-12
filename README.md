# satgapfillr

**satgapfillr** is an open-source Python toolkit for training and applying **UNet-based models** to gap-fill missing data in remote sensing datasets.  It helps you handle common gaps due to clouds, sensor dropouts, or quality screening — and lets you fit *location-specific models* using your own time series.

UNet models are particularly powerful for this problem because they combine local context with multi-scale information, thanks to their encoder–decoder architecture with skip connections. This structure lets the model learn both fine-grained spatial features (like sharp edges or local patterns) and broad regional structures, which is critical when inferring missing values in gridded Earth observation data. UNets have become a go-to approach for tasks like cloud removal, image inpainting, and environmental data reconstruction because they naturally preserve spatial coherence while filling gaps.

Gaps in remote sensing data — like clouds or sensor dropouts — can severely limit what you can analyze or model. In many cases, reliable gap-filled products simply don’t exist at all, which means you’re stuck masking out large swaths of valuable information. satgapfillr puts powerful, customizable gap-filling directly in your hands — so you don’t have to wait (and hope) for someone else to publish a global gap-filled dataset that may not fit your region, timeframe, or research question. Instead, you can train your own local model on your own data, tailored to your specific needs, with flexibility to adjust and improve the training approach over time.

*With satgapfillr, you can easily*:

- Load your own data in Zarr or NetCDF formats
- Preprocess with xarray and xbatcher to handle larger-than-memory time series
- Train a UNet on your domain with realistic missingness (e.g. simulate clouds)
- Validate performance with skill metrics
- Provide metrics for your gap-filling performance to demonstrate (to readers and reviewers) that you gap-filling process is working effectively
- Apply the trained model to fill gaps
- Save the results back to Zarr or NetCDF for analysis and re-use

---

## Installation

```bash
# Clone and install in editable/development mode
git clone https://github.com/ocean-satellite-tools/satgapfillr.git
cd satgapfillr
pip install -e .
```

Requirements: Python 3.8+, `xarray`, `zarr`, `xbatcher`, `TensorFlow`, `scikit-learn`.

---

## Quickstart

Below is an example of how you might train a gap-filling UNet on your data.

```python
import satgapfillr as sgf

# Load your dataset from Zarr or NetCDF
ds = sgf.read_data("my-data.zarr")

# Simulate realistic missingness using time-lagged cloud mask
mask = sgf.simulate_cloud_mask(ds, lag_days=10)

# Preprocess: extract patches (small) or use xbatcher (large)
if ds.nbytes < 1e10:  # ~10GB
    patches = sgf.extract_patches(ds, patch_size=(64, 64))
    norm_patches = sgf.normalize(patches)
    X_train = ...  # your code to get X, y from patches
else:
    batcher = sgf.make_batcher(ds, input_dims={"time": 1}, input_overlap={"time": 0}, batch_size=32)
    tf_dataset = sgf.make_tf_dataset(batcher)

# Build your UNet
model = sgf.UNetGapFiller(input_shape=(64, 64, 4))  # adjust channels!

# Fit the model
if ds.nbytes < 1e10:
    model.fit(X_train, y_train, epochs=10)
else:
    model.fit(tf_dataset, epochs=10)

# Apply the model to fill gaps
gapfilled = sgf.fill_gaps(model, ds)

# Save the result
sgf.save_data(gapfilled, "my-gapfilled-data.zarr")
```

---

## Example Notebooks

See the [`examples/`](examples/) folder for a live notebook showing how to:
- Load a sample Zarr
- Simulate missingness
- Use `xbatcher` for large training sets
- Train a UNet
- Validate results and visualize skill scores

See https://ocean-satellite-tools.github.io/mind-the-chl-gap for a tutorials.

---

## Project Status

First Skeleton.

---

## License

Apache 2.0 — free to use, share, and adapt.

---

## Contributing

This project is open to improvements and extensions — PRs welcome!  
Please file an issue if you find a bug or have an idea to make it more useful for your regional remote sensing workflows.

---

## Contact

Eli Holmes NOAA Fisheries

Original models and concept developed by Yifei Hang, Shridhar Sinha, and Jiarui Yu using an Indian Ocean dataset developed by Minh Phan. All were University of Washington CS and Applied Math undergrads participating in the Varanasi Internship. Their work was extended by Eli Holmes into the satgapfillr package.

