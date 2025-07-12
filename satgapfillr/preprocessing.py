from xbatcher import BatchGenerator

def make_batcher(ds, input_dims, input_overlap, batch_size):
    return BatchGenerator(ds, input_dims=input_dims, input_overlap=input_overlap, batch_size=batch_size)

def compute_norm_stats(ds, var_names):
    mean = ds[var_names].mean(dim='time')
    std = ds[var_names].std(dim='time')
    return mean, std

def make_tf_dataset(batcher):
    return batcher.to_tf_dataset()

def extract_patches(ds, patch_size):
    # Placeholder for patchify fallback
    return ds

def normalize(ds):
    # Placeholder for simple normalization
    return ds
