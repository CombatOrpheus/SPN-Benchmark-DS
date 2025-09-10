# Configuration Options

This file explains the configuration options available in the `*.toml` files in this directory.

## `PartitionGrid.toml`

This file is used by the `ObtainGridDS.py` script to control the generation of grid-based datasets.

- `raw_data_location`: The path to the raw data JSON file.
- `temporary_grid_location`: The directory to store the temporary grid data.
- `accumulation_data`: A boolean indicating whether to accumulate data or start fresh.
- `places_upper_limit`: The upper limit for the number of places in the Petri net.
- `markings_upper_limit`: The upper limit for the number of markings in the Petri net.
- `samples_per_grid`: The number of samples to take from each grid cell.
- `lambda_variations_per_sample`: The number of lambda variations to generate for each sample.
- `output_grid_location`: The directory to save the final grid dataset.

## `SPNGenerate.toml`

This file is used by the `SPNGenerate.py` script to control the generation of Stochastic Petri Net (SPN) datasets.

- `output_data_location`: The directory where the generated data will be saved.
- `place_upper_bound`: The upper bound for the number of places in the SPN.
- `marks_lower_limit`: The lower limit for the number of markings in the SPN.
- `marks_upper_limit`: The upper limit for the number of markings in the SPN.
- `number_of_parallel_jobs`: The number of parallel jobs to use for data generation.
- `number_of_samples_to_generate`: The number of SPN samples to generate.
- `enable_pruning`: A boolean indicating whether to prune the SPN.
- `enable_token_addition`: A boolean indicating whether to add tokens to the SPN.
- `maximum_number_of_places`: The maximum number of places in the SPN.
- `minimum_number_of_places`: The minimum number of places in the SPN.
- `enable_visualization`: A boolean indicating whether to generate visualizations of the SPNs.
- `visualization_output_location`: The directory to save the visualizations.
- `enable_transformations`: A boolean indicating whether to apply transformations to the SPNs.
- `maximum_transformations_per_sample`: The maximum number of transformations to apply to each SPN.
- `output_file`: The name of the output HDF5 file.
