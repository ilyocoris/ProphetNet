## Plan

1. Build algorithmic datasets in the proper format
2. Optional: tokenization, reduce embedding layer to only relevant tokens?
3. Change `CrossAttention_Diffusion_LM` in `create_model` to have only 1 layer in the encoder & 1 layer in the denoiser.

## Things that have been done

Delete the `org_data`