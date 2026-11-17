# OSCD100 real satellite sample

This folder contains a small RGB preview pair made from the real Sentinel-2
imagery in `train_000` of the [OSCD100 dataset](https://huggingface.co/datasets/hkristen/oscd100).
The original pair is dated 1 August 2015 and 1 August 2017 and is 256 × 256 pixels.

The PNGs combine the native 10 m bands B04 (red), B03 (green), and B02 (blue).
A shared 2nd–98th percentile stretch was applied across both dates for display.
The dashboard uses 10 m per pixel when this sample is loaded. They are intended
for interactive EarthChange demos; they are not the original multispectral
GeoTIFFs and should not be used for benchmark scoring.

Attribution: OSCD by Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch,
and Yann Gousseau (2018); OSCD100 subset curated by hkristen. The dataset card
identifies the data as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This adapted image pair is provided under the same license. Changes: converted
three Sentinel-2 bands to RGB PNG and applied the shared display stretch above.

Original paper: Daudt et al., “Urban Change Detection for Multispectral Earth
Observation Using Convolutional Neural Networks,” IGARSS 2018.
