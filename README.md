# Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization

This is the code to reproduce the experiments of the ICML 2022 paper "[Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization](https://proceedings.mlr.press/v162/javaloy22a.html)". If you have any doubts/problems, feel free to open an issue. If you want to cite the paper, you can find a bibtex entry [here](#Citing).

## Datasets

The datasets for the heterogeneous experiments are taken from [UCI](https://archive.ics.uci.edu/ml/datasets.php) and [R package datasets](https://vincentarelbundock.github.io/Rdatasets/datasets.html).

## Dependencies

- Python v3.8
- PyTorch v1.10.1
- Pytorch-lightning v1.5.9

## Requirements
To set up the conda environment, run:

```{bash}
conda env create -f environment.yml    # create conda env
conda activate impartial               # activate conda env
```

## Usage

To run the heterogeneous experiment we first need to enter that folder:

```{bash}
cd heterogeneous
```

You can find information about all the available arguments via `python main.py --help`.  For example:

- To train the baseline _ELBO-VAE_ on the _Wine_ dataset (vanilla):

  ```{bash}
  python main.py -model=vae -dataset=datasets/Wine -batch-size=128 -latent-perc=50 -hidden-size=50 \
  -seed=1 -model=vae -miss-perc=20 -miss-suffix=1
  ```
  
- To train the baseline _ELBO-VAE_ on the _Wine_ dataset with GradNorm (α=0) (our framework):

    ```{bash}
    python main.py -model=vae -dataset=datasets/Wine -batch-size=128 -latent-perc=50 -hidden-size=50 \
    -seed=1 -model=vae -miss-perc=20 -miss-suffix=1 -methods gradnorm -alpha 0.0
    ```
  
- To train _ELBO-IWAE_ on the _Wine_ dataset with _GradNorm-PCGrad (α=1)_ as an MTL method (our framework), run:

  ```{bash}
  python main.py -model=iwae -dataset=datasets/Wine -batch-size=128 -latent-perc=50 -hidden-size=50 \
  -seed=1 -model=vae -miss-perc=20 -miss-suffix=1 -methods gradnorm pcgrad -alpha=1.0
  ```
  
- To train _ELBO-DReG_ on the _Wine_ dataset with _IMTL-G_ as an MTL method (our framework), run:
   ```{bash}
  python main.py -model=dreg -dataset=datasets/Wine -batch-size=128 -latent-perc=50 -hidden-size=50 \
  -seed=1 -model=vae -miss-perc=20 -miss-suffix=1 -methods imtl-g
  ```
  
- To train _HI-VAE_ on the _Wine_ dataset with _MGDA-UB_ as an MTL method (our framework), run:
   ```{bash}
  python main.py -model=hivae -dataset=datasets/Wine -batch-size=1000 -max-epochs=2000 -size-z=10 -size-y=5 -size-s=10 \
  -seed=1 -miss-perc=20 -miss-suffix=1 -methods mgda
  ```



## Multi-modal

We also provide the code for multi-modal data (adapted from [the mmvae repository](https://github.com/iffsid/mmvae)). More dependencies would need to be installed in order to run this code (the same dependencies as those from [the original mmvae code](https://github.com/iffsid/mmvae)). Moreover, as seen in the main paper, these experiments take a significant amount of hours to run.



## Citing
```bibtex
@InProceedings{pmlr-v162-javaloy22a,
  title = {Mitigating Modality Collapse in Multimodal {VAE}s via Impartial Optimization},
  author = {Javaloy, Adrian and Meghdadi, Maryam and Valera, Isabel},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages =  {9938--9964},
  year = {2022},
  volume = {162},
  series = {Proceedings of Machine Learning Research},
  month = {17--23 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v162/javaloy22a/javaloy22a.pdf},
  url = {https://proceedings.mlr.press/v162/javaloy22a.html}
}
```
