# Joint image restoration and contour detection using Discrete Mumford-Shah with Ambrosio-Tortorelli penalization
> **Hoang Trieu Vy Le, [Nelly Pustelnik](https://perso.ens-lyon.fr/nelly.pustelnik/), [Marion Foare](https://perso.ens-lyon.fr/marion.foare/),**
*Proximal Based Strategies for Solving Discrete Mumford-Shah With Ambrosio-Tortorelli Penalization on Edges,*
EUSIPCO 2022, [Download](https://ieeexplore.ieee.org/abstract/document/9723590)

## <div align="center">Quick start Examples </div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt]() in a
[**Python>=3.7.0**](https://www.python.org/) environment,


```bash
git clone https://github.com/HoangTrieuVy/GGS-DMS # clone
cd your_repo
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Denoisers</summary>
Quick denoising with *clean image* and *noisy image*

  
</details>


## <div align="center">Reproducing results </div>

<details open>
<summary>Comparing on real images BSDS500</summary>

Running DMS on Python
```bash
cd SPL-fig6
python dms_real_std_0_05
python trof_ggs_real_std_0_05
```
Running [Home et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig6
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_figure6_std_0_05 ; exit"
```
  
</details>

