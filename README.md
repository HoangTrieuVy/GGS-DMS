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

```python
optional arguments:
  -h, --help   show this help message and exit
  --z Z        noisy image path
  --x X        original image path
  --b B        beta
  --l L        lambda
  --algo ALGO  PALM, SLPAM,PALM-eps-descent,SLPAM-eps-descent
  --norm NORM  l1, AT
  --eps EPS    epsilon
  --it IT      number of iteration
 ```
 
```bash
cd demo
python example.py --n 

```

</details>




## <div align="center">Reproducing results </div>

<details open>
<summary>Comparison on  one synthetic only AWGN noisy image </summary>
 
Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python

```bash
cd SPL-fig4
python dms_spl_fig4.py
```
  
Running [Home et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig4
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_ggs ; exit"
```
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig4/Screenshot%202022-10-18%20032114.png" >
</details>

<details open>
<summary>Comparison on a set of synthetic noisy and blur images </summary>
 
Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python

```bash
cd SPL-fig5
python dms_ggs # running different schemes on dms
python trof_ggs # TV and T-ROF 
```
  
Running [Home et al.](https://iopscience.iop.org/article/10.1088/0266-5611/31/11/115011/pdf?casa_token=1EtwyHOFYqIAAAAA:7KNljR8MVKVeHvoB3wqw1eWDDzgYFHc860UrQ7bm69d6MpeA5UU9fHkUdCgLsC4uKAXoOfbwWzC2) on matlab

```bash
cd SPL-fig5
matlab -nodisplay -r "./setPath ; exit"
matlab -nodisplay -r "./hohm_ggs ; exit"
```
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig5/Screenshot%202022-10-18%20032056.png" >
</details>

<details open>
  
  
<summary>Comparison on real images BSDS500</summary>

Running [DMS with different schemes](https://ieeexplore.ieee.org/abstract/document/9723590) on Python
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
 <img align="center" width="1500" src="https://github.com/HoangTrieuVy/GGS-DMS/blob/main/SPL-fig6/Screenshot%202022-10-18%20031023.png" >

</details>



