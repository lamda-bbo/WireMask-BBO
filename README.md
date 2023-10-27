# NeurIPS'23 Macro Placement by Wire-Mask-Guided Black-Box Optimization

Official implementation of NeurIPS'23 paper "Macro Placement by Wire-Mask-Guided Black-Box Optimization"

This repository contains the Python code for WireMask-BBO, a black-box optimization framework for macro placement by using a wire-mask-guided greedy procedure for objective evaluation. Equipped with different BBO algorithms, WireMask-BBO empirically achieves significant improvements over previous methods.

## Requirements

+ gpytorch==1.8.1
+ matplotlib==2.2.3
+ numpy==1.15.1
+ opencv_python==4.1.2.30
+ scipy==1.1.0
+ setuptools==40.2.0
+ torch==1.13.1

## File structure

+ `result` directory stores the output results of optimzation. `result/MaskPlace` placement results are provided by the author of MaskPlace [https://openreview.net/forum?id=T2DBbSh6_uY].
+ `ISPD2005.py` serves the benchmark download script.
+ `place_db.py` serves the netlist information extraction script, originally borrowed from [https://github.com/laiyao1/maskplace].
+ `common.py` defines several constants.
+ `utils.py` defines functions to be used for optimization.
+ `TuRBO` directory is borrowed from [https://github.com/uber-research/TuRBO] for TuRBO implemention for WireMask-BO.

+ `EA_swap_only.py` , `RS.py` and `BO.py` are the main entrances for WireMask-EA, WireMask-RS and WireMask-BO optimization procedure.
+ `EA_finetune.py` implements the finetuning procedure based on MaskPlace [https://openreview.net/forum?id=T2DBbSh6_uY] placement results.
+ `plot.py` provides a script for reproducing the Figure 4 (HPWL v.s. wall clock time) in the main paper.

## Usage

You should first build the environment according to the requirements.

Then download the ISPD2005 benchmark.

```
python ispd2005.py
mv ispd2005 benchmark
```

To run WireMask-EA on chip $adaptec1$ using random seed 2023, with 100 rounds of random search for initialzation, run the following code

```
python EA_swap_only.py --dataset adaptec1 --seed 2023 --init_round 100
```

To run WireMask-RS on chip $bigblue1$ using random seed 2024, with 200 rounds of random search, run the following code

```
python RS.py --dataset bigblue1 --seed 2024 --stop_round 200
```

To run WireMask-BO on chip $adaptec3$ using random seed 2025, run the following code

```
python BO.py --dataset adaptec1 --seed 2025
```

To finetune MaskPlace on chip $adaptec4$ using 1000 rounds of WireMask-EA, with random seed 2026, run the following code

```
python EA_finetune.py --dataset adaptec4 --seed 2026 --stop_round 1000
```

To reproducing the Figure 4 (HPWL v.s. wall clock time) in the main paper, you should first run EA, RS and BO for minutes, respectively. Then run

```
python plot.py
```

The results will be shown in `all.pdf`.

It is worth noting that our evaluation functions are included in `utils.py`. All results in our paper are generated based on `cal_hpwl()` and `write_placement_and_overlap()`.