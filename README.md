# MOEA/D with Wasserstein
This repository contains the code of the algorithm MOEA/D/W used in the following paper:  

Ponti A, Candelieri A, Giordani I, Archetti F. **Intrusion Detection in Networks by Wasserstein Enabled Many-Objective Evolutionary Algorithms.** _Mathematics. 2023; 11(10)_:2342. https://doi.org/10.3390/math11102342

## Python dependencies
Use the `requirements.txt` file as reference.  
You can automatically install all the dependencies using the following command. 
````bash
pip install -r requirements.txt
````

## How to use the code
There are two entrypoints:
- `run_benchmark.py`: run the experiments on the benchmark functions. Here it is possible to modify the test function as well as the number of variables and objectives.
- `run_osp.py`: run the experiments on the Optimal Sensor Placement problem. Here is possible to modify the number of objective functions (2 or 4) and the bedget of sensors.

## How to cite us
If you use this repository, please cite the following paper:
> [Ponti A, Candelieri A, Giordani I, Archetti F. Intrusion Detection in Networks by Wasserstein Enabled Many-Objective Evolutionary Algorithms. Mathematics. 2023; 11(10):2342. https://doi.org/10.3390/math11102342](https://www.mdpi.com/2227-7390/11/10/2342)

```
@Article{math11102342,
  AUTHOR = {Ponti, Andrea and Candelieri, Antonio and Giordani, Ilaria and Archetti, Francesco},
  TITLE = {Intrusion Detection in Networks by Wasserstein Enabled Many-Objective Evolutionary Algorithms},
  JOURNAL = {Mathematics},
  VOLUME = {11},
  YEAR = {2023},
  NUMBER = {10},
  ARTICLE-NUMBER = {2342},
  URL = {https://www.mdpi.com/2227-7390/11/10/2342},
  ISSN = {2227-7390},
  DOI = {10.3390/math11102342}
}
```
