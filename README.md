# Redis Vector Index Sizing Simulator


## Summary
Tool for generating VSS scenarios and then analyzing memory usage


## Features
- Calculates memory footprint of a Redis vector index scenario
## Prerequisites
- Python
- Docker
## Installation
1. Clone this repo.
```bash
git clone https://github.com/Redislabs-Solution-Architects/vss-sizer.git && cd vss-sizer
```
2. Install Python requirements (either in a virtual env or global)
```bash
pip install -r requirements.txt
```
3.  Start Redis Enterprise Docker environment
```bash
./start.sh
```

## Usage
### Options
- --url. URL connect string.  Default = redis://@localhost:12000
- --nkeys. Number of keys to be generated.  Default = 100,000.
- --indextype. Vector Index Type.  Default = hnsw.
- --metrictype.  Vector Metric Type.  Default = l2.
- --floattype.  Vector Float Type.  Default = f32.
- --vecdim.  Vector Dimension.  Default = 1536.
- --vecm.  HNSW M Param.  Default = 16.
### Execution
```bash
$ python3 sizer.py --vecdim 1536
```
### Output
Sample output for the test above.
```bash
Vector Index Test
 
*** Parameters ***
nkeys: 100000
objecttype: hash
indextype: hnsw
metrictype: cosine
floattype: float32
vecdim: 1536
knn: 10
iterations: 10
vecm: 16
 
*** Results ***
index ram used: 1126.67 MB
data ram used: 350.57 MB
index to data ratio: 321.38%
document size: 7376 B
average query latency: 3.18 ms
```
