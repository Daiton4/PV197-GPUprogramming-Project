# PV197-GPUprogramming-Project

### Semestral project from PV179 GPU Programming course

In this course basics of theory, algorithms and technologies in GPU computation field were explained. Part of evaluation was independent project. We were given problem suited to be solved in parallel. Our task was to implement as efficient algorithm as possible (there were given minimal limits, it was also a contest). Results were tested on one machine with given configuration and compared to solution for CPU.

#### Structure
Originally project was splitted into files framework.cu (testing framework for problem), kernel_CPU.cu (implementation of solution in CPU, used for comparison and given beforehand) and kernel.cu (my implementation for GPU). Because this program was not working on PC I was debugging from, Code is slightly edited (with no effect on computation, that would disqualify me from course) and included in its whole length in framework.cu

#### Problem Description
The project is focused to computing similarity of galaxies. Your task is to implement a CUDA version an algorithm, which compare distances of all pairs of stairs in two representations of the same galaxy. The galaxy is represented as a vector of stars, each star has Cartesian coordinates in 3D space. The algorithm needs to evaluate following formula:

dist = sqrt( 1/(n(n-1)) * sum_{i=1}^{n-1} sum_{j=i+1}^{n} (d_{ij}^A - d_{ij}^B)^2 )

where n is number of stars and d_{ij}^A is Euclidean distance between star at index i and star at index j in galaxy A. Thus, the formula computes differences of all-to-all distances, subtracts them, squares them and sums them and normalizes the number when summed to not be influenced by the number of stars.
- the input size can be any number fitting into GPU memory
- the code must run on CUDA card with compute capability 3.0 or newer
- the performance will be measuted using 2 000 stars per galaxy or more
- the highest acceptable error of GPU version is 1%

#### Solution (just the idea)
Final version of my solution uses matrix multiplying algorithm edited for counting on values distributed through upper right matrix corner. (Subproblems are solved in square counts with one nonuniform part). Shared block memory is used for faster loading and numbers for blocks are optimized. Also dummy threads and formula optimization is used. Program was optimized only for specific counts and machine, it was tested on.
