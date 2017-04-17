## CSCI 4971 - Large Scale Matrix Computation and Machine Learning

### Time and Space Efficient Spectral Clustering via Column Sampling by Li et al.
by Matt Poegel


## Motivation

* Clustering algorithms are useful to uncover natural groupings in the data
* $k$-means is great but spectral clustering can be better
    * allows for clusters with non-convex boundaries


## Spectral Clustering

$\begin{array}{|l|} \hline
\textbf{Input: } X \in \mathbb{R}^{n \times d},\ k \text{ clusters} \\
\textbf{Output: } \hat{y} \text{ cluster label for each row of } X \\ \hline
\text{1. Form affinity matrix, } W \in \mathbb{R}^{n \times n} \\
\text{2. Form degree matrix, } D \leftarrow \texttt{diag}(W \mathbf{1}) \\
\text{3. } L \leftarrow I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}} \\
\text{4. Take the bottom } k \text{ eigenvectors of } L \text{ to form } Z \\
\text{5. } \hat{y} \leftarrow \texttt{KMeans}(Z,\ k) \\ \hline
\end{array}$


## Intuition for Randomness

* Computing $W \in \mathbb{R}^{n \times n}$ is expensive
    * Use randomness to estimate $W$
* $W$ is also expensive to store
    * Use randomness to only compute and store parts of $W$


## Nystrom-based Spectral Clustering

$\begin{array}{|l|} \hline
\textbf{Input: } X \in \mathbb{R}^{n \times d},\ k \text{ clusters, } m \text{ sampled columns} \\
\textbf{Output: } \hat{y} \text{ cluster label for each row of } X \\ \hline
\text{1. } C \leftarrow A S \in \mathbb{R}^{n \times m} \text{ where } A \text{ is the affinity matrix} \\
\text{2. } W \leftarrow S^{\intercal} A S \\
\text{2. } \hat{D} \leftarrow \texttt{diag}(C W^{\dagger} C^{\intercal} \mathbf{1}) \\
\text{3. } D_{11} \leftarrow \texttt{diag}(W \mathbf{1}) \\
\text{4. } \hat{M}_{:1} = [M_{11} \hat{M}_{21}^{\intercal}]^{\intercal} \leftarrow \hat{D}^{-\frac{1}{2}} C D_{11}^{-\frac{1}{2}} \\
\text{5. } S \leftarrow M_{11} + M_{11}^{-\frac{1}{2}} \hat{M}_{21}^{\intercal} \hat{M}_{21} M_{11}^{-\frac{1}{2}} \\
\text{6. SVD: } S = U \Lambda T^{\intercal} \\
\text{7. } V \leftarrow \hat{M}_{:1} M_{11}^{-\frac{1}{2}} U \Lambda^{-\frac{1}{2}} \\
\text{8. } \hat{y} \leftarrow \texttt{KMeans}(V_k,\ k) \\ \hline
\end{array}$


## Time and Space Complexity

|        | Nystrom                   |
| ------ |:-------------------------:|
| Time   | $\mathcal{O}(nm^2 + m^3)$ |
| Space  | $\mathcal{O}(nm)$         |


## Proposed Algorithm

$\begin{array}{|l|} \hline
\textbf{Input: } X \in \mathbb{R}^{n \times d},\ k \text{ clusters, } Z \text{ m points from } X \\
\textbf{Output: } \hat{y} \text{ cluster label for each row of } X \\ \hline
\text{1. Form the affinity matrix } A_{11} \in \mathbb{R}^{m \times m} \text{ using } Z \\
\text{2. } D_* \leftarrow \texttt{diag}(A_{11} \mathbf{1}) \\
\text{3. } M_* \leftarrow D_*^{-\frac{1}{2}} A_{11} D_*^{-\frac{1}{2}} \\
\text{4. Compute the } k \text{ largest eigenvectors and eigenvalues of } M_* \text{ as } \\
\hspace{1.2em} V \text{ and } \Lambda \text{ respectively.} \\
\text{5. } B \leftarrow D_*^{-\frac{1}{2}} V \Lambda^{-1} \\
\text{6. } \textbf{for } i = 1 \text{ to } n \textbf{ do} \\
\text{7. } \hspace{2em} \text{Form the affinity vector } a \in \mathbb{R}^{1 \times m} \text{ between } x_i \text{ and points} \\
\hspace{3.2em} \text{in } Z \\
\text{8. } \hspace{2em} Q_i \leftarrow a B \\
\text{9. } \textbf{end for} \\
\text{10. } \hat{D} \leftarrow \texttt{diag}(Q \Lambda Q^{\intercal} \mathbf{1}) \\
\text{11. } U \leftarrow \hat{D}^{-\frac{1}{2}} Q \\
\text{12. } \hat{y} \leftarrow \texttt{KMeans}(U,\ k) \\ \hline
\end{array}$


## Proposed Algorithm to Orthogonalize $U$

$\begin{array}{|l|} \hline
\textbf{Input: } U \in \mathbb{R}^{n \times k},\ \Lambda \in \mathbb{R}^{k \times k} \\
\textbf{Output: } \text{orthogonalized } \tilde{U} \text{ and } \tilde{\Lambda} \\ \hline
\text{1. } P \leftarrow U^{\intercal} U \\
\text{2. eigen-decomposition: } P = V \Sigma V^{\intercal} \\
\text{3. } B \leftarrow \Sigma^{\frac{1}{2}} V^{\intercal} \Lambda V \Sigma^{\frac{1}{2}} \\
\text{4. eigen-decomposition: } B = \tilde{V} \tilde{\Lambda} \tilde{V}^{\intercal} \\
\text{5. } \tilde{U} \leftarrow U V \Sigma^{-\frac{1}{2}} \tilde{V} \\ \hline
\end{array}$


## Time and Space Complexity

|        | Nystrom                   | Proposed           |
| ------ |:-------------------------:|:------------------:|
| Time   | $\mathcal{O}(nm^2 + m^3)$ | $\mathcal{O}(nmk)$ |
| Space  | $\mathcal{O}(nm)$         | $\mathcal{O}(nk)$  |


## Theoretical Guarentees

### Proposition 1
_Assume that graph $G$ has no more than $k$ connected components. Then $A_{:1}1 = \hat{A}_{:1}1$._

### Proposition 2
_In the orthogonalization algorithm, $U \Lambda U^{\intercal} = \tilde{U} \tilde{\Lambda} \tilde{U}^{\intercal}$
and $\tilde{V}^{\intercal} \tilde{V} = I$._


## Emperical Evaluation

### Data Sets

* USPS handwritten digits, $n = 6647$, $d = 256$, $k = 9$

### Results in Python:
Parameters: $m = 300$, $\gamma = 4$, iterations = 1

| Algorithm | Time (sec)  | Accuracy (%) |
| --------- |:-----------:|:------------:|
| NCut      | 756.507     | 72.432       |
| $k$-means | -           | 71.055       |
| KASP      | **18.814**  | 48.217       |
| Nystrom   | 24.499      | 72.905       |
| CSSP      | 21.55       | **73.627**   |


## More Results

### Data Sets

* MNIST handwritten digits, $n = 18831$, $d = 256$, $k = 3$

Parameters: $m = 500$, iterations = 1

| Algorithm | Time (sec)  | Accuracy (%) |
| --------- |:-----------:|:------------:|
| Nystrom   | 174.019     | 82.423       |
| CSSP      | **112.667** | **85.981**   |


## Memory Efficient Implementation

### Amazon EC2 Compute Optimized c4.2xlarge (8 CPUs, 15GB RAM)

Parameters: $m = 300$

| $n$      | $k$  | Time (sec)  | Accuracy (%) |
| ---------| ---- |:-----------:|:------------:|
| 18,623   |  3   | 50.605      | 89.438       |
| 24,754   |  4   | 65.130      | 83.109       |
| 30,596   |  5   | 82.712      | 81.671       |
| 36,017   |  6   | 93.855      | 72.569       |
| 41,935   |  7   | 111.408     | 57.014       |
| 211,127  |  2   | -           | -            |


## Conclusion

* The proposed algorithm performs very well in practice
    * outperforms normalized cut at scale
    * outperforms Nystrom spectral clustering at scale 
        * time
        * accuracy
        * space
* Authors also showed success with image segmentation
    * computed the eigenvectors of a $4752 \times 3168$ image in only 23 seconds


[https://github.com/mpoegel/spectral-clustering](https://github.com/mpoegel/spectral-clustering)


## References

Li, M., Lian, X. C., Kwok, J. T., & Lu, B. L. (2011, June). Time and space efficient spectral clustering via column sampling. In Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on (pp. 2297-2304). IEEE.

Yan, D., Huang, L., & Jordan, M. I. (2009, June). Fast approximate spectral clustering. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 907-916). ACM.
