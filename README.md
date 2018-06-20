# PFCM
Possiblistic Fuzzy C-Means Algorithm in Python

- Algorithm explanation : https://www.researchgate.net/publication/3336300_A_Possibilistic_Fuzzy_C-Means_Clustering_Algorithm
- Implementation of the algorithm MATLAB : https://www.ijser.org/researchpaper/implementation-of-possibilistic-fuzzy-cmeans-clustering-algorithm-in-matlab.pdf


    <b>Parameters for the main function</b> (pfcm): <ul>
    <li><u>data</u>: Dataset to be clustered, with size M-by-N, where M is the number of data points and N is the number of coordinates for each data point.</li>
    <li><u>c</u> : Number of clusters</li>
    <li><u>expo</u> : exponent for the U matrix (default = 2)</li>
    <li><u>max_iter</u> : Maximum number of iterations (default = 1000)</li>
    <li><u>min_impor</u> : Minimum amount of imporvement (default = 0.005)</li>
    <li><u>a</u> : User-defined constant a (default = 1)</li>
    <li><u>b</u> : User-defined constant b that should be greater than a (default = 4)</li>
    <li><u>nc</u> : User-defined constant nc (default = 2)</li>
    </ul>
    
    
    The clustering process stops when the maximum number of iterations is
    reached, or when objective function improvement or the maximum centers
    imporvement between two consecutive iterations is less
     than the minimum amount specified.
     
     
    <b>Return values :</b><ul>
    <li><u>cntr</u> : The clusters centers</li>
    <li><u>U</u> : The C-Partionned Matrix (used in FCM)</li>
    <li><u>T</u> : The Typicality Matrix (used in PCM)</li>
    <li><u>obj_fcn</u> : The objective function for U and T</li>
    </ul>
    <hr>
 
    <b>Parameters of the PFCM Prediction function (pfcm_predict)</b><ul>
    <li><u>data</u>: Dataset to be clustered, with size M-by-N,
    where M is the number of data points
    and N is the number of coordinates for each data point.</li>
    <li><u>cntr</u> : centers of the dataset previoulsy calculated</li>
    <li><u>expo</u> : exponent for the U matrix (default = 2)</li>
    <li><u>a</u> : User-defined constant a (default = 1)</li>
    <li><u>b</u> : User-defined constant b that should be
    greater than a (default = 4)</li>
    <li><u>nc</u> : User-defined constant nc (default = 2)</li>
    </ul>
    The algortihm predicts which clusters the new dataset belongs to<br><br>
    <b>Return values :</b><ul>
    <li><u>new_cntr</u> : The new clusters centers</li>
    <li><u>U</u> : The C-Partionned Matrix (used in FCM)</li>
    <li><u>T</u> : The Typicality Matrix (used in PCM)</li>
    <li><u>obj_fcn</u> : The objective function for U and T</li>
    </ul>
