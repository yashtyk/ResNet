# ResNet
 Authors: Yana Shtyk, Olga Taran, André Csillaghy, Jonathan Donzallaz
 
The work addresses the problem of prediction of solar flares. To research this problem a machine learning technique called recurrent neural network was used.
We trained and tested our model on the SDOBenchmark dataset.  It is a time series dataset created by FHNW. 
It consists  of images of active regions cropped from SDO data. 
As the dataset has 10 different SDO channels, we were able to investigate prediction capabilities of all of them. Best TSS of 0.64 was obtained using a magnetogram. Furthermore, we found out that channel aggregation can help to improve prediction capabilities of the model. Using this technique we managed to achieve TSS of 0.7.
