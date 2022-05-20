# Pipeline of post processing

### 1. plot the loss to see if the model converge.
    
- add all the losses together to be the total loss
- plot training loss with validation loss with `loss_visualization.py`
- determine if the validation loss has qualitative change during training.
  - the validation loss should gradually go up. The turning point of validation loss is supposed to be the best model
  - check in `summary.json` that  best.pth corresponds to the visual judgement

### 2. generate the reconstruction, embedding and clusters
- check svd result. Compare original vs. original-svd-keypoint result.
- check reconstruction in both training and testing data.
  - compare the original data with reconstruction data. We assert the training data performs better than testing data.
- generate embeddings for all the sessions. The embedding is the mean value of the encoder output.
- generate the clusters for all the sessions. Here is the codebook vectors
  - check the length of bouts in the clusters

### 3. plotting
#### the plotting codes are all under `util/visualzation`
- plot original vs. reconstruction comparision with `animtation.py`
- (if using cluster method like GMM/KMeans), plot the cluster number vs. different scores using `cluster_num_visualization.py`
- plot the embedding trajectories with `cluster_traja_illustration.py`.
- plot the histogram of the bout length for all sessions
- plot the trajectories sampled from different labels
- plot in BENTO
- plot the heatmap