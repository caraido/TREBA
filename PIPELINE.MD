# Pipeline of post processing

### 1. plot the loss to see if the model converge.
    
- add all the losses together to be the total loss
- plot training loss with validation loss
- determine if the validation loss has qualitative change during training.
  - the validation loss should gradually go up. The turning point of validation loss is supposed to be the best model
  - check in `summary.json` that  best.pth corresponds to the visual judgement

### 2. generate the reconstruction, embedding and classification
- check svd result
- check reconstruction in testing data
- generate embedding

### 3. generate the labels

### 4. plotting
- plot the embedding trajectories
- plot the trajectories sampled from different labels
- plot in BENTO
- plot the heatmap
5.