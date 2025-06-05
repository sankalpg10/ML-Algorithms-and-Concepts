"""
The moving average of standard deviation is calculated as follows:

1)Compute the standard deviation of the current batch of gradients
2)Update the moving average of the standard deviation using a weighted average of the current batch's 
standard deviation and the previous moving average.
"""

"""batch_of_gradients is an array of arrays, where each inner array contains the gradients for a single batch.
    The moving average of the standard deviation is updated after each batch of gradients is processed, 
    and the gradients are clipped using the moving average of the standard deviation. 
    The clipped gradients are then used to update the model weights."""

import numpy as np

# Initialize moving average of standard deviation to 0


def calculate_moving_avg(numbatches,batch_of_gradients):
    moving_avg_std = 0.0

    # Hyperparameter for weighting the moving average
    alpha = 0.9

    for i in range(numbatches):

        #calculate SD of current batch

        batch_std = np.std(batch_of_gradients[i])

        #update moving avg

        moving_avg_std = alpha*moving_avg_std + (1-alpha)*batch_std

        #clipping gradients

        clipped_gradients  = np.clip(batch_of_gradients[i],-moving_avg_std,moving_avg_std)

        

        return moving_avg_std,clipped_gradients

def update_weights(weights,gradients):
    # Perform gradient update using the clipped gradients
    "Use clipped gradients to update weights"

    pass







