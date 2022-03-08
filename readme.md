# Object Detection
This repository is my practice codes from [Object Detection Series](https://www.youtube.com/playlist?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq)
from Aladdin Persson.

# YOLOv1
I made two minor adjustments following the video tutorial.
1. In the `loss.py`, it is unnecessary to call `torch.flatten` since `self.mse` calculates the Mean Squared Error of every
   entry and adds together that results in a single number. As a result, I did not utilize it.
2. In the `dataset.py`, I have created the `collate_fn` method to transform the batch of images instead of a single image as I believe that results in a faster process.

