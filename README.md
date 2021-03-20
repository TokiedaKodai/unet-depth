# unet-depth

Learning based surface shape estimator using .
You can increase accuracy of depth image measured by active stereo method which losts high frequency shape or unevenness smaller than the density of the projection pattern of object surface.
Network predicts the surface shape from shading image, focused on Shape from Shading methods.

## Data

Input a low-resolution depth image, a shading image, and optionally a pattern projected image used when low-res depth measured by a active stereo method.
Network outputs the accurate depth of object surface.
You can get target depth by adding the low-res depth to the network output.

## Training

You need synthetic input images (low-res depth, shading, and optionally pattern projected) and ground truth depth image for learning.
You can fine-tune the trained model using real images.

## Test

It can be estimated and evaluated on test data using the trained model at the end of training, at the point with the lowest validation loss, or at any point in time.
It automatically outputs evaluation indices such as RMSE and distance images and error maps.
