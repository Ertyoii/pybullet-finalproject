# Final Project robot manipulation
OK, result not so good... Most of the time it fails even though MSE loss hit 1e-8.

TODO:
1. Try larger datasets.
2. Amplify data x100.
3. (maybe) Write some test cases.
 
DONE:
1. Find the center of object.
2. Figure out the x, y input's relationship with center.
3. Generate images and corresponding x, y, a.
4. Compute loss.
5. Build a naive model to infer x, y, a from images.
6. Train the model and integrate it into grasp operation.