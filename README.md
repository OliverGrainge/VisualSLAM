# VisualSLAM
This is a full visual slam pipeline built in python. The repo is built to provide a simple but comprehesive VSLAM system to support quick experimentation and research, particularly in the field of visual place recognition. The code base is extensible where each aspect of the VSLAM pipeline can be altered in a systematic way such that it's effect on performance can be evaluated. 

The codebase uses a stereo odometry pipeline with 3d-2d transformation estimation through the PnP algorithm with RANSAC for improved structural data association. Additionaly pose graph optimization over the last N poses is implemented in addition to global optimization with a loop closing detector. The algorithmic steps can be outlined below. 

1. Extract point features in Left and Right Stereo Images $I^{l}_k$,  $I^{r}_k$.
2. Match the points between left and right stereo images.
3. Triangulate points between stereo images to get 3D points.
4. Extract Point features from the next stereo pair in the sequence $I^{l}_{k+1}$ , $I^{r}_{k+1}$
5. Estimate relative transformation between $I^{l}_k$, $I^{l}_{k+1}$ with the PnP and RANSAC algorithm.
