## Experiments

- No sensors in franka simulation

### Hypothesis 1 (GMM on single-point LiDAR)

- Setup simple trajectory (circle)
- Sample signals at equidistant steps (10) for some runs (iid; 2 - static and dynamic environment)
- Visualize all signals within runs in histograms
- Visualize signals timesteps
- Prepare data for GMM like this: One datapoint is one histogram of values at one timestep within a run


## Setup

### Connect To RasberryPi

Connect
```
ssh pi@10.250.144.42
password: rootroot1
```

Manual start (debugging)

```
cd code
./start_lidar.sh
```

Start/Stop, Status services
```
sudo systemctl start lidar.service
sudo systemctl stop lidar.service
systemctl status lidar.service
```

## Resources

### Classification of Outdoor 3D Lidar Data Based on Unsupervised Gaussian Mixture Models

#### Key Points
- two-layer classification model (requires labelling by expert in the second stage)
- first stage is clustering the 3d point cloud points (after featurization) using GMM
- second stage human expert assigns each point to a smaller number of classes (supervised)

#### Notes
