## This file contains a description of the input parameters used by PetaVision objects.


### HyPerCol parameters:

   * nx - number of neurons in the x direction
   * ny - number of neurons in the y direction
   * dt - time step: units of microseconds

### HyPerLayer parameters:

   * spikingFlag (default 1)
     - layer update is non spiking if 0
     - layer update is spiking if 1
   * writeNonspikingActivity (default set by WRITE_NONSPIKING_ACTIVITY HyPerLayer.cpp)
     - Only final activity is written to disk if 0
     - Activity is written to disk at each timestep if 1

### HyPerConn parameters:

   * initFromLastFlag
      - initializes internally if 0
      - initializes from last saved file if 1

### KernelConn parameters:
(The params.pv file still uses the HyPerConn keyword; KernelConn is not a keyword)

   * symmetrizeWeights
      - symmetrizes weights during normalization if symmetrizeWeights is nonzero
      - does not symmetrize weights if symmetrizeWeights = 0 