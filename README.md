# Terra Melody
This is a Deep Learning Neural Network to generate Midi-files.

## Setup
We first need a config file that is setting our model up. We provide an example config file in the repository `config.yaml`.

To install the package use:
```
!pip install git+https://github.com/timowendner/Terra-Melody
```

Then the model can be run like:
```
import terra
model = terra.run('/path/to/config.yaml')
```