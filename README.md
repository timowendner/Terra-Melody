# Terra Melody
This is a Deep Learning Neural Network to generate Midi-files.

# Intuition
At first the midi-files get encoded. Every single note will get turned into a list of `[starting point, difference to the last starting point, duration, velocity, key, octave]`. This encoding is then basically a sequence of embeddings. With those embeddings the LSTM tries to predict the next embedding in the sequence.

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