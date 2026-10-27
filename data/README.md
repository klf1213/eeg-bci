# Data

The EEG recordings are not distributed with this repository. Download BCI
Competition IV Dataset 2a from its official source and place the files as
follows:

```text
data/
├── raw/
│   ├── A01T.mat ... A09T.mat
│   └── A01E.mat ... A09E.mat
└── evaluation_labels/
    └── A01E.mat ... A09E.mat
```

The evaluation-label files are used only to verify and score the held-out E
sessions. They must not be used for training, early stopping or hyperparameter
selection.
