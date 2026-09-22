# LeafNet Demo

This folder contains the Flask-based inference prototype developed for LeafNet.

The primary prototype:

1. accepts an uploaded leaf image,
2. applies the DenseNet201 inference transform,
3. predicts one of the configured disease classes,
4. looks up crop/disease metadata, and
5. renders the result in the browser.

The trained `BestDenseNet.pth` checkpoint is not included in this repository.

`archive/` contains earlier experiments, including a database-backed prototype. These files are preserved for project history and may contain environment-specific paths from the original development setup.
