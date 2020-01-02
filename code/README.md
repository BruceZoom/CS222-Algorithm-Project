# Code Directory
This is the directory for code.
## Usage
### utils.py
Convert json files into two txt file.
- data.txt: each row is the gate weights for one sample
- labels.txt: each row is the label for the sample
### subclasses_encoder.py
Apply clustering to the data.txt and encode each sample with the cluster index. Can also merge two encoding together.
- encoding.txt: each row is contains encodings, and the last element in each row is the original label for the sample in CIFAR-100
