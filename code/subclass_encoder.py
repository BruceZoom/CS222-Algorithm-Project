from utils import *
import pickle
from copy import deepcopy
from sklearn.cluster import KMeans, MiniBatchKMeans


class SubclassEncoder(object):
    def __init__(self, data, labels, cluster_per_class=2, threshold=None, verbose=0):
        classes = np.unique(labels)
        if verbose: print("Clustering...")
        kmeans = KMeans(n_clusters=int(len(classes) * cluster_per_class), n_jobs=-1, verbose=verbose).fit(data)
        if threshold is not None:
            if verbose: print("Removing anomalies...")
            cluster_cnts = np.histogram(kmeans.labels_, bins=len(np.unique(kmeans.labels_)))[0]
            print(cluster_cnts)
            kmeans.cluster_centers_ = kmeans.cluster_centers_[cluster_cnts > 10]
            elim_labels = np.where(cluster_cnts <= 10)
            if type(elim_labels) is tuple and len(elim_labels) == 1:
                elim_labels = elim_labels[0]
            print(elim_labels)
            for label in elim_labels:
                print(kmeans.labels_, label)
                print(sum(kmeans.labels_ == label))
                print(data[kmeans.labels_ == label, :].shape)
                kmeans.labels_[kmeans.labels_ == label] = kmeans.predict(
                    data[kmeans.labels_ == label, :])
        self.encoders = [kmeans, ]
        self._labels = labels.copy()

    def merge(self, other):
        encoder = deepcopy(self)
        encoder.encoders += deepcopy(other.encoders)
        return encoder

    def encode(self, data=None, labels=None):
        if data is None and labels is None:
            encoding = np.array([encoder.labels_ for encoder in self.encoders]).T
            print(encoding.shape, self._labels.shape)
            encoding = np.concatenate([encoding, self._labels.reshape([-1, 1])], axis=1)
        else:
            encoding = np.array([encoder.predict(data) for encoder in self.encoders]).T
            encoding = np.concatenate([encoding, labels.reshape([-1, 1])], axis=1)
        return encoding

    @staticmethod
    def merge_encoding(encodings):
        res = encodings[0]
        for i, encoding in enumerate(encodings):
            if i == 0: continue
            res = np.concatenate([encoding[:, :-1], res], axis=1)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Subclass Encoder")
    parser.add_argument("prog", help="One of the 'encode', 'merge'.")
    parser.add_argument(
        "--data_dir", help="Directory containing 'data.txt' and 'labels.txt'",
        default=None, type=str)
    parser.add_argument(
        "--encoder_save_path", help="The file to save the encoder.",
        default=None, type=str)
    parser.add_argument(
        "--encoding_save_path", help="The file to save the encoding.",
        default="data/sub_encoding.txt", type=str)
    parser.add_argument(
        "--cluster_per_class", help="Clusters per class.",
        default=2, type=float)
    parser.add_argument(
        "--threshold", help="Minimum samples a cluster should have.",
        default=None, type=float)
    parser.add_argument(
        "--verbose", help="Verbose mode.",
        default=0, type=int)
    parser.add_argument(
        "--input_encodings", help="Encodings to be merged.",
        action="append", type=str)
    args = parser.parse_args()

    if args.prog == 'encode':
        # create an encoding
        if args.verbose: print("Loading data...")
        data = np.loadtxt(os.path.join(args.data_dir, 'data.txt'))
        labels = np.loadtxt(os.path.join(args.data_dir, 'labels.txt'))
        print(data.shape)
        if args.verbose: print("Encoding...")
        encoder = SubclassEncoder(data, labels, cluster_per_class=args.cluster_per_class,
                                  threshold=args.threshold, verbose=args.verbose)
        encoding = encoder.encode()

        if args.verbose: print("Saving encoding...")
        if not os.path.exists(os.path.split(args.encoding_save_path)[0]):
            os.mkdir(os.path.split(args.encoding_save_path)[0])
        np.savetxt(args.encoding_save_path, encoding)

        if args.encoder_save_path is not None:
            if args.verbose: print("Saving encoder...")
            if not os.path.exists(os.path.split(args.encoder_save_path)[0]):
                os.mkdir(os.path.split(args.encoder_save_path)[0])
            outfile = open(args.encoder_save_path, "wb")
            pickle.dump(encoder, outfile)
            outfile.close()

    elif args.prog == 'merge':
        # merge encodings
        if args.verbose: print("Merging encodings...")
        encoding = SubclassEncoder.merge_encoding([
            np.loadtxt(fname) for fname in args.input_encodings
        ])

        if args.verbose: print("Saving encoding...")
        if not os.path.exists(os.path.split(args.encoding_save_path)[0]):
            os.mkdir(os.path.split(args.encoding_save_path)[0])
        np.savetxt(args.encoding_save_path, encoding)
