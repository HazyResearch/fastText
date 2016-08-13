# fastText, forked for Hazy Research

fastText is a library for efficient learning of word representations and sentence classification.

## Requirements

**fastText** builds on modern Mac OS and Linux distributions.
Since it uses C++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.

### Text classification

This library can also be used to train supervised text classifiers, for instance for sentiment analysis.
In order to train a text classifier using the method described in [1](#bag-of-tricks-for-efficient-text-classification), use:

```
$ ./fasttext train -input train.txt -output model
```

where `train.txt` is a text file containing a training sentence per line along with the labels.
By default, we assume that labels are words that are prefixed by the string `__label__`.
This will output two files: `model.bin` and `model.vec`.
Once the model was trained, you can evaluate it by computing the precision at 1 (P@1) on a test set using:

```
$ ./fasttext test model.bin test.txt
```

In order to obtain the most likely label for a piece of text, use:

```
$ ./fasttext predict model.bin test.txt
```

where `test.txt` contains a piece of text to classify per line.
Doing so will output to the standard output the most likely label per line.
See `classification-example.sh` for an example use case.
In order to reproduce results from the paper [2](#bag-of-tricks-for-efficient-text-classification), run `classification-results.sh`, this will download all the datasets and reproduce the results from Table 1.

## Full documentation

Invoke a command without arguments to list available arguments and their default values:

```
$ ./fasttext supervised
Empty input or output path.

The following arguments are mandatory:
  -input      training file path
  -output     output file path

The following arguments are optional:
  -lr         learning rate [0.05]
  -dim        size of word vectors [100]
  -epoch      number of epochs [5]
  -minCount   minimal number of word occurences [1]
  -wordNgrams max length of word ngram [1]
  -bucket     number of buckets [2000000]
  -minn       min length of char ngram [3]
  -maxn       max length of char ngram [6]
  -thread     number of threads [12]
  -verbose    how often to print to stdout [10000]
  -t          sampling threshold [0.0001]
  -label      labels prefix [__label__]
```

## References

Please cite [1](#bag-of-tricks-for-efficient-text-classification) if using this code for text classification.

### Bag of Tricks for Efficient Text Classification

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/pdf/1607.01759v2.pdf)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

## License

fastText is BSD-licensed. We also provide an additional patent grant.
