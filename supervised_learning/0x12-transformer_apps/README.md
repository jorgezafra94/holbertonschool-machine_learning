# Transformer Applications
```
$ cat load_dataset.py
#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()
pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./load_dataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
$
```

## Task0 - Dataset
Create the class Dataset that loads and preps a dataset for machine translation:<br>
<br>
* Class constructor `def __init__(self)`:
* creates the instance attributes:
* data_train, which contains the ted_hrlr_translate/pt_to_en tf.data.Dataset train split, loaded as_supervided
* data_valid, which contains the ted_hrlr_translate/pt_to_en tf.data.Dataset validate split, loaded as_supervided
* tokenizer_pt is the Portuguese tokenizer created from the training set
* tokenizer_en is the English tokenizer created from the training set
* Create the instance method `def tokenize_dataset(self, data):` that creates sub-word tokenizers for our dataset:
* data is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
* pt is the tf.Tensor containing the Portuguese sentence
* en is the tf.Tensor containing the corresponding English sentence
* The maximum vocab size should be set to 2**15

Returns: tokenizer_pt, tokenizer_en
* tokenizer_pt is the Portuguese tokenizer
* tokenizer_en is the English tokenizer

```
$ cat 0-main.py
#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
$ ./0-main.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
tinham comido peixe com batatas fritas ?
did they eat fish and chips ?
<class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'>
<class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'>
$
```

## Task1 - Encode Tokens
Update the class Dataset:<br>
<br>
* Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens:
* * pt is the tf.Tensor containing the Portuguese sentence
* * en is the tf.Tensor containing the corresponding English sentence
* The tokenized sentences should include the start and end of sentence tokens
* The start token should be indexed as vocab_size
* The end token should be indexed as vocab_size + 1

Returns: pt_tokens, en_tokens
* pt_tokens is a np.ndarray containing the Portuguese tokens
* en_tokens is a np.ndarray. containing the English tokens

```
$ cat 1-main.py
#!/usr/bin/env python3

Dataset = __import__('1-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
for pt, en in data.data_train.take(1):
    print(data.encode(pt, en))
for pt, en in data.data_valid.take(1):
    print(data.encode(pt, en))
$ ./1-main.py
([30138, 6, 36, 17925, 13, 3, 3037, 1, 4880, 3, 387, 2832, 18, 18444, 1, 5, 8, 3, 16679, 19460, 739, 2, 30139], [28543, 4, 56, 15, 1266, 20397, 10721, 1, 15, 100, 125, 352, 3, 45, 3066, 6, 8004, 1, 88, 13, 14859, 2, 28544])
([30138, 289, 15409, 2591, 19, 20318, 26024, 29997, 28, 30139], [28543, 93, 25, 907, 1366, 4, 5742, 33, 28544])
$
```

## Task2 - TF Encode
Update the class Dataset:<br>
<br>
* Create the instance method `def tf_encode(self, pt, en):` that acts as a tensorflow wrapper for the encode instance method
* Make sure to set the shape of the pt and en return tensors
* Update the class constructor `def __init__(self):`
* update the data_train and data_validate attributes by tokenizing the examples

```
$ cat 2-main.py
#!/usr/bin/env python3

Dataset = __import__('2-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
data = Dataset()
print('got here')
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./2-main.py
tf.Tensor(
[30138     6    36 17925    13     3  3037     1  4880     3   387  2832
    18 18444     1     5     8     3 16679 19460   739     2 30139], shape=(23,), dtype=int64) tf.Tensor(
[28543     4    56    15  1266 20397 10721     1    15   100   125   352
     3    45  3066     6  8004     1    88    13 14859     2 28544], shape=(23,), dtype=int64)
tf.Tensor([30138   289 15409  2591    19 20318 26024 29997    28 30139], shape=(10,), dtype=int64) tf.Tensor([28543    93    25   907  1366     4  5742    33 28544], shape=(9,), dtype=int64)
$
```
## Task3 - Pipeline
Update the class Dataset to set up the data pipeline:<br>
<br>
* Update the class constructor `def __init__(self, batch_size, max_len)`:
* batch_size is the batch size for training/validation
* max_len is the maximum number of tokens allowed per example sentence
* update the data_train attribute by performing the following actions:
* filter out all examples that have either sentence with more than max_len tokens
* cache the dataset to increase performance
* shuffle the entire dataset
* split the dataset into padded batches of size batch_size
* prefetch the dataset using tf.data.experimental.AUTOTUNE to increase performance
* update the data_validate attribute by performing the following actions:
* filter out all examples that have either sentence with more than max_len tokens
* split the dataset into padded batches of size batch_size

```
$ cat 3-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
$ ./3-main.py
tf.Tensor(
[[30138  1029   104 ...     0     0     0]
 [30138    40     8 ...     0     0     0]
 [30138    12    14 ...     0     0     0]
 ...
 [30138    72 23483 ...     0     0     0]
 [30138  2381   420 ...     0     0     0]
 [30138     7 14093 ...     0     0     0]], shape=(32, 39), dtype=int64) tf.Tensor(
[[28543   831   142 ...     0     0     0]
 [28543    16    13 ...     0     0     0]
 [28543    19     8 ...     0     0     0]
 ...
 [28543    18    27 ...     0     0     0]
 [28543  2648   114 ... 28544     0     0]
 [28543  9100 19214 ...     0     0     0]], shape=(32, 37), dtype=int64)
tf.Tensor(
[[30138   289 15409 ...     0     0     0]
 [30138    86   168 ...     0     0     0]
 [30138  5036     9 ...     0     0     0]
 ...
 [30138  1157 29927 ...     0     0     0]
 [30138    33   837 ...     0     0     0]
 [30138   126  3308 ...     0     0     0]], shape=(32, 32), dtype=int64) tf.Tensor(
[[28543    93    25 ...     0     0     0]
 [28543    11    20 ...     0     0     0]
 [28543    11  2850 ...     0     0     0]
 ...
 [28543    11   406 ...     0     0     0]
 [28543     9   152 ...     0     0     0]
 [28543     4   272 ...     0     0     0]], shape=(32, 35), dtype=int64)
$
```

## Task4 - Create Masks
Create the function `def create_masks(inputs, target):` that creates all masks for training/validation:<br>
<br>
* inputs is a tf.Tensor of shape (batch_size, seq_len_in) that contains the input sentence
* target is a tf.Tensor of shape (batch_size, seq_len_out) that contains the target sentence
* This function should only use tensorflow operations in order to properly function in the training step

Returns: encoder_mask, look_ahead_mask, decoder_mask
* encoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) to be applied in the encoder
* look_ahead_mask is the tf.Tensor look ahead mask of shape (batch_size, 1, seq_len_out, seq_len_out) to be applied in the decoder
* decoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) to be applied in the decoder

```
$ cat 4-main.py
#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for inputs, target in data.data_train.take(1):
    print(create_masks(inputs, target))
$ ./4-main.py
(<tf.Tensor: id=414557, shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: id=414589, shape=(32, 1, 37, 37), dtype=float32, numpy=
array([[[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.],
         [0., 0., 0., ..., 0., 1., 1.]]],


       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         ...,
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.],
         [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>), <tf.Tensor: id=414573, shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       ...,


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]],


       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>
$
```

## Task5 - Train
Take your implementation of a transformer from our previous project and save it to the file 5-transformer.py. Note, you may need to make slight adjustments to this model to get it to functionally train.<br>
<br>
Write a the function def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs): that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:
<br>
* N the number of blocks in the encoder and decoder
* dm the dimensionality of the model
* h the number of heads
* hidden the number of hidden units in the fully connected layers
* max_len the maximum number of tokens per sequence
* batch_size the batch size for training
* epochs the number of epochs to train for
* Your model should be trained with Adam optimization with beta_1=0.9, beta_2=0.98, epsilon=1e-9
* The learning rate should be scheduled using the following equation with warmup_steps=4000:

* Your model should use sparse categorical crossentropy loss, ignoring padded tokens
* Your model should print the following information about the training:
* Every 50 batches, you should print Epoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}
* Every epoch, you should print Epoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}

Returns the trained model

```
$ cat 5-main.py
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
print(type(transformer))
$ ./5-main.py
Epoch 1, batch 0: loss 10.26855754852295 accuracy 0.0
Epoch 1, batch 50: loss 10.23129940032959 accuracy 0.0009087905054911971

...

Epoch 1, batch 1000: loss 7.164522647857666 accuracy 0.06743457913398743
Epoch 1, batch 1050: loss 7.076988220214844 accuracy 0.07054812461137772
Epoch 1: loss 7.038494110107422 accuracy 0.07192815840244293
Epoch 2, batch 0: loss 5.177524089813232 accuracy 0.1298387050628662
Epoch 2, batch 50: loss 5.189461708068848 accuracy 0.14023463428020477

...

Epoch 2, batch 1000: loss 4.870367527008057 accuracy 0.15659810602664948
Epoch 2, batch 1050: loss 4.858142375946045 accuracy 0.15731287002563477
Epoch 2: loss 4.852652549743652 accuracy 0.15768977999687195
<class '5-transformer.Transformer'>
$
```
