# Grokking experiments done with Random Order AutoRegressive (ROAR) models on Sum Equalities

Data is valuable and finite. How much performance is lost when you only train right-to-left in your autoregressive models?

These experiments try to quantify this in the very simple case of two digit sum equalities. Sum equalities take the form a±b==c±d. Two sets of numbers are added together to make the same sum. 93-32==? has many possible answers. This is more complex that most grokking tasks and requires that the model learn the underlying structure of the sequence.

We can easily test if the model has learned the underlying structure of the data by checking if models prompted with a±b==? successfully complete the sequences such that a±b==c±d.

## Sequence Traversal Orders

We explore 3 sequence order traversals. 
* Conventional left-to-right training.
* Alternating left-to-right and right-to-left
* Pseudo random sequence order traversal where the "next" token in the random sequence is likely to be located near the "previous" token in the original unscrambled sequence

## Training Details

### Data:
* Training data consisted of 10,000 randomly generated two digit sum equality in the form a±b==c±d consisting of integers in the range -99 - +99
* A leading and trailing padding token to indicate the start/end of each line, resulting in a maximum of 14 tokens

### Model/Training:
* Utilized the ROAR architecture to allow random order autoregressive modeling
* layers = 32
* model dim = 512
* ff dim = 1024
* attention heads = 8
* Dropout = Randomly dropped between 0% and 100% of activations for each training sample. For each training sample/iteration the exact same nodes were dropped across all layers to encourage useful ensembling. 
* LR = 1e-3
* AdamW WD = 1
* Batch = 1024


# Results: Proportion of Sequence Completions Which are Valid by Training Strategy

![Percent Valid](resources/percent_valid.png)

We observe that left-to-right training leaves a significant amount of potential performance on the table given a finite amount of data. Forcing the model to learn a more comprehensive model of the structure of its training data improves performance even when the model is only evaluated left-to-right.

# Left-to-Right Test Loss During Training

![Percent Valid](resources/test_loss.png)

The proportion of samples completed in a valid way is almost uncorrelated with test loss, though sharp drops can be observed when phase changes occur.


# Model L2 During Training

![Percent Valid](resources/l2.png)

More complex sequence traversal orders are more difficult to learn initially.

# Usage
See [the demo notebook](grok_demo.ipynb) for the exact code used to generate these results.
