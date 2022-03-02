"""
Description: This program performs logistic regression to find the best and
                worst words
Authors: edited by Zach Perry, source code provided by Alvin Grissom, cs360
Date: 3/1/2022
"""

import random
import numpy as np
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict

import argparse
import operator

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score : float, threshold=20.0) -> float:
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label : str, words : list, vocab):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1


class LogReg:
    def __init__(self, num_features : int, mu : float, step=lambda x: 0.05):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        self.beta = zeros(num_features)
        self.iterations_since_update = np.ones(len(self.beta))
        #self.iterations_since_update = []
        self.mu = mu
        self.step = step
        self.last_update = defaultdict(int)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples : list) -> (float, float):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def I(p : float, example : Example) -> int:
        if abs(example.y - p) < 0.5:
            return 1
        return 0

    """
    TODO: finish worst_words() and best_words()
    Hint: The index of words in vocab aligns with the corresponding index in beta.
    """

    def worst_words(self, vocab : list,num_worst=20) -> list:
        indices = np.argpartition(np.abs(self.beta), num_worst)[:num_worst]
        return np.array(vocab)[indices]

    def best_words(self, vocab : list,num_best=20) -> list:
        indices = np.argpartition(np.abs(self.beta), -1*num_best)[-1*(num_best+1):]
        return np.array(vocab)[indices]




    def sg_update(self, train_example : Example, iteration : int, use_tfidf=False) -> np.ndarray:
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """

        # TODO: Implement updates in this function

        p = sigmoid(np.dot(self.beta, train_example.x))

        for i in range(len(self.beta)):
            if train_example.x[i] != 0:
                self.beta[i] += self.step(i) * (train_example.y - p) * train_example.x[i]
                self.beta[i] *= pow(1 - 2 * self.step(i) * self.mu, self.iterations_since_update[i])
                self.iterations_since_update[i] = 0

            self.iterations_since_update[i] += 1


        """
        Hints:
        train_example is an instance of Example
        iterations_since_update shou track of how long it has been since a parameter was updated.

        See https://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf for efficient/lazy regularized updates.
        Basic idea: skip non-zero updates, but keep track of how many times they've been skipped.
        Then catch up when they're finally updated.
        """

        return self.beta


def read_dataset(positive : list, negative : list, vocab : list, test_proportion=.1) -> (list, list, list):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def step_update(iteration : int) -> float:
    # TODO (extra credit): Update this function to provide an
    # effective iteration dependent step size
    return 0.1

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.mu, lambda x: args.step)

    # Iterations
    update_number = 0
    update = ['update']
    tps = ['TP']
    hps = ['HP']
    tas = ['TA']
    has = ['HA']
    for pp in range(args.passes):
        for ii in train:
            update_number += 1
            lr.sg_update(ii, update_number, use_tfidf=False)
            of = open('results.csv','w')
            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
                update.append(update_number)
                tps.append(train_lp)
                hps.append(ho_lp)
                tas.append(train_acc)
                has.append(ho_acc)


        of.write(",".join([str(i) for i in update]) + "\n")

        of.write(",".join([str(i) for i in tps]) + "\n")

        of.write(",".join([str(i) for i in hps]) + "\n")

        of.write(",".join([str(i) for i in tas]) + "\n")

        of.write(",".join([str(i) for i in has]) + "\n")
    of.close()

    print("Best", lr.best_words(vocab))
    print("Worst", lr.worst_words(vocab))
