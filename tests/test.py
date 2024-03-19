# %%
import unittest
import numpy as np
import sys
import pandas as pd

from scipy import stats


class TestRNN(unittest.TestCase):

    def setUp(self):
        self.root = ".."

    def test_rnn_loss(self):

        df = pd.read_csv(f"{self.root}/assignments/rnn_loss_values.csv")
        loss_values = df["0"].values.reshape(-1)
        # Check if the loss is decreasing
        rho = stats.pearsonr(loss_values, np.arange(len(loss_values)))[0]
        self.assertTrue(rho < 0)

    def test_rnn_prediction(self):
        df = pd.read_csv(f"{self.root}/assignments/rnn_test_predictions.csv")
        predictions = df["0"].values.reshape(-1)
        y = pd.read_csv(f"{self.root}/data/test.csv")["country"].values
        acc = np.mean(predictions == y)
        self.assertTrue(acc > 1.0 / 52)


class TestLSTM(unittest.TestCase):

    def setUp(self):
        self.root = ".."

    def test_lstm_loss(self):

        df = pd.read_csv(f"{self.root}/assignments/lstm_loss_values.csv")
        loss_values = df["0"].values.reshape(-1)
        # Check if the loss is decreasing
        rho = stats.pearsonr(loss_values, np.arange(len(loss_values)))[0]
        self.assertTrue(rho < 0)

    def test_lstm_prediction(self):
        with open(
            f"{self.root}/assignments/lstm_test_predictions.txt", "r", encoding="utf8"
        ) as fp:
            predictions = fp.read()
        predictions = list(predictions)

        with open("../data/the-foundation-test.txt", "r", encoding="utf8") as fp:
            eval_text = fp.read()

        eval_text = eval_text.replace("\n", " ")
        eval_text = eval_text.replace("\r", " ")

        _, targets = seq2input_target(eval_text, window_length=30)

        acc = np.mean(predictions == np.array(targets))
        rand_acc = 0.0120

        self.assertTrue(acc > rand_acc)


def seq2input_target(seq, window_length):
    input_text = [
        list(seq[i : i + window_length]) for i in range(len(seq) - window_length)
    ]
    target_text = list(seq[window_length:])
    return input_text, target_text


if __name__ == "__main__":
    unittest.main()

# %%
