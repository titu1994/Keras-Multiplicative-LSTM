# Multiplicative LSTM in Keras
Implementation of the paper [Multiplicative LSTM for sequence modelling](https://arxiv.org/pdf/1609.07959.pdf) for Keras 2.0+. 

Multiplicative LSTMs have been shown to achieve state-of-the-art or close to SotA results for sequence modelling datasets. They also perform better than stacked LSTM models for the Hutter-prize dataset and the raw wikipedia dataset.

## Equation for mLSTM
From the paper, the change in the equations of the general LSTM are : 

<img src="https://github.com/titu1994/Keras-Multiplicative-LSTM/blob/master/images/mlstm_equations.PNG?raw=true" height=100% width=100%>

The size of m_t selected is same as that of h_t, therefore the size of all mLSTM models should be 1.25 times that of the equivalent LSTM model.

## Usage
Add the `multiplicative_lstm.py` script into your repository, and import the MultiplicativeLSTM layer.

Eg. You can replace Keras LSTM layers with MultiplicativeLSTM layers.
```
from multiplicative_lstm import MultiplicativeLSTM
```

## Comparison to LSTM on IMDB dataset
While IMDB is not an idea dataset to compare LSTM models since they overfit rapidly, the weights for the two models have been provided to show the comparison.

1) LSTM Score : 83.20% (overfits after 3 epochs)
2) mLSTM Score : 83.27% (overfits after 7 epochs)
