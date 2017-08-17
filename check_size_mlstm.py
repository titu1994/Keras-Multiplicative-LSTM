import multiplicative_lstm

from keras.layers import Input, LSTM
from keras.models import Model

ip = Input(shape=(1, 100))

lstm = LSTM(128)(ip)
mlstm = multiplicative_lstm.MultiplicativeLSTM(128)(ip)

lstm_model = Model(ip, lstm)
mlstm_model = Model(ip, mlstm)

lstm_model.summary()
print('\n' * 3)

mlstm_model.summary()
print('\n' * 3)

params_count_lstm = lstm_model.count_params()
params_count_mlstm = mlstm_model.count_params()

param_ratio = params_count_mlstm / float(params_count_lstm)
if param_ratio != 1.25:
    print("Param count (mlstm) / Param count (lstm) = %0.2f, should be close to 1.25" % (param_ratio))

print("Size ratio of mLSTM to LSTM is %0.2f!" % (param_ratio))