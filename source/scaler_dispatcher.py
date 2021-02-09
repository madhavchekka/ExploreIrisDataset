# scaler_dispatcher.py

from sklearn import preprocessing

scalers = {
	"standard": preprocessing.StandardScaler(),
	"minmax": preprocessing.MinMaxScaler(),
	"maxabs": preprocessing.MaxAbsScaler()
}