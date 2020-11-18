import requests
import pandas
import scipy
import numpy as np
import sys
import matplotlib.pyplot as plt

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:

    response = requests.get(TRAIN_DATA_URL)
    test=pd.read_csv("linreg_test.csv")
    train=pd.read_csv("linreg_train.csv")
    n=np.size(train['area'])
    mx,my=np.mean(train['area'],np.mean(train[price])
    sy=np.sum(train['area']*train['price']-n*my*mx
    sx=np.sum(train['area']*train['area']-n*mx*mx
    a=sy/sx
    b=my-a*mx
    pred=a+area*b
    return pred
    
    # YOUR IMPLEMENTATION HERE
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
