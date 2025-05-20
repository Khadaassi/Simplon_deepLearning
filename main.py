from utils.load_data import load_data
from utils.preprocessor import preprocess_data
from utils.build_pipeline import *
from utils.metrics  import *
import sys

data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def main():

    original_data = preprocess_data(data)
    #print(original_data.shape)
    [X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test] = build_pipeline(original_data)
    model = build_model(X_train_preprocessed, X_val_preprocessed,y_train, y_val )
    ROC, F1, Recall = metrics(model, X_test_preprocessed, y_test)
    """ print(X_train_preprocessed.shape)
    print(X_val_preprocessed.shape)
    print(X_test_preprocessed.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)"""




if __name__ == "__main__":
    main()
