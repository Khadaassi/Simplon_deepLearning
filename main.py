from utils.load_data import load_data
from utils.preprocessor import preprocess_data
from utils.build_pipeline import *
from utils.metrics  import *
import sys

data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def main():
    try:
        data = load_data("model/data/dataset_assurance.csv")
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}", file=sys.stderr)
        sys.exit(1)
    pass




if __name__ == "__main__":
    main()
