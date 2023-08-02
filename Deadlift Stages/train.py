import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pickle

def get_args():
    parser = argparse.ArgumentParser(description="Train deadlift bot")
    parser.add_argument("--csv", type=str, required=True, help="Path to csv file")
    return parser.parse_args()

def main(path_to_csv):

    df = pd.read_csv(path_to_csv)
    df_copy = df.copy(deep=True)

    X = df_copy.drop('class', axis=1)
    y = df_copy['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo)
        print('---')
        print('accuracy:', accuracy_score(y_test, yhat))
        print('precision:', precision_score(y_test, yhat, average='micro'))
        print('recall:', recall_score(y_test, yhat, average='micro'))
        print()

    with open('deadlift_bot_rf.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)

if __name__ == '__main__':
    args = get_args()
    main(args.csv)
    