from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# Normalize the titles
def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ["Jonkheer","Don",'the Countess', 'Dona', 'Lady',"Sir"]:
        return 'Royalty'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

def get_xy_train_model():
    df = pd.read_csv('train.csv')

    # A list with the all the different titles
    titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))

    df['Title'] = df['Name'].map(lambda x: get_title(x))
    df['Title'] = df.apply(replace_titles, axis=1)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna("S", inplace=True)
    df.drop("Cabin", axis=1, inplace=True)
    df.drop("Ticket", axis=1, inplace=True)
    df.drop("Name", axis=1, inplace=True)
    df.Sex.replace(('male','female'), (0,1), inplace = True)
    df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
    df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'),
    (0,1,2,3,4,5,6,7), inplace = True)

    predictors = df.drop(['Survived', 'PassengerId'], axis=1)
    target = df["Survived"]

    return predictors, target

def run_ml_train_model():
    predictors, target = get_xy_train_model()
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.1, random_state = 0)

    randomforest = RandomForestClassifier()
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_val)

    acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
    print("Accuracy: {}".format(acc_randomforest))

    filename = 'titanic_model.sav'
    pickle.dump(randomforest, open(filename, 'wb'))

def run_ml_test_model():
    df_test = pd.read_csv('test.csv')

    df_test['Title'] = df_test['Name'].map(lambda x: get_title(x))
    df_test['Title'] = df_test.apply(replace_titles, axis=1)
    df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
    df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)
    df_test['Embarked'].fillna("S", inplace=True)
    df_test.drop("Cabin", axis=1, inplace=True)
    df_test.drop("Ticket", axis=1, inplace=True)
    df_test.drop("Name", axis=1, inplace=True)
    df_test.drop("PassengerId", axis=1, inplace=True)
    df_test.Sex.replace(('male','female'), (0,1), inplace = True)
    df_test.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
    df_test.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'),
    (0,1,2,3,4,5,6,7), inplace = True)

    ids = df_test['PassengerId']

    predictions = randomforest.predict(df_test)
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
