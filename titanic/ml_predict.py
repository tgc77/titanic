def prediction_model(data):
    import pickle

    x = [[
        data['pclass'],
        data['sex'],
        data['age'],
        data['sibsp'],
        data['parch'],
        data['fare'],
        data['embarked'],
        data['title']
    ]]
    randomforest = pickle.load(open('titanic_model.sav', 'rb'))
    prediction = randomforest.predict(x)
    if prediction == 0:
        return 'Not survived'
    elif prediction == 1:
        return 'Survived'
    else:
        return 'Error'
