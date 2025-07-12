from sklearn.metrics import r2_score, mean_absolute_error

def skill_scores(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
    }
