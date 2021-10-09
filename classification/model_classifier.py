keys, val, classifiers = [], [], []
ns_probs = [0 for _ in range(len(y_test_balanced))]
def model_classifier(classifier, name):
    classifier.fit(X_balanced, y_balanced)
    classifiers.append(classifier)
    prediction = classifier.predict(X_test_balanced)
    report = classification_report(y_test_balanced, prediction)#, output_dict=True)
    print(report)
    score = float(accuracy_score(y_test_balanced, prediction))
    print('Accuracy Score is {:.3%}'.format(score))
    keys.append(name)
    val.append(float(score))
    
    ns_probs = [0 for _ in range(len(y_test_balanced))]
    lr_probs = classifier.predict_proba(X_test_balanced)
    lr_probs = lr_probs[:, 1]
    
    ns_auc = roc_auc_score(y_test_balanced, ns_probs)
    lr_auc = roc_auc_score(y_test_balanced, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print(name + ': ROC AUC=%.3f' % (lr_auc))
