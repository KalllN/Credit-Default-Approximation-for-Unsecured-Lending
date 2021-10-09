ax = plt.gca()
for i in classifiers:
    i.fit(X_balanced, y_balanced)
    plot_roc_curve(i, X_test_balanced, y_test_balanced, ax = ax)
    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    
#keys = pd.Series(data = keys, index = val).sort_index().tolist()
accuracy_table = {'Models' : keys, 'Accuracy Score': val}
final_score = pd.DataFrame(accuracy_table)
ax = final_score.plot('Models', 'Accuracy Score', color = 'black', linestyle = 'solid', linewidth = 2, 
                 marker = 'o', markersize = 10, figsize = (20, 5), ylim = (0.4, 1), 
                 fontsize = 12, rot = 0, xlabel = 'Models').legend(loc = 'upper left', prop = {'size': 15})
