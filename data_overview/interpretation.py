from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, plot_roc_curve, roc_auc_score
import seaborn as sns

# Income Distribution on the basis of age
df.plot.scatter(x = 'age', y = 'income', figsize = (25, 5), color = '#A1A1A1', xlim = (20, 65), ylim = (50000, 800000), 
                alpha = 0.35, marker = '^', grid = True, fontsize = 12)
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Income', fontsize = 15)

# Employment Duration
labels = ['0 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years']
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#colors = ['#FFDB49', '#FF8C00', '#FF6361', '#00CC66', '#58508D', '#FDF78E', '#AD1F35', '#F1C9C0', '#E6759E', '#ACF5E1']
colors = ['white', '#bdbdbd', '#bababa', '#b0b0b0', '#a1a1a1', '#969696', '#949494', '#8f8f8f', '#878787', '#828282']
df['employment_duration'].value_counts().head(10).plot.pie(explode = explode, autopct='%1.1f%%', 
                        startangle = 90, figsize = (10, 10), labels = labels, colors = colors, fontsize = 12)
plt.ylabel('Employment Duration', fontsize = 15)

# Mean Income Distribution by Status of Loan
new_df = df.groupby(['STATUS']).mean()
new_df.reset_index(inplace = True)
new_df.plot.bar(x = 'STATUS', y = 'income', figsize = (5, 5), fontsize = 10, rot = 0, color = '#A1A1A1', title = 'Mean Income Distribution by Status of Loan')
plt.xlabel('Status', fontsize = 15)

# Median Income Distribution by Status of Loan
med_df = df.groupby(['STATUS']).median()
med_df.reset_index(inplace = True)
med_df.plot.bar(x = 'STATUS', y = 'income', figsize = (5, 5), fontsize = 10, rot = 0, color = '#A1A1A1', title = 'Median Income Distribution by Status of Loan')
plt.xlabel('Status', fontsize = 15)

labels = ['work phone', 'phone', 'email', 'car', 'realty']
new_df.plot.bar(x = 'STATUS', y = ['FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'car', 'FLAG_OWN_REALTY'], 
  figsize = (15, 10), rot = 0, label = labels, fontsize = 12)#, color = ['#E5FF00', '#B2E3B3', '#569D9D', '#CEC9DF', '#FF85DA']
plt.ylabel('Status', fontsize = 15)

# correlation
plt.figure(figsize=(25, 5), dpi = 80)
matrix = np.triu(df.corr())
corr_plot = sns.heatmap(df.corr(), xticklabels = df.corr().columns, yticklabels = df.corr().columns, 
                        cmap = 'gist_gray', center=0, annot=True, mask = matrix)
plt.title('Correlation plot', fontsize = 22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
fig = corr_plot.get_figure()
