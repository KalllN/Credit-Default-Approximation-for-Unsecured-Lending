from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

mms = MinMaxScaler()
X_scaled = pd.DataFrame(mms.fit_transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(mms.transform(X_test), columns = X_test.columns)

oversample = SMOTE()
X_balanced, y_balanced = oversample.fit_resample(X_scaled, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scaled, y_test)
