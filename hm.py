import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# مسیر فایل
file_path = r"E:\new folder1\household_power_consumption.txt"

# خواندن فایل
data = pd.read_csv(file_path, sep=';')

# نمایش چند خط اول داده‌ها
print(data.head())

# نمایش اطلاعات کلی فایل
print(data.info())

# بررسی داده‌های گمشده
print(data.isnull().sum())  # نمایش تعداد مقادیر گمشده در هر ستون

# نمایش چند ردیف اول برای بررسی نادرستی داده‌ها
print(data.head())

# جایگزینی مقادیر گمشده با میانگین
data['Global_active_power'].fillna(data['Global_active_power'].mean(), inplace=True)

# حذف ردیف‌هایی که دارای مقادیر گمشده هستند
data = data.dropna()

# حذف ردیف‌هایی که مصرف انرژی آنها منفی است
data = data[data['Global_active_power'] >= 0]
# حذف ویژگی‌های غیرضروری مانند 'Date' و 'Time' (که ممکن است برای تحلیل نهایی مهم نباشند)
data = data.drop(columns=['Date', 'Time'])

# نمایش داده‌ها پس از حذف ویژگی‌ها
print(data.head())
from sklearn.preprocessing import MinMaxScaler

# انتخاب ویژگی‌های عددی
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# نمایش داده‌های نرمال‌شده
print(data.head())

# تقسیم داده‌ها به ویژگی‌ها و هدف
X = data.drop(columns=['Global_active_power', 'Date', 'Time'])  # ویژگی‌ها
y = data['Global_active_power']  # هدف

# تقسیم داده‌ها به مجموعه آموزش و تست
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بررسی ابعاد داده‌های آموزش و تست
print(f"Dimensions of X_train: {X_train.shape}")
print(f"Dimensions of X_test: {X_test.shape}")
# مدل رگرسیون خطی
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# پیش‌بینی با مدل رگرسیون خطی
y_pred_lr = lr_model.predict(X_test)

# ارزیابی مدل رگرسیون خطی
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Linear Regression Evaluation:")
print(f"R2 Score: {r2_score(y_test, y_pred_lr)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_lr)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_lr)}")

# مدل درخت تصمیم
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# پیش‌بینی با مدل درخت تصمیم
y_pred_dt = dt_model.predict(X_test)

# ارزیابی مدل درخت تصمیم
print("Decision Tree Evaluation:")
print(f"R2 Score: {r2_score(y_test, y_pred_dt)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_dt)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_dt)}")

# مدل شبکه عصبی
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# پیش‌بینی با مدل شبکه عصبی
y_pred_mlp = mlp_model.predict(X_test)

# ارزیابی مدل شبکه عصبی
print("MLP Evaluation:")
print(f"R2 Score: {r2_score(y_test, y_pred_mlp)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_mlp)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_mlp)}")

# رسم نمودار پیش‌بینی‌ها و مقادیر واقعی برای رگرسیون خطی
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Predicted (Linear Regression)', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Linear Regression)')
plt.legend()
plt.show()

# ذخیره مدل‌ها
import joblib

# ذخیره مدل رگرسیون خطی
joblib.dump(lr_model, 'lr_model.pkl')

# ذخیره مدل درخت تصمیم
joblib.dump(dt_model, 'dt_model.pkl')

# ذخیره مدل شبکه عصبی
joblib.dump(mlp_model, 'mlp_model.pkl')

