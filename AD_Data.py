import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. โหลดข้อมูลจากไฟล์ Excel
df = pd.read_excel('advertising_data.xlsx')

# 2. กำหนดตัวแปรต้น (Features - 3 ฟีเจอร์) และ ตัวแปรตาม (Target)
X = df[['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']] # 3 ฟีเจอร์งบประมาณในการขายสินค้าผ่านทาง....
y = df['Sales'] # สิ่งที่ต้องการทำนาย ยอดขายสินค้า

# 3. แบ่งข้อมูลเป็นชุดฝึกสอน (Train) 80% และชุดทดสอบ (Test) 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ทำ Feature Scaling (สำคัญมากสำหรับ KNN เพราะใช้ระยะทาง Euclidean Distance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. สร้างและฝึกสอนโมเดล KNN (กำหนดค่า K เช่น k=5)
k = 5
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# 6. ทำนายผลและประเมินประสิทธิภาพโมเดล
y_pred = knn_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- ผลการประเมินโมเดล KNN (K={k}) ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R2): {r2:.4f}")

# 7. ตัวอย่างการนำโมเดลไปทำนายข้อมูลใหม่ (เช่น TV=150, Radio=30, Newspaper=20)
new_data = pd.DataFrame([[150, 30, 20]], columns=['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget'])
new_data_scaled = scaler.transform(new_data)
predicted_sales = knn_model.predict(new_data_scaled)

print(f"\nทำนายยอดขายสำหรับงบโฆษณาใหม่: {predicted_sales[0]:.2f}")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

# โหลดข้อมูลและเตรียมข้อมูล (ใช้ตัวแปรเดิมจากโค้ดก่อนหน้า)
df = pd.read_excel('advertising_data.xlsx')
X = df[['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']]
y = df['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ทดลองรันค่า K ตั้งแต่ 1 ถึง 15
for k in range(1, 16):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f"K = {k:2d} | MSE = {mse:.4f}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าธีมกราฟให้ดูสะอาดและสวยงาม
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_excel('advertising_data.xlsx')

features = ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
colors = ['#FF5733', '#33FF57', '#3357FF']

for i, feature in enumerate(features):
    # สร้าง Scatter Plot พร้อมเส้นแนวโน้ม (Regression Line)
    sns.scatterplot(data=df, x=feature, y='Sales', ax=axes[i], color=colors[i], alpha=0.7, s=70)
    sns.regplot(data=df, x=feature, y='Sales', ax=axes[i], scatter=False, color='black', line_kws={'linestyle':'--'})
    
    axes[i].set_title(f'Sales vs {feature}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Sales (ยอดขาย)', fontsize=12)

plt.tight_layout()
plt.show()