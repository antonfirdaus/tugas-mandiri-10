import flask
import os
from flask import send_from_directory


app = flask.Flask(__name__)

#!/usr/bin/env python
# coding: utf-8

# ![header%20ipynb.png](attachment:header%20ipynb.png)

# # Hands-On
# ---
# 
# Hands-On ini digunakan pada kegiatan Microcredential Associate Data Scientist 2021

# # Tugas Mandiri Pertemuan 10
# ---
# Pertemuan 10 (sepuluh) pada Microcredential Associate Data Scientist 2021 menyampaikan materi mengenai Membangun Model (Dasar Regresi dan Regresi Linier). silakan Anda kerjakan Latihan 1 s/d 20. Output yang anda lihat merupakan panduan yang dapat Anda ikuti dalam penulisan code :)

# # Latihan (1)
# 
# ### Melakukan import library yang dibutuhkan

# In[1]:


# import library pandas
import pandas as pd

# Import library numpy
import numpy as np

# Import library matplotlib dan seaborn untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Import Module LinearRegression digunakan untuk memanggil algoritma Linear Regression.
from sklearn.linear_model import LinearRegression

# import Module train_test_split digunakan untuk membagi data kita menjadi training dan testing set.
from sklearn.model_selection import train_test_split

# import modul mean_absolute_error dari library sklearn
from sklearn.metrics import mean_absolute_error

#import math agar program dapat menggunakan semua fungsi yang ada pada modul math.(ex:sqrt)
import math

# me-non aktifkan peringatan pada python
import warnings 
warnings.simplefilter("ignore")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.joint(app.root.path,'static'),  'favicon.ico', mimetype='image/favicon.png')

@app.route('/')
@app.route('/home')
def home():
    # ### Load Dataset

    # In[2]:


    #Panggil file (load file bernama CarPrice_Assignment.csv) dan simpan dalam dataframe Lalu tampilkan 10 baris awal dataset dengan function head()
    data = pd.read_csv('CarPrice_Assignment.csv')
    dataset = pd.DataFrame(data)
    dataset.head(10)


    # # Latihan (2)
    # 
    # ### Review Dataset

    # In[3]:


    # melihat jumlah baris dan jumlah kolom (bentuk data) pada data df dengan fungsi .shape 
    dataset.shape


    # Data kita mempunyai 26 kolom dengan 205 baris.

    # In[4]:


    # Melihat Informasi lebih detail mengenai struktur DataFrame dapat dilihat menggunakan fungsi info()
    dataset.info()


    # In[5]:


    # melihat statistik data untuk data numeric seperti count, mean, standard deviation, maximum, mininum, dan quartile.
    dataset.describe()


    # In[6]:


    # cek nilai yang hilang / missing values di dalam data
    dataset.isna().sum()


    # Ternyata data kita tidak ada missing values.

    # Simple linear regression atau regresi linear sederhana merupakan jenis regresi yang paling sederhana karena hanya melibatkan satu variabel bebas atau variabel independen X.

    # # Visualisasi data untuk pemilihan fitur / variabel independen X
    # ---
    # 1. Variabel y atau variabel dependent adalah 'price'
    # 2. Lakukan Visualisasi dalam penerapannya agar dapat terlihat jelas / mempermudah dalam membaca data tsb
    # 3. Untuk dapat menentukan variabel X yaitu dapat melihat korelasi antar variabel dengan variabel y / kolom 'price'

    # # Latihan (3)
    # 
    # ### untuk dapat menentukan lebih detail / akurat dalam pemilihan fitur dapat dilihat dari hubungan korelasi nya dengan function corr() 

    # In[7]:


    dataset.corr()


    # tampaknya enginesize, boreratio, horsepower, wheelbase memiliki korelasi yang signifikan dengan harga/price.

    # # Latihan (4)
    # 
    # ### Buat Visualisasi scater plot dari kolom:
    # 
    # 'enginesize', 'boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'wheelbase', 'citympg', 'highwaympg'

    # In[8]:


    def pp(x,y,z):
        sns.pairplot(dataset, x_vars=[x, y, z], y_vars='price', size=4, aspect=1, diag_kind=None)
        plt.show()


    pp('enginesize', 'boreratio', 'stroke')
    pp('compressionratio', 'horsepower', 'peakrpm')
    pp('wheelbase', 'citympg', 'highwaympg')


    # # Latihan (5)
    # 
    # ### Buat Visualisasi Heatmap dari kolom:
    # 
    # 'enginesize', 'boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'wheelbase', 'citympg', 'highwaympg'

    # In[9]:


    plt.figure(figsize = (8,8))
    data_fitur = dataset[['enginesize', 'boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'wheelbase', 'citympg', 'highwaympg']]
    sns.heatmap(data_fitur.corr(),annot=False,fmt="f").set_title("Korelasi Heatmap Calon Variabel X")
    plt.show()


    # Dari hasil visualisasi diatas bahwa fitur/kolom enginesize memiliki korelasi yang tinggi terhadap kolom price / variabel dependent sehingga kita mengambil fitur/kolom enginesize untuk di training
    # * Independent variabel(x) adalah enginesize.
    # * Dependent variabel(y) adalah price.

    # # Latihan (6)
    # 
    # ### Buat Visualisasi Scatter Plot antara calon variabel X(enginesize) dan y(price):

    # In[10]:


    plt.scatter(dataset['enginesize'], dataset['price'])
    plt.xlabel('enginesize')
    plt.ylabel('price')
    plt.title('Scatter Plot enginesize vs Price')
    plt.show()


    # Scatter plot menunjukkan dengan jelas hubungan antarvariabel serta sebarannya di dataset. Selain itu, dengan scatter plot juga kita dapat mengindikasikan bahwa variabel enginesize dan price memiliki hubungan linear.

    #     Catatan : korelasi 0.874145 adalah nilai yang cukup tinggi, artinya nilai price benar-benar sangat dipengaruhi oleh nilai enginesize, karena korelasi tinggi maka algoritma Regresi Linier ini cocok digunakan untuk data tersebut.

    # # Latihan (7)
    # 
    # ### definisi variabel X(enginesize) dan y(price):

    # In[11]:


    # Prepare data
    # Pertama, buat variabel x dan y.
    x = dataset['enginesize'].values.reshape(-1,1)
    y = dataset['price'].values.reshape(-1,1)


    # Formula Regresi Linear
    # ![gambar.png](attachment:gambar.png)
    # 
    # 
    # "Jika kita melihat formula regresi linear di atas, kita pasti ingat rumus persamaan garis yang pernah dipelajari di bangku sekolah, yaitu y = mx + c, dimana m merupakan gradien atau kemiringan garis dan c merupakan konstanta."

    # * from scratch
    # * y = ax + b atau y = w1x + w0 atau y = mx + c
    # * x = input
    # * y = output
    # * b atau w0 = intercept / bias
    # * a atau w1 = slope / gradient / coefficient

    # # Latihan (8)
    # 
    # ### definisi variabel nilai mean/rata-rata X(enginesize) dan nilai mean/rata-rata y(price):

    # In[12]:


    x_mean = np.mean(x)
    y_mean = np.mean(y)
    print('nilai mean var x: ', x_mean,'\n'
        'nilai mean var y: ', y_mean)


    # # Latihan (9)
    # 
    # ### carilah nilai koefisien korelasi nya dengan rumus dibawah:

    # ![gambar.png](attachment:gambar.png)

    # In[13]:


    atas = sum((x - x_mean)*(y - y_mean))
    bawah = math.sqrt((sum((x - x_mean)**2)) * (sum((y - y_mean)**2)))
    correlation = atas/bawah
    print('Nilai Correlation Coefficient: ', correlation)


    # 
    # ### carilah nilai parameter theta 1 dan theta 0 dengan rumus dibawah:

    # 
    # ![gambar.png](attachment:gambar.png)

    #     theta_1 = ((111-104.11) * (13495-13276.71)) + ... + ((114-104.11) * (22625-13276.71)) / ((111-104.11)^2 + ... + (114-104.11)^2)

    # # Latihan (10)
    # 
    # ### carilah nilai theta_1 atau nilai slope

    # In[14]:


    # slope
    # Slope adalah tingkat kemiringan garis, intercept 
    # adalah jarak titik y pada garis dari titik 0
    variance = sum((x - x_mean)**2)
    covariance = sum((x - x_mean) * (y - y_mean))
    theta_1 = covariance/variance
    print('Nilai theta_1: ',theta_1)


    # # Latihan (11)
    # 
    # ### carilah nilai theta_0 atau nilai intercept 

    # In[15]:


    # intercept
    theta_0 = y_mean - (theta_1 * x_mean)
    print('Nilai theta_1: ',theta_0)


    # ### Maka persamaan garis : 
    # 
    # #     **y = 167.69x - 8005.44**
    # 
    # Jadi persamaan garis diatas dapat digunakan untuk melakukan prediksi apabila kita memiliki data enginesize yang baru, price dapat diperkirakan dengan rumus tersebut, masukkan nilai enginesize baru ke x, maka perkiraan nilai y (price) akan didapat.

    # # Latihan (12)
    # 
    # ### carilah nilai prediksi secara manual dan buatlah visualisasi scater plot nya

    # In[16]:


    # prediction manual
    y_pred = theta_0 + (theta_1 * 130)

    print(y_pred)


    # In[17]:


    # visualisasi prediksi dengan scatter plot
    y_pred = theta_0 + (theta_1 * x)

    plt.scatter(dataset['enginesize'], dataset['price'])
    plt.plot(x, y_pred, c='r')
    plt.xlabel('enginesize')
    plt.ylabel('Price')
    plt.title('Plot enginesize vs Price')


    # Linier Regression digunakan untuk Prediksi dengan mencari pola garis terbaik antara variable independent dan dependen
    # 
    # Pros:
    # 
    #     Mudah diimplementasikan
    #     Digunakan untuk memprediksi nilai numerik/ continous /data jenis interval dan ratio
    # 
    # Cons :
    # 
    #     Cenderung mudah Overfitting
    #     Tidak dapat digunakan bila relasi antara variabel independen dan dependen tidak linier atau korelasi variabel rendah

    # # Linier Regression dengan menggunakan library sklearn

    # 1. Pertama yang kita lakukan adalah split data, Train/test split adalah salah satu metode yang dapat digunakan untuk mengevaluasi performa model machine learning. Metode evaluasi model ini membagi dataset menjadi dua bagian yakni bagian yang digunakan untuk training data dan untuk testing data dengan proporsi tertentu. Train data digunakan untuk fit model machine learning, sedangkan test data digunakan untuk mengevaluasi hasil fit model tersebut.
    # 
    # ![image.png](attachment:image.png)
    # 
    # 
    # Python memiliki library yang dapat mengimplementasikan train/test split dengan mudah yaitu Scikit-Learn. Untuk menggunakannya, kita perlu mengimport Scikit-Learn terlebih dahulu, kemudian setelah itu kita dapat menggunakan fungsi train_test_split().

    # # Latihan (13)
    # 
    # ### split data train dan test dengan function  train_test_split() dengan train_size=0.8, test_size=0.2 dan random_state=100

    # In[18]:


    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 100)


    # * X_train: Untuk menampung data source yang akan dilatih.
    # * X_test: Untuk menampung data target yang akan dilatih.
    # * y_train: Untuk menampung data source yang akan digunakan untuk testing.
    # * y_test: Untuk menampung data target yang akan digunakan untuk testing.
    # 
    # X dan y adalah nama variabel yang digunakan saat mendefinisikan data source dan data target. Parameter *test_size* digunakan untuk mendefinisikan ukuran data testing. Dalam contoh di atas, test_size=0.2 berarti data yang digunakan sebagai data testing adalah sebesar 20% dari keseluruhan dataset.
    # 
    # Perlu diketahui bahwa metode ini akan membagi train set dan test set secara random atau acak. Jadi, jika kita mengulang proses running, maka tentunya hasil yang didapat akan berubah-ubah. Untuk mengatasinya, kita dapat menggunakan parameter *random_state*

    # # Latihan (14)
    # 
    # ### buat object variabel linier regression

    # In[19]:


    regressor = LinearRegression()


    # # Latihan (15)
    # 
    # ### training the model menggunakan training data yang sudah displit sebelumnya.

    # In[20]:


    regressor.fit(X_train, y_train)


    # # Latihan (16)
    # 
    # ### cari tau nilai slope/koefisien (m) dan intercept (b), dengan menggunakan function dari library sklearn -> LinierRegression

    # In[21]:


    print(regressor.coef_)
    print(regressor.intercept_)


    # Dari nilai m dan b diatas, kalau dimasukan ke dalam rumus persamaan menjadi:
    # #     **y = 168.17x - 8037.06**

    # # Latihan (17)
    # 
    # ### cari tahu accuracy score dari model kita menggunakan testing data yang sudah displit sebelumnya. Dan nilai korelasinya

    # In[22]:


    regressor.score(X_test, y_test)


    # Model kita mendapatkan accuracy score sebesar 80.68%

    # In[23]:


    print('Correlation: ', math.sqrt(regressor.score(X_test,y_test)))


    # # Latihan (18)
    # 
    # ### visualisasi Regression Line menggunakan data testing.

    # In[24]:


    y_prediksi = regressor.predict(X_test)

    plt.scatter(X_test,y_test)
    plt.plot(X_test, y_prediksi, c='r')
    plt.xlabel('enginesize')
    plt.ylabel('Price')
    plt.title('Plot enginesize vs Price')


    # Garis merah merupakan Regression Line dari model yang telah dibuat sebelumnya.

    # # Latihan (19)
    # 
    # ### Setelah kita yakin dengan model yang dibuat, selanjutnya adalah prediksi dari harga mobil dengan enginesize 100, 150, dan 200.

    # In[25]:


    #Prediksi harga mobil dengan enginesize 130.

    print('nilai prediksi harga dengan enginesize 100 : ',regressor.predict([[100]]))
    print('nilai prediksi harga dengan enginesize 150 : ',regressor.predict([[150]]))
    print('nilai prediksi harga dengan enginesize 200 : ',regressor.predict([[200]]))


    # In[26]:


    np_table = np.concatenate((X_test,y_test,y_prediksi), axis=1)
    new_dataframe = pd.DataFrame(data=np_table, columns=['x_test','y_test','y_predict'])


    # In[27]:


    new_dataframe


    # Semakin tinggi nilai error, semakin besar errornya

    # # Latihan (20)
    # 
    # ### Cetak nilai Mean Absolute Error, Mean Squared Error, dan Root Mean Squared Error

    # In[28]:


    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prediksi))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_prediksi))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_prediksi)))    


    # In[29]:


    plt.title('Comparison of Y values in test and the Predicted values')
    plt.ylabel('Test Set')
    plt.xlabel('Predicted values')
    plt.plot(y_prediksi, '.', y_test, 'x')
    plt.show()

    return "Hasil Eksekusi"

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()
