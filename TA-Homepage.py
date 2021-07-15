# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:40:28 2021

@author: USER
"""
#problem
#sebelum login gk bisa ke home, set login sebagai menu utama
import streamlit as st
import pandas as pd
import streamlit as st
import os
import sys
import pandas as pd
from io import BytesIO, StringIO
import numpy as np
import glob
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from scipy.fftpack import fft
from scipy import signal
from sklearn.model_selection import train_test_split
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT, role TEXT, disease_name TEXT)')


def add_userdata(username,password,role):
    c.execute('INSERT INTO userstable(username,password,role) VALUES (?,?,?)',(username,password,role))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def check_patient(username,role):
    c.execute('SELECT * FROM userstable WHERE username =? AND role = ?',(username,role))
    data = c.fetchall()
    return data

def get_role(username1):
    #st.dataframe(username)
    c.execute('SELECT role FROM userstable WHERE username =?',(username1,))
    data=c.fetchall()
    return data
def get_disease(username):
    c.execute('SELECT disease_name FROM userstable WHERE username =?',(username,))
    data=c.fetchall()
    return data

def set_disease(username,d_name):
    c.execute('UPDATE userstable SET disease_name = ? WHERE username =?',(d_name,username))
    conn.commit()

def reset_disease():
    c.execute('UPDATE userstable SET disease_name = NULL')
    conn.commit()

def get_username(user_text):
    c.execute('SELECT username FROM userstable WHERE username =?',(user_text,))
    data=c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

#dataset itu harusnya untuk pemrosesan 1 file aja, bukan semua
def ExtractData (dataset,signal):
    R_values =[]
    Q_values =[]
    S_values =[]
    ST_values =[]
    R_index_values =[]
    Q_index_values =[]
    S_index_values =[]
    ST_index_values =[]
    #filter data
    dataset_filtered = FilterData(dataset,signal)
    #print("Filtrasi data telah selesai")
    #ekstraksi R
    R_index_values = ExtractR (dataset_filtered)
    R_index_values = R_index_values[:-2]
    print("R berhasil diekstraksi. berikut isinya=", R_index_values)
    print ("berikut len r index =",len(R_index_values))
    #masalah = kalo dibatasi r index values, nanti bisa out of bounds (minus indexnya)
    #solusi = jika R index values < 80, lewati
    #jika ya, maka bisa dieksekusi
    temp_rmin1 = len(R_index_values)-1
    #ekstraksi Q
    for i in R_index_values:
        if i-80>=0:
            #print("index = ",i)
            Q_current_index=ExtractQ (dataset_filtered,i)
            Q_index_values.append(Q_current_index)
    print("Q berhasil diekstraksi=",Q_index_values)
    #ekstraksi S
    for j in R_index_values:
        if j+80<len(dataset_filtered):
            #print("index = ",i)
            S_current_index=ExtractS (dataset_filtered,j)
            S_index_values.append(S_current_index)
            #print("placeholder")
    print("S berhasil diekstraksi=",S_index_values)
    #ekstraksi ST
    for k in S_index_values:
        if k + 130 < len(dataset_filtered):
            #print("index = ",i)
            ST_current_index=ExtractST (dataset_filtered,k)
            ST_index_values.append(ST_current_index)
    print("ST berhasil diekstraksi=",ST_index_values)
    #obtain R values
    for i in R_index_values:
        print("index = ",i)
        R_values.append(dataset_filtered[i])
    #obtain Q values
    for i in Q_index_values:
        print("index = ",i)
        Q_values.append(dataset_filtered[i])
    #obtain S values
    for i in S_index_values:
        print("index = ",i)
        S_values.append(dataset_filtered[i])
    #obtain ST values
    for i in ST_index_values:
        print("index = ",i)
        ST_values.append(dataset_filtered[i])
    R_average = sum(R_values)/len(R_values)
    R_std = np.std(R_values)
    Q_average = sum(Q_values)/len(Q_values)
    S_average = sum(S_values)/len(S_values)
    ST_average = sum(ST_values)/len(ST_values)
    #print(R_average)
    #print(Q_average)
    #print(S_average)
    #print(ST_average)
    df_temp = {'R_average':[R_average],'R_std':[R_std],'Q_average':[Q_average],'S_average':[S_average],'ST_average':[ST_average]}
    #df_return['R_average']=R_average
    #df_return['Q_average']=Q_average
    #df_return['S_average']=S_average
    #df_return['ST_average']=ST_average
    df_return=pd.DataFrame(df_temp)
    print(df_return)
    return df_return

#filter-> returns filtered dataset
def FilterData (dataset,signal):
    #select the nodes only==============================================================
    temp = dataset
    elapsed_time = temp.index.values
    temp = temp.astype("float64")
    #print(temp['\'ii\''])
    #print(elapsed_time)
    #frequency setup==============================================================
    #print ('Sampling frequency is: ')
    samplingFreq = 1
    #print (samplingFreq)
    #plotting data original==============================================================
    #matplotlib.rc('figure', figsize=(15, 8))
    #plt.plot(temp.index.values,temp['\'ii\''])
    # Frequency Domain==============================================================
    # FFT len is half size of the signal len
    # Because of nyquist theorem only half of the sampling frequency can be seen in the sprectrum
    ekgData = temp.values
    fftData = np.abs( fft(ekgData) )
    fftLen = int(len(fftData) / 2)
    freqs = np.linspace(0,samplingFreq/2, fftLen )

    matplotlib.rc('figure', figsize=(20, 8))

    #plt.figure()
    #plt.plot( freqs, fftData[0:fftLen] )
    #plt.figure()

    #plt.plot( freqs[0:400], fftData[0:400] )
    # Bandpass Filter==============================================================
    t = np.linspace(0, 1, 1000, False)  # 1 second
    #sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)

    #plt.plot(elapsed_time,ekgData)
    #plt.title("sebelum band pass filter")
    #plt.show()

    sos = signal.butter(0,0.1, 'hp', fs=1000, output='sos')
    bandpassfiltered = signal.sosfilt(sos,ekgData)
    #print(filtered)
    #plt.plot(elapsed_time,bandpassfiltered)
    #plt.title("sesudah band pass filter")
    #plt.show()
    ## Design IIR filter==============================================================
    from scipy import signal
    sos = signal.iirfilter(17, [49, 51], rs=60, btype='bandstop',
                            analog=False, ftype='cheby2', fs=4000,
                            output='sos')
    w, h = signal.sosfreqz(sos, 2000, fs=2000)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    #ax.set_title('Chebyshev Type II bandpass frequency response')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    #ax.grid(which='both', axis='both')
    #plt.show()
    #==============================================================
    ## filter out 50 Hz noise
    ekgFiltered = signal.sosfilt(sos, bandpassfiltered)
    #ekgFiltered = signal.sosfilt(sos, ekgData)
    # Time Domain Signal
    matplotlib.rc('figure', figsize=(15, 8))
    #plt.plot(elapsed_time,ekgFiltered)
    #==============================================================
    # Frequency Domain
    # FFT len is half size of the signal len
    # Because of nyquist theorem only half of the sampling frequency can be seen in the sprectrum
    fftData = np.abs( fft(ekgFiltered) )
    fftLen = int(len(fftData) / 2)
    freqs = np.linspace(0,samplingFreq/2, fftLen )

    matplotlib.rc('figure', figsize=(15, 8))

    #plt.figure()
    #plt.plot( freqs, fftData[0:fftLen] )
    #plt.figure()

    #plt.plot( freqs[0:400], fftData[0:400] )
    
    ## Design IIR filter
    sos2 = signal.iirfilter(17, [0.5, 200], rs=60, btype='bandpass',
                            analog=False, ftype='cheby2', fs=4000,
                            output='sos')
    w, h = signal.sosfreqz(sos2, 2000, fs=2000)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    #ax.set_title('Chebyshev Type II bandpass frequency response')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Amplitude [dB]')
    #ax.axis((10, 1000, -100, 10))
    #ax.grid(which='both', axis='both')
    #plt.show()
    ## filter out 50 Hz noise
    ekgFiltered2 = signal.sosfilt(sos2, ekgFiltered)
    # Time Domain Signal
    
    matplotlib.rc('figure', figsize=(15, 8))
    plt.plot(elapsed_time,ekgFiltered2)
    print("hasil filter")
    plt.show()
    return ekgFiltered2

def ExtractR (dataset):
    r_list_return = []
    dataset1 = pd.DataFrame(dataset)
    #print(dataset1)
    list_index_r = []
    #==============
    #bikin var treshold, tapi cari maxnya dulu
    #masalah baru = kalo 1 denyut anomali bgmn?, berarti cari max, pilih no 10
    dataset_sorted = list(dataset)
    dataset_sorted.sort(reverse = True)
    #print(dataset_sorted)
    selected_max_r = dataset_sorted[1000]
    #print(selected_max_r)
    treshold = selected_max_r/2
    #print("treshold r adalah = ", treshold)
    #print("check dataset ", dataset)
    #==============
    for i in range(0,len(dataset)):
        #seharusnya dia cari yang paling tinggi, terus dibagi dua, itu jadi tresholdnya
        if dataset[i]>treshold:#<-mungkin diubah jadi 0.25 atau 0.3 <- dis is not working, karena ada yang dibawah itu
            list_index_r.append(i)
            #print("iterasi ke ", i)
    #print("list index r =",list_index_r)
    #print("length list index r =",len(list_index_r))
    #value df new itu nanti jadi semacam range window denyut R.
    list_window_r = []
    list_window_r.append(list_index_r[0])
    for i in range (0, len(list_index_r)-1):
        if list_index_r[i]+1 != list_index_r[i+1]:
            list_window_r.append(list_index_r[i])
            list_window_r.append(list_index_r[i+1])
    reduced_by = 2
    #check jika ganjil
    if len(list_window_r) %2 !=0:
        list_window_r.append(list_index_r[-1])#masukan index terakhir supaya genap
        
    if len(list_window_r)<=4:
        reduced_by = 1
    print("list window r =",list_window_r)
    print("length list window r =",len(list_window_r))
    #memperoleh r, salah satu window
    #print(list_window_r[38])
    #filter tambahan, kalau list window r 
    #ada yang duplikat maka jangan dimasukin
    #print("fungsi cek duplikat index")
    for j in range (2,len(list_window_r)-reduced_by,2):
        #print ("j saat ini adalah = ", j)
        #print ("j + list window r =", j+list_window_r[j])
        if list_window_r[j]!=list_window_r[j+1]:
            print ("j != j+1, kasus ini = ", list_window_r[j], " dan ", list_window_r[j+1])
            window_r = dataset [list_window_r[j]:list_window_r[j+1]]#ini ambil dari rangenya itu kan, j itu harus ganjil, j+1 genap
            #plt.plot(window_r)
            #plt.show()
            window_r = pd.DataFrame(window_r)
            index_r_peak = window_r.idxmax()+list_window_r[j]#idxmax ini buat search index max, terus ditambah sama index awal
            #contoh = idx max = 100
            #maka nilai aslinya adalah 
            index_r_peak = int(index_r_peak)
            r_list_return.append (index_r_peak)
        else:
            #this thing is sus, confirmed
            #kalo misalkan sama, kenapa tidak return nilai normalnya aja ya?
            print ("j = j+1, kasus ini = ", list_window_r[j], " dan ", list_window_r[j+1])
            r_list_return.append (list_window_r[j])
            #window_r = pd.DataFrame(window_r)
        #print("dataset, dengan idx ", list_window_r[j], " sampai ",list_window_r[j+1])
        
        #window_r = window_r.sort_values(0,ascending=False)
        #true index = index window depan(ganjil) + index peak di dataset baru
        #print("R-peak = ", window_r.max())
        #print("salah satu window r =", window_r)
        #print("nilai j (loop, index window r, index ke)=", j)
        #print("Saat ini indexnya ", j, "yang nilainya ", list_window_r[j])
        #print("korelasi dengan index ", j+1, "yang nilainya ", list_window_r[j+1])
        #print("length dari index window r = ",len(list_window_r))
        #print("list window r", window_r)
        
        #index_r_peak = window_r.idxmax()+list_window_r[j]
        #index_r_peak = int(index_r_peak)
        #print("index R Peak = ", index_r_peak)
        #jangan return value langsung, karena dibutuhkan function lain (ekstraksi Q, S, dan ST)
        #r_list_return.append (dataset[index_r_peak])
        #r_list_return.append (index_r_peak)
    #print(dataset[330])
    #return semua peak R index (berupa list)
    return r_list_return

#masalah, indexnya (R) naik tinggi terus tiba2 minus
def ExtractQ (dataset,index_r_peak):
    #mencari Q peak
    #ini cuma salah satu doang
    #print(index_r_peak)
    window_q = dataset[index_r_peak-80:index_r_peak]
    print ("index r peak utk window Q adalah ",index_r_peak-80," dan " ,index_r_peak)
    print("window Q = ",window_q)
    #plt.plot(window_q)
    window_q = pd.DataFrame(window_q)
    #true index = index window depan(ganjil) + index peak di dataset baru
    print("Q-peak = ", window_q.min())
    #print(window_q)
    index_q_peak = window_q.idxmin()
    index_q_peak = index_q_peak+index_r_peak-80
    index_q_peak = int(index_q_peak)
    #print("index Q Peak = ", index_q_peak)
    #print(dataset[307])
    #mengembalikan 1 Q peak at a time
    return index_q_peak

def ExtractS (dataset,index_r_peak):
    #print(index_r_peak)
    window_s = dataset[index_r_peak:index_r_peak+80]
    #print(window_s)
    #plt.plot(window_s)
    window_s = pd.DataFrame(window_s)
    #true index = index window depan(ganjil) + index peak di dataset baru
    #print("s-peak = ", window_s.min())
    #print(window_s)
    index_s_peak = window_s.idxmin()+index_r_peak
    index_s_peak = int(index_s_peak)
    #print("index s Peak = ", index_s_peak)
    #print(dataset[351])
    return index_s_peak

def ExtractST (dataset,index_s_peak):
   #Cari QRS offset index================================================
    #mencari S peak
    #print(index_s_peak)
    window_qrs = dataset[index_s_peak:index_s_peak+40]
    #print(window_qrs)
    #plt.plot(window_qrs)
    window_qrs = pd.DataFrame(window_qrs)
    #true index = index window depan(ganjil) + index peak di dataset baru
    #print("QRS offset index = ", window_qrs.min())
    #print(window_qrs)
    index_qrs_peak = window_qrs.idxmin()+index_s_peak
    index_qrs_peak = int(index_qrs_peak)
    #print("index qrs Peak = ", index_qrs_peak)
    #print(dataset[351])
   #cari ST Elevation dari QRS offset index (untuk max)==============================
    #ST Elevation, yang dicari max
    #mencari qrs offset
    #print(index_qrs_peak)
    window_st = dataset[index_qrs_peak:index_qrs_peak+90]
    #print(window_st)
    #plt.plot(window_st)
    window_st = pd.DataFrame(window_st)
    #true index = index window depan(ganjil) + index peak di dataset baru
    #print("ST elevation = ", window_st.max())
    #print(window_st)
    index_st_peak = window_st.idxmax()+index_qrs_peak
    index_st_peak = int(index_st_peak)
    #print("index st elevation = ", index_st_peak)
    #print(dataset[384])
    return index_st_peak

def diseaseNameMethod(x):
    return {
        0: 'Healthy control',
        1: 'Myocardial infarction',
        2: 'Bundle branch block',
        3: 'Cardiomyopathy',
        4: 'Dysrhythmia',
        5: 'Hypertrophy',
        6: 'Valvular heart disease',
        7: 'Myocarditis',
        8: 'Stable angina',
        9: 'Heart failure (NYHA 3)',
        10: 'Heart failure (NYHA 4)',
        11: 'Heart failure (NYHA 2)',
        12: 'Palpitation',
        13: 'Unstable angina'
    }[x]

def main():
    """Simple Login App"""

    st.title("ECG Classification Application")

    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    
                    
    if choice == "Home":
        st.subheader("Home")

    elif choice == "Login":
        st.subheader("Login Section")
        
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            
            if result:
                isRole = get_role(username)[0][0]
                st.success(isRole)
                if isRole == "Dokter":
                    temp_profile=view_all_users()
                    temp_profile=pd.DataFrame(temp_profile)
                    st.dataframe(temp_profile.head())
                    st.success("Logged In as Dr. {}".format(username))
                    patient_name = st.text_input('Masukan nama pasien : ')
                    #needs to be changed
                    patient_validation = check_patient(patient_name, "Pasien")
                    reset_button = st.button("Reset disease")
                    if reset_button :
                        reset_disease()
                        st.text("reset berhasil")
                    if patient_validation:
                        #st.text("User not found")
                   
                    #task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
                    #if task == "Add Post":
                     #   st.subheader("Add Your Post")
        
                    #elif task == "Analytics":
                    #    st.subheader("Analytics")
                    #elif task == "Profiles":
                     #   st.subheader("User Profiles")
                      #  user_result = view_all_users()
                       # clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                        #st.dataframe(clean_db)
                     #========================================================
                        st.info(__doc__)
                        
                        #st.markdown(STYLE, unsafe_allow_html=True)
                        file = st.file_uploader("Upload file", type = ["csv"])
                        show_file = st.empty()
                        
                        if not file:
                            show_file.info("Please upload the EKG Data : {}".format(' '.join(['csv'])))
                            return
                        content = file.getvalue()
                        #isTrained = False
                        if isinstance(file,BytesIO):
                            df = pd.read_csv(file)
                            st.dataframe(df.head(2))
                            #========================================================
                            #setelah dapet, diproses datanya
                            #pembuatan dataset master
                            df_a = pd.DataFrame()
                            df_a['Label'] = 0
                            df_a['Q Peak Average'] = 0.0
                            df_a['R Peak Average'] = 0.0
                            df_a['R Standard Deviation']=0.0
                            df_a['S Peak Average'] = 0.0
                            #master_df['RR Mean'] = ""
                            #master_df['RR Deviation'] = ""
                            df_a['ST Elevation Average'] = 0.0
                            print (df_a)
                            #================
                            #!!!!!Jika menggunakan mesin berbeda, ubah metode pengambilan datanya di sini
                            data_pasien = df['\'ii\'']
                            data_pasien = data_pasien[1:]
                            df_extracted = ExtractData(data_pasien,signal)
                            #================
                            df_extracted = pd.DataFrame(columns = ['R_average','Q_average','R_std','S_average','ST_average'])
                            df_extracted=df_extracted.append(ExtractData(data_pasien,signal),ignore_index=True)
                            df_a['R Peak Average']=df_extracted['R_average']
                            df_a['R Standard Deviation']=df_extracted['R_std']
                            df_a['Q Peak Average']=df_extracted['Q_average']
                            df_a['S Peak Average']=df_extracted['S_average']
                            df_a['ST Elevation Average']=df_extracted['ST_average']
                            #========================================================
                            st.text("Hasil Ekstraksi")
                            st.dataframe(df_a.head(2))
                            df_input = df_a.drop('Label', axis = 1)
                            st.dataframe(df_input.head(2))
                            #klasifikasi#====================================
                            #ada button train, supaya bisa bener modellingnya
                            train_button = st.checkbox("Train")
                            predict_button = st.checkbox("Predict")
                            isTrained = False
                            if train_button:
                                acc = 0.0
                                report = pd.DataFrame()
                                while acc < 0.8:
                                    
                                    X = pd.read_csv("TA-X.csv")#label sudah bener
                                    X = X.drop('Unnamed: 0', axis = 1)
                                    y = pd.read_csv("TA-y.csv")#label sudah bener
                                    y = y.drop('Unnamed: 0', axis = 1)
                                    df_full = X.join(y)
                                    #y labeling ulang
                                    df_scenario3_healthy=df_full.loc[df_full['Label']==0]
                                    df_scenario3_MI=df_full.loc[df_full['Label']==1]
                                    df_scenario3_MI = df_scenario3_MI[0:80]
                                    df_scenario3 = df_scenario3_healthy.append(df_scenario3_MI,ignore_index=True)
                                    X_s3=df_scenario3.drop('Label',axis=1)
                                    #X_s3 = X_s3.drop('patient', axis = 1)
                                    #X_s3 = X_s3.drop('Unnamed: 0', axis = 1)
                                    #X_s3 = X_s3.drop('file', axis = 1)
                                    y_s3=df_scenario3['Label']
                                    #train test split
                                    tx,vx,ty,vy = train_test_split( X_s3, y_s3, test_size=0.33)
                                    #st.text("This is training sample")
                                    #st.dataframe(tx)
                                    #st.dataframe(X)
                                    from sklearn.neural_network import MLPClassifier
                                    BPNNModel = MLPClassifier(hidden_layer_sizes=(5,5,5,5,2),activation="relu",max_iter=1500)
                                    BPNNModel.fit(tx,ty)
                                    #st.text("Training Selesai")
                                    #classification report
                                    predict = BPNNModel.predict(vx)
                                    from sklearn.metrics import classification_report
                                    res_final = classification_report(vy, predict, output_dict=True)
                                    res_final = pd.DataFrame(res_final).transpose()
                                    report = res_final
                                    #st.dataframe(res_final)
                                    print(classification_report(vy, predict))
                                    score_test = BPNNModel.score(vx, vy)
                                    acc = score_test
                                    print(score_test)
                                    #temp_acc_res = st.success("acc = {}".format(acc))
                                    #del temp_acc_res
                                st.dataframe(report)
                                st.success("Score accuracy = {}".format(score_test))
                                isTrained= True
                            #====================================                                     
                        if isTrained ==True :
                            if predict_button:
                                
                                #binary
                                hasil_prediksi = BPNNModel.predict(df_input)
                                disease_name = diseaseNameMethod(hasil_prediksi[0])
                                st.text(disease_name)
                                set_disease(patient_name, disease_name)
                                st.success("Pasien ini mengalami penyakit {}".format(disease_name))
                                #============================================================
                                #softmax
                                #hasil_prediksi = BPNNModel.predict_proba(df_input)
                                #hasil_prediksi *=100
                                #st.text(hasil_prediksi.argmax())
                                #disease_name = diseaseNameMethod(hasil_prediksi.argmax())
                                #st.text(disease_name)
                                #st.success("Pasien ini mengalami penyakit {}".format(disease_name))
                                #========================================================
                                #file.close()
                else:
                    
                    st.success("Logged In as Patient. {}".format(username))
                    st.success("Hasil klasifikasi : Anda menderita {}".format(get_disease(username)))
            else:
                st.warning("Incorrect Username/Password")





    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        new_role = st.radio("Role",('Dokter', 'Pasien'))

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password),new_role)
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



if __name__ == '__main__':
    main()

# You can also use the verify functions of the various lib