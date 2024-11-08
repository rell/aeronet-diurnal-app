from flask import Flask, request, send_file
import openpyxl
from io import BytesIO
import zipfile
import os
from bs4 import BeautifulSoup
import requests
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lxml
from datetime import timedelta
from timezonefinder import TimezoneFinder
from pytz import timezone

app = Flask(__name__)

'''
Terrell Credle
Back-end Issues:
 - Directories were improperly set Dir + String != Dir/Name -> also instead use ./ notation or join.path from . (universal home dir)
 - Calling bad table name -- when referencing df2['Daily_Occurence'] which caused bad reference error 
 - Bad incoming app.route
 - Did not assign df_full locally before calling it within the outter try of the function.
 - df_full should be re-evaluated - removed its conditional as I saw no previous reference for it -
    compare against your original file to see if it was previously reference and I removed it.
 - Use os.path.join instead of main DIR writting

'''

@app.route('/process_diurnal')
def process_data_route():
    try:
        print("Success: Python server reached")
        # Extract the URL sent from the client
        url = request.get_json() 
        url = url.get("url")
        
        # url = reques.args.get('url')        
        # Set output directory for generated files
        output_dir = "temp_output"
        os.makedirs(output_dir, exist_ok=True)

        # Call the data processing function with the URL
        process_diurnal(url, output_dir)

        # Zip the generated files
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), arcname=file)
        
        zip_buffer.seek(0)

        # Return zip file to the client
        return send_file(zip_buffer, as_attachment=True, download_name="data_output.zip", mimetype="application/zip")
    except Exception as e:
        print(f"Exception occured:\n{e}")

def tz_diff(date, tz1, tz2):              #Returns the difference in hours between timezone1 and timezone2 for a given date.
    date = pd.to_datetime(date)
    return (tz1.localize(date) - tz2.localize(date).astimezone(tz1)).seconds/3600

def process_diurnal(url, output_dir):
    print(url) 
    DIR = os.getcwd()
    TEMP_DIR = os.path.join(DIR, "temp.txt")
        
    print(TEMP_DIR)

    features = ['AOD_340nm','AOD_380nm','AOD_440nm','AOD_500nm','AOD_675nm','AOD_870nm','AOD_1020nm','AOD_1640nm','440-870_Angstrom_Exponent','Precipitable_Water(cm)']
    try:
        request = requests.get(url, stream=True) 
        soup =  BeautifulSoup(request.text, "html.parser")
  
    except Exception as e:
        print(f"R\n{e}")
 
    with open(os.path.join(DIR, "temp.txt"),"w") as oFile:
        oFile.write(str(soup.text))
        oFile.close()
    
    try:
        df = pd.read_csv(os.path.join(DIR, "temp.txt"),skiprows = 6) #reads data into Pandas dataframe
        os.remove(os.path.join(DIR, "temp.txt"))
        url = url.split("/")[4]
        if df.empty:
            print("No data to display. Please retry with different parameters.")    
        else:

            df = df.replace(-999.0, np.nan)             #replaces all -999.9 values with NaN so that averages are properly calculated
            df[['Day','Month','Year']] = df['Date(dd:mm:yyyy)'].str.split(':',expand=True)                            #splits Date column into separate columns
            df['Date'] = df[['Year','Month','Day']].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")   #joins those columns back in a different format
            df['Date']= pd.to_datetime(df['Date']) 
            df['Time(hh:mm:ss)'] = pd.to_datetime(df['Time(hh:mm:ss)'], format='%H:%M:%S').dt.time #converts time column to datetime format for easier manipulation
            diff = tz_diff('2000-01-01',timezone('UTC'), timezone(TimezoneFinder().timezone_at(lng=pd.unique(df[['Site_Longitude(Degrees)']].values.ravel('K'))[0],
                                                        lat=pd.unique(df[['Site_Latitude(Degrees)']].values.ravel('K'))[0])))    #computes the time difference between site and GMT based on coordinates
            if diff > 12:
                diff = diff-24          #this correction converts time difference range from 0-24 hours to [-12, +12] hours
            
            df.insert(3,"Time_Diff", diff)                                       #inserts blank column that will store the computed time difference between GMT and the local time at that location
            df['GMT'] = df.apply(lambda x: pd.Timestamp.combine(x['Date'], x['Time(hh:mm:ss)']), axis=1) #combines date and time together into one timestamp in GMT
            df['LST'] = df['GMT'] + df['Time_Diff'].apply(lambda x: timedelta(hours=x)) #calculates local standard time based on time zone difference from GMT   
            df['B'] = (df['Day_of_Year']-1)*(360/365) #calculates B parameter
            df['E'] = 229.2*(0.000075 + 0.001868*df['B'].apply(math.cos) - 0.032077*df['B'].apply(math.sin) - 0.014615*(2*df['B']).apply(math.cos) - 0.04089*(2*df['B']).apply(math.sin)) #calculates equation of time in minutes
            df['Lst'] = 15*abs((df['GMT']-df['LST']).dt.total_seconds())/3600 #calculates standard meridian
            df['Correction'] = 4*(df['Lst'] - abs(df['Site_Longitude(Degrees)'])) + df['E'] #calculates time difference between solar time and local standard time
            df['Timestamp (solar)'] = (df['LST'] + pd.to_timedelta(df['Correction'], unit='m')).dt.strftime("%Y-%m-%d %H:%M:%S") #calculates local solar time as datetime stamp 
            df['Date_Solar'] = pd.to_datetime(df['Timestamp (solar)']).dt.date
            df['Hour'] = df['GMT'].dt.strftime('%H')   #isolates the hours from the timestamp
            df['Date_Solar'] = pd.to_datetime(df['Date_Solar']) 
            print(df.head)
            for i in range(len(features)):
                df1 = df[['Date_Solar','Hour','Time(hh:mm:ss)']].copy()
                df1.loc[:, features[i]] = df[features[i]]
                df2 = df1.rename(columns={"Date_Solar": "Date", "Time(hh:mm:ss)": "Time"})

                df2['Date'] = pd.to_datetime(df2['Date'])
                df2['Daily_Occurence'] = df2.groupby('Date')['Date'].transform('size')              #creates new Daily Occurence column, which counts number of records for each day
                df_hist = df2[['Date','Daily_Occurence']].drop_duplicates().reset_index(drop=True)
                
                os.makedirs(os.path.join(DIR, "Histograms"), exist_ok=True)
               
                # df.columns['Daily_Occurence'] - Issue header becasue it was dropped
                plt.hist(df_hist['Daily_Occurence'])
                plt.xlabel('Number of Daily Measurements')
                plt.ylabel('Count')
                
                hist_dir = os.path.join(DIR,'Histograms',f'Hist_{url}.png')
                plt.savefig(hist_dir,dpi=200)
                print(hist_dir)
                print(output_dir)
                # plt.savefig(os.path.join(output_dir,hist_dir),dpi=200)
                plt.close()

                df_averaged = df2[df2['Daily_Occurence'] >= 10]    #creates new dataframe that only contains records having more than 10 occurrences per day
                df_averaged = df_averaged.reset_index(drop = True).drop(columns='Daily_Occurence') #resets index and removes Daily Occurence column
                numeric_cols = df2.select_dtypes(include='number').columns.tolist()
                df_daily = df2.groupby('Date')[numeric_cols].mean().reset_index().drop(columns='Daily_Occurence') #creates a new dataframe with just daily averages, whose values will be used to compute absolute differences
                df_daily = df_daily.rename(columns={features[i]:str(features[i])+'_Daily'})
                df_combined = pd.merge(df_averaged, df_daily, on='Date')              #merges daily data frame with the concatenated dataframe from above

                df_combined['Absolute_Diff_'+str(features[i])] = np.nan    #creates blank columns, which represent the absolute differences for AOD, 440-870 Angstrom, WVC
                for j in range(len(df_combined)):      #calculates absolute differences between instantaneous (hourly) bins and daily values.
                    df_combined.loc[j, 'Absolute_Diff_' + str(features[i])] = df_combined.loc[j, features[i]] - df_combined.loc[j, str(features[i]) + '_Daily']

                df_absolute = df_combined.filter(regex='Absolute')
                df_combined = df_combined[['Date','Hour']]
                df_combined = pd.concat([df_combined,df_absolute], axis=1)

                os.makedirs(os.path.join(DIR,"New_Algorithm"), exist_ok=True)

                df_stdev = df_combined.groupby(['Date','Hour']).std().reset_index()
                df_miu = df_combined.groupby(['Date','Hour']).mean().reset_index()

                df_stdev = df_stdev.rename(columns={'Absolute_Diff_'+str(features[i]):str(features[i])+'_Sigma'})
                df_miu = df_miu.rename(columns={'Absolute_Diff_'+str(features[i]):str(features[i])+'_Miu'})
                df_statistics = pd.merge(df_stdev, df_miu)

                df_final = df_statistics.dropna()
                df_final.loc[:, 'Hour'] = df_final['Hour'].astype(int)       #cuts zero from hourly bin column
                df_final = df_final.drop(columns=['Date'])
                df_final.insert(len(df_final.columns),"N_Diff",1)
                df_N_Diff = df_final.groupby(['Hour'])['N_Diff'].sum().reset_index() #sums all number of differences per given hour
                df_final = df_final.drop(columns=['N_Diff'])
                df_final = df_final.groupby(['Hour']).mean().reset_index()   #takes average of all means and standard deviations per given hour
                df_diurnal = pd.merge(df_final,df_N_Diff)
                df_diurnal = df_diurnal.rename(columns={'N_Diff':str(features[i])+'_Diff'})
                if df_full.empty:
                    df_full = df_diurnal
                else:
                     df_full = pd.merge(df_full, df_diurnal, on="Hour", how="outer")

            print(df_diurnal)
            os.makedirs(os.path.join(DIR,"New_Algorithm","Output"), exist_ok=True)
            outdir = os.path.join(DIR,'New_Algorithm',"Output",f"Output_{url}.xlsx")
            df_full = df_full.sort_values(by=['Hour']).reset_index(drop=True)
            df_full.to_excel(outdir,index=False)   #saves table as Excel file
            os.makedirs(os.path.join(DIR,"New_Algorithm","Plots"), exist_ok=True)
            
            for i in range(len(features)):
                # Exception now occuring here: Cant detect -> AOD_340nm_Miu |
                plt.plot(df_full['Hour'], df_full[str(features[i])+'_Miu'])
                plt.xlabel('Hourly Bin')
                plt.ylabel('Diurnal Variability of '+str(features[i]))
                #plt.title("Site Name: "+site)
                plt.fill_between(df_full['Hour'], df_full[str(features[i])+'_Miu']-df_full[str(features[i])+'_Sigma'], df_full[str(features[i])+'_Miu']+df_full[str(features[i])+'_Sigma'], alpha = 0.2)
                plot_dir = os.path.join(DIR,"New_Algorithm","Plots",f"Plot_{url}_{features[i]}.png")
                plt.savefig(os.path.join(output_dir,plot_dir),dpi=200)
                plt.close()

    except Exception as e:
        print(e)
        os.remove(os.path.join(DIR,"temp.txt"))
        print("\nNo data to display. Please retry with different parameters.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
