# final-exam

Running This File
To run the file, you will need to download the Chicago crime data set from https://data.cityofchicago.org/Public-Safety/Chicago-Police-Department-Illinois-Uniform-Crime-R/c7ck-438e. The file is quite large, so I was not able to upload it to github, but if you look at the
early lines of my program, you can see that I reference the file in a different directory with the path name to read it into a Pandas
Dataframe. I chose to only pull variables about date/time, location, beat, community, and crime. The nrows refers to the rows in the 
dataset that only contain 2017-2019 (April 2019). I chose these years to make the data more manageable because few other economics factors seem to affect these years (as compared to, perhaps, 2008-2009 during the Great Recession). 

Once you have downloaded the dataset, the file can by run with the command 'python3 crime_analysis.py'. As a note of warning, I had run times of up to 5 minutes. Furthermore, random forests--which seems like it would work for this dataset--produces memory failures or simply gets killed by my computer for various reasons. Therefore, running my file may work on a different machine, but be warned that it failed on mine.
