

# In every script, make sure all hyperparameters are enabled 
# Just check a line that looks like : 

#for n_estimators in [1, 10, 25, 35, 40, 50, 100, 200, 500, 1000, 5000]:

# If running script 2 for the first time (i.e. datasets are absent)
# In script 2,  uncomment the following lines : 

# dataset_url = 'https://www.kaggle.com/minbavel/cifar-100-images'
# od.download(dataset_url)

# and use the credentials from 
# kaggle_api = {"username":"varunrawal","key":"16c2afe690b4b18e912ad53dc7424900"}


# If running script 2 for the next (second) time onwards (i.e. datasets are present)
# In script 2,  comment out those lines to avoid getting stuck at the download part

# dataset_url = 'https://www.kaggle.com/minbavel/cifar-100-images'
# od.download(dataset_url)




# Run the scripts : 




python RF_Part1.py > RFP1.log & 
python RF_Part2.py > RF_P2.log & 
python RF_Part3.py > RF_P3.log &

