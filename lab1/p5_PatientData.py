import pandas as pd

# Refer to PatientData.csv: each row is a patient and the last column is the condition
# that the patient has. Do data exploration using Pandas and other visualization tools
# to understand whatever you can about the dataset.
#
# a) How many patients and how many features are there?
#
# b) What is the meaning of the first 4 features? See if you can understand what they mean.
#
# c) Are there missing values? Replace them with the average of the corresponding feature column.
#
# d) How could you test which features strongly influence the patient condition and which do not?



# Part A
results = pd.read_csv('PatientData.csv')

print("Number of patients: ", str(len(results)))
num_features = results.shape[1]-1
print("Number of features ", str(num_features))



# Part B
# First feature is age, second feature is gender (0 for male, 1 for female), third feature is height in cm, fourth feature is weight in kg.



# Part C
columnAvs = []
for column in results.columns:
    columnIter = results[column]
    curSum = 0
    for i in columnIter:
        if i != "?":
            curSum += float(i)
    columnAvs.append(curSum/len(results))
    
for index, row in results.iterrows():
    i = 0
    for column in results.columns:
        if row[column] == "?":
            results.at[index, column] = str(columnAvs[i])
        i+=1

modified_csv = 'modifiedPatientData.csv'
results.to_csv(modified_csv, index=False)



# Part D
# We could use linear regression techniques to see if there is any strong correlation between certain features and the corresponding illnesses.
# Similarly could be used to check which do not have an effect, i.e. which features do not create a linear curve.