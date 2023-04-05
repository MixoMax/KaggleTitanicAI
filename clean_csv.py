import csv

def to_int(s):
    try:
        return int(s)
    except:
        int_str = ""
        for i in range(len(str(s))):
            if str(s)[i] in "0123456789.":
                int_str += str(s)[i]
        if int_str == "":
            return 0
        elif int_str.count(".") > 1:
            return int(int_str.replace(".", ""))
        else:
            return int(float(int_str))

input_file = open("train.csv", "r")

output_csv_array = []

for row in csv.reader(input_file):
    
    output_row = []
    
    for i in range(len(row)):
        if row[0] == "PassengerId":
            print("header")
            if i in [0,1,2,4,5,6,7,8,9,11]:
                output_row.append(row[i])
            
        
        elif i in [0,1,2,5,6,7,9]:
            try:
                output_row.append(to_int(row[i]))
            except:
                print("Error: ", row[i])
        
        elif i == 4:
            output_row.append(0 if row[i] == "male" else 1)
        
        elif i == 8:
            output_row.append(to_int(row[i]))
            
        elif i == 11:
            output_row.append(0 if row[i] == "S" else 1 if row[i] == "C" else 2)
    
    output_csv_array.append(output_row)

with open("clean_train.csv", "w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerows(output_csv_array)

input_file.close()
