import csv, json, os
rootdir= 'AmazonReviews'
outputFileDict = {
    "outputFileProductInfo" : 'preprocessed_productinfo.csv',
    "outputFileReviews" : 'preprocessed_reviewinfo.csv'
}

def readJsonData(rootdir):
    dataArray=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files[:100]: 
            #print(os.path.join(subdir, file))
            with open(os.path.join(subdir, file)) as infile:
                jsonObj = json.loads(infile.read())
                jsonObj['category'] = subdir.split('\\')[1]
                dataArray.append(jsonObj) 
    return dataArray

def getHeaders(dataArray):
    header = {}
    for i in dataArray[0].keys():
        try:
            if(type(dataArray[0][i])==list):
                e= dataArray[0][i][0].keys()
            else:
                e = dataArray[0][i].keys()
            header[i] = list(e)
        except:
            pass
    return header

def writeToFile(dataArray, header,outputFileDict):
    for typeOfFile in header.keys():
        outputFile = "outputFile"+ typeOfFile
        with open(outputFileDict[outputFile],'w', newline='') as outputFile:            
            f = csv.writer(outputFile)
            file_header = ['category']
            headers= header[typeOfFile]
            file_header.extend(headers)
            f.writerow(file_header)
            
            for data in dataArray:
                dataReview = [data['category']]
                try:
                    if type(data[typeOfFile])==list:
                        for review in data[typeOfFile]:
                            for subheader in headers:
                                dataReview.append(review[subheader])
                            if "" not in dataReview:
                                f.writerow(dataReview)
                            dataReview = [data['category']]
                    else:
                        for subheader in headers:
                            dataReview.append(data[typeOfFile][subheader])
                        if None not in dataReview:
                            f.writerow(dataReview)
                except:
                    pass
    
dataArray = readJsonData(rootdir)
header = getHeaders(dataArray)
writeToFile(dataArray, header,outputFileDict)