import numpy as np
import pandas as pd
from sklearn import linear_model,neighbors
from sklearn.ensemble import RandomForestRegressor

def predictDataPre():
    avg_data = pd.read_table('data/weibo_train_data.txt', sep='\t', header=None, names=['user', 'blog', 'time', 'forward', 'comment', 'like', 'content'], encoding='utf-8').groupby('user').mean()
    predictData=open('data/weibo_predict_data.txt',encoding='utf-8')
    writer = open('data/training_data_predict_1.txt', 'w+', encoding='utf-8')
    reader=open('data/word_count_test.txt','r',encoding='gbk')
    words=[]
    while 1:
        line=reader.readline()
        if not line:
            break
        params=line.split('\n')
        words.append(params[0])
    # words=['红包','有','？','分享','就','块','一个','不','吧']
    print(words)
    j=0
    lines=predictData.readlines()
    for line in lines:
        params = line.split('\t')
        user = params[0]
        content = params[3]
        wordExist=[0]*len(words)
        i=0
        for word in words:
            if str(word) in str(content):
                wordExist[i]=1
            i=i+1
        writeString=""
        for w in wordExist:
            writeString=writeString+str(w)+"\t"
        if user in avg_data.index:
            predict = avg_data.loc[user]
            writeString=writeString+str(predict.forward)+"\t"+str(predict.comment)+"\t"+str(predict.like)+"\t"
        else:
            writeString=writeString+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"
        writer.write(writeString+"\n")
        writer.flush()
        j=j+1
        if j%10000==0:
            print(str(j/len(lines)*100)+"%")
    print(j)

def dataPre():
    predictData = pd.read_table('data/weibo_train_data.txt', sep='\t', header=None, names=['user', 'blog', 'time', 'forward', 'comment', 'like', 'content'], encoding='utf-8')
    writer = open('data/training_data_test_1.txt', 'w+', encoding='utf-8')
    reader=open('data/word_count_test.txt','r',encoding='gbk')
    words=[]
    while 1:
        line=reader.readline()
        if not line:
            break
        params=line.split('\n')
        words.append(params[0])
    # words=['红包','有','？','分享','就','块','一个','不','吧']
    print(words)
    avg_data = predictData.groupby('user').mean()
    j=0
    for content in predictData.content:
        wordExist=[0]*len(words)
        i=0
        for word in words:
            if str(word) in str(content):
                wordExist[i]=1
            i=i+1

        writeString=""
        for w in wordExist:
            writeString=writeString+str(w)+"\t"
        if predictData.loc[j].user in avg_data.index:
            predict = avg_data.loc[predictData.loc[j].user]
            writeString=writeString+str(predict.forward)+"\t"+str(predict.comment)+"\t"+str(predict.like)+"\t"
        else:
            writeString=writeString+str(0)+"\t"+str(0)+"\t"+str(0)+"\t"
        writeString=writeString+str(predictData.loc[j].forward)+"\t"+str(predictData.loc[j].comment)+"\t"+str(predictData.loc[j].like)+"\t"
        writer.write(writeString+"\n")
        writer.flush()
        j=j+1
        if j%10000==0:
            print(str(j/len(predictData)*100)+"%")

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def learn(train_num,test_num,word_num):
    output=[]
    output.append(word_num)
    data=[]
    trainData= open("data/training_data_test_1.txt",'r',encoding='utf-8')
    i=0
    while 1:
        line = trainData.readline()
        if not line:
            break
        params = line.split("\t")
        if "\n" in params:
            params.remove("\n")
        row=[]
        j=0
        for p in params:
            if j>=word_num and j<=len(params)-7:
                j=j+1
                continue
            row.append(num(p))
            j=j+1
        data.append(row)
        if(i%10000==0):
            print(i)
        if i>train_num+test_num:
            break
        i=i+1
    trainData.close()

    data=np.array(data)

    training_set_X=data[0:train_num,0:-3]
    training_set_Y=data[0:train_num,-3:]
    test_set_X=data[-test_num:,0:-3]
    test_set_Y=data[-test_num:,-3:]
    # print(training_set_X)
    # print(training_set_Y)
    # print(test_set_X)
    # print(test_set_Y)
    print("=====================Average Data============================")
    predict=data[-test_num:,-6:-3]
    predict=np.around(predict).astype(int)
    # print(predict)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    output.append(count_precision(predict,test_set_Y))
    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % regr.score(test_set_X, test_set_Y))

    print("=====================Ordinary Least Squares============================")
    regr = linear_model.LinearRegression()
    regr.fit(training_set_X, training_set_Y)
    # print('Coefficients: \n', regr.coef_)

    predict=regr.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    # print(predict)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))

    print("=====================Ridge Regression============================")
    regr2 = linear_model.Ridge (alpha = .5)
    regr2.fit(training_set_X, training_set_Y)
    predict=regr2.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr2.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))

    print("=====================Lasso============================")
    regr3 = linear_model.Lasso (alpha = .5)
    regr3.fit(training_set_X, training_set_Y)
    predict=regr3.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr3.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))

    print("=====================LARS Lasso============================")
    regr4 = linear_model.LassoLars(alpha=.1)
    regr4.fit(training_set_X, training_set_Y)
    predict=regr4.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr4.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))

    print("=====================Random Forest============================")
    regr5 = RandomForestRegressor(max_depth=14,n_jobs=-1,min_samples_split=3)
    regr5.fit(training_set_X, training_set_Y)
    predict=regr5.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr5.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))


    print("=====================KNN============================")
    regr6 = neighbors.KNeighborsRegressor(n_jobs=-1)
    regr6.fit(training_set_X, training_set_Y)
    predict=regr6.predict(test_set_X)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    print("Mean squared error: %.2f"
          % np.mean((predict - test_set_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr6.score(test_set_X, test_set_Y))
    output.append(count_precision(predict,test_set_Y))
    writePredict("rf",regr5,word_num)
    writePredict("knn",regr6,word_num)
    # output_writer=open("data/output.txt","a+")
    # output_writer.write(str(output)+"\n")
def writePredict(name,reg,word_num):
    data=[]
    predictData= open("data/training_data_predict_1.txt",'r',encoding='utf-8')
    while 1:
        line = predictData.readline()
        if not line:
            break
        params = line.split("\t")
        if "\n" in params:
            params.remove("\n")
        row=[]
        i=0
        for p in params:
            if(i>=word_num and i<=len(params)-4):
                i=i+1
                continue
            row.append(num(p))
            i=i+1
        data.append(row)
    predictData.close()

    predict=reg.predict(data)
    predict[predict<0]=0
    predict=np.around(predict).astype(int)

    reader = open('data/weibo_predict_data.txt', 'r', encoding='utf-8')
    if name=="rf":
        writer = open('data/predict_linear_rf.txt', 'w', encoding='utf-8')
    else:
        writer = open('data/predict_linear_knn.txt', 'w', encoding='utf-8')
    line = reader.readline()
    j=0
    while 1:
        if not line:
            break;
        params = line.split('\t')
        user = params[0]
        content = params[3]
        forward = 0
        comment = 0
        like = 0


        forward=predict[j][0]
        comment=predict[j][1]
        like=predict[j][2]
        writer.write('%s\t%s\t%d,%d,%d\n' % (params[0], params[1], forward, comment, like))
        line = reader.readline()
        # if j%1000==0:
        #     print(str(j/len(data)*100)+"%")
        j=j+1
        #if i>10:
           #break;

    reader.close()
    writer.close()

    print(j)
def count_precision(predict_data,real_data):
    fenzi = 0
    fenmu = 0
    for i in range(len(predict_data)):
        predict_forward=predict_data[i][0]
        predict_comment=predict_data[i][1]
        predict_like=predict_data[i][2]
        real_forward=real_data[i][0]
        real_comment=real_data[i][1]
        real_like=real_data[i][2]

        deciation_forward=abs(predict_forward-real_forward)/(real_forward+5)
        deciation_comment=abs(predict_comment-real_comment)/(real_comment+3)
        deciation_like=abs(predict_like-real_like)/(real_like+3)

        precision=1-0.5*deciation_forward-0.25*deciation_comment-0.25*deciation_like

        if precision-0.8>0:
            sgn=1
        else:
            sgn=0
        count_cur=real_forward+real_comment+real_like+1
        if count_cur>100:
            count_cur=100

        fenzi+=count_cur*sgn
        fenmu+=count_cur

    print("precision:"+str(fenzi/fenmu*100)+"%")
    return(str(fenzi/fenmu*100)+"%")
if __name__=="__main__":
    # dataPre()
    # predictDataPre()
    # for i in range(270):
    #     print("==========================================")
    #     print("i="+str(i))
    #     print("==========================================")
    #     learn(90000,10000,i)
    learn(10000,1000,10)
