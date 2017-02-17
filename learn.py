import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import threading
import matplotlib.pyplot as plt

def predictDataPre(): #测试集中每条微博内容包含多少关键词
    avg_data = pd.read_table('data/weibo_train_data.txt', sep='\t', header=None, names=['user', 'blog', 'time', 'forward', 'comment', 'like', 'content'], encoding='utf-8').groupby('user').mean()
    predictData=open('data/weibo_predict_data.txt',encoding='utf-8')
    writer = open('data/training_data_predict_1.txt', 'w+', encoding='utf-8')
    reader=open('data/word_count_test.txt','r',encoding='gbk')
    words=[]
    while 1:
        line=reader.readline()
        if not line:
            break
        params=line.split('\t')
        words.append(params[0])
    # words=['红包','有','？','分享','就','块','一个','不','吧']
    j=0
    lines=predictData.readlines()
    for line in lines:
        params = line.split('\t')
        user = params[0]
        content = params[3]
        wordExist=[-1]*len(words) #初始化为0
        i=0
        for word in words:
            if str(word) in str(content):
                wordExist[i]=1
            i=i+1
        writeString=""
        for w in wordExist:
            writeString=writeString+str(w)+"\t"
        if user in avg_data.index: #如果该用户在训练集中
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

def dataPre(): #训练集中每条微博内容包含的关键词
    predictData = pd.read_table('data/weibo_train_data.txt', sep='\t', header=None, names=['user', 'blog', 'time', 'forward', 'comment', 'like', 'content'], encoding='utf-8')
    writer = open('data/training_data_test_1.txt', 'w+', encoding='utf-8')
    reader=open('data/word_count_test.txt','r',encoding='gbk')
    words=[]
    while 1:
        line=reader.readline()
        if not line:
            break
        params=line.split('\t')
        words.append(params[0])
    # words=['红包','有','？','分享','就','块','一个','不','吧']
    print(words)
    avg_data = predictData.groupby('user').mean()
    j=0
    for content in predictData.content:
        wordExist=[-1]*len(words)
        i=0
        for word in words:
            if str(word) in str(content):
                wordExist[i]=1
            i=i+1
        #每一条微博内容包含多少关键词
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

def learn(data_num,train_num):
    data=[]
    trainData= open("data/training_data_test_1.txt",'r',encoding='utf-8') #读出训练集的结果
    i=0
    while 1:
        line = trainData.readline()
        if not line:
            break
        params = line.split("\t")
        if "\n" in params:
            params.remove("\n")
        row=[]
        for p in params:
            row.append(num(p)) #int(p)
        data.append(row)
        if(i%10000==0):
            print(i)
        if i>data_num:
            break
        i=i+1
    trainData.close()

    data=np.array(data)  #转化成矩阵

    precision_list = []
    for q in range(1,101):
    # t1 = threading.Thread(target=randomFroest,args=(10,data,train_num))
    # threads.append(t1)
    # t2 = threading.Thread(target=randomFroest,args=(None,data,train_num))
    # threads.append(t2)
        precision_list.append(randomFroest(q,data,train_num))
    #     t=threading.Thread(target=randomFroest,args=(q,data,train_num))
    #     threads.append(t)
    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()
    #t.join()
    #writePredict(regr5)
    print(precision_list)
    x=np.linspace(1,100,100)
    y=precision_list
    plt.figure(1)
    plt.plot(x,y)
    plt.show()

def randomFroest(depth,data,train_num):
    training_set_X=data[0:train_num,0:-3]
    training_set_Y=data[0:train_num,-3:]
    test_set_X=data[train_num:,0:-3]
    test_set_Y=data[train_num:,-3:]
    regr5 = RandomForestRegressor(n_estimators =depth)
    regr5.fit(training_set_X, training_set_Y)
    predict=regr5.predict(test_set_X)   #用随机森林进行预测
    predict[predict<0]=0
    predict=np.around(predict).astype(int)
    # print("=====================Random Forest============================")
    # print("depth:"+str(depth))
    # print("Mean squared error: %.2f"
    #       % np.mean((predict - test_set_Y) ** 2))
    # print('Variance score: %.2f' % regr5.score(test_set_X, test_set_Y))
    precision=count_precision(predict,test_set_Y)
    # print("precision:"+str(precision)+"%")
    return precision

def writePredict(reg):
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
        for p in params:
            row.append(num(p))
        data.append(row)
    predictData.close()

    predict=reg.predict(data)  #用训练好的模型进行预测
    predict[predict<0]=0
    predict=np.around(predict).astype(int)

    reader = open('data/weibo_predict_data.txt', 'r', encoding='utf-8')
    writer = open('data/predict_linear_4.txt', 'w', encoding='utf-8')
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

    return fenzi/fenmu*100

if __name__=="__main__":
    # dataPre()
    # predictDataPre()
    learn(10000,1000)
