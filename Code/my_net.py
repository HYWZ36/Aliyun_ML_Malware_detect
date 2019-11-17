import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, CuDNNLSTM, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import time
import numpy as np
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import csv
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold


my_security_train = './my_security_train.pkl'
my_security_test = './my_security_test.pkl'
my_result = './my_result1.pkl'
my_result_csv = './my_result1.csv'
inputLen = 5000
# config = K.tf.ConfigProto()
# # 程序按需申请内存
# config.gpu_options.allow_growth = True
# session = K.tf.Session(config = config)


# 读取文件到变量中
with open(my_security_train, 'rb') as f:
    train_labels = pickle.load(f)
    train_apis = pickle.load(f)
with open(my_security_test, 'rb') as f:
    test_files = pickle.load(f)
    test_apis = pickle.load(f)


# print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
# tensorboard = TensorBoard('./Logs/', write_images=1, histogram_freq=1)
# print(train_labels)
# 将标签转换为空格相隔的一维数组
train_labels = np.asarray(train_labels)
# print(train_labels)


tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)
# print(train_apis)
# 通过训练和测试数据集丰富取词器的字典，方便后续操作
tokenizer.fit_on_texts(train_apis)
# print(train_apis)
# print(test_apis)
tokenizer.fit_on_texts(test_apis)
# print(test_apis)
# print(tokenizer.word_index)
# #获取目前提取词的字典信息
# # vocal = tokenizer.word_index
train_apis = tokenizer.texts_to_sequences(train_apis)
# 通过字典信息将字符转换为对应的数字
test_apis = tokenizer.texts_to_sequences(test_apis)
# print(test_apis)
# 序列化原数组为没有逗号的数组，默认在前面填充,默认截断前面的
train_apis = pad_sequences(train_apis, inputLen, padding='post', truncating='post')
# print(test_apis)
test_apis = pad_sequences(test_apis, inputLen, padding='post', truncating='post')




# print(test_apis)


def SequenceModel():
    # Sequential()是序列模型，其实是堆叠模型，可以在它上面堆砌网络形成一个复杂的网络结构
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=6000))
    model.add(Dense(8, activation='softmax'))
    return model




def lstm():
    my_inpuy = Input(shape=(6000,), dtype='float64')
    # 在网络第一层，起降维的作用
    emb = Embedding(len(tokenizer.word_index) + 1, 5, input_length=6000)
    emb = emb(my_inpuy)
    net = Conv1D(16, 3, padding='same', kernel_initializer='glorot_uniform')(emb)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Conv1D(32, 3, padding='same', kernel_initializer='glorot_uniform')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPool1D(pool_size=4)(net)


    net1 = Conv1D(16, 4, padding='same', kernel_initializer='glorot_uniform')(emb)
    net1 = BatchNormalization()(net1)
    net1 = Activation('relu')(net1)
    net1 = Conv1D(32, 4, padding='same', kernel_initializer='glorot_uniform')(net1)
    net1 = BatchNormalization()(net1)
    net1 = Activation('relu')(net1)
    net1 = MaxPool1D(pool_size=4)(net1)


    net2 = Conv1D(16, 5, padding='same', kernel_initializer='glorot_uniform')(emb)
    net2 = BatchNormalization()(net2)
    net2 = Activation('relu')(net2)
    net2 = Conv1D(32, 5, padding='same', kernel_initializer='glorot_uniform')(net2)
    net2 = BatchNormalization()(net2)
    net2 = Activation('relu')(net2)
    net2 = MaxPool1D(pool_size=4)(net2)


    net = concatenate([net, net1, net2], axis=-1)
    net = CuDNNLSTM(256)(net)
    net = Dense(8, activation='softmax')(net)
    model = Model(inputs=my_inpuy, outputs=net)
    return model




def textcnn():
    kernel_size = [1, 3, 3, 5, 5]
    acti = 'relu'
    # 可看做一个文件的api集为一句话，然后话中的词总量是6000
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index) + 1, 20, input_length=inputLen)(my_input)
    emb = SpatialDropout1D(0.2)(emb)


    net = []
    for kernel in kernel_size:
        # 32个卷积核
        con = Conv1D(32, kernel, activation=acti, padding="same")(emb)
        # 滑动窗口大小是2,默认输出最后一维是通道数
        con = MaxPool1D(2)(con)
        net.append(con)
    # print(net)
    # input()
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    # print(net)
    # input()
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(8, activation='softmax')(net)
    model = Model(inputs=my_input, outputs=net)
    return model


test_result = np.zeros(shape=(len(test_apis),8))


# print(train_apis.shape)
# print(train_labels.shape)
# 5折交叉验证，将训练集切分成训练和验证集
skf = StratifiedKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(skf.split(train_apis, train_labels)):
    # print(i)
    # model = SequenceModel()
    model = textcnn()


    # metrics默认只有loss，加accuracy后在model.evaluate(...)的返回值即有accuracy结果
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #模型保存规则
    model_save_path = './my_model/my_model_{}.h5'.format(str(i))
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    #早停规则
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', baseline=None,
                            restore_best_weights=True)
    #训练的过程会保存模型并早停
    model.fit(train_apis[train_index], train_labels[train_index], epochs=100, batch_size=1000,
              validation_data=(train_apis[valid_index], train_labels[valid_index]), callbacks=[checkpoint, earlystop])
    model.load_weights(model_save_path)
    # print(train_index, valid_index)


    test_tmpapis = model.predict(test_apis)
    test_result = test_result + test_tmpapis


# loss, acc = model.evaluate(train_apis, train_labels)
# print(loss)
# print(acc)
# print(model.predict(train_apis))


# print(test_files)
# print(test_apis)
test_result = test_result/5.0
with open(my_result, 'wb') as f:
    pickle.dump(test_files, f)
    pickle.dump(test_result, f)


# print(len(test_files))
# print(len(test_apis))




result = []
for i in range(len(test_files)):
    # #     print(test_files[i])
    #     #之前test_apis不带逗号的格式是矩阵格式，现在tolist转为带逗号的列表格式
    #     print(test_apis[i])
    #     print(test_apis[i].tolist())
    #     result.append(test_files[i])
    #     result.append(test_apis[i])
    tmp = []
    a = test_result[i].tolist()
    tmp.append(test_files[i])
    # extend相比于append可以添加多个值
    tmp.extend(a)
    #     print(tmp)
    result.append(tmp)
# print(1)
# print(result)


with open(my_result_csv, 'w') as f:
    #     f.write([1,2,3])
    result_csv = csv.writer(f)
    result_csv.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"])
    result_csv.writerows(result)