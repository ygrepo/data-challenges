from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn import model_selection

#from scoring import compute_bias_metrics_for_model, get_final_metric, calculate_overall_auc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

MAX_NUM_WORDS = 10000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

EMBEDDINGS_PATH = '../input/Glove_6B/glove.6B.100d.txt'
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 1
BATCH_SIZE = 128

MAX_NUM_WORDS = 10000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
MODEL_NAME = 'my_model'

# All comments must be truncated or padded to be the same length.
MAX_SEQUENCE_LENGTH = 250

print(os.listdir("../input"))
print(os.listdir("../input/Glove_6B"))
print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
# train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/small_train.csv')
train = train.head(100)
print('loaded %d records' % len(train))

# Make sure all comment_text values are strings
train['comment_text'] = train['comment_text'].astype(str)

# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# Convert target and identity columns to booleans
def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df


train_df = convert_dataframe_to_bool(train)

# Create a text tokenizer.
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)


def pad_text(texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


def get_embeddings():
    # Load embeddings
    print('loading embeddings')
    embeddings_index = {}
    with open(EMBEDDINGS_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,
                                 EMBEDDINGS_DIMENSION))
    num_words_in_embedding = 0
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            num_words_in_embedding += 1
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        # else:
        #     print("Word not found in embeddings:" + word)
    return embedding_matrix


# Create model layers.
def get_convolutional_neural_net_layers():
    """Returns (input_layer, output_layer)"""

    embedding_matrix = get_embeddings()
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                EMBEDDINGS_DIMENSION,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    x = embedding_layer(sequence_input)
    x = Conv1D(128, 2, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(5, padding='same')(x)
    x = Conv1D(128, 4, activation='relu', padding='same')(x)
    x = MaxPooling1D(40, padding='same')(x)
    x = Flatten()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)
    return sequence_input, preds


def train_model(train_df):
    # Prepare data
    train_data_df, validate_df = model_selection.train_test_split(train_df, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
    tokenizer.fit_on_texts(train_data_df[TEXT_COLUMN])

    train_text = pad_text(train_data_df[TEXT_COLUMN])
    train_labels = to_categorical(train_data_df[TOXICITY_COLUMN])
    validate_text = pad_text(validate_df[TEXT_COLUMN])
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    # Compile model.
    print('compiling model')
    input_layer, output_layer = get_convolutional_neural_net_layers()
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=LEARNING_RATE),
                  metrics=['acc'])
    print(model.summary())
    # Train model.
    print('training model')
    model.fit(train_text,
              train_labels,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_labels),
              verbose=2)

    validate_df[MODEL_NAME] = model.predict(pad_text(validate_df[TEXT_COLUMN]))[:, 1]
    # bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
    # print(bias_metrics_df)
    # final_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))
    # print(final_metric)
    return model


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        embedding_matrix = get_embeddings()
        embedding_tensor = torch.from_numpy(embedding_matrix).double()
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, verbose=False):
        x = self.embedding_layer(x.long())
        # print("1.x={}".format(x.shape))
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        # print("1.x={}".format(x.shape))
        x = self.conv1(x.float())
        x = F.relu(x)
        # print("2.x={}".format(x.shape))
        x = F.max_pool1d(x, 5)
        # print("3.x={}".format(x.shape))
        x = self.conv2(x)
        # print("4.x={}".format(x.shape))
        x = F.relu(x)
        x = F.max_pool1d(x, 5)
        # print("5.x={}".format(x.shape))
        x = self.conv3(x)
        # print("6.x={}".format(x.shape))
        x = F.relu(x)
        x = F.max_pool1d(x, 40)
        # print("7.x={}".format(x.shape))
        x = x.view(x.size(0), -1)
        # print("8.x={}".format(x.shape))
        x = F.dropout(x, DROPOUT_RATE)
        x = self.fc1(x)
        # print("9.x={}".format(x.shape))
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        # print("10.x={}".format(x.shape))
        # x = torch.sigmoid(x)
        # x = F.log_softmax(x, dim=1)
        # print("11.x={}".format(x.shape))
        return x


def xentropy_cost(x_pred, x_target):
    assert x_target.size() == x_pred.size(), "size fail ! " + str(x_target.size()) + " " + str(x_pred.size())
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    return cost_value


def train_pytorch_model(epoch, model, optimizer, train_loader, criterion):
    model.train()  # Prepare data
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def convert_value_to_binary(v):
    return 1. if v > 0.5 else 0.


def convert_array_to_binary(a):
    return np.array([convert_value_to_binary(v) for v in a])


def compare_tensors(a, b):
    a = a.cpu()
    b = b.cpu()
    correct = 0
    for x, y in zip(a, b):
        if torch.equal(x.float(), y.float()):
            correct += 1
    return correct


def predict(data, model):
    data = data.to(device)
    output = model(data)
    return np.apply_along_axis(convert_array_to_binary, 0, output.cpu().detach().numpy())


def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    prediction_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        # send to device
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = np.apply_along_axis(convert_array_to_binary, 0, output.cpu().detach().numpy())
        prediction_list.append([x.tolist() for x in pred])
        pred = torch.from_numpy(pred)
        correct += compare_tensors(pred, target)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset), accuracy))


# function to count number of parameters
def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
cudnn.deterministic = True
cudnn.benchmark = False


def train(device, train_df):
    print('%d train comments' % len(train_df))

    train_data_df, validate_df = model_selection.train_test_split(train_df, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
    tokenizer.fit_on_texts(train_data_df[TEXT_COLUMN])

    train_text = pad_text(train_data_df[TEXT_COLUMN])
    train_labels = np.array(train_data_df[TOXICITY_COLUMN], dtype='int')
    # train_labels = to_categorical(train_data_df[TOXICITY_COLUMN])
    train_tensor = torch_data.TensorDataset(torch.FloatTensor(train_text), torch.FloatTensor(train_labels))
    train_loader = torch_data.DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)

    validate_text = pad_text(validate_df[TEXT_COLUMN])
    validate_labels = np.array(validate_df[TOXICITY_COLUMN], dtype='int')
    # validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])
    validate_tensor = torch_data.TensorDataset(torch.FloatTensor(validate_text), torch.FloatTensor(validate_labels))
    validate_loader = torch_data.DataLoader(validate_tensor, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().float()
    model.to(device)
    print('Number of parameters: {}'.format(get_n_params(model)))
    criterion = nn.BCEWithLogitsLoss()
    # criterion = xentropy_cost
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    # criterion = torch.nn.BCELoss(reduction='sum')
    for epoch in range(0, NUM_EPOCHS):
        train_pytorch_model(epoch, model, optimizer, train_loader, criterion)
        test(model, criterion, validate_loader)

    validate_df.loc[:, MODEL_NAME] = predict(torch.FloatTensor(validate_text), model)
    # print(validate_df)
    # bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
    # print(bias_metrics_df)
    # final_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))
    # print(final_metric)
    return model


def submit_keras_version(train_df):
    model = train_model(train_df)
    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    test_text = pad_text(test[TEXT_COLUMN])
    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv',
                             index_col='id')
    submission['prediction'] = predict(torch.FloatTensor(test_text), model)
    submission.to_csv('submission.csv')

def submit_pytorch(device, train_df):
    try:
        model = train(device, train_df)
        test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
        test = test.head(100)
        test_text = pad_text(test[TEXT_COLUMN])
        submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv',
                             index_col='id')
        submission = submission.head(100)
        submission['prediction'] = predict(torch.FloatTensor(test_text), model)
        submission.to_csv('submission.csv')
    except Exception as ex:
        print(ex)


submit_pytorch(device, train_df)

#submit_keras_version(train_df)
