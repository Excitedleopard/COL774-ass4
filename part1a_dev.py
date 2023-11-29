import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
import csv
import pandas as pd
import os
import sys
from collections import Counter
import torch.nn.functional as F
from PIL import Image
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 23
torch.manual_seed(seed)
random.seed(seed)

START_TOKEN = "<start>"
END_TOKEN = "<end>"

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_file_path = os.path.join(sys.argv[1], 'SyntheticData/train.csv')
vocab_init = []
images = []

a = time.time()
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    i = 0
    for row in csv_reader:
        print(i)
        if i == 0:
            i += 1
            continue
        if i == 16000:
            break
        i += 1
        vocab_init.append(row[1])
        image_path = os.path.join(sys.argv[1], f"SyntheticData/images/{row[0]}")
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = transformer(image)
        images.append(preprocessed_image)
print(time.time() - a)

START_TOKEN = "<start>"
END_TOKEN = "<end>"


def create_vocab(latex_expressions):
    all_tokens = []
    for latex_expr in latex_expressions:
        tokens = latex_expr.split()
        all_tokens.extend(tokens)
    all_tokens.append(START_TOKEN)
    all_tokens.append(END_TOKEN)
    token_counter = Counter(all_tokens)
    vocab1 = {token: idx for idx, (token, _) in enumerate(token_counter.items(), start=1)}
    vocab2 = {idx: token for idx, (token, _) in enumerate(token_counter.items(), start=1)}
    return vocab1, vocab2


latex_vocab, inv_vocab = create_vocab(vocab_init)
batch_size = 64
learning_rate_encoder = 0.001
learning_rate_decoder = 0.001
embedding_dim = 512
hidden_dim = 512
output_size = len(latex_vocab) + 1  # Add 1 for padding token
num_epochs = 10


class LatexEncoder(nn.Module):
    def __init__(self):
        super(LatexEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)


class LatexDecoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, vocab_size):
        super(LatexDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, input.size(0), -1)
        # Ensure that the hidden tensor is a tuple with hidden state and cell state
        hidden = (hidden, torch.zeros_like(hidden)) if isinstance(hidden, torch.Tensor) else hidden
        combined = torch.cat((embedded, hidden[0]), 2)
        output, hidden = self.lstm(combined)
        output = F.log_softmax(self.fc(output[0]), dim=1)
        return output, hidden


    def generate_latex(self, context_vector, max_length=250):
        # Initialize hidden state with the context vector
        hidden = (context_vector.unsqueeze(0), torch.zeros(1, 1, self.hidden_dim))
        start_token = torch.tensor([latex_vocab[START_TOKEN]], dtype=torch.long)
        output_sequence = [start_token.tolist()]

        for _ in range(max_length):
            embedded = self.embedding(start_token).view(1, 1, -1)
            lstm_output, hidden = self.lstm(embedded, hidden)
            output = F.log_softmax(self.fc(lstm_output[0]), dim=1)
            _, next_token = output.topk(1)
            next_token = next_token.squeeze().detach().view(-1)
            output_sequence.append(next_token.tolist())

            # Check if the sequence has produced the END_TOKEN
            if next_token == latex_vocab[END_TOKEN]:
                break

            start_token = next_token

        return output_sequence





encoder = LatexEncoder().cuda()
decoder = LatexDecoder(output_size, embedding_dim, hidden_dim, len(latex_vocab) + 1).cuda()

criterion_encoder = nn.MSELoss().cuda()
optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate_encoder)
criterion_decoder = nn.CrossEntropyLoss().cuda()
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate_decoder)


class LatexDataset(Dataset):
    def __init__(self, latex_expressions, images):
        self.latex_expressions = latex_expressions
        self.images = images

    def __len__(self):
        return len(self.latex_expressions)

    def __getitem__(self, idx):
        latex_expr = self.latex_expressions[idx]
        image = self.images[idx]
        return {'latex_expr': latex_expr, 'image': image}


# Create dataset and dataloader
latex_dataset = LatexDataset(vocab_init, images)
dataloader = DataLoader(latex_dataset, batch_size=batch_size, shuffle=True)


def train_decoder(decoder, target_tensor, context_vector, criterion, optimizer, use_teacher_forcing=True):
    target_length = target_tensor.size(1)
    batch_size = target_tensor.size(0)

    # Initialize the hidden state with the context vector
    decoder_hidden = context_vector.unsqueeze(0)

    # Use the START_TOKEN as the initial input for all sequences in the batch
    decoder_input = torch.tensor([[latex_vocab[START_TOKEN]]] * batch_size, dtype=torch.long).cuda()

    loss = 0

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        loss += criterion(decoder_output, target_tensor[:, di])

        if use_teacher_forcing:
            decoder_input = target_tensor[:, di].view(batch_size, -1)
        else:
            _, top_index = decoder_output.topk(1)
            decoder_input = top_index.squeeze().detach().view(batch_size, -1)

    optimizer.zero_grad()
    optimizer.step()

    return loss / target_length


def prepare_decoder_input(target_sequences, vocab):
    tokenized_sequences = [seq.split() for seq in target_sequences]

    token_indices = [
        [vocab[token] for token in tokens] + [vocab[END_TOKEN],]
        for tokens in tokenized_sequences
    ]
    # Pad sequences to have the same length
    max_length = max(len(seq) for seq in token_indices)
    padded_indices = [
        seq + [0, ] * (max_length - len(seq))
        for seq in token_indices
    ]
    return torch.tensor(padded_indices, dtype=torch.long)



for epoch in range(10):
    total_loss_decoder = 0.0
    batch_no = 0
    for batch in dataloader:
        print(batch_no)
        batch_no += 1
        images, targets = batch['image'].cuda(), batch['latex_expr']
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        # Forward pass through the encoder
        context_vector = encoder(images)

        # Teacher forcing for 50% of the time during training
        use_teacher_forcing = True if random.random() < 0.5 else False

        # Prepare the target tensor for the entire batch
        target_tensor = prepare_decoder_input(targets, latex_vocab).cuda()

        # Calculate loss for the batch
        loss_decoder = train_decoder(decoder, target_tensor, context_vector, criterion_decoder, optimizer_decoder, use_teacher_forcing)

        # Backward pass and optimization after processing each batch
        loss_decoder.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        # Accumulate the loss
        total_loss_decoder += loss_decoder

    # Print or log the average loss for the epoch
    average_loss_decoder = total_loss_decoder / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Decoder Loss: {average_loss_decoder:.4f}')


def predict(self, images, maxlength=250):
        context_vectors = encoder(images)
        prediction_size = context_vectors.size(0)
        hidden = (context_vectors.unsqueeze(0),)
        decoder_inp = torch.tensor([latex_vocab[START_TOKEN] for _ in range(prediction_size)]).cuda()

        output_sequence = [[torch.tensor([latex_vocab[START_TOKEN]]).cuda()] for _ in range(prediction_size)]

        for ind in range(maxlength):
            decoder_out, hidden = self.forward(decoder_inp, hidden[0])
            _, top_index = decoder_out.topk(1)
            decoder_inp = top_index.squeeze().detach().view(prediction_size, -1)
            curr_output_seq = [list(decoder_inp)]
            for i in range(prediction_size):
                if curr_output_seq[0][i].item() != 0:
                    output_sequence[i].extend(inv_vocab[curr_output_seq[0][i].item()])
#                 else:
#                     output_sequence[i].extend('0') 
            if all(token == latex_vocab[END_TOKEN] for token in decoder_inp):
                break
        for i in range(len(output_sequence)):
            s = ""
            for j in range(1, len(output_sequence[i])):
                s += output_sequence[i][j] + ' '
            s += '$'
            output_sequence[i] = s
        
        return output_sequence



# print(images[0:8].size())
# print(predict(decoder, images[0:8]))
df = pd.read_csv(os.path.join(sys.argv[1],'HandwrittenData/val_hw.csv'))
batch_size = 100
pred = []
image2 = []
with open(os.path.join(sys.argv[1],'HandwrittenData/val_hw.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    i = 0
    for row in csv_reader:
        # print(i)
        if i == 0:
            i += 1
            continue
        i += 1
#         print(row[0])
        image_path = os.path.join(sys.argv[1], f"HandwrittenData/images/train/{row[0]}")
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = transformer(image)
        image2.append(preprocessed_image)


pred = []
for i in range(0, len(df), batch_size):
#     print(len(image2[i:i+batch_size]))
    pred.extend(predict(decoder, torch.stack(image2[i:i+batch_size]).cuda()))
#     print(i)
# print(pred)
df['formula'] = pred
df.to_csv('pred1a.csv', index=False)


df = pd.read_csv(os.path.join(sys.argv[1],'SyntheticData/test.csv'))
batch_size = 100
pred = []
image2 = []
with open(os.path.join(sys.argv[1],'SyntheticData/test.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    i = 0
    for row in csv_reader:
        # print(i)
        if i == 0:
            i += 1
            continue
        i += 1
#         print(row[0])
        image_path = os.path.join(sys.argv[1], f"SyntheticData/images/{row[0]}")
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = transformer(image)
        image2.append(preprocessed_image)


pred = []
for i in range(0, len(df), batch_size):
#     print(len(image2[i:i+batch_size]))
    pred.extend(predict(decoder, torch.stack(image2[i:i+batch_size]).cuda()))
#     print(i)
# print(pred)
df['formula'] = pred
df.to_csv('pred1b.csv', index=False)


def creating_csv(path, out_name):
    batch_size = 100
    df_1 = pd.DataFrame()
    img = []

    img_path = Path(path)
    cnt = 0
    img_names = []
    for image_path in img_path.iterdir():
        cnt += 1
        img.append(image_path)

    df_1['image'] = img

    pred_1 = []
    for i in range(0, cnt, batch_size):
        pred_1.extend(predict(decoder, torch.stack(img[i:i+batch_size]).cuda()))
    df_1['formula'] = pred_1
    df_1.to_csv(out_name, index=False)

# print(creating_csv('dataset/col_774_A4_2023/HandwrittenData/val_hw.csv', 'pred1a.csv'))
