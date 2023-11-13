import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

## Hyperparameters
ps = 64
enc_chs = 5*4*2
stages_count = 5
dec_chs = enc_chs // stages_count
SNR = 13

epochs = 500
batch_size = int(32*(32/ps)**2)
loss_func = nn.MSELoss()

data_dir = "data/raw/imagenet-a/"
JSSC_channel_name = "Channel"

train_count = int(50000 * (32/ps)**2)
test_count = int(10000 * (32/ps)**2)

# Verificar si CUDA está disponible
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PowerNormParts(nn.Module):
    def __init__(self, parts_count, cplx=False, part_last_dim=True):
        super(PowerNormParts, self).__init__()
        self.parts_count = parts_count
        self.cplx = cplx
        self.part_last_dim = part_last_dim

    def forward(self, inputs):
        shape = inputs.shape
        if self.part_last_dim:
            inputs = inputs.reshape(shape[0], -1, shape[-1])
            inputs = inputs.transpose(1, 2)

        flatp = inputs.reshape(shape[0], self.parts_count, -1)
        if self.cplx:
            dsize = flatp.shape[2] // 2
        else:
            dsize = flatp.shape[2]
        dsize_f = float(dsize)

        norm = torch.norm(flatp, dim=2, keepdim=True)
        out = torch.sqrt(torch.tensor(dsize_f)) * flatp / norm
        if self.part_last_dim:
            out = out.reshape(shape[0], shape[-1], -1)
            out = out.transpose(1, 2)
        out = out.reshape(shape)
        return out


## Channel Layer
class Channel(nn.Module):
    def __init__(self, snr, cplx=False):
        super(Channel, self).__init__()
        self.cplx = cplx
        self.set_snr(snr)

    def forward(self, inputs):
        shape = inputs.shape
        gnoise = torch.randn(shape) * self.noise_std
        return inputs + gnoise.to(device)

    def get_snr(self):
        return self.snr

    def set_snr(self, snr):
        self.snr = snr
        if self.cplx:
            self.noise_std = np.sqrt(10**(-snr/10)) / np.sqrt(2)
        else:
            self.noise_std = np.sqrt(10**(-snr/10))


## PSNR Plotter

# PSNR Calculation Function for PyTorch
def psnr(target, prediction, max_val=1.0):
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# PSNR Plotter Function for PyTorch
def PSNR_plotter(x_axis, model, channel, testX, epoch,stages_count=1, goal=None):
    sc = stages_count
    PSNRs = np.zeros((sc, len(x_axis)))
    pre_snr = channel.get_snr()

    for i, snr in enumerate(x_axis):
        channel.set_snr(snr)
        preds = model(testX)
        for j in range(sc):
            PSNRs[j, i] = psnr(testX, preds[j]).mean().item()

    channel.set_snr(pre_snr)

    if sc == 1:
        plt.plot(x_axis, PSNRs[0], label='Model')
    else:
        for i in range(sc):
            plt.plot(x_axis, PSNRs[i], label='Stage_' + str(i+1))
    if goal is not None:
        plt.plot(x_axis, goal, label='Goal')

    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    plt.savefig(f"models/test1/images/{epoch}.png")

# Note: The function assumes that 'testX' and the output of 'model' are PyTorch tensors.

## PSNR Metric
class PSNR_metric(nn.Module):
    def __init__(self):
        super(PSNR_metric, self).__init__()
        self.PSNR_additive = torch.tensor(0.0)
        self.counter = torch.tensor(0.0)

    def forward(self, y_true, y_pred):
        self.PSNR_additive += psnr(y_true, y_pred).mean()
        self.counter += 1
        return self.PSNR_additive / self.counter

    def reset(self):
        self.PSNR_additive = torch.tensor(0.0)
        self.counter = torch.tensor(0.0)

    def compute(self):
        return self.PSNR_additive / self.counter if self.counter != 0 else torch.tensor(0.0)

## Learning rate scheduler
def lr_scheduler(epoch, lr):
  if epoch == 0:
    print("\nlearning_rate: 0.001")
  elif epoch == 20:
    print("\nlearning_rate: 0.0005")
  elif epoch == 30:
    print("\nlearning_rate: 0.0001")

  if epoch < 20:
    return 0.001
  elif epoch < 30:
    return 0.0005
  else:
    return 0.0001
  
## Function caller
class FuncCaller:
    def __init__(self, period, function, *args, **kwargs):
        self.period = period
        self.fn = function
        self.args = args
        self.kwargs = kwargs

    def on_epoch_end(self, epoch):
        if epoch % self.period == 0:
            self.fn(*self.args, **self.kwargs)


## Data Loader
# Load the data from the directory data/raw/imagenet-a/ and return the train and test loaders
# Ruta al conjunto de datos ImageNet-A

def load_data(data_dir, ps, train_count, test_count):
    
    # Transformaciones para el conjunto de datos
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convierte las imágenes a tensores de PyTorch
    ])

    # Cargar el conjunto de datos ImageNet-A
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # DataLoader para PyTorch
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Proceso y división del conjunto de datos en parches
    trainX, testX = [], []
    total_patches = 0
    for img, label in data_loader:
        img = img.squeeze(0)  # Eliminar la dimensión del batch

        # Proceso en parches
        shape = img.shape
        tile_dim0 = shape[1] // ps
        tile_dim1 = shape[2] // ps
        patches = img[:, :tile_dim0 * ps, :tile_dim1 * ps]
        patches = patches.unfold(1, ps, ps).unfold(2, ps, ps)
        patches = patches.contiguous().view(3, -1, ps, ps).permute(1, 0, 2, 3)

        # Añadir parches a la lista
        trainX.extend(patches)
        total_patches += len(patches)
        if total_patches > (train_count + test_count):
            break

    # Convertir la lista a tensor y dividir en entrenamiento y prueba
    trainX = torch.stack(trainX)
    trainX = trainX[:train_count + test_count]
    trainX, testX = trainX[:train_count], trainX[train_count:]

    # Normalizar los datos
    trainX = trainX.float() / 255.0
    testX = testX.float() / 255.0

    return trainX, testX


### Model
## Encoder

class Encoder(nn.Module):
    def __init__(self, out_chs):
        super(Encoder, self).__init__()
        # Definir las capas del encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # padding ajustado para 'same' efecto
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(32, out_chs, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape, img_chs=3):
        super(Decoder, self).__init__()
        # Asumiendo que input_shape es un triple (C, H, W)
        self.conv1 = nn.ConvTranspose2d(input_shape[0], 32, kernel_size=5, stride=1, padding=2)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.ConvTranspose2d(16, img_chs, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))  # Activación Sigmoid en la última capa
        return x

class Model(nn.Module):
    def __init__(self, enc_chs, dec_chs, stages_count, SNR, img_chs=3):
        super(Model, self).__init__()
        # Crear el encoder
        self.encoder = Encoder(out_chs=enc_chs)
        # Crear el normalizador de potencia
        self.powernorm = PowerNormParts(parts_count=stages_count, cplx=True)
        # Crear el canal
        self.channel = Channel(snr=SNR, cplx=True)
        # Crear los decodificadores
        self.decoders = nn.ModuleList()
        for i in range(stages_count):
            self.decoders.append(Decoder(input_shape=(dec_chs*(i+1), None, None), img_chs=img_chs))

    def forward(self, x):
        encoder_out = self.encoder(x)
        power_out = self.powernorm(encoder_out)
        channel_out = self.channel(power_out)
        outputs = []
        for i, decoder in enumerate(self.decoders):
            # Seleccionar las características relevantes para cada decodificador
            decoder_input = channel_out[:, :dec_chs*(i+1), :, :]
            outputs.append(decoder(decoder_input))
        return outputs

known_SNRs  = [1, 4, 7, 13, 19]
test_SNRs   = [1, 4, 7, 10, 13, 16, 19, 22, 25]
known_goals = [[24.636, 25.964, 26.618, 26.945, 27.109, 27.182, 27.236, 27.255, 27.273],
              [23.836, 26.655, 28.091, 28.909, 29.309, 29.527, 29.636, 29.673, 29.709],
              [21.836, 25.945, 28.545, 30.091, 31.055, 31.618, 31.945, 32.091, 32.182],
              [21.036, 23.836, 26.455, 28.854, 30.836, 32.345, 33.382, 34.036, 34.382],
              [20.291, 23.127, 25.891, 28.473, 30.745, 32.655, 34.073, 35.036, 35.600]]

model = Model(enc_chs, dec_chs, stages_count, SNR).to(device)

trainX, testX = load_data(data_dir, ps, train_count, test_count)
trainX = trainX.to(device)
testX = testX.to(device)
train_loader = DataLoader(trainX, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testX, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler(epoch, 0.001))
psnr_metric = PSNR_metric()

def train(model, train_loader, optimizer, loss_func, epoch):
    model.train()
    total_loss = 0
    for data in train_loader:
        inputs = data.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs[-1], inputs)  # Considerar la salida del último decodificador
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

def evaluate_psnr(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_psnr = 0.0
        count = 0
        for data in test_loader:
            inputs = data.to(device)
            outputs = model(inputs)[-1]
            total_psnr += psnr(inputs, outputs).mean().item()
            count += 1

        avg_psnr = total_psnr / count
    model.train()
    return avg_psnr

for epoch in range(epochs):
    min_psnr = 0.0
    train(model, train_loader, optimizer, loss_func, epoch)
    scheduler.step()
    # If the model is the best, save it


    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"models/test_1/{JSSC_channel_name}_model_{epoch}.pth")       

    if epoch % 10 == 0:
        avg_psnr = evaluate_psnr(model, test_loader)
        
        if avg_psnr > min_psnr:
            min_psnr = avg_psnr
            torch.save(model.state_dict(), f"models/test_1/models/{JSSC_channel_name}_model_best.pth")

        PSNR_plotter(x_axis=test_SNRs, model=model, channel=model.channel, testX=testX, epoch=epoch, stages_count=stages_count, goal=None) 
        print(f"Epoch {epoch}, Average PSNR: {avg_psnr}")

model.eval()
with torch.no_grad():
    total_psnr = 0.0
    count = 0
    for data in test_loader:
        inputs = data.to(device)
        outputs = model(inputs)[-1]
        total_psnr += psnr(inputs, outputs).mean().item()
        count += 1
    avg_psnr = total_psnr / count
    PSNR_plotter(x_axis=test_SNRs, model=model, channel=model.channel, testX=testX.to('CPU'), epoch=epoch, stages_count=stages_count, goal=None)
        