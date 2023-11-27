import numpy as np
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt

## Hyperparameters
# Hyperparameters
ps = 64  # Pixel size
enc_chs = 5 * 4 * 2  # Encoder channels, calculated as a product of factors
stages_count = 5  # Number of stages in the model
dec_chs = enc_chs // stages_count  # Decoder channels, derived from encoder channels
SNR = 13  # Signal-to-Noise Ratio

# Training parameters
epochs = 500  # Total number of training epochs
batch_size = int(32 * (32 / ps) ** 2)  # Calculated batch size based on pixel size
mse = nn.MSELoss()  # Mean Squared Error loss function

# Data directory and channel name
data_dir = "data/raw/small_imagenet-a/"  # Directory for the ImageNet-A dataset
JSSC_channel_name = "Channel"  # Name of the channel for JSSC

# Dataset size configuration
train_count = int(50000 * (32 / ps) ** 2)  # Calculated number of training samples
test_count = int(10000 * (32 / ps) ** 2)  # Calculated number of test samples

# Device configuration for PyTorch
global DEVICE  # Declaring DEVICE as a global variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Using GPU if available
print(f"DEVICE: {DEVICE}")  # Printing the device information
psnr = PeakSignalNoiseRatio().to(DEVICE)  # Initializing PSNR metric and moving it to the device

# Class definition for PowerNormParts
class PowerNormParts(nn.Module):
    def __init__(self, parts_count, cplx=False, part_last_dim=True):
        """
        Initialize the PowerNormParts module.
        :param parts_count: Number of parts to divide the input into.
        :param cplx: Boolean indicating if complex numbers are used.
        :param part_last_dim: Boolean indicating if the last dimension is used for parts.
        """
        super(PowerNormParts, self).__init__()
        self.parts_count = parts_count
        self.cplx = cplx
        self.part_last_dim = part_last_dim

    def forward(self, inputs):
        """
        Forward pass of the PowerNormParts module.
        :param inputs: Input tensor to be processed.
        :return: Normalized output tensor.
        """
        # Reshaping and transposing inputs if necessary
        shape = inputs.shape
        if self.part_last_dim:
            inputs = inputs.reshape(shape[0], -1, shape[-1])
            inputs = inputs.transpose(1, 2)

        # Processing inputs
        flatp = inputs.reshape(shape[0], self.parts_count, -1)
        if self.cplx:
            dsize = flatp.shape[2] // 2
        else:
            dsize = flatp.shape[2]
        dsize_f = float(dsize)

        # Normalizing the inputs
        norm = torch.norm(flatp, dim=2, keepdim=True)
        out = torch.sqrt(torch.tensor(dsize_f)) * flatp / norm
        if self.part_last_dim:
            out = out.reshape(shape[0], shape[-1], -1)
            out = out.transpose(1, 2)
        out = out.reshape(shape)
        return out



class Channel(nn.Module):
    def __init__(self, snr, cplx=False):
        """
        Initialize the Channel module.
        :param snr: Signal-to-Noise Ratio for the channel.
        :param cplx: Boolean indicating if complex numbers are used.
        """
        super(Channel, self).__init__()
        self.cplx = cplx  # Complex number flag
        self.set_snr(snr)  # Setting the SNR

    def forward(self, inputs):
        """
        Forward pass of the Channel module, simulating noise addition.
        :param inputs: Input tensor to be processed.
        :return: Input tensor with added Gaussian noise.
        """
        shape = inputs.shape
        gnoise = torch.randn(shape) * self.noise_std  # Generating Gaussian noise
        device = inputs.device  # Getting the device of the inputs
        return inputs + gnoise.to(device)  # Adding noise to the inputs

    def get_snr(self):
        """ Return the current SNR of the channel. """
        return self.snr

    def set_snr(self, snr):
        """
        Set the SNR of the channel and calculate the corresponding noise standard deviation.
        :param snr: New Signal-to-Noise Ratio value.
        """
        self.snr = snr
        if self.cplx:
            self.noise_std = np.sqrt(10 ** (-snr / 10)) / np.sqrt(2)
        else:
            self.noise_std = np.sqrt(10 ** (-snr / 10))

def PSNR_plotter(x_axis, model, testloader, epoch, stages_count=1, goal=None):
    """
    Plot the PSNR values for different SNR levels.
    :param x_axis: SNR values to evaluate the model on.
    :param model: The model to evaluate.
    :param testloader: Test loader.
    :param epoch: Current training epoch.
    :param stages_count: Number of stages in the model.
    :param goal: Optional goal line to plot.
    """
    psnr = PeakSignalNoiseRatio().cpu()
    sc = stages_count
    PSNRs = np.zeros((sc, len(x_axis)))
    pre_snr = model.channel.get_snr()  # Storing the initial SNR
    psnr_val = []
    for i, snr in enumerate(x_axis):
        model.channel.set_snr(snr)
        psnr_val.append(evaluate_psnr(model, testloader, psnr, device='cpu'))

    model.channel.set_snr(pre_snr)  # Resetting the SNR to its initial value
    PSNRs = np.asarray(psnr_val).squeeze().transpose()
    if sc == 1:
        plt.plot(x_axis, PSNRs[0], label='Model')
    else:
        for i in range(sc):
            plt.plot(x_axis, PSNRs[i], label=f'Stage_{i+1}')
    if goal is not None:
        plt.plot(x_axis, goal, label='Goal')

    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    plt.savefig(f"models/test_1/images/epoch_{epoch}.png")
    plt.clf()



class PSNR_metric(nn.Module):
    def __init__(self):
        """
        Initialize the PSNR_metric module.
        This module accumulates PSNR values over multiple batches and computes their average.
        """
        super(PSNR_metric, self).__init__()
        self.PSNR_additive = torch.tensor(0.0)  # Sum of PSNR values
        self.counter = torch.tensor(0.0)  # Counter for the number of batches

    def forward(self, y_true, y_pred):
        """
        Forward pass to accumulate PSNR value.
        :param y_true: Ground truth tensor.
        :param y_pred: Predicted tensor.
        :return: Current average PSNR over all batches.
        """
        self.PSNR_additive += psnr(y_true, y_pred).mean()  # Accumulating PSNR
        self.counter += 1  # Incrementing counter
        return self.PSNR_additive / self.counter  # Returning average PSNR

    def reset(self):
        """ Reset the PSNR additive and counter to zero. """
        self.PSNR_additive = torch.tensor(0.0)
        self.counter = torch.tensor(0.0)

    def compute(self):
        """
        Compute the average PSNR.
        :return: Average PSNR if counter is not zero; otherwise, returns zero.
        """
        return self.PSNR_additive / self.counter if self.counter != 0 else torch.tensor(0.0)

# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    """
    Adjusts the learning rate based on the epoch number.
    :param epoch: Current epoch number.
    :param lr: Current learning rate.
    :return: Adjusted learning rate.
    """
    if epoch == 0:
        print("\nlearning_rate: 0.001")
    elif epoch == 20:
        print("\nlearning_rate: 0.0005")
    elif epoch == 30:
        print("\nlearning_rate: 0.0001")

    # Adjusting learning rate based on epoch
    if epoch < 20:
        return 0.001
    elif epoch < 30:
        return 0.0005
    else:
        return 0.0001

  
def load_data(data_dir, ps, train_count, test_count):
    """
    Load and preprocess data from the specified directory.
    :param data_dir: Directory of the dataset.
    :param ps: Size of each image patch.
    :param train_count: Number of training samples to generate.
    :param test_count: Number of test samples to generate.
    :return: Tuple of training and testing datasets.
    """
    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Load the ImageNet-A dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # DataLoader for PyTorch
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Process and split the dataset into patches
    trainX, testX = [], []
    total_patches = 0
    for img, label in data_loader:
        img = img.squeeze(0)  # Remove the batch dimension

        # Patch processing
        shape = img.shape
        tile_dim0 = shape[1] // ps
        tile_dim1 = shape[2] // ps
        patches = img[:, :tile_dim0 * ps, :tile_dim1 * ps]
        patches = patches.unfold(1, ps, ps).unfold(2, ps, ps)
        patches = patches.contiguous().view(3, -1, ps, ps).permute(1, 0, 2, 3)

        # Add patches to the list
        trainX.extend(patches)
        total_patches += len(patches)
        if total_patches > (train_count + test_count):
            break

    # Convert list to tensor and split into train and test
    trainX = torch.stack(trainX)
    train_count = int(0.7 * len(trainX))
    test_count = int(len(trainX) - train_count)
    trainX = trainX[:train_count + test_count]
    trainX, testX = trainX[:train_count], trainX[train_count:]

    # Normalize the data
    trainX = trainX.float() / 255.0
    testX = testX.float() / 255.0

    return trainX, testX

class Encoder(nn.Module):
    def __init__(self, out_chs):
        """
        Initialize the Encoder module.
        :param out_chs: Number of output channels.
        """
        super(Encoder, self).__init__()
        # Define encoder layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(32, out_chs, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        """
        Forward pass of the Encoder.
        :param x: Input tensor.
        :return: Encoded output tensor.
        """
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape, img_chs=3):
        """
        Initialize the Decoder module.
        :param input_shape: Shape of the input tensor.
        :param img_chs: Number of image channels.
        """
        super(Decoder, self).__init__()
        # Assuming input_shape is a tuple (C, H, W)
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
        """
        Forward pass of the Decoder.
        :param x: Input tensor.
        :return: Decoded output tensor.
        """
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))  # Sigmoid activation in the last layer
        return x

class Model(nn.Module):
    def __init__(self, enc_chs, dec_chs, stages_count, SNR, img_chs=3):
        """
        Initialize the Model.
        :param enc_chs: Number of encoder channels.
        :param dec_chs: Number of decoder channels.
        :param stages_count: Number of stages in the model.
        :param SNR: Signal-to-Noise Ratio.
        :param img_chs: Number of image channels.
        """
        super(Model, self).__init__()
        # Create the encoder
        self.encoder = Encoder(out_chs=enc_chs)
        # Create the power normalizer
        self.powernorm = PowerNormParts(parts_count=stages_count, cplx=True)
        # Create the channel
        self.channel = Channel(snr=SNR, cplx=True)
        # Create the decoders
        self.decoders = []
        for i in range(stages_count):
            self.decoders.append(Decoder(input_shape=(dec_chs * (i + 1), None, None), img_chs=img_chs).to(DEVICE))

    def forward(self, x):
        """
        Forward pass of the Model.
        :param x: Input tensor.
        :return: List of outputs from each decoding stage.
        """
        encoder_out = self.encoder(x)
        power_out = self.powernorm(encoder_out)
        channel_out = self.channel(power_out)
        outputs = []
        losses = []
        for i, decoder in enumerate(self.decoders):
            # Select relevant features for each decoder
            decoder_input = channel_out[:, :dec_chs * (i + 1), :, :]
            output = decoder(decoder_input)
            outputs.append(output)
            # Calculate the loss
            loss = mse(x, output)  # Compute MSE loss against the original input
            losses.append(loss)

        return outputs, losses
    
# Define SNR values and target PSNR goals for performance measurement
known_SNRs  = [1, 4, 7, 13, 19]
test_SNRs   = [1, 4, 7, 10, 13, 16, 19, 22, 25]
known_goals = [[24.636, 25.964, 26.618, 26.945, 27.109, 27.182, 27.236, 27.255, 27.273],
              [23.836, 26.655, 28.091, 28.909, 29.309, 29.527, 29.636, 29.673, 29.709],
              [21.836, 25.945, 28.545, 30.091, 31.055, 31.618, 31.945, 32.091, 32.182],
              [21.036, 23.836, 26.455, 28.854, 30.836, 32.345, 33.382, 34.036, 34.382],
              [20.291, 23.127, 25.891, 28.473, 30.745, 32.655, 34.073, 35.036, 35.600]]

# Initialize the model and move it to the appropriate device
model = Model(enc_chs, dec_chs, stages_count, SNR).to(DEVICE)

# Load and preprocess data
trainX, testX = load_data(data_dir, ps, train_count, test_count)
trainX = trainX.to(DEVICE)
testX = testX.to(DEVICE)

# Create data loaders for training and testing
train_loader = DataLoader(trainX, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testX, batch_size=batch_size, shuffle=False)

# Setup the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler(epoch, 0.001))

# Initialize PSNR metric
psnr_metric = PSNR_metric()

def plot_loss(loss_per_channel, epochs, stages_count):
    """
    Plot the loss values for each channel.
    :param loss_per_channel: Loss values for each channel.
    :param epochs: Number of training epochs.
    :param stages_count: Number of stages in the model.
    """
    plt.figure(figsize=(10, 5))
    for i in range(stages_count):
        plt.plot(range(epochs+1), loss_per_channel[:, i], label=f"Channel {i + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"models/test_1/images/loss_{epoch}.png")
    plt.clf()

def loss_function(outputs, inputs):
    loss_list = []
    lossess = []
    for output in outputs:
        mse_out = mse(output, inputs)
        loss = mse_out
        lossess.append(loss)
        loss_list.append(mse_out.cpu().detach().numpy())
    return torch.stack(lossess).mean(), loss_list

def train(model, train_loader, optimizer, epoch):
    """
    Train the model for one epoch.
    :param model: The neural network model.
    :param train_loader: DataLoader for training data.
    :param optimizer: Optimizer for model parameters.
    :param loss_func: Loss function.
    :param epoch: Current epoch number.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    first = True
    for data in train_loader:
        inputs = data.to(DEVICE)
        optimizer.zero_grad()  # Zero out any existing gradients
        outputs, losses = model(inputs)  # Get model outputs
        loss, loss_list = loss_function(outputs, inputs)  # Calculate loss
        if first == True:
            total_loss_epoch = loss_list
        if first == False:
            total_loss_epoch = np.vstack((total_loss_epoch, loss_list))
        loss = sum(losses)/len(losses)            
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item()
        first = False
    model.eval()  # Set the model to evaluation mode
    loss_per_channel = np.sum(total_loss_epoch, axis=0) / len(train_loader)
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}, Loss_per_channel: {loss_list}")
    return loss_per_channel

def get_psnr(outputs, input, psnr):
    """
    Calculate PSNR for each output channel.
    :param outputs: List of output tensors from the model.
    :param input: Input tensor.
    :param psnr: PSNR calculation function.
    :return: Array of PSNR values for each channel.
    """
    psnr_values = [psnr(output, input).cpu() for output in outputs]
    return np.array(psnr_values)

def evaluate_psnr(model, test_loader, psnr, device):
    """
    Evaluate the model on test data and calculate average PSNR.
    :param model: The neural network model.
    :param test_loader: DataLoader for testing data.
    :param psnr: PSNR calculation function.
    :param device: Device to run the evaluation on.
    :return: Average PSNR value.
    """
    psnr_val = np.zeros([1, 5])
    psnr = PeakSignalNoiseRatio().to(device)
    model = model.to(device)
    with torch.no_grad():  # No gradient calculation for evaluation
        count = 0
        total_loss = 0
        for data in test_loader:
            inputs = data.to(device)
            outputs, losses = model(inputs)
            psnr_val += get_psnr(outputs, inputs, psnr)
            count += 1
            loss = sum(losses) / len(losses)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader)}")
        avg_psnr = psnr_val / count
    model.to(DEVICE)  # Move model back to the original device
    return avg_psnr

# Training loop
total_loss_per_channel = []
first = True
for epoch in range(epochs):
    min_psnr = 0.0
    loss_per_channel = train(model, train_loader, optimizer, epoch)
    if first == True:
        total_loss_per_channel = loss_per_channel
    if first == False:
        total_loss_per_channel = np.vstack((total_loss_per_channel, loss_per_channel))
    scheduler.step()  # Update the learning rate
    first = False
    # Save the model and evaluate PSNR at specified intervals
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"models/test_1/models/{JSSC_channel_name}_model_{epoch}.pth")

    if epoch % 10 == 0:
        avg_psnr = evaluate_psnr(model, test_loader, psnr, DEVICE).squeeze()
        if avg_psnr.mean() > min_psnr:
            min_psnr = avg_psnr.mean()
            torch.save(model.state_dict(), f"models/test_1/models/{JSSC_channel_name}_model_best.pth")

        # PSNR_plotter(test_SNRs, model, test_loader, epoch, stages_count)
        if epoch > 0:
            plot_loss(total_loss_per_channel, epoch, stages_count)
        print(f"Epoch {epoch}, PSNR channel_1: {avg_psnr[0]}, PSNR channel_2: {avg_psnr[1]}, PSNR channel_3: {avg_psnr[2]}, PSNR channel_4: {avg_psnr[3]}, PSNR channel_5: {avg_psnr[4]}")

# Final model evaluation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    total_psnr = 0.0
    count = 0
    for data in test_loader:
        inputs = data.to(DEVICE)
        outputs = model(inputs)
        total_psnr += psnr(inputs, outputs[-1]).mean().item()  # Calculate PSNR for the last output
        count += 1
    avg_psnr = total_psnr / count
    PSNR_plotter(test_SNRs, model, testX, epochs, stages_count)  # Plot PSNR for the final epoch

        