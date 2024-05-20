from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import entropy
import numpy as np
from scipy.linalg import sqrtm, norm

class LeNet(nn.Module):
    def __init__(self):
        """The original LeNet used Sigmoid activations, but we use ReLU by default here. 
        
        Based on: https://en.wikipedia.org/wiki/LeNet
        """
        super(LeNet, self).__init__()
        
        # feature extraction
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=576, out_features=120)
        
        self.act3 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        self.act4 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.softmax = nn.Softmax()
        
        # python3 dictionary is ordered, retrain the sequential order
        # of the layers here
        self.layers = {
            'conv1': self.conv1,
            'act1': self.act1,
            'pool1': self.pool1,
            'conv2': self.conv2,
            'act2': self.act2,
            'pool2': self.pool2,
            'flatten': self.flatten,
            'fc1': self.fc1,
            'act3': self.act3,
            'fc2': self.fc2,
            'act4': self.act4,
            'fc3': self.fc3,
            'softmax': self.softmax
        }

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# apply this to all MNIST images before using in train/test
mnist_transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ]
)



def train_lenet_on_mnist(
        train_loader : torch.utils.data.DataLoader,
        test_loader : torch.utils.data.DataLoader,
        save_model_to: Path = Path('model.pth')   
    ) -> Path:
    """Code adapted from: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python
    """
    learning_rate = 0.001
    num_epochs = 10
    
    # set up lenet5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        # train
        tot_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):  

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_f(outputs, labels)
            tot_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print('\rEpoch [{}/{}], Step [{}], Avg loss: {:.4f}'.format(
                    epoch+1, 
                    num_epochs, 
                    i+1, 
                    tot_loss / i
                ), end='')
                
        # create a new line after each epoch
        print("")

        # compute test accuracy
        num_correct, num_total = 0.0, 0.0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            num_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).item()
            num_total += outputs.shape[0]
        
        print(f"Accuracy: {num_correct / num_total}")

                
    # done training, save it
    torch.save(model.state_dict(), str(save_model_to))
    
    # loaded_model = torch.load('model.pth')
    return save_model_to, model

def get_from_layer(layer_name: str, model, images):
    _hard_coded_layers = {
        'conv1': model.conv1,
        'act1': model.act1,
        'pool1': model.pool1,
        'conv2': model.conv2,
        'act2': model.act2,
        'pool2': model.pool2,
        'flatten': model.flatten,
        'fc1': model.fc1,
        'act3': model.act3,
        'fc2': model.fc2,
        'act4': model.act4,
        'fc3': model.fc3,
    }
    layers = getattr(model, 'layers', _hard_coded_layers)
    
    if layer_name not in layers:
        raise ValueError(f"Invalid layer name: {layer_name}")
        
    l_n = ""
    i = 0
    layers_iter = list(layers.items())
    feat = images.to("cuda:0")
    while l_n != layer_name:
        l_n, l_f = layers_iter[i]
        feat = l_f(feat)
        i += 1
        
    return feat

def extract_features(model, dataloader, layer_name):
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            features.append(get_from_layer(layer_name, model, images).cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features


# ---------

# # the ground truth data + a loader for it
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# # get lenet features for ground truth (real mnist)
# layer_name = 'fc2'  
# features_real = extract_features(model, test_loader, layer_name)

# # get lenet features for generated images
# # generated_samples_loader is an iterator over batch_size=bs groups of generated images
# features_generated = extract_features(model, generated_samples_loader, layer_name)

def compute_FID_score_from_features(features_real, features_generated):

    mean_real = np.mean(features_real, axis=0) # 84
    mean_generated = np.mean(features_generated, axis=0) # 84
    cov_real = np.cov(features_real, rowvar=False) # 84, 84
    cov_generated = np.cov(features_generated, rowvar=False) # 84, 84
    # please check!!!!

    fid = np.sum((mean_real - mean_generated)**2) + np.trace(cov_real + cov_generated - 2 * sqrtm(np.matmul(cov_real, cov_generated)))
    return fid.real # might have small imaginary component

def compute_diversity_score_from_features(features_real, features_synthetic):

    num_samples = features_synthetic.shape[0]
    diversity_times = int(num_samples/2)

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = norm(features_synthetic[first_indices] - features_synthetic[second_indices], axis=1)

    return float(dist.mean())

def get_fid_score(ddpm, guide_w, n_samples_to_generate=1000):

    assert(n_samples_to_generate % 10 == 0)

    # STEP 1: sample from ddpm
    ddpm.eval()
    with torch.no_grad():
        
        # synthesize new images
        x_gen_contexts, x_gens = [], []
        samples_per_batch = 40 # must be a multiple of n_classes=10, chosen to optimize run time
        n_batches = int(np.ceil(n_samples_to_generate / samples_per_batch))
        for batch in range(n_batches):
            print(f"    synthesizing image batch {batch+1}/{n_batches} for FID computation...")
            x_gen_context, x_gen, x_gen_store = ddpm.sample(samples_per_batch, (1, 28, 28), "cuda:0", guide_w=guide_w)
            x_gen_contexts.append(x_gen_context)
            x_gens.append(x_gen)
            
        x_gen_context = torch.cat(x_gen_contexts)[:n_samples_to_generate]
        x_gen = torch.cat(x_gens)[:n_samples_to_generate]

    x_gen_normalized = x_gen[:,0].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)

    # STEP 2: compute LeNet features for synthesized images
    lenet = LeNet().to("cuda:0")
    lenet.load_state_dict(torch.load("exp_outputs/mnist/lenet_sigmoid.pth"))
    dataset_synthetic = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=mnist_transform)
    # replace images, context with our images, context
    dataset_synthetic.data = x_gen_normalized.to("cpu")
    dataset_synthetic.targets = x_gen_context.to("cpu")
    dataloader_synthetic = DataLoader(dataset_synthetic, batch_size=64, shuffle=False, num_workers=0)
    layer_name = 'fc2' 
    # layer_name = 'act4'
    features_synthetic = extract_features(lenet, dataloader_synthetic, layer_name)

    # STEP 3: compute LeNet features for real images
    dataset_real = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=mnist_transform)
    dataloader_real = DataLoader(dataset_real, batch_size=64, shuffle=False, num_workers=0)
    features_real = extract_features(lenet, dataloader_real, layer_name)

    # STEP 4: compute FID
    fid_score = compute_FID_score_from_features(features_real, features_synthetic)

    # STEP 5: compute diversity
    diversity_score = compute_diversity_score_from_features(features_real, features_synthetic)

    return fid_score, diversity_score


def train_lenet():

    batch_size = 64

    train_dataset = torchvision.datasets.MNIST(root = './dataset', train = True, transform = mnist_transform, download = True)
    test_dataset = torchvision.datasets.MNIST(root = './dataset', train = False, transform = mnist_transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    model_save_path = "exp_outputs/mnist/lenet_sigmoid.pth"
    save_model_to, model = train_lenet_on_mnist(train_loader, test_loader, model_save_path)
    print(f"Lenet model saved to: {save_model_to}")


def main():

    train_lenet() # only need to run this one time

if __name__ == "__main__":
    main()