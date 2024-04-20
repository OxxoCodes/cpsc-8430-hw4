print("Importing dependencies...")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import wandb

wandb.init(project="GAN_training", entity="OxxoCodes", config={
    "epochs": 200,
    "batch_size": 32,
    "learning_rate_g": 0.0002,
    "learning_rate_d": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "latent_dim": 100,
    "img_width": 64,
    "img_height": 64,
    "num_classes": 40,
    "generator_hidden_dim": 64,
    "discriminator_hidden_dim": 64
})

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes=40):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.emb = nn.Embedding(num_classes, hidden_dim)
        self.upsize = nn.Linear(hidden_dim, input_size)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_size, hidden_dim * 2, 4, 1, 0, bias=False),
            nn.LayerNorm([hidden_dim * 2, 4, 4]),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim * 4, 8, 8]),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim * 2, 16, 16]),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim, 32, 32]),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x, classes):
        # x has shape (batch_size, input_size, 1, 1)
        class_embeddings = self.emb(classes) # Get class embeddings
        class_embeddings = self.upsize(class_embeddings) # Match dim of latent input via MLP upsizing
        class_embeddings = class_embeddings.view(-1, self.input_size, 1, 1).expand_as(x)

        output = x * class_embeddings # Embeddings and latent input affect each other
        output = self.model(output) # Get final image
        return output

        input(x.shape)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x.expand(-1, -1, self.emb(classes).size(2), self.emb(classes).size(3))
        print("Shape after reshaping x:", x.shape)
    
        output = torch.mul(self.emb(classes), x)
        print("Shape after multiplying with embedding:", output.shape)
        
        output = self.model(output)
        print("Shape after passing through the model:", output.shape)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, kernel_size=4, stride=2, padding=1, num_classes=40):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size, stride, padding, bias=False),
            nn.LayerNorm([hidden_dim*2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size, stride, padding, bias=False),
            nn.LayerNorm([hidden_dim*4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.real_or_fake_layer = nn.Sequential(
            nn.Linear(hidden_dim*4*8*8, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4*8*8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        model_output = self.model(x) # Get latent space representation
        model_output = model_output.view(model_output.shape[0], -1)
        img_class = self.classifier(model_output) # Classify image labels using MLP (40 classes)
        real_or_fake = self.real_or_fake_layer(model_output) # Classify realness using MLP
        return real_or_fake, img_class

class CelebADataset(Dataset):
    def __init__(self, dataset_path, transform):
        super(CelebADataset, self).__init__()
        print("Locating all data files...")
        filenames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)[:10000]]
        print("Loading data...")
        self.images = [Image.open(f).convert("RGB") for f in tqdm(filenames) if ".png" in f]
        self.flip_index = len(self.images)
        self.images += self.images
        self.transform = transform
        self.flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])

        print("Parsing labels...")
        self.class_labels = []
        with open("celeba_labels.txt", "r") as f:
            lines = [x.strip() for x in f.readlines()]
            # self.labels = lines[1].split(" ")
            # label_indexes = [self.labels.index(x) for x in [""]]
            # input(label_indexes)

        for line in lines[2:]:
            line = line.replace("  ", " ")
            self.class_labels.append(
                torch.tensor([0 if x == "-1" else 1 for x in line.split(" ")[1:]], dtype=torch.float32)
                )
    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if idx >= self.flip_index:
            image = self.flip_transform(image)
        labels = self.class_labels[idx]
        return image, labels


device = "cuda" if torch.cuda.is_available() else "cpu"

img_width, img_height = 64, 64
img_dim = img_width*img_height*3
latent_dim = 100

generator = Generator(input_size=latent_dim, hidden_dim=64).to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))

discriminator = Discriminator(input_size=img_dim, hidden_dim=64).to(device)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))

transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.CenterCrop((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = CelebADataset("/scratch/nbrown9/cpsc-8650/CelebA/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png", transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def save_image(tensor):
    tensor = tensor.detach().cpu()
    img = tensor.view(3, img_width, img_height)
    img = (img + 1) / 2 * 255
    img = img.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(img)

def save_image_grid(generator, suffix, n_images=25):
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        z = torch.randn(1, latent_dim, 1, 1, device=device)
        gen_inputs = torch.randn(1, latent_dim, 1, 1, device=device)
        gen_rand_class = torch.randint(0,39,(1,),dtype=torch.int64).to(device)
        with torch.no_grad():
            generated_image = generator(z, gen_rand_class).view(3, img_width, img_height)
        image = save_image(generated_image)
        plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')
    filename = f'image_saves/img_grid_{suffix}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved outputs to {filename}")

bceloss = nn.BCELoss()
bcewithlogitsloss = nn.BCEWithLogitsLoss()

def train(n_epochs):
    idx = 0
    for epoch in range(n_epochs):
        for batch_idx, (inputs, classes) in enumerate(dataloader):
            inputs = inputs.to(device)
            classes = classes.to(device)

            real_labels = torch.ones(inputs.size(0), 1).to(device)
            fake_labels = torch.zeros(inputs.size(0), 1).to(device)

            # Train discriminator
            optimizer_d.zero_grad()

            # Get discriminator output on real images
            discrim_real_rf, discrim_real_classes = discriminator(inputs)
            
            # discrim_real_loss = (bceloss(discrim_real_rf, real_labels) + bcewithlogitsloss(discrim_real_classes, classes))/2

            # Get generator outputs
            gen_inputs = torch.randn(inputs.size(0), latent_dim, 1, 1, device=device)
            gen_rand_classes = torch.randint(0,39,(inputs.shape[0],),dtype=torch.int64).to(device)
            gen_outputs = generator(gen_inputs, gen_rand_classes).view(-1, 3, img_width, img_height)

            # Get discriminator outputs on fake images (detach to prevent gradient flow from gen->disc)
            discrim_fake_rf, discrim_fake_classes = discriminator(gen_outputs.detach())
            # discrim_fake_loss = bceloss(discrim_fake_rf, fake_labels) + bcewithlogitsloss(discrim_fake_classes, classes)

            # discrim_loss = (discrim_real_loss + discrim_fake_loss) /2
            discrim_loss = ((-torch.mean(discrim_real_rf) + torch.mean(discrim_fake_rf)) + bcewithlogitsloss(discrim_fake_classes, classes)) / 2
            
            discrim_loss.backward()
            optimizer_d.step()

            # Gradient clipping
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # n_critic = 5, as per the wgan paper
            if batch_idx % 5 == 0:
                # Train generator
                optimizer_g.zero_grad()
                
                # Get discriminator output on fake images
                discrim_fake_rf, discrim_fake_classes = discriminator(gen_outputs)

                # Loss is a combination of Wasserstein distance (WGAN) and BCEWithLogitsLoss (AC-GAN)
                gen_loss = (-torch.mean(discrim_fake_rf) + bcewithlogitsloss(discrim_fake_classes, classes)) / 2

                gen_loss.backward()
                optimizer_g.step()

            print(f"Epoch {epoch}/{n_epochs}, Iter {batch_idx}/{len(dataloader)}, G-Loss:{gen_loss}, D-Loss:{discrim_loss}")

            wandb.log({
                "Generator Loss": gen_loss.item(),
                "Discriminator Loss": discrim_loss.item(),
                "Wasserstein Distance": -torch.mean(discrim_fake_rf).item() + -torch.mean(discrim_real_rf).item(),
                "Real Score": discrim_real_rf.mean().item(),
                "Fake Score": discrim_fake_rf.mean().item()
            })

            if idx % (len(dataloader)*n_epochs // 1000) == 0:
                generator.eval()
                save_image_grid(generator, f"{epoch}_{batch_idx}")
                generator.train()
            idx += 1

train(200)