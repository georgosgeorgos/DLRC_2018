import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.encoder_layer1 = nn.Conv2d(1  , 32 , 3, stride=1, padding=1)
        self.encoder_layer2 = nn.Conv2d(32 , 128, 3, stride=1, padding=1)
        self.encoder_layer3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.decoder_layer4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.decoder_layer3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.decoder_layer2 = nn.Conv2d(128, 32 , 3, stride=1, padding=1)
        self.decoder_layer1 = nn.Conv2d(32 , 1  , 3, stride=1, padding=1)

        self.mode = "nearest"
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def routine(self, x):
        x = self.relu(x)
        dim = x.size()
        return self.pool(x), dim[2:]

    def forward(self, x):
        #print("encoder")
        #print(x.size())
        x = self.encoder_layer1(x)
        x, x_size_layer1 = self.routine(x) 
        #print(x.size())
        
        x = self.encoder_layer2(x)
        x, x_size_layer2 = self.routine(x)
        #print(x.size())
        x = self.encoder_layer3(x)
        x, x_size_layer3 = self.routine(x)
        #print(x.size())
        x = self.encoder_layer4(x)
        #print(x.size())

        x_hidden = x.clone()

        #print("decoder")
        x = self.decoder_layer4(x)
        #print(x.size())

        x = self.relu(x)
        x = nn.Upsample(size=x_size_layer3, mode=self.mode)(x)
        x = self.decoder_layer3(x)
        #print(x.size())

        x = self.relu(x)
        x = nn.Upsample(size=x_size_layer2, mode=self.mode)(x)
        x = self.decoder_layer2(x)
        #print(x.size())

        x = self.relu(x)
        x = nn.Upsample(size=x_size_layer1, mode=self.mode)(x)
        x = self.decoder_layer1(x)
        #print(x.size())

        return x, x_hidden

