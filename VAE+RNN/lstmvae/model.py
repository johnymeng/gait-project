import torch
from torch import nn
from torch.nn import functional

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        """x: tensor of shape (batch_size, seq_length, hidden_size)"""
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
    
class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, device):
        """LSTM based VAE

        Args:
            input_size (int): batch_size * sequence_length * input dim
            hidden_size (int): output size of LSTMVAE
            latent_size (int): size of latent z-layer
            num_layers (int): number of layers in VAE
            num_lstm_layer (int): number of layers in the LSTM
        """
        
        super(LSTMVAE, self).__init__()
        self.device = device
        
        #setting model dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        
        #create LSTM Auto Encoder
        self.lstm_encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers
        )
        #create decoder
        self.lstm_decoder = Decoder(
            input_size=input_size,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers
        )
        
        #fully connected layers
        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.hidden_size, self.latent_size)
        
    def reparmeterize(self, mu, logvar):
        """employing the reparameterization trick for backprop

        Args:
            mu (_type_): _description_
            logvar (_type_): _description_

        Returns:
            _type_: _description_
        """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)
        
        z = mu + noise * std
        
        return z
    

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        #encodes input sapce to hidden space
        encoder_hidden = self.lstm_decoder(x)
        encoder_h = encoder_hidden[0].view(batch_size, self.hidden_size).to(self.device)
        
        #extract the latent variable z from hidden space to latent space
        mean = self.fc21(encoder_h)
        logvar = self.fc22(encoder_h)
        z = self.reparmeterize(mean, logvar) #batch_size * latent_size
        
        #init hidden state as inputs
        h_ = self.fc3(z)
        
        #decode latent space to input space
        z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
        
        #init hidden state
        hidden = (h_.contiguous(), h_.contiguous())
        reconstructed_output, hidden = self.lstm_decoder(z, hidden)
        
        x_hat = reconstructed_output
        
        #calculate VAE loss
        losses = self.loss_function(x_hat, x, mean, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["reconstruction_loss"],
            losses["kld"]
        )
        
        return m_loss, x_hat, (recon_loss, kld_loss)      
    
    def loss_function(self, *args, **kwargs)->dict:
        """
        computes the VAE loss fucntion
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
         
        Returns:
            dict: _description_
        """
        
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        kld_weight = 0.00025
        recons_loss = functional.mse_loss(recons, input)
        
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu*2 - log_var.exp(), dim=1), 
            dim=0
        )    
        
        loss = recons_loss + kld_weight * kld_loss
        
        return {
            "loss": loss,
            "reconstruction_loss": recons_loss.detach(),
            "kld": -kld_loss.detach()
        }   
        
class LSTMAE(nn.Module):
    """LSTM-based Auto Encoder"""

    def __init__(self, input_size, hidden_size, latent_size, device=torch.device("cuda")):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size of LSTM AE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMAE, self).__init__()
        self.device = device

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # lstm ae
        self.lstm_enc = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.lstm_dec = Decoder(
            input_size=input_size,
            output_size=input_size,
            hidden_size=hidden_size,
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        enc_hidden = self.lstm_enc(x)

        temp_input = torch.zeros((batch_size, seq_len, feature_dim), dtype=torch.float).to(
            self.device
        )
        hidden = enc_hidden
        reconstruct_output, hidden = self.lstm_dec(temp_input, hidden)
        reconstruct_loss = self.criterion(reconstruct_output, x)

        return reconstruct_loss, reconstruct_output, (0, 0)