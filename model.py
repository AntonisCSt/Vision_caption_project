import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super().__init__()
        
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs, hidden_size layers, number of lstm layers (this time is 1)
        # Also, the dropout is set to 0 (no dropout) 
        self.lstm = nn.LSTM(input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = self.num_layers, 
                             dropout = 0, 
                             batch_first=True
                           )

        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 1 tag)
        self.linear_fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):
        
        # not include <end>
        captions = captions[:, :-1]
        
        #Pass the captions through the embedding layer so that the model can find the relationships between the word tokens better.
        embed = self.word_embeddings(captions)
        
        #Concatenate the image features as you have indicated
        inputs = torch.cat((features.unsqueeze(1), embed), dim=1)
        
        #setup the device
        #device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        #pass concatinated info through the lstm
        outputs, _ = self.lstm(inputs)
        
        # pass through the linear unit
        outputs = self.linear_fc(outputs)
        
        return outputs
    
 
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        hidden = (torch.randn(self.num_layers, 1, self.hidden_dim).to(inputs.device),torch.randn(self.num_layers, 1, self.hidden_dim).to(inputs.device))
        
        #initiate an empty list
        sample_output = []
        
        while len(sample_output) < max_len:
            #Run the lstm and get the output
            lstm_out, states = self.lstm( inputs, hidden)
            outputs = self.linear_fc(lstm_out)
            
            #choose the word with the highest propability
            outputs = outputs.squeeze(1) 
            predicted_word = outputs.argmax(dim = 1)
            
            #add it to the caption sentence list
            sample_output.append(predicted_word.tolist()[0])
            
            #embed the input so it can be fed in the loop next  
            inputs = self.word_embeddings(predicted_word)
            inputs = inputs.unsqueeze(1)
     
            
            if predicted_word.tolist()[0] == 1:
                break
        
        return sample_output
        
        
        
        