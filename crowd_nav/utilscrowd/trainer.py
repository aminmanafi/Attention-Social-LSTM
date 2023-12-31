import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                data_input1 = open("Amin_Manafi_data_input1","a+")
                data_value1 = open("Amin_manafi_data_value1","a+")
        
                inputs, values = data
                # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
                #  0     1      2     3      4        5     6      7         8       9     10      11     12
                
                # data_input1.write(str(inputs))
                # data_input1.write("\n")
                
                # data_value1.write(str(values))
                # data_value1.write("\n")
                
                
                #np.savetxt(data_input1,np.array(inputs.cpu()))
                #np.savetxt(data_value1,np.array(values.cpu()))
                
                inputs = torch.cat((inputs[:,:,:5],inputs[:,:,8:]),dim=2)
                
                
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
                
                data_input1.close()
                data_value1.close()
        
            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
        
        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        
        for _ in range(num_batches):
            data_input2 = open("Amin_Manafi_data_input2","a+")
            data_value2 = open("Amin_Manafi_data_value2","a+")
            
            inputs, values = next(iter(self.data_loader))
            
            # data_input2.write(str(inputs))
            # data_input2.write("\n")
            # data_value2.write(str(values))
            # data_value2.write("\n")
            #np.savetxt(data_input2,np.array(inputs.cpu()))
            #np.savetxt(data_value2,np.array(values.cpu()))

            
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            
            data_input2.close()
            data_value2.close()
        
        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
