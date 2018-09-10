# implementation-for-neuron-network
First, we define the network as usual. For example, 

	class Net(nn.Module):
		def __init__(self):
        		super(Net, self).__init__()
        		self.conv1 = nn.Conv2d(3, 128, 3,padding=2)
			self.conv2 = nn.Conv2d(128, 128, 3,padding=2)
        		self.pool1 = nn.MaxPool2d(2, 2)
        		self.conv3 = nn.Conv2d(128, 256, 3,padding=2)
			self.conv4 = nn.Conv2d(256, 256, 3,padding=2)
			self.conv5 = nn.Conv2d(256, 512, 3,padding=2)
			self.conv6 = nn.Conv2d(512, 512, 3,padding=2)
			self.pool2 = nn.MaxPool2d(2, 2,padding=1)
        		self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        		self.fc2 = nn.Linear(1024, 1024)
        		self.fc3 = nn.Linear(1024, 10)
		def forward(self, x):
			x = F.relu(self.conv1(x))
        		x = self.pool1(F.relu(self.conv2(x)))
			x = F.relu(self.conv3(x))
			x = self.pool1(F.relu(self.conv4(x)))
			x = F.relu(self.conv5(x))
			x = self.pool2(F.relu(self.conv6(x)))
        		x = x.view(-1, 512 * 8 * 8)
        		x = F.relu(self.fc1(x))
        		x = F.relu(self.fc2(x))
        		x = self.fc3(x)
        return x

Then we can start load the data into the network.


outputs = net(inputs)   //forward            
loss = criterion(outputs, labels)   //compute the loss


Then, we detached  all the weights from the neuron network using the detach function. This means instead of calculating  all the gradients in the network  during the back-propagate process, we only calculate everything except the gradient of weights since we need to calculate the weights with quantized value of activations.  What's more, setting the grad_require  attribute of tensor of weights False before the backward process can be another way. But both of them need to modify the convolution layer function source code since we cannot see the weights directly by just call the function like nn.Conv2d(). 

 
To calculate the gradient of weight, we need to have the error of neuron and the value of activations, according to the equation ∆ w (ji) = ηδ (j) y (i) as is described in that paper. 
In pytorch, it's easy to get those two value using the hook function. Although we can't use the hook function to modify the value of the network directly, what we need is just the origin value of y(i). We can transfer its value to another variable and do the quantized operation on it. Finally, we can calculate the error of weights.

After we get the error of weights, we can update them according to the learning rate we set.
Then we can start another epoch of training.
