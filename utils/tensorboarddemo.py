# demo.py
#https://pytorch.apachecn.org/docs/1.4/6.html
#https://blog.csdn.net/leviopku/article/details/108985530 
#https://pytorch.org/docs/stable/tensorboard.html
#https://github.com/lanpa/tensorboardX
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002


sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

for n_iter in range(100):
    """
    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)

    dummy_img = torch.rand(32, 3, 64, 64)  # output from network
    """
    if n_iter % 10 == 0:
        """
        x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)
    
        dummy_audio = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # amplitude of sound should in [-1, 1]
            dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
        writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)
   
        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
        """

        for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

        # needs tensorboard 0.4RC or later
        #writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)
"""
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = dset.CIFAR10(root='/data/tmpexec/cifar', train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=4,
                    shuffle=False, num_workers=0)
dataiter = iter(test_loader)
images, labels = dataiter.next()
#features = images.view(100, 784)
#writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

#dataiter = iter(train_loader)
#images, labels = dataiter.next()
writer.add_graph(resnet18, images)
"""
# export scalar data to JSON for external processing
#writer.export_scalars_to_json("./all_scalars.json")
writer.close()