import torch.nn as nn
import math
import numpy as np
import torch 


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            # residual = self.downsample(x)
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=44):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=[4])
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(4608, num_classes) #367:5632  #305:4608
        # self.s = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * 1 * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print(x.shape)  #b,64,4959
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape) #79360
        x = self.fc(x)
        # print(x)
        # x = self.s(x)
        # print(x)

        return x
        
class ResNet_3c(nn.Module):

    def __init__(self, block, layers, num_classes=44):
        self.inplanes = 64
        super(ResNet_3c, self).__init__()
        self.conv = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=[4])
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(4608, num_classes) #367:5632  #305:4608  #266:4096
        # self.s = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * 1 * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print(x.shape)  #b,64,4959
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape) #79360
        x = self.fc(x)
        # print(x)
        # x = self.s(x)
        # print(x)

        return x        
        
class ResNet_d(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet_d, self).__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.cbam  = CBAM(64, 64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=[4])
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(4608, num_classes) #79360   #367:5632  #167:2560 #200:3072
        self.s = nn.Sigmoid()   #[0,1]
        self.l = nn.ReLU6()
        self.rl = nn.ReLU()
        
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * 1 * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # def forward(self, x):
        # x = self.conv(x)
        # x = self.bn1(x)
        # x = self.relu(x)   #[b, 64, 4959]
        # # print(x.shape)
        # x = self.cbam(x)  #
        # # print(x.shape)
        # x = self.layer1(x)
        # # print(x.shape)
        # x = self.layer2(x)
        # # print(x.shape)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # # print(x.shape)

        # x = self.avgpool(x)        
        # x = x.view(x.size(0), -1)
        # feature = x
        # # print(x.shape) #79360
        # x = self.fc(x)
        # # x = self.rl(x)   #[0,]
        # # x = self.l(x)
        # # print(x.shape)
        # # x = self.s(x)   #[0,1]
        # # print(x)

        # return x, feature

    def forward(self, target, all, all_label, target_label ):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return: 
        """
        # produce embeddings for support set images
        encoded_images = []
        pred_results = []
        for i in np.arange(all.size(0)):
            # gen_encode = self.g(all[:,i,:,:,:])
            # encoded_images.append(gen_encode)
            # print(all.shape)
            x = all[i,:,:].unsqueeze(0)
        # print(x.shape)
        # x = all
            x = self.conv(x)
            x = self.bn1(x)
            x = self.relu(x)   #[b, 64, 4959]
            # x = self.cbam(x)  #
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)        
            x = x.view(x.size(0), -1)
            feature = x
            # print(feature.shape)  #torch.Size([30, 4608])
            encoded_images.append(feature)
            # outputs1 = torch.stack(encoded_images)            

        # produce embeddings for target images
        for i in np.arange(target.size(0)):
            # gen_encode = self.g(target_image[:,i,:,:,:])
            # encoded_images.append(gen_encode)
            # outputs = torch.stack(encoded_images)
            x=target[i,:,:].unsqueeze(0) #torch.Size([1, 3, 305])
            x = self.conv(x)            
            x = self.bn1(x)
            x = self.relu(x)   #[b, 64, 4959]
            # x = self.cbam(x)  #
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)        
            x = x.view(x.size(0), -1)
            
            encoded_images.append(x)
            outputs = torch.stack(encoded_images)  #torch.Size([31, 1, 4608])


            # get similarity between support set embeddings and target
            similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = similarities.t()
            print(similarities)

            # produce predictions for target probabilities
            preds = self.classify(similarities,support_set_y=all_label)
            print(preds)
            fx = self.fc(x)
            fx = self.s(fx)
            fpreds = (fx + preds)/2
            print(fpreds)
            pred_results.append(fpreds)
            pred = torch.stack(pred_results) 
            encoded_images.pop()
            
        return pred
            # # calculate accuracy and crossentropy loss
            # values, indices = preds.max(1)
            # if i == 0:
                # accuracy = torch.mean((indices.squeeze() == target_label[:,i]).float())
                # crossentropy_loss = F.cross_entropy(preds, target_label[:,i].long())
            # else:
                # accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, i]).float())
                # crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, target_label[:, i].long())

            # # delete the last target image encoding of encoded_images
            # encoded_images.pop()

        # return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1)        

def resnet(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
def resnet_3c(**kwargs):
    return ResNet_3c(BasicBlock, [2, 2, 2, 2], **kwargs)    
    
    
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        sum_input = torch.sum(torch.pow(input_image, 2))
        input_magnitude = sum_input.clamp(eps, float("inf")).rsqrt()
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2))
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            # print(sum_support)
            # print(support_magnitude)
            # print(input_image.shape)   #torch.Size([1, 4608])
            # print((support_image.squeeze().unsqueeze(1)).shape)   #torch.Size([1, 4608])
            # flat_inputs = torch.flatten(inputs)
            # dot_product = input_image.unsqueeze(2).mm(support_image.unsqueeze(2)).squeeze()
            dot_product = input_image.mm(support_image.squeeze().unsqueeze(1))
            # print(dot_product)
            cosine_similarity = dot_product * support_magnitude *input_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        similarities = similarities.squeeze(1) #torch.Size([30, 1])
        # print(similarities)
        return similarities

        
class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):

        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        sigmoid = nn.Sigmoid()
        softmax_similarities = softmax(similarities)  #torch.Size([1, 30])
        # print(softmax_similarities.shape) 
        # print(support_set_y.shape)  #torch.Size([30, 2])
        # preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        preds = softmax_similarities.mm(support_set_y)
        # print(softmax_similarities)
        # print(preds)
        return preds        