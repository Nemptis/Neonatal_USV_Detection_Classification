import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

import numpy as np
import copy

import torch
from torch.utils.data import DataLoader

from classification_net import classification_net



class testable_dataset(ABC):
    def __init__(self):
        self.name = type(self).__name__
        self.num_classes = 0

    @abstractmethod
    def __len__(self):
        pass



colPal = sns.color_palette('pastel')


class net_tester():
    """Written by Arkenberg & Walther.
        Takes a model and up to three datasets (testset, trainset and validationset), passes all the data in these set through the model and then prints or plots all kinds of test results.
    """

    def __init__(self, model:classification_net, sets:list, data_loaders:list[DataLoader] = None):
        self.model = model
        #TODO: Allow datamodule, not just dataloader, to be passed
        if self.model is not None:
            self.model.eval()

        self.sets = sets
        if data_loaders is None:
            self.data_loaders = [DataLoader(S, batch_size=len(S), shuffle=False, collate_fn=collate_fn) for S in self.sets]
        else:
            self.data_loaders = data_loaders

        self.numSets = len(sets)
        self.totalSize = sum([len(S) for S in sets])

        sns.set_theme(font_scale=2)
        self.colors = sns.color_palette('pastel')


    def printSetsStats(self):
        print("----------- SETS STATS -----------")
        print("Total size of {:5} divided into:".format(self.totalSize))

        for S, D in zip(self.sets, self.data_loaders):
            self.printSetStats(S, self.totalSize, self.getlabelCounts(D))


    def plotSetsStats(self, fig:plt.Figure):
        sub_plots = fig.subplots(1, self.numSets)

        for S, D, ax in zip(self.sets, self.data_loaders, sub_plots):
            self.plotSetStats(ax, S, self.totalSize, self.getlabelCounts(D))


    def printModelStats(self, dataset_name="dsname", num_classes=4):
        print("---------- TEST RESULTS ----------")

        for S, D in zip(self.sets, self.data_loaders):
            stats = self.runModelTest(self.model, D)
            self.printTestStats(dataset_name, stats, num_classes)


    def printModelMosaic(self):
        print("---------- TEST MOSAIC ----------")
        for S, D in zip(self.sets, self.data_loaders):
            print('--- ' + "S.name" + ' ---')
            print(self.runModelMosaic(self.model, D))


    def plotModelStats(self, fig, num_classes=4):
        for k, S, D in zip(range(self.numSets), self.sets, self.data_loaders):
            mosaic = self.runModelMosaic(self.model, D)
            piePlot = fig.add_subplot(2, self.numSets, k+1)
            self.plotTestStats(piePlot, "S.name", mosaic, num_classes)
            mosaicPlot = fig.add_subplot(2, self.numSets, self.numSets + k+1)
            self.plotMosaic(mosaicPlot, "S.name", mosaic, num_classes)

        fig.suptitle('Results for ' + "cnn", fontdict={'weight': 'bold'})
        fig.tight_layout()



    def getlabelCounts(self, dataL: DataLoader):
        _, labels = self.get_all_data(dataL)
        possLabs = torch.eye(len(labels[0]))
        labelCounts = []
        for L in possLabs:
            s = torch.sum(torch.min(labels == L, dim=1)[0]).item()
            labelCounts.append(s)
        return labelCounts

    @staticmethod
    def get_all_data(data_loader: DataLoader):
        """
        Concatenate all batches from a DataLoader into a single batch.

        This function is useful when you want to process the entire dataset at once,
        instead of working with individual batches.

        Parameters:
        data_loader (DataLoader): The DataLoader to get the data from.

        Returns:
        tuple: A tuple containing two tensors. The first tensor contains all the
        features and the second tensor contains all the labels.
        """
        all_features = []
        all_labels = []

        for features, labels in data_loader:
            all_features.append(features)
            all_labels.append(labels)

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        return all_features, all_labels


    @staticmethod
    def runModelTest(model, dataL: DataLoader):
        #features, labels = net_tester.get_all_data(dataL)
        labels = []
        results = []
        for features, targets in dataL:
            with torch.no_grad():
                preds = model(features.to("cuda")).cpu()
                for i, pred in enumerate(preds):
                    #pred[0] += pred[-1]
                    results.append(pred)
                    labels.append(targets[i])

        results = torch.stack(results, dim=0)
        labels = torch.stack(labels, dim=0)

        #results = model(features).detach()

        possLabs = torch.eye(len(labels[0]))

        totalSizes = []
        numsRight = []
        convVals = []

        for L in possLabs:
            mask = torch.min(labels == L, dim=1)[0]
            relevantLabels = labels[mask]
            relevantResults = results[mask]
            rightOnes = torch.argmax(relevantResults, dim=1) == torch.argmax(
                relevantLabels, dim=1)

            numRight = torch.sum(rightOnes)

            if numRight >= 1:
                rightConfidences = torch.max(relevantResults[rightOnes], dim=1)[0]
            else:
                rightConfidences = torch.tensor([0.])
            maxLow1Pct = torch.max(torch.sort(rightConfidences)[0][:int(max(1, 0.01*len(rightConfidences)))])
            meanRightConf = torch.mean(rightConfidences)
            minRightConf = torch.min(rightConfidences)
            stdConf = torch.std(rightConfidences)


            totalSizes.append(len(relevantLabels))
            numsRight.append(numRight)
            convVals.append((meanRightConf, minRightConf, stdConf, maxLow1Pct))
        return totalSizes, numsRight, convVals


    @staticmethod
    def runModelMosaic(model, dataL: DataLoader):
        #features, labels = net_tester.get_all_data(dataL)

        labels = []
        results = []
        for features, targets in dataL:
            with torch.no_grad():
                preds = model(features.to("cuda")).cpu()
                for i, pred in enumerate(preds):
                    #pred[0] += pred[-1]
                    results.append(pred)
                    labels.append(targets[i].cpu())

        results = torch.stack(results, dim=0)
        labels = torch.stack(labels, dim=0)


        #results = model(features.cuda()).detach()
        possLabs = torch.eye(len(labels[0])).cpu()

        mosaic = []
        for rightL in possLabs:
            rightMask = torch.min(labels == rightL, dim=1)[0]
            relevantResults = results[rightMask]
            lableResults = []
            for testL in possLabs:
                testL = testL.expand_as(relevantResults)
                hits = torch.argmax(
                    relevantResults, dim=1) == torch.argmax(testL, dim=1)
                numHits = torch.sum(hits)
                lableResults.append(numHits)
            mosaic.append(lableResults)
        return np.array(mosaic).transpose()


    @staticmethod
    def printTestStats(name, stats, num_classes):
        totalSizes, numsRight, convVals = stats
        print('--- ' + name + ' ---')
        print("   In total  got {:4} out of {:4} right, {:7.2%}. Confidence: mean: {:7.2%}, std: {:7.2%}, max-low-1%: {:7.2%}".format(sum(numsRight), sum(
            totalSizes), sum(numsRight)/sum(totalSizes), np.mean(np.array(convVals)[:, 0]), np.mean(np.array(convVals)[:, 2]), np.mean(np.array(convVals)[:, 3])))
        for n in range(num_classes):
            l = str(n+1)
            print("    For '{:2}' got {:4} out of {:4} right, {:7.2%}. Confidence: mean: {:7.2%}, std: {:7.2%}, max-low-1%: {:7.2%}".format(
                l, numsRight[n], totalSizes[n], numsRight[n]/totalSizes[n], convVals[n][0], convVals[n][2], convVals[n][3]))


    @staticmethod
    def printSetStats(set, total_size, label_counts):
        print('--- ' + set.name + ' ---')
        print(f"  Size: {len(set):4}, {len(set)/total_size:7.2%}")
        for n in range(set.num_classes):
            l = str(n+1)
            print(f"    Count of  category '{l:2}': {label_counts[n]:4}, {label_counts[n]/len(set):7.2%}")


    @staticmethod
    def plotSetStats(sub_plot, set, total_size, label_counts):
        possibleLabels = [str(n+1) for n in range(set.num_classes)]

        sub_plot.set_title(set.name, fontdict={'fontweight': 'bold'})
        sub_plot.pie(label_counts, labels=possibleLabels, wedgeprops=dict(
            width=0.6), colors=colPal, autopct=make_autopct(label_counts), pctdistance=0.7)
        middleText = f"{len(set)/total_size:.0%}\n({len(set)})"
        sub_plot.text(0., 0., middleText, horizontalalignment='center',
                     verticalalignment='center', fontdict={'fontweight': 'bold'})


    @staticmethod
    def plotTestStats(subPlot, name, mosaicData, num_classes):
        M = mosaicData
        right = int(np.sum(np.identity(len(M)) * M))
        total = np.sum(M)
        rightFrac = right / total
        subPlot.set_title(name + ' | Right predictions',
                          fontdict={'fontweight': 'bold'})
        subPlot.pie((rightFrac, 1-rightFrac),
                    wedgeprops=dict(width=0.3), colors=colPal)
        middleText = "{:.2%}\n({} of {})".format(rightFrac, right, total)
        subPlot.text(0., 0., middleText, horizontalalignment='center',
                     verticalalignment='center', fontdict={'fontweight': 'bold'})


    @staticmethod
    def plotMosaic(subPlot, name, mosaicData, num_classes):
        possibleLabels = [str(n+1) for n in range(num_classes)]

        colMap = copy.copy(sns.color_palette(
            "blend:#A1C9F4,#8DE5A1", as_cmap=True))
        colMap.set_under(color='white')
        subPlot.set_title(name + ' | Mosaic', fontdict={'fontweight': 'bold'})
        big = not len(possibleLabels) > 10
        plot = sns.heatmap(mosaicData, annot=big, fmt="d", cbar=False, xticklabels=possibleLabels,
                           yticklabels=possibleLabels, cmap=colMap, vmin=0.1, ax=subPlot, linewidths=.5, linecolor='#010203')
        plot.set_xlabel("Category")
        plot.set_ylabel("Prediction")


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct


