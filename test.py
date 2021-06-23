import numpy as np

from deepvac import Deepvac, ClassifierReport, LOG

class DeepvacClsTest(Deepvac):
    def doTest(self):
        cls_report = ClassifierReport(ds_name="cls_dataset",total_num=self.config.test_loader.__len__(), cls_num=self.config.cls_num, threshold=0.99)
        for i, (inp, labels) in enumerate(self.config.test_loader):
            inp = inp.to(self.config.device)
            prediction = self.config.net(inp).cpu()
            prediction = np.argmax(prediction.data, axis=1)
            for gt, pred in zip(np.array(labels), np.array(prediction)):
                cls_report.add(gt,pred)
            if i%100 ==0:
                LOG.logI('Process {} samples. '.format(i))
        cls_report()
        self.config.sample = inp

if __name__ == '__main__':
    from config import config as deepvac_config
    cls = DeepvacClsTest(deepvac_config)
    cls()
