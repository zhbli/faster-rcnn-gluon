from mxnet import nd, gluon, image
batch_size=64
train_augs = [
    image.HorizontalFlipAug(.5),
    image.RandomCropAug((224,224))
]
test_augs = [
    image.CenterCropAug((224,224))
]
data_dir = '/home/zhbli/temp/dog_dataset'
train_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/train',
    transform=lambda X, y: transform(X, y, train_augs)
)
test_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/test',
    transform=lambda X, y: transform(X, y, test_augs)
)
def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')
def getDataset():
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)
    return train_data, test_data