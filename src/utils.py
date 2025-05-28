from dataset import ChestXrayDataset

def train_test_val_splits(train_transform=None, val_transform=None):
    train = ChestXrayDataset(
        split_file='data/xray_images/train.txt',
        image_root_dir='data/xray_images/',
        bbox_csv='data/xray_images/BBox_List_2017.csv',
        metadata_csv='data/xray_images/preprocessed_metadata.csv',
        transform=train_transform
    )

    val = ChestXrayDataset(
        split_file='data/xray_images/val.txt',
        image_root_dir='data/xray_images/',
        bbox_csv='data/xray_images/BBox_List_2017.csv',
        metadata_csv='data/xray_images/preprocessed_metadata.csv',
        transform=val_transform

    )

    test = ChestXrayDataset(
        split_file='data/xray_images/test_list.txt',
        image_root_dir='data/xray_images/',
        bbox_csv='data/xray_images/BBox_List_2017.csv',
        metadata_csv='data/xray_images/preprocessed_metadata.csv',
        transform=val_transform
    )

    return {
        'train': train,
        'val': val,
        'test': test
    }

