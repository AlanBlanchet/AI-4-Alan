from ai.configs.ai import AI


def test_load_custom_model():
    config = dict(_="CustomModel", name="ResNet")

    # Build model
    AI.from_config(config)

    assert True


def test_train_clf():
    config = dict(
        _="train",
        task=dict(
            _="Classification",
            input="image",
            labels="label",
            type="multiclass",
            params=dict(img_size=300),
        ),
        dataset=dict(
            _="HuggingFaceDataset",
            path="ILSVRC/imagenet-1k",
            split_train=dict(name="train"),
            split_val=dict(name="validation"),
        ),
        model=dict(_="CustomModel", name="ResNet"),
        trainer=dict(_="AITrainer"),
    )

    # Build trainer
    AI.from_config(config)

    assert True


def test_train_det():
    config = dict(
        task=dict(
            _="Detection",
            input="image",
            labels="objects.classes",
            bboxes="objects.boxes",
            type="multiclass",
            params=dict(
                img_size=300,
                anchors=dict(
                    feature_maps=[38, 19, 10, 5, 3, 1], num_anchors=[4, 6, 6, 6, 4, 4]
                ),
            ),
        ),
        dataset=dict(
            _="HuggingFaceDataset",
            path="fuliucansheng/pascal_voc",
            name="voc2007_main",
            split_train=dict(name="train"),
            split_val=dict(name="validation"),
        ),
        model=dict(_="CustomModel", name="SSD"),
        trainer=dict(_="AITrainer"),
    )

    # Build trainer
    AI.from_config(config)

    assert True
