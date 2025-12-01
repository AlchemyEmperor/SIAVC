import torch
from torch import nn

from all_model.models import C3DNet, SqueezeNet, MobileNet, ShuffleNet, MobileNetV2, ShuffleNetV2, ResNeXt, resnet


def generate_model(opt):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet',
                         'shufflenet', 'mobilenetv2', 'shufflenetv2']


    if opt.model == 'c3d':
        from all_model.models.C3DNet import get_fine_tuning_parameters
        model = C3DNet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'squeezenet':
        from all_model.models.SqueezeNet import get_fine_tuning_parameters
        model = SqueezeNet.get_model(
            version=opt.version,
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'shufflenet':
        from all_model.models.ShuffleNet import get_fine_tuning_parameters
        model = ShuffleNet.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)
    elif opt.model == 'shufflenetv2':
        from all_model.models.ShuffleNetV2 import get_fine_tuning_parameters
        model = ShuffleNetV2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenet':
        from all_model.models.MobileNet import get_fine_tuning_parameters
        model = MobileNet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenetv2':
        from all_model.models.MobileNetV2 import get_fine_tuning_parameters
        model = MobileNetV2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from all_model.models.ResNeXt import get_fine_tuning_parameters
        if opt.model_depth == 50:
            model = ResNeXt.resnext50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = ResNeXt.resnext101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = ResNeXt.resnext152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from all_model.models.resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)



    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
