def get_model(network):
    """
    加载指定的模型并将其移至CUDA设备

    参数:
        config: 配置对象，包含network等必要参数

    返回:
        已加载并移至CUDA的模型实例
    """
    # network = network
    # print(f"#----------正在加载模型: {network}----------#")
    #
    # # 基础参数
    # num_classes = getattr(config, 'num_classes', 1)
    # input_channels = getattr(config, 'input_channels', 3)

    #本人网络---------------------------------------------------------------------------------------

    #对比实验模型包括以下，cmd上注意名字
    #CMUNeXt
    #CPF_Net
    #egeunet
    #MALUNet
    #Rolling_Unet
    #SCSONet
    #SegNet
    #UCM_Net
    #UNet
    #UNext_S

    if network == 'MALUNet':
        from models.models.MALUNet import MALUNet
        model = MALUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                        bridge=True).cuda()

    if network == 'SegNet':
        from models.models.SegNet import SegNet
        model = SegNet(1)



    if network == 'UNet':
        from models.models.UNet import Unet
        model = Unet(num_classes=1)


    if network == 'UNext_S':
        from models.models.UNext_S import UNext_S
        model = UNext_S(num_classes=1, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                    embed_dims=[32, 64, 128, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    # attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1]).cuda()

    if network == 'UCM_Net':
        from models.models.UCM_Net import UCM_Net
        model = UCM_Net(num_classes=1)

    if network == 'CMUNeXt':
        from models.models.CMUNeXt import CMUNeXt
        model = CMUNeXt(num_classes=1).cuda()

    if network == 'SCSONet':
        from models.models.SCSONet import SCSONet
        model = SCSONet().cuda()
    if network == 'Rolling_Unet':
        from models.models.Rolling_Unet import Rolling_Unet_S
        model = Rolling_Unet_S(1,3).cuda()

    if network == 'egeunet':
        from models.models.egeunet import EGEUNet
        model = EGEUNet().cuda()

    if network == 'Rolling_Unet':
        from models.models.Rolling_Unet import Rolling_Unet_L
        model = Rolling_Unet_L(1,3).cuda()

    #消融模型包括以下，cmd上注意名字

    ## 下采样上的消融实验   CARecon
    if network == 'CAGSNet_GRGCM_EUCB_PSFE':
        from models.CARecon.CAGSNet_GRGCM_EUCB_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_OFU_PSFE':
        from models.CARecon.CAGSNet_GRGCM_EUCB_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_upsample_PSFE':
        from models.CARecon.CAGSNet_GRGCM_EUCB_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()

    ## GRGCM的消融实验
    if network == 'CAGSNet_CBAM_CARecon_PSFE':
        from models.GRGCM.CAGSNet_CBAM_CARecon_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_CoordAtt_CARecon_PSFE':
        from models.GRGCM.CAGSNet_CoordAtt_CARecon_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_DSAMBlock_CARecon_PSFE':
        from models.GRGCM.CAGSNet_DSAMBlock_CARecon_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_EMA_CARecon_PSFE':
        from models.GRGCM.CAGSNet_EMA_CARecon_PSFE import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()

    ##  PSFE的消融实验
    if network == 'CAGSNet_GRGCM_CARecon_CondConv2D':
        from models.PSFE.CAGSNet_GRGCM_CARecon_CondConv2D import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_CARecon_ConvBNPReLU':
        from models.PSFE.CAGSNet_GRGCM_CARecon_ConvBNPReLU import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_CARecon_DCNv2':
        from models.PSFE.CAGSNet_GRGCM_CARecon_DCNv2 import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_CARecon_GhostModule':
        from models.PSFE.CAGSNet_GRGCM_CARecon_GhostModule import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()
    if network == 'CAGSNet_GRGCM_CARecon_LDConv':
        from models.PSFE.CAGSNet_GRGCM_CARecon_LDConv import CAGSNet
        model = CAGSNet(classes=1, M=3, N=21,).cuda()

    return model