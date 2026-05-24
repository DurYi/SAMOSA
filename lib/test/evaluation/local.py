from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.lasot_path = 'data/LaSOT/LaSOTTesting'
    settings.lasot_extension_subset_path = 'data/LaSOT/LaSOT_extension_subset'
    settings.nfs_path = 'data/NFS'
    settings.otb_path = 'data/OTB100'
    settings.uav_path = 'data/uav'
    settings.results_path = './output'
    settings.result_plot_path = './evaluation_results'
    settings.save_dir = './evaluation_results'

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'data/got10k_lmdb'
    settings.got10k_path = 'data/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = 'data/itb'
    settings.lasot_lmdb_path = 'data/lasot_lmdb'
    settings.network_path =  ''   # Where tracking networks are stored.
    settings.prj_dir = ''
    settings.segmentation_path = '/data1/os/test/segmentation_results'
    settings.tc128_path = 'data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'data/TrackingNet'
    settings.vot18_path = 'data/vot2018'
    settings.vot22_path = 'data/vot2022'
    settings.vot_path = 'data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

