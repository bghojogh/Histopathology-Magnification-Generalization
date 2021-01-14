from StainingUtils import StainingUtils

from ReinhardNormalizer import ReinhardNormalizer
from BreakHisOrganizer import BreakHisOrganizer


def main():
    # dataset_root = 'D:/Datasets/square_breast/'
    # reinhard_normalizer = ReinhardNormalizer.ReinhardNormalizer()    
    # normalization_ref_image = StainingUtils.read_image('reference.png')
    # path_extension = '*/*/*/*/*/*'
    # image_extension = '.png'
    # data_frame_name = 'for_test'
    # break_his_organizer = BreakHisOrganizer.BreakHisOrganizer(dataset_root,
    #                                                         reinhard_normalizer,
    #                                                         normalization_ref_image,
    #                                                         path_extension,
    #                                                         image_extension,
    #                                                         data_frame_name)
    
    # break_his_organizer.build_dataframe()
    # break_his_organizer.write_dataframe()
    # break_his_organizer.save_dataset(new_dataset_name= 'custom_dataset')
    
    # del break_his_organizer, reinhard_normalizer, normalization_ref_image
    
    dataset_root = 'D:/Datasets/masf_organized_breakhis_dataset/Best_after_normalization/'
    reinhard_normalizer = None
    normalization_ref_image = None
    path_extension = '*/*/*'
    image_extension = '.png'
    
    break_his_organizer_new_dataset = BreakHisOrganizer.BreakHisOrganizer(dataset_root,
                                                            reinhard_normalizer,
                                                            normalization_ref_image,
                                                            path_extension,
                                                            image_extension)
    
    break_his_organizer_new_dataset.build_organized_dataframe()
    break_his_organizer_new_dataset.write_dataframe(dataframe_name='organized_dataframe')
    
    
    
    


if __name__=='__main__':
    main()

